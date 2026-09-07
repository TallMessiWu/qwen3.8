#!/usr/bin/env python3
"""不起模型、8 卡复现 MoE All2All 的 token 错位——判据是闭式的。

397B 的空输出只在 EP + aclgraph + 单步 token 数 > mc2_tokens_capacity 时出现，
超过容量那一步从 MC2 切到 ALLTOALL。但至今所有结论都是从"服务吐 EOS"倒推的，
**ALLTOALL 这条路本身算得对不对，从来没有被直接量过**。这个脚本就量这一件事，
不加载任何权重、不起 vLLM 服务，几分钟跑完。

判据：把 dispatch 的输出原样喂回 combine（等价于让每个专家做恒等映射），
combine 的结果必须等于 x 各 topk 权重之和的加权，即 x * sum_k(w_k)；
把 topk 权重归一化成 sum_k(w_k)=1，期望输出就是 x 自己。于是不需要参考实现、
不需要专家权重、不需要跑 MLP——token 一旦在 all2all 里错位、丢失或被 padding
污染，这个恒等式立刻不成立，且偏差是 O(1) 而不是精度级的。

两层各测各的：
  dispatcher  只测 TokenDispatcherWithAll2AllV 的 dispatch→combine 往返。
  full        再套上 AlltoAllCommImpl 的 prepare/finalize，覆盖那个
              pad_size = tp_size - num_tokens 的笔误（batch > tp 时为负、
              padding 从不发生，tensor_split 于是切出不等长的 8 份）。

用法（服务器容器内）：
    torchrun --nproc_per_node=8 scripts/debug/probe_moe_alltoall.py --smoke
    torchrun --nproc_per_node=8 scripts/debug/probe_moe_alltoall.py
    torchrun --nproc_per_node=8 scripts/debug/probe_moe_alltoall.py --tokens 405 --verbose

判读——先看 dispatcher 层，它最干净：
  dispatcher 有 RED  → all2all 的 dispatch/combine 本身就把 token 搞错了，与图无关，
                       收工去修它。这是最好的结果，一个不用起模型的最小复现。
  dispatcher 全 GREEN→ 分发回收是对的，往下看 full 层。

full 层的 RED 要按 token 数 %8 分开读，别一见红就下结论：
  只有 %8 != 0 的尺寸红（9/100/387/405/423），%8 == 0 的全绿（8/64/400/1024）
                     → 就是 pad_size = tp_size - num_tokens 那个笔误：num_tokens 一
                       超过 tp 它就是负数，padding 从不发生，tensor_split 切出不等长
                       的 8 份，finalize 的 dist.all_gather 要求等长，多半直接抛
                       "Tensors must be the same size"。
                       ⚠️ 这条红**不等于**已经解释了真机现象：真机 ≤400 走的是 MC2 的
                       prepare（pad 到 padded_num_tokens），根本不碰这段；而真机 405
                       是静默吐 EOS，不是抛异常。两者对不上，说明真机那条路上这段被
                       跳过了——最可能是 replace_allreduce=True 整段短路，正好是
                       [moe-prep] 插桩里那一列要回答的。
  %8 == 0 的尺寸也红 → 与整除性无关，问题在 splits 或通信缓冲区容量，看偏差行号分布。

两层都全 GREEN     → ALLTOALL 的数值通路是干净的，问题在它之外。回到 aclgraph 那条
                     线：capture 阶段（启动时 dummy run 跑遍 4..400，全部走 MC2）给
                     之后 eager 的 ALLTOALL 留了副作用。下一步用 [moe-fp] 指纹对拍
                     GRAPH=1/0 找分叉层。

注意本脚本走 QuantType.NONE。真机 MoE 是 w4a4 MXFP4，量化分支在 all_to_all
之前还会把 scale 再单独换一次；这里全 GREEN 不能替量化那条分支背书。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        # 覆盖三类边界：能否被 tp=8 整除、真机实测的 387 通过 / 405 失败那一对、
        # 以及远超容量的大尺寸。1/7/9 专门盯 pad_size 笔误。
        default=[1, 7, 8, 9, 64, 100, 387, 400, 405, 423, 1024],
        help="要扫的 token 数",
    )
    p.add_argument("--hidden", type=int, default=1024, help="hidden_size")
    p.add_argument("--experts", type=int, default=512, help="全局专家数（默认 512，EP8 下每卡 64，与真机同量级）")
    p.add_argument("--topk", type=int, default=10, help="每 token 选几个专家（真机是 10）")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--stage", choices=["dispatcher", "full", "both"], default="both")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true", help="只建环境并打印配置，不做数值测试")
    p.add_argument("--verbose", action="store_true", help="RED 时额外打印每行的偏差分布")
    return p.parse_args()


def log(rank: int, msg: str) -> None:
    """只让 rank 0 说话，八份重复日志没法读。"""
    if rank == 0:
        print(msg, flush=True)


def main() -> int:
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        print("必须用 torchrun 起多卡：torchrun --nproc_per_node=8 " + sys.argv[0], file=sys.stderr)
        return 2

    import torch_npu  # noqa: F401  # 注册 npu 后端，必须在建 device 之前

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    import torch.distributed as dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    # tp=world_size + EP 打开，对齐真机的单机 TP8+EP。model_config 留空：
    # initialize_model_parallel 只在 model_config 为 None 或 is_moe 时才建 EP group，
    # 留空正好命中前者，也省掉拉一份真模型配置。
    parallel_config = ParallelConfig(tensor_parallel_size=world_size, enable_expert_parallel=True)
    vllm_config = VllmConfig(parallel_config=parallel_config)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}",
            local_rank=local_rank,
            backend="hccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size, backend="hccl")

        from vllm.distributed.parallel_state import get_ep_group, get_tp_group
        from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank_in_group
        num_local_experts, expert_map, _ = determine_expert_map(ep_size, ep_rank, args.experts)
        if expert_map is not None:
            expert_map = expert_map.to(device)

        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        log(
            rank,
            f"[env] world={world_size} ep_size={ep_size} experts={args.experts} "
            f"local_experts={num_local_experts} topk={args.topk} hidden={args.hidden} dtype={args.dtype}",
        )
        if args.smoke:
            log(rank, "[smoke] 环境建起来了，EP group / expert_map 都正常。去掉 --smoke 跑数值测试。")
            dist.barrier()
            dist.destroy_process_group()
            return 0

        from vllm_ascend.ops.fused_moe.dataclass.moe_quant import build_quant_params
        from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
        from vllm_ascend.ops.fused_moe.moe_comm_method import AlltoAllCommImpl
        from vllm_ascend.quantization.quant_type import QuantType

        moe_config = _build_moe_config(
            num_experts=args.experts,
            num_local_experts=num_local_experts,
            topk=args.topk,
            hidden=args.hidden,
            ep_size=ep_size,
            tp_size=world_size,
            dtype=dtype,
            device=device,
            tp_group=get_tp_group(),
        )
        comm = AlltoAllCommImpl(moe_config)
        dispatcher = comm.token_dispatcher
        # 这两个 dataclass 的字段全是必填，没有默认值——本机用 tests/ut 的 torch_npu
        # mock 实测过一遍才敢这么写，少一个就是 TypeError。
        quant = build_quant_params(
            quant_type=QuantType.NONE,
            comm_quant_mode=None,
            mxfp_act_quant_type=None,
            mxfp_weight_quant_type=None,
            mxfp_scale_dtype=None,
            mxfp_per_token_scale_dtype=None,
            mxfp_use_bf16=None,
            is_per_channel_weight=False,
        )
        routing = MoeRouterInput(
            expert_map=expert_map,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
            pertoken_scale=None,
        )

        # bf16 的往返本身就有精度损耗，阈值必须宽到不会把它误判成错位；
        # 而真正的 token 错位是 O(1) 量级，宽阈值一样拦得住。
        tol = 5e-2 if dtype is torch.bfloat16 else 1e-4

        stages = ["dispatcher", "full"] if args.stage == "both" else [args.stage]
        failures = 0
        for stage in stages:
            log(rank, f"\n=== stage: {stage} ===")
            log(rank, f"{'tokens':>8} {'max_abs_diff':>14} {'rel':>10}  verdict")
            for num_tokens in args.tokens:
                try:
                    max_abs, rel = _run_case(
                        stage=stage,
                        num_tokens=num_tokens,
                        hidden=args.hidden,
                        topk=args.topk,
                        num_experts=args.experts,
                        dtype=dtype,
                        device=device,
                        seed=args.seed,
                        comm=comm,
                        dispatcher=dispatcher,
                        routing=routing,
                        quant=quant,
                        verbose=args.verbose,
                        rank=rank,
                    )
                except Exception:  # noqa: BLE001 - 单个用例炸掉不该中断整轮扫描
                    failures += 1
                    log(rank, f"{num_tokens:>8} {'EXCEPTION':>14} {'-':>10}  RED")
                    if rank == 0:
                        traceback.print_exc()
                    continue

                ok = max_abs <= tol
                failures += 0 if ok else 1
                log(rank, f"{num_tokens:>8} {max_abs:>14.6f} {rel:>10.2e}  {'GREEN' if ok else 'RED'}")
                # 每个用例后对齐一次。形状类错误通常八个 rank 同时抛，barrier 能让
                # 下一个用例从干净状态开始；万一只有部分 rank 抛，这里会卡住——那也
                # 比带着失步的集合通信继续跑、把后面的结果全污染要好。
                dist.barrier()

        # 各 rank 自己判自己的，但结论要合并：任一 rank 红就是红。
        verdict = torch.tensor([failures], dtype=torch.int32, device=device)
        dist.all_reduce(verdict)
        total = int(verdict.item())
        log(rank, "")
        if total == 0:
            log(rank, "GREEN：All2All 的 dispatch/combine 与 prepare/finalize 都没搞错 token。")
            log(rank, "       问题不在 All2All 的数值通路，回到 aclgraph 那条线（capture 副作用）。")
        else:
            log(rank, f"RED：{total} 个用例对不上（跨 rank 合计）。All2All 本身就错，与图无关。")
            log(rank, "     看 RED 的 token 数分布：只在非 8 的倍数上红 → pad_size 笔误；")
            log(rank, "     只在大尺寸上红 → splits 或通信缓冲区容量。")

        dist.barrier()
        dist.destroy_process_group()
        return 1 if total else 0


def _build_moe_config(
    *,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    hidden: int,
    ep_size: int,
    tp_size: int,
    dtype: torch.dtype,
    device: torch.device,
    tp_group,
):
    """搭一个够用的 moe_config。

    真造 FusedMoEConfig 要连带 FusedMoEParallelConfig 和一串 __post_init__ 断言，
    而这里被读到的只有下面这几个字段。未设的属性一律返回 None，让漏配当场暴露成
    AttributeError-free 的 None 而不是 MagicMock——MagicMock 会让 `if cfg.xxx`
    恒真，把配置错误伪装成正常路径。
    """

    class _MoEConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return None

    return _MoEConfig(
        num_experts=num_experts,
        original_num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        in_dtype=dtype,
        device=device,
        ep_size=ep_size,
        tp_size=tp_size,
        dp_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
        # finalize 拿 moe_config.tp_group.device_group 做 all_gather，少了它 full
        # 阶段直接崩在 None.device_group 上。
        tp_group=tp_group,
    )


def _select_topk(router_logits, topk):
    """从 router_logits 取 topk，并把权重归一化成每 token 和为 1。

    归一化是这个探针成立的前提：恒等 MLP 下 combine 做的是按权重求和，
    权重和为 1 时期望输出才正好等于输入。
    """
    weights, ids = torch.topk(torch.softmax(router_logits.float(), dim=-1), topk, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(torch.float32), ids.to(torch.int32)


def _make_inputs(*, num_tokens, hidden, topk, num_experts, dtype, device, seed):
    """造输入。seed 由调用方决定要不要掺 rank。

    在 CPU 上用固定 generator 生成再搬过去，不用 NPU RNG：要让各 rank 拿到"说好
    一样"或"说好不一样"的数据，就不能依赖各卡自己的随机状态。

    full 阶段各 rank 必须拿同一个 x（TP 下 hidden_states 是复制的，finalize 之后
    要跟它对账）；dispatcher 阶段则故意让各 rank 不同，好让 all2all 的 splits 不
    对称——各卡数据一样时 splits 也一样，错位反而可能对称地抵消掉。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + num_tokens)
    x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float32)
    logits = torch.randn(num_tokens, num_experts, generator=gen, dtype=torch.float32)

    weights, ids = _select_topk(logits, topk)
    return (
        x.to(device=device, dtype=dtype),
        logits.to(device=device, dtype=dtype),
        weights.to(device),
        ids.to(device),
    )


def _roundtrip(dispatcher, hidden_states, topk_weights, topk_ids, routing, quant):
    """dispatch → （恒等 MLP）→ combine。"""
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput

    out = dispatcher.token_dispatch(
        token_dispatch_input=MoETokenDispatchInput(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            routing=routing,
            quant=quant,
        )
    )
    # 专家什么都不做，把收到的 token 原样还回去，于是 combine 之后只剩加权求和。
    return dispatcher.token_combine(
        hidden_states=out.hidden_states,
        combine_metadata=out.combine_metadata,
    )


def _run_case(
    *,
    stage,
    num_tokens,
    hidden,
    topk,
    num_experts,
    dtype,
    device,
    seed,
    comm,
    dispatcher,
    routing,
    quant,
    verbose,
    rank,
):
    from vllm_ascend.quantization.quant_type import QuantType

    x, router_logits, topk_weights, topk_ids = _make_inputs(
        num_tokens=num_tokens,
        hidden=hidden,
        topk=topk,
        num_experts=num_experts,
        dtype=dtype,
        device=device,
        # dispatcher 阶段各 rank 掺 rank 进 seed 拿到不同数据；full 阶段各 rank
        # 必须同源，finalize 之后才能跟同一个 x 对账。
        seed=seed + (rank * 100003 if stage == "dispatcher" else 0),
    )

    if stage == "dispatcher":
        got = _roundtrip(dispatcher, x, topk_weights, topk_ids, routing, quant)
        expected = x
    else:
        # 走完整通路：prepare 切分 → 往返 → finalize 拼回。照 routed_experts 的真实
        # 顺序，topk 在 prepare 之后、用被切分过的 router_logits 现算——行数必须跟
        # 切分后的 hidden 对齐，拿切分前的 topk 直接喂进去行数就错了。
        prep = comm.prepare(hidden_states=x, router_logits=router_logits, quant_type=QuantType.NONE)
        h = prep.hidden_states
        w_shard, ids_shard = _select_topk(prep.router_logits, topk)
        combined = _roundtrip(dispatcher, h, w_shard, ids_shard, routing, quant)
        got = comm.finalize(
            hidden_states=combined,
            reduce_results=False,
            padded_hidden_states_shape=prep.padded_hidden_states_shape,
        )
        # finalize 把各 rank 的分片 all_gather 回来，期望仍是完整的 x。
        expected = x
        if got.shape != expected.shape:
            # 形状都对不上就不必比数值了，这本身就是错位的铁证。
            raise RuntimeError(f"finalize 形状 {tuple(got.shape)} != 输入 {tuple(expected.shape)}")

    diff = (got.float() - expected.float()).abs()
    max_abs = float(diff.max())
    denom = float(expected.float().abs().max()) or 1.0
    if verbose and max_abs > 0 and rank == 0:
        per_row = diff.max(dim=-1).values
        bad = (per_row > 1e-2).nonzero().flatten()
        print(f"    偏差行数={bad.numel()}/{num_tokens} 前若干行号={bad[:16].tolist()}", flush=True)
    return max_abs, max_abs / denom


if __name__ == "__main__":
    sys.exit(main())
