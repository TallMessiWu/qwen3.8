#!/usr/bin/env python3
"""不起模型、8 卡验 ALLGATHER+EP 的专家路由是否自洽——判据闭式，几分钟出结果。

验的是一个恒等式，与具体权重、具体故障都无关，所以换 EP 尺寸、换专家数、
rebase 了 vllm-ascend 的路由代码之后都值得重跑。

判据：把 dispatch 的输出原样喂回 combine（等于让每个专家做恒等映射），
combine 用 `npu_moe_token_unpermute(probs=topk_weights)` 按权重加权求和，于是
单个 rank 拿到的是「本 rank 名下那些专家」的权重和乘 x。跨 EP group all_reduce
之后，每个 token 的系数就是它全部 topk 权重之和；把权重归一化成 1，
**期望输出就是 x 自己**。

不需要参考实现、不需要专家权重、不需要跑 MLP。而 expert_map 只要漏掉或重复认领
任何一个专家，这个等式立刻不成立：

    if expert_map is not None:                 # EP=1 才进
        mask = expert_map[topk_ids] != -1
        topk_weights = topk_weights * mask      # 不属于本 rank 的权重置零

两个阶段，都很便宜：
  map        纯逻辑检查，不跑 MoE：把八个 rank 的 expert_map 拼起来，验每个全局专家
             恰好被一个 rank 认领一次。秒级。
  roundtrip  上面那个恒等式，按 token 数扫描。

用法（服务器容器内）：
    torchrun --nproc_per_node=8 scripts/bench/test_moe_ep_routing.py --smoke
    torchrun --nproc_per_node=8 scripts/bench/test_moe_ep_routing.py

判读：
  map 阶段 RED        → expert_map 本身就错（漏认领/重复认领），直接锁定，
                        `determine_expert_map(ep_size, ep_rank, num_experts)` 去查。
  roundtrip 某尺寸 RED → 路由在那个尺寸上丢/错 token。看 RED 的分布：
                        全尺寸都红 → 与 token 数无关，是 EP 切分或掩码的问题；
                        只有大尺寸红 → 查 npu_moe_init_routing 的 active_num
                        （= num_tokens * top_k）有没有上限。
  全 GREEN            → ALLGATHER+EP 的路由在 eager 下是干净的。

⚠️ 覆盖边界（决定了「全 GREEN」能推出什么）：本脚本走 QuantType.NONE 且全程 eager。
所以全 GREEN 只能排除「路由逻辑本身写错了」，不能替量化路径或图模式背书。
2026-09-07 用它查 397B 长 prompt 空输出时就是这个结果：路由全 GREEN，真因在别处——
MoE 的 TP 规约判据被 dynamo 烘成了捕获期的常量（见 AGENTS.md「当前状态」）。
这是它的典型用法：**便宜地砍掉一整片嫌疑区，而不是指认真凶。**

选通信方式的分支与芯片代次有关（A5 上 EP8 命中 `world_size <= num_experts_per_tok`
就走不到 ALLTOALL），所以换芯片之后先确认这一步实际选的是哪条路再判读结果。
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
        # 387/405 是真机实测的通过/失败一对，423 是复现空输出的那个长度。
        # 容量 400 两侧各取几个，另加小尺寸看是否与规模无关。
        default=[1, 8, 64, 387, 400, 405, 423, 1024],
        help="要扫的 token 数",
    )
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--experts", type=int, default=512, help="全局专家数（EP8 下每卡 64，与真机同量级）")
    p.add_argument("--topk", type=int, default=10, help="真机 num_experts_per_tok=10")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--stage", choices=["map", "roundtrip", "capture", "all"], default="all")
    p.add_argument("--capture-tokens", type=int, default=400, help="capture 阶段捕获的图尺寸（模拟真机 capture_model）")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--smoke", action="store_true", help="只建环境并打印配置")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def log(rank: int, msg: str) -> None:
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

    parallel_config = ParallelConfig(tensor_parallel_size=world_size, enable_expert_parallel=True)
    vllm_config = VllmConfig(parallel_config=parallel_config)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=(
                f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:{os.environ.get('MASTER_PORT', '29500')}"
            ),
            local_rank=local_rank,
            backend="hccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world_size, backend="hccl")

        from vllm.distributed.parallel_state import get_ep_group
        from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        ep_rank = ep_group.rank_in_group
        num_local_experts, expert_map, _ = determine_expert_map(ep_size, ep_rank, args.experts)
        expert_map = expert_map.to(device) if expert_map is not None else None

        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        log(
            rank,
            f"[env] world={world_size} ep_size={ep_size} experts={args.experts} "
            f"local_experts={num_local_experts} topk={args.topk} hidden={args.hidden} dtype={args.dtype}",
        )
        if args.smoke:
            log(rank, "[smoke] 环境正常，EP group 与 expert_map 都建起来了。去掉 --smoke 跑实测。")
            dist.barrier()
            dist.destroy_process_group()
            return 0

        failures = 0
        stages = ["map", "roundtrip", "capture"] if args.stage == "all" else [args.stage]

        if "map" in stages:
            failures += _check_expert_map(
                rank=rank,
                ep_group=ep_group,
                expert_map=expert_map,
                num_experts=args.experts,
                num_local_experts=num_local_experts,
                device=device,
            )

        if "capture" in stages:
            failures += _run_capture_probe(
                args=args, rank=rank, expert_map=expert_map,
                num_local_experts=num_local_experts, device=device, dtype=dtype,
            )

        if "roundtrip" in stages:
            failures += _run_roundtrips(
                args=args,
                rank=rank,
                ep_group=ep_group,
                expert_map=expert_map,
                num_local_experts=num_local_experts,
                device=device,
                dtype=dtype,
            )

        verdict = torch.tensor([failures], dtype=torch.int32, device=device)
        dist.all_reduce(verdict)
        total = int(verdict.item())
        log(rank, "")
        if total == 0:
            log(rank, "GREEN：ALLGATHER+EP 的专家路由在 eager 下是干净的。")
            log(rank, "       397B 那个只在 GRAPH=1 出现的空输出与这段无关，")
            log(rank, "       回去查 aclgraph 本身改变了什么（本脚本全程 eager，覆盖不到）。")
        else:
            log(rank, f"RED：{total} 个用例对不上（跨 rank 合计）。路由本身就有问题，与图无关。")

        dist.barrier()
        dist.destroy_process_group()
        return 1 if total else 0


def _check_expert_map(*, rank, ep_group, expert_map, num_experts, num_local_experts, device) -> int:
    """纯逻辑检查：每个全局专家应当恰好被一个 rank 认领一次。

    不跑 MoE、不碰算子，秒级。放在最前面是因为它一旦红，后面所有数值实验都不必做了：
    expert_map 错了，`mask = expert_map[topk_ids] != -1` 的结果自然全错。
    """
    import torch.distributed as dist

    claimed = (
        (expert_map != -1).to(torch.int32)
        if expert_map is not None
        else torch.ones(num_experts, dtype=torch.int32, device=device)
    )
    total = claimed.clone()
    dist.all_reduce(total, group=ep_group.device_group)

    missing = int((total == 0).sum())
    duplicated = int((total > 1).sum())
    local_count = int(claimed.sum())

    ok = missing == 0 and duplicated == 0 and local_count == num_local_experts
    log(rank, "\n=== stage: map ===")
    log(
        rank,
        f"global_experts={num_experts} local={local_count}(期望 {num_local_experts}) "
        f"未被认领={missing} 被多认领={duplicated}  {'GREEN' if ok else 'RED'}",
    )
    if not ok and rank == 0:
        bad = (total != 1).nonzero().flatten()
        print(f"    异常专家 id 前若干={bad[:32].tolist()}", flush=True)
    return 0 if ok else 1



def _build_dispatcher_and_inputs(*, args, expert_map, num_local_experts, device, dtype, num_tokens, seed_off=0):
    """把 dispatcher 与一次 dispatch 所需的输入凑齐，capture / roundtrip 两个阶段共用。"""
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import build_quant_params
    from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
    from vllm_ascend.quantization.quant_type import QuantType

    dispatcher = TokenDispatcherWithAllGather(
        top_k=args.topk, num_experts=args.experts, num_local_experts=num_local_experts
    )
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
    x, w, ids = _make_inputs(
        num_tokens=num_tokens, hidden=args.hidden, topk=args.topk,
        num_experts=args.experts, dtype=dtype, device=device, seed=args.seed + seed_off,
    )

    def run():
        out = dispatcher.token_dispatch(
            token_dispatch_input=MoETokenDispatchInput(
                hidden_states=x, topk_weights=w, topk_ids=ids, routing=routing, quant=quant
            )
        )
        return dispatcher.token_combine(
            hidden_states=out.hidden_states, combine_metadata=out.combine_metadata
        )

    return run, x


def _run_capture_probe(*, args, rank, expert_map, num_local_experts, device, dtype) -> int:
    """本脚本里唯一直接对着 GRAPH 这个变量的实验。

    真机上 EP、MTP 一直开着，唯一在变的就是 aclgraph 开不开。所以这里不比"算得对不对"，
    而比**同一段 eager 代码在图捕获前后会不会给出不同结果**：

        1. eager 跑一遍大尺寸，记下 before
        2. 捕获一张小尺寸的图（模拟 GRAPH=1 启动时的 capture_model，那时全部 <= 容量）
        3. replay 几次
        4. eager 再跑一遍同样的大尺寸，记下 after

    before 与 after 必须逐位相同——输入没变、走的也是同一段 eager 代码。一旦不同，
    就说明"捕获这个动作本身"污染了后续 eager 执行（内存池、通信域、算子工作区之类），
    那正是 GRAPH=1 坏而 GRAPH=0 好的形状，且完全不需要起模型就复现了。

    ALLGATHER 的 dispatch/combine 内部不含集合通信（all_gather 在 prepare 里，
    这里够不到），所以捕获它不涉及 HCCL，可行性比捕获整个模型高得多。
    捕获失败不影响其余阶段的结论，按 SKIP 记、不算 RED。
    """
    log(rank, "\n=== stage: capture ===")
    probe_tokens = max(args.tokens)
    try:
        run_big, _ = _build_dispatcher_and_inputs(
            args=args, expert_map=expert_map, num_local_experts=num_local_experts,
            device=device, dtype=dtype, num_tokens=probe_tokens,
        )
        torch.npu.synchronize()
        before = run_big().float().clone()
        torch.npu.synchronize()

        run_small, _ = _build_dispatcher_and_inputs(
            args=args, expert_map=expert_map, num_local_experts=num_local_experts,
            device=device, dtype=dtype, num_tokens=args.capture_tokens, seed_off=31,
        )
        # 预热必须在独立 stream 上，否则捕获会把预热的残留一起录进去（CUDA graph
        # 的老规矩，NPU 同理）。
        side = torch.npu.Stream()
        side.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(side):
            for _ in range(3):
                run_small()
        torch.npu.current_stream().wait_stream(side)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run_small()
        for _ in range(3):
            graph.replay()
        torch.npu.synchronize()

        after = run_big().float().clone()
        torch.npu.synchronize()

        max_abs = float((after - before).abs().max())
        ok = max_abs == 0.0
        log(rank, f"capture_size={args.capture_tokens} probe_size={probe_tokens} "
                  f"max|after-before|={max_abs:.6e}  {'GREEN' if ok else 'RED'}")
        if not ok:
            log(rank, "  RED：图捕获改变了之后 eager 的结果——不起模型复现了 GRAPH 变量的影响。")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 - 捕获失败不该带倒其余阶段
        log(rank, f"SKIP：这个环境下捕获跑不起来（{type(exc).__name__}: {exc}）")
        if rank == 0:
            traceback.print_exc()
        return 0


def _run_roundtrips(*, args, rank, ep_group, expert_map, num_local_experts, device, dtype) -> int:
    import torch.distributed as dist
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import build_quant_params
    from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
    from vllm_ascend.quantization.quant_type import QuantType

    dispatcher = TokenDispatcherWithAllGather(
        top_k=args.topk,
        num_experts=args.experts,
        num_local_experts=num_local_experts,
    )
    # 这两个 dataclass 的字段全部必填，没有默认值。
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

    # bf16 往返本身有精度损耗，阈值放宽到不会把它误判成错位；真正的丢/错 token
    # 是 O(1) 量级的偏差，宽阈值一样拦得住。
    tol = 5e-2 if dtype is torch.bfloat16 else 1e-4

    log(rank, "\n=== stage: roundtrip ===")
    log(rank, f"{'tokens':>8} {'max_abs_diff':>14} {'rel':>10}  verdict")
    failures = 0
    for num_tokens in args.tokens:
        try:
            x, topk_weights, topk_ids = _make_inputs(
                num_tokens=num_tokens,
                hidden=args.hidden,
                topk=args.topk,
                num_experts=args.experts,
                dtype=dtype,
                device=device,
                seed=args.seed,
            )
            out = dispatcher.token_dispatch(
                token_dispatch_input=MoETokenDispatchInput(
                    hidden_states=x,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    routing=routing,
                    quant=quant,
                )
            )
            # 专家什么都不做，收到什么还回什么，于是 combine 之后只剩按权重加权求和。
            combined = dispatcher.token_combine(
                hidden_states=out.hidden_states,
                combine_metadata=out.combine_metadata,
            )
            # 每个 rank 只算了自己名下专家的那部分贡献，跨 EP 求和才凑齐一个 token
            # 的全部 topk 权重。这一步正是真机上 finalize(reduce_results=True) 做的事。
            summed = combined.float()
            dist.all_reduce(summed, group=ep_group.device_group)

            diff = (summed - x.float()).abs()
            max_abs = float(diff.max())
            denom = float(x.float().abs().max()) or 1.0
            ok = max_abs <= tol
            failures += 0 if ok else 1
            log(rank, f"{num_tokens:>8} {max_abs:>14.6f} {max_abs / denom:>10.2e}  {'GREEN' if ok else 'RED'}")
            if args.verbose and not ok and rank == 0:
                per_row = diff.max(dim=-1).values
                bad = (per_row > tol).nonzero().flatten()
                print(f"    偏差行数={bad.numel()}/{num_tokens} 前若干行号={bad[:16].tolist()}", flush=True)
        except Exception:  # noqa: BLE001 - 单个用例炸掉不该中断整轮扫描
            failures += 1
            log(rank, f"{num_tokens:>8} {'EXCEPTION':>14} {'-':>10}  RED")
            if rank == 0:
                traceback.print_exc()
        # 形状类错误通常八个 rank 同时抛；对齐一次让下个用例从干净状态开始。
        dist.barrier()
    return failures


def _make_inputs(*, num_tokens, hidden, topk, num_experts, dtype, device, seed):
    """造各 rank 完全一致的输入。

    ALLGATHER 语义下 prepare 会把 tokens 聚齐，dispatcher 收到的是完整序列，
    所以八个 rank 必须拿到同一份 x 和同一份 topk——各卡自行 randn 会让
    all_reduce 之后的对账无从谈起。固定 generator 在 CPU 上生成再搬过去。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + num_tokens)
    x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float32)
    logits = torch.randn(num_tokens, num_experts, generator=gen, dtype=torch.float32)

    weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    # 归一化成每 token 权重和为 1：恒等专家下 all_reduce 的期望结果才正好是 x。
    weights = weights / weights.sum(dim=-1, keepdim=True)

    return (
        x.to(device=device, dtype=dtype),
        weights.to(device=device, dtype=torch.float32),
        ids.to(device=device, dtype=torch.int32),
    )


if __name__ == "__main__":
    sys.exit(main())
