# scripts/local —— 本机脚本（不是服务器脚本）

`scripts/` 下其余内容都是交给服务器（昇腾 NPU）执行的；这个目录相反，只在**本机**跑：
x86_64 + NVIDIA GPU，没有 NPU、没有 CANN。目的是在把改动推上真机之前，先用静态检查
和纯 PyTorch 模拟拦掉一批错误。

## 环境

```bash
bash scripts/local/setup-devenv.sh
source .venv/bin/activate
```

环境刻意对齐真机容器，而不是对齐本地 `vllm/` submodule 的工作区：

| | 真机（容器） | 本机（.venv） |
| --- | --- | --- |
| Python | 3.11 | 3.11 |
| vllm | 0.28.0（pip 装） | `vllm/` submodule 当前的 checkout（`VLLM_TARGET_DEVICE=empty` 装，不编译 kernel） |
| torch | 2.10.0 + torch-npu | 2.10.0（PyPI 默认 wheel，带 CUDA） |
| vllm-ascend | editable `junlin-c8-mxfp-16278` | editable `junlin-c8-mxfp-16278`（同一个 worktree） |
| torch_npu | 真 NPU | `tests/ut/conftest.py` 自动注入的 MagicMock |

`vllm/` 跟着 vllm-ascend 的 `.github/vllm-main-verified.commit` 走（当前 main
`84030bbe3d`）。真机是 pip 装的 0.28.0，两者用起来差别不大，不必纠结；真要对某个确切
版本，覆盖 `VLLM_REF` + `VLLM_WORKTREE` 从 submodule 临时派生一个只读 worktree
（用完 `git -C vllm worktree remove` 删掉）。换版本重建会清掉 `.venv`，两套环境没法并存。

**2026-09-15 从 0.27.1 升上来。** 近期 upstream/main 的 vllm-ascend 只支持 vllm 恰好
0.28.0（101 处 `vllm_version_is("0.28.0")` 守卫）或 main `84030bbe3d`；0.27.1 会在
`patch_kv_cache_utils.py` 就因 `_get_packed_kv_cache_groups` 缺失而 import 失败。
`junlin-c8-mxfp` / `junlin-qfa` 这两条老分支还钉着 `ba07e4a48f`，要验它们得先把 `vllm/`
切过去再重建 venv。

## 能拦住什么

| 能拦 | 拦不住 |
| --- | --- |
| patch 目标被改名/挪走（import 期就炸） | 昇腾算子的真实数值行为 |
| patch 替换实现与 vllm 原函数签名漂移 | NPU 内存/图模式/多卡通信 |
| `tests/ut` 里 CPU 那部分的逻辑回归 | 性能（FD 编译期开关那类问题） |
| shape / 切分轴 / tiling 参数的算术错误 | CANN 版本相关的行为差异 |
| import 期错误、语法错误、类型错误 | 101 处 `vllm_version_is("0.28.0")` 守卫里的真机分支 |

## 用法

```bash
# patch 目标体检：目标还在不在、参数列表有没有漂
python scripts/local/check_patch_targets.py
python scripts/local/check_patch_targets.py --worktree vllm-ascend/main -v

# CPU 单测 + 跟已知基线比对（自动排除 NPU 专属目录）
bash scripts/local/run_cpu_ut.sh
bash scripts/local/run_cpu_ut.sh tests/ut/ops        # 只跑一部分
UPDATE_BASELINE=1 bash scripts/local/run_cpu_ut.sh   # 确认过之后刷新基线
```

`run_cpu_ut.sh` 的退出码：0 是与基线一致，1 是比基线新增了失败**或者 pytest 压根没
跑起来**——pytest 退出码不是 0/1（2 中断、3 内部错误、4 用法/conftest 错误、5 没收集到
用例）时直接判 RED 并跳过基线比对，否则"一条 FAILED 都没有"会被读成基线里的用例都好了。
带路径参数只跑一部分时没有基线可比，退出码就是 pytest 自己的——路径打错、用例红了都会
非 0，不再一律 0。

`tests/ut/<module>/a2|a3_2|310p/` 这些子目录是 NPU 专属的，本机跑不了，也不该跑——
路由规则见 vllm-ascend 的 `.github/workflows/scripts/test_config.yaml`。

### 当前实测（2026-09-15，vllm `84030bbe3d` + `junlin-c8-mxfp-16278`）

`run_cpu_ut.sh`：**4374 passed / 1 failed / 37 skipped**，约 30 秒。基线已刷成这 1 条
（`test_ascend_config.py::test_config_modules_do_not_load_vllm_config`）——原来 4 条里
有 3 条在新 vllm 上自己好了。

`check_patch_targets.py`：84 处 patch 里 **GREEN 60 / AMBER 13 / NEW 3 / SKIP 8 / RED 0**。

- **AMBER**（13 条）是参数列表对不上，多数是有意适配，但每条都该能说出为什么。
  当前这批：`FusedMoEFactory` ×2、`DeepseekV2MLAAttention.__init__` 少
  `non_causal_multi_token_decode`、`preprocess_mamba` 少 `align_ctx`、
  `Qwen3NextAttention.forward` 多 `output`、`apply_sampling_constraints` 多 `top_k`、
  `rejection_sample` 换入参、`build_attn_metadata`、`DFlashCudaGraphManager`、
  `SpeculatorCudaGraphManager`、`InputBatch` ×2 多
  `seq_lens_np`/`attn_state`/`is_dummy`、`DeepseekV32IndexerCache.get_attn_backend`
  的 `self`→`_self`。
- **NEW**（3 条）全在 `patch_mamba_utils.py:546-548`（`prepare_mamba_copy_by_layer` /
  `do_mamba_copy_block_for_layer` / `finish_mamba_copy_by_layer`）。NEW 不等于安全：
  monkeypatch 赋值一定会把属性创建出来，所以"目标已改名、patch 往废名字上赋值而静默
  失效"看起来跟"有意新增属性"一模一样。工具靠「patch 前 vllm 上有没有这个名字」把它
  拎出来，但是哪一种得人来判。
- **SKIP**（8 条）全是 `patch_v2/patch_triton.py`，承载它的模块本机没加载：
  `patch/worker/__init__.py` 有 `if HAS_TRITON:` 守卫，而 conftest 把 `triton.runtime`
  换成了 MagicMock，本机 `HAS_TRITON` 恒为 False。这批只有真机能判。

0.27.1 时代的对照（2026-09-03，`junlin-c8-mxfp`）：patch 体检 75 处 GREEN 55 /
AMBER 12 / SKIP 8 / RED 0，单测 2703 passed / 4 failed / 12 skipped。

## 已知短板

- **`vllm_version_is("0.28.0")` 在本机恒为 False**，也就是那 101 处守卫本机走的全是
  另一条分支。真机 pip 装的是 0.28.0，守卫全 True；本机装的是 `vllm/` 的 main
  checkout，自报 `0.28.1rc1.dev676+g84030bbe3`，`Version(...) == Version("0.28.0")`
  不成立，守卫全 False。想用 `VLLM_VERSION=0.28.0` 强行对齐也不行——0.28.0 分支的
  `attention_v1.py` 要 import `vllm.model_executor.layers.attention.pcp`，而 main 已
  把它挪到 `vllm.v1.attention.ops.pcp`，pytest 直接以退出码 4 死在 collection
  （`utils.py:_vllm_empty_device_matches_release` 的注释就是拿 PCP 的位置区分两条 lane
  的）。要验 0.28.0 那条分支，只能把 `vllm/` 切到 v0.28.0 再重建 venv。
- **子进程拿不到 mock**。conftest 是往当前进程的 `sys.modules` 里塞 mock，
  测试里 `subprocess.run([sys.executable, "-c", ...])` 起的子进程一律看不到，
  一 import `torch_npu` 就炸。
- **triton 相关的一切都判不了**（见上面 SKIP）。
- **torch 是 2.10.0 但没有 torch_npu**，凡是靠 `torch.npu.*` 真实行为的路径都是空转。
- **数值模拟只能验算法，不能验算子**。5080 上跑的是你自己写的 PyTorch 参考实现，
  跟昇腾算子的真实数值行为（累加顺序、量化舍入）不是一回事。
