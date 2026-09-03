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
| vllm | 0.27.1 | 0.27.1（同一份源码，`VLLM_TARGET_DEVICE=empty` 装，不编译 kernel） |
| torch | 2.10.0 + torch-npu | 2.10.0（PyPI 默认 wheel，带 CUDA） |
| vllm-ascend | editable `junlin-qfa` | editable `junlin-qfa`（同一个 worktree） |
| torch_npu | 真 NPU | `tests/ut/conftest.py` 自动注入的 MagicMock |

vllm 源码取自 `.dev/vllm-0.27.1`——从 `vllm/` submodule 派生的只读 worktree。
`vllm/` 本身的 checkout 不受影响（它现在停在 v0.28.1rc0，比真机领先 1300+ commit，
直接拿它当参考会得出跟真机对不上的结论）。

## 能拦住什么

| 能拦 | 拦不住 |
| --- | --- |
| patch 目标被改名/挪走（import 期就炸） | 昇腾算子的真实数值行为 |
| patch 替换实现与 vllm 原函数签名漂移 | NPU 内存/图模式/多卡通信 |
| `tests/ut` 里 CPU 那部分的逻辑回归 | 性能（FD 编译期开关那类问题） |
| shape / 切分轴 / tiling 参数的算术错误 | CANN 版本相关的行为差异 |
| import 期错误、语法错误、类型错误 | |

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

`tests/ut/<module>/a2|a3_2|310p/` 这些子目录是 NPU 专属的，本机跑不了，也不该跑——
路由规则见 vllm-ascend 的 `.github/workflows/scripts/test_config.yaml`。

### 建立环境时的实测基线（2026-09-03）

`check_patch_targets.py`：75 处 patch 里 GREEN 55 / AMBER 12 / SKIP 8 / RED 0。

- **AMBER** 是参数列表对不上，多数是有意适配（`rejection_sample` 换了入参、
  `InputBatch` 多了 `seq_lens_np`/`attn_state`/`is_dummy` 等），但每条都该能说出为什么。
- **SKIP** 是承载 patch 的模块本机没加载：`patch/worker/__init__.py` 有
  `if HAS_TRITON:` 守卫，而 conftest 把 `triton.runtime` 换成了 MagicMock，
  本机 `HAS_TRITON` 恒为 False。这批只有真机能判。
- **NEW** 不等于安全：monkeypatch 赋值一定会把属性创建出来，所以"目标已改名、
  patch 往废名字上赋值而静默失效"看起来跟"有意新增属性"一模一样。工具靠
  「patch 前 vllm 上有没有这个名字」把它拎出来，但是哪一种得人来判。

`run_cpu_ut.sh`：2703 passed / 4 failed / 12 skipped，约 20 秒。4 条失败见
[ut_baseline.txt](ut_baseline.txt)，其中只有一条已确定是本机假象（测试起了子进程，
子进程不继承 conftest 的 mock），另外三条要跟真机 CI 比对才能定性。

## 已知短板

- **子进程拿不到 mock**。conftest 是往当前进程的 `sys.modules` 里塞 mock，
  测试里 `subprocess.run([sys.executable, "-c", ...])` 起的子进程一律看不到，
  一 import `torch_npu` 就炸。
- **triton 相关的一切都判不了**（见上面 SKIP）。
- **torch 是 2.10.0 但没有 torch_npu**，凡是靠 `torch.npu.*` 真实行为的路径都是空转。
- **数值模拟只能验算法，不能验算子**。5080 上跑的是你自己写的 PyTorch 参考实现，
  跟昇腾算子的真实数值行为（累加顺序、量化舍入）不是一回事。
