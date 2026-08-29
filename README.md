# qwen3.8

在昇腾 NPU 上调试和测试 Qwen3.8 的工作区，服务栈为 `vllm` + `vllm-ascend`。

## 一键初始化仓库

在 Linux 服务器执行下面整段命令。它会克隆或更新主仓，初始化 `vllm` 和
`vllm-ascend/main`，配置只读的官方 `upstream`，并创建常驻的
`vllm-ascend/upstream-main` 与
`vllm-ascend/junlin-qfa` worktree。

```bash
set -euo pipefail

###############################################################################
# 请按需将 /home/hajimi 替换为自己的工作目录，后续路径会自动跟随。          #
###############################################################################
QWEN38_ROOT=/home/hajimi/qwen3.8
VLLM_ASCEND_ROOT="${QWEN38_ROOT}/vllm-ascend"
VLLM_ASCEND_MAIN="${VLLM_ASCEND_ROOT}/main"
VLLM_ASCEND_UPSTREAM_MAIN="${VLLM_ASCEND_ROOT}/upstream-main"
VLLM_ASCEND_QFA_BRANCH=junlin-qfa
VLLM_ASCEND_QFA="${VLLM_ASCEND_ROOT}/${VLLM_ASCEND_QFA_BRANCH//\//-}"
VLLM_ASCEND_UPSTREAM_URL=https://github.com/vllm-project/vllm-ascend.git

mkdir -p "$(dirname "${QWEN38_ROOT}")"

if git -C "${QWEN38_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${QWEN38_ROOT}" switch main
else
  git clone https://github.com/TallMessiWu/qwen3.8.git "${QWEN38_ROOT}"
fi

git -C "${QWEN38_ROOT}" config submodule.recurse true
git -C "${QWEN38_ROOT}" config fetch.recurseSubmodules on-demand
git -C "${QWEN38_ROOT}" pull --ff-only --recurse-submodules
git -C "${QWEN38_ROOT}" submodule sync --recursive
git -C "${QWEN38_ROOT}" submodule update --init --recursive

git -C "${VLLM_ASCEND_MAIN}" fetch origin
if git -C "${VLLM_ASCEND_MAIN}" show-ref --verify --quiet refs/heads/main; then
  git -C "${VLLM_ASCEND_MAIN}" switch main
else
  git -C "${VLLM_ASCEND_MAIN}" switch -c main --track origin/main
fi
git -C "${VLLM_ASCEND_MAIN}" branch --set-upstream-to=origin/main main
git -C "${VLLM_ASCEND_MAIN}" config submodule.recurse true
git -C "${VLLM_ASCEND_MAIN}" config fetch.recurseSubmodules on-demand
git -C "${VLLM_ASCEND_MAIN}" pull --ff-only --recurse-submodules

if git -C "${VLLM_ASCEND_MAIN}" remote get-url upstream >/dev/null 2>&1; then
  git -C "${VLLM_ASCEND_MAIN}" remote set-url upstream "${VLLM_ASCEND_UPSTREAM_URL}"
else
  git -C "${VLLM_ASCEND_MAIN}" remote add upstream "${VLLM_ASCEND_UPSTREAM_URL}"
fi
git -C "${VLLM_ASCEND_MAIN}" fetch upstream --prune --recurse-submodules=on-demand

if git -C "${VLLM_ASCEND_UPSTREAM_MAIN}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${VLLM_ASCEND_UPSTREAM_MAIN}" switch upstream-main
elif git -C "${VLLM_ASCEND_MAIN}" show-ref --verify --quiet refs/heads/upstream-main; then
  git -C "${VLLM_ASCEND_MAIN}" worktree add "${VLLM_ASCEND_UPSTREAM_MAIN}" upstream-main
else
  git -C "${VLLM_ASCEND_MAIN}" worktree add \
    -b upstream-main "${VLLM_ASCEND_UPSTREAM_MAIN}" upstream/main
fi

git -C "${VLLM_ASCEND_UPSTREAM_MAIN}" branch \
  --set-upstream-to=upstream/main upstream-main
git -C "${VLLM_ASCEND_UPSTREAM_MAIN}" pull --ff-only --recurse-submodules

if git -C "${VLLM_ASCEND_QFA}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${VLLM_ASCEND_QFA}" switch "${VLLM_ASCEND_QFA_BRANCH}"
elif git -C "${VLLM_ASCEND_MAIN}" show-ref --verify --quiet \
  "refs/heads/${VLLM_ASCEND_QFA_BRANCH}"; then
  git -C "${VLLM_ASCEND_MAIN}" worktree add \
    "${VLLM_ASCEND_QFA}" "${VLLM_ASCEND_QFA_BRANCH}"
else
  git -C "${VLLM_ASCEND_MAIN}" worktree add --track \
    -b "${VLLM_ASCEND_QFA_BRANCH}" "${VLLM_ASCEND_QFA}" \
    "origin/${VLLM_ASCEND_QFA_BRANCH}"
fi

git -C "${VLLM_ASCEND_QFA}" branch \
  --set-upstream-to="origin/${VLLM_ASCEND_QFA_BRANCH}" \
  "${VLLM_ASCEND_QFA_BRANCH}"
git -C "${VLLM_ASCEND_QFA}" pull --ff-only --recurse-submodules

for worktree in \
  "${VLLM_ASCEND_MAIN}" \
  "${VLLM_ASCEND_UPSTREAM_MAIN}" \
  "${VLLM_ASCEND_QFA}"; do
  git -C "${worktree}" submodule sync --recursive
  git -C "${worktree}" submodule update --init --recursive
done

git -C "${QWEN38_ROOT}" submodule status --recursive
git -C "${VLLM_ASCEND_MAIN}" worktree list
git -C "${VLLM_ASCEND_MAIN}" status -sb
git -C "${VLLM_ASCEND_UPSTREAM_MAIN}" status -sb
git -C "${VLLM_ASCEND_QFA}" status -sb
git -C "${VLLM_ASCEND_MAIN}" submodule status --recursive
git -C "${VLLM_ASCEND_UPSTREAM_MAIN}" submodule status --recursive
git -C "${VLLM_ASCEND_QFA}" submodule status --recursive
```

三个分支状态应分别显示 `main...origin/main`、
`upstream-main...upstream/main` 和
`junlin-qfa...origin/junlin-qfa`。
`upstream` 只用于 fetch/pull，不要向它 push。

## 创建 A5 容器

### 前置条件

- 在宿主机执行脚本，不要在已有容器内执行。
- 已安装 Docker，并确保最终选择的镜像在本机存在。未指定 `--image` 或
  `IMAGE` 环境变量时，才要求加载默认镜像
  `vllm-ascend:dev-26.1.0.day20260817-A5-py311-openEuler24.03-lts-aarch64`；
  也可以通过这两种方式使用其他镜像。
- 宿主机有 `/dev/davinci0` 至 `/dev/davinci7`、`/dev/davinci_manager`、
  `/dev/hisi_hdc`、`/dev/ummu` 和 `/dev/uburma`。
- 宿主机已准备 `/usr/local/Ascend/driver`、`/usr/local/Ascend/firmware`、
  `/usr/local/sbin/npu-smi`、`/usr/local/dcmi`、`/etc/hccl_rootinfo.json`、
  `/etc/hixlep` 和 `/usr/lib64`。
- `/home`、`/mnt`、`/data` 会按相同绝对路径挂载进容器。自定义项目路径、
  checkout、代理或安装脚本时，路径必须位于容器可见的挂载目录中。
- 安装脚本默认位于
  `/home/hajimi/qwen3.8/scripts/install-vllm-ascend.sh`，可通过
  `--install-script` 覆盖。
- editable 安装使用的 checkout 默认为
  `/home/hajimi/qwen3.8/vllm-ascend/junlin-qfa`，
  可通过 `--vllm-ascend-repo` 覆盖。如果传入
  `--vllm-ascend-version` 从镜像源安装指定包版本，则不要求本地 checkout
  存在。
- 代理脚本默认为 `/home/hajimi/proxy.sh`，可通过 `--proxy-file` 覆盖；选定
  的代理文件不存在时会直接跳过。
- 交互 shell 的默认目录和回退目录可分别通过 `--shell-workdir` 和
  `--shell-fallback-dir` 覆盖；Python 路径可通过 `--python-bin` 覆盖。
- 默认容器名为 `hajimi-vllm`，可通过 `--container-name` 或
  `CONTAINER_NAME` 环境变量覆盖。脚本不会替换最终选定的同名容器。

使用默认镜像时，可用下面的命令确认镜像和设备：

```bash
docker image inspect \
  vllm-ascend:dev-26.1.0.day20260817-A5-py311-openEuler24.03-lts-aarch64
ls /dev/davinci{0..7}
```

### 使用默认配置

```bash
cd /home/hajimi/qwen3.8
bash scripts/create-container.sh
```

脚本会创建 privileged、host network、host PID 的 8 卡容器，挂载宿主机
目录，配置 root 的 `.bashrc`，然后调用 `install-vllm-ascend.sh`。默认不会
解析 vLLM 依赖，而是以 `--no-deps` 强制安装 `vllm==0.27.1`；vLLM-Ascend
从默认 checkout 以 editable 模式安装。

创建完成后进入容器：

```bash
docker exec -it hajimi-vllm bash
```

### 自定义路径和包版本

两个脚本同时支持 `--参数 值` 和 `--参数=值`。完整列表：

```bash
bash scripts/create-container.sh --help
bash scripts/install-vllm-ascend.sh --help
```

常用参数：

| 参数 | 默认值或行为 |
| --- | --- |
| `--image` | 默认 vendor A5 镜像 |
| `--container-name` | `hajimi-vllm` |
| `--install-script` | `/home/hajimi/qwen3.8/scripts/install-vllm-ascend.sh` |
| `--vllm-ascend-repo` | `/home/hajimi/qwen3.8/vllm-ascend/junlin-qfa` |
| `--proxy-file` | `/home/hajimi/proxy.sh`，不存在时跳过 |
| `--shell-workdir` | `/home/hajimi/qwen3.8/scripts` |
| `--python-bin` | `python3` |
| `--vllm-version` | `0.27.1`，可覆盖为其他版本 |
| `--vllm-ascend-version` | 不传时 editable 安装 checkout；传入时安装指定包版本 |
| `--pip-index-url` | `https://mirrors.aliyun.com/pypi/simple` |
| `--pytorch-index-url` | `https://download.pytorch.org/whl/cpu` |

示例：

```bash
bash scripts/create-container.sh \
  --image vllm-ascend:custom-a5 \
  --container-name qwen38-test \
  --install-script /mnt/qwen3.8/scripts/install-vllm-ascend.sh \
  --vllm-ascend-repo /mnt/qwen3.8/vllm-ascend/junlin-qfa \
  --proxy-file /mnt/proxy.sh \
  --shell-workdir /mnt/qwen3.8/scripts \
  --shell-fallback-dir /mnt \
  --python-bin /usr/bin/python3 \
  --vllm-version 0.27.1
```

如果传入 `--vllm-ascend-version VERSION`，安装器会从指定 Python 镜像源
安装该版本，不再 editable 安装 checkout。原有的 `IMAGE` 和
`CONTAINER_NAME` 环境变量覆盖方式仍然可用，显式命令行参数优先。

## 完成后的结构与操作说明

### 宿主机目录结构

使用默认路径完成一键初始化后，关键目录结构如下。任务过程中额外创建的
其他功能 worktree 不属于这个基础结构。

```text
/home/hajimi/
├── proxy.sh                         # 可选的宿主机代理脚本，不属于本仓
└── qwen3.8/
    ├── .git/                        # 主仓 Git 元数据
    ├── .gitmodules                  # vllm 与 vllm-ascend/main 的 submodule 配置
    ├── AGENTS.md                    # agent 开发与仓库约束
    ├── CLAUDE.md -> AGENTS.md       # 共享规则的符号链接
    ├── README.md
    ├── .agents/skills/              # agent 技能真身
    ├── .claude/skills -> .agents/skills/
    ├── skills-lock.json
    ├── pics/                        # 架构页面使用的图片
    ├── qwen3.8-architecture.html
    ├── scripts/
    │   ├── README.md                # 运行时辅助和本地测试说明
    │   ├── create-container.sh      # 宿主机容器创建入口
    │   ├── install-vllm-ascend.sh   # 容器内 Python 包安装入口
    │   ├── 27B.sh                   # 单机服务入口
    │   ├── 2.4T-0.sh ... 2.4T-3.sh # 四机服务入口
    │   ├── serve_qwen3.8_2.4t_4node.sh # 四机共用服务启动脚本
    │   ├── runtime/                 # 裁层 checkpoint 运行时过滤器
    │   └── tests/                   # 脚本静态回归测试
    ├── vllm/                        # 上游 vLLM submodule，只读参考
    └── vllm-ascend/
        ├── main/                    # 个人 fork submodule，跟踪 origin/main
        ├── upstream-main/           # 本地 worktree，跟踪 upstream/main
        └── junlin-qfa/              # 默认容器使用，跟踪 origin/junlin-qfa
```

完成容器创建后，容器不会再复制一份源码，而是通过 bind mount 看到宿主机
目录。默认关键结构如下：

```text
hajimi-vllm 容器
├── /home/                            # bind mount：宿主机 /home
│   └── hajimi/qwen3.8/               # 与上面的宿主机 checkout 是同一份文件
├── /mnt/                             # bind mount：宿主机 /mnt
├── /data/                            # bind mount：宿主机 /data
├── /usr/local/Ascend/driver/         # bind mount：宿主机驱动
├── /usr/local/Ascend/firmware/       # bind mount：宿主机固件
├── /root/.bashrc                     # 追加代理加载和默认工作目录配置
└── Python 环境
    ├── vllm                          # 默认安装 0.27.1，或安装指定版本
    └── vllm-ascend                   # editable checkout，或指定的包版本
```

### 一键初始化仓库做了什么

1. 创建 `/home/hajimi`，在 `/home/hajimi/qwen3.8` 不存在时克隆本仓；已存在
   时切换到 `main` 并执行 fast-forward 更新。
2. 开启主仓的递归 submodule 配置，随后同步并初始化 `vllm` 和
   `vllm-ascend/main` 及其递归 submodule。
3. 将 `vllm-ascend/main` 切到本地 `main` 分支，并确保它跟踪个人 fork 的
   `origin/main`。
4. 为 vLLM-Ascend 配置官方只读远端 `upstream`，获取最新的
   `upstream/main`。该远端只用于 fetch/pull。
5. 创建或更新 `vllm-ascend/upstream-main` worktree，让它跟踪
   `upstream/main`，与个人 fork 的 `main` checkout 分开维护。
6. 创建或更新 `vllm-ascend/junlin-qfa` worktree，让它跟踪个人
   fork 的 `junlin-qfa` 分支（QFA 算子接入，基于 upstream/main）。
7. 在三个 vLLM-Ascend worktree 中同步递归 submodule，最后打印 submodule、
   worktree 和分支状态，供人工确认初始化结果。

### 创建容器和安装包做了什么

1. `create-container.sh` 解析默认值和命令行覆盖，检查 Docker、目标镜像，
   并拒绝覆盖已有的同名容器。
2. 使用 root 用户创建 detached 容器，启用 host network、host PID、
   privileged 模式和 2 GiB shared memory。
3. 将 8 张 NPU、NPU 管理设备、驱动、固件、HCCL 配置以及 `/home`、
   `/mnt`、`/data` 等宿主机路径挂载进容器。
4. 向容器的 `/root/.bashrc` 写入代理加载和工作目录逻辑。代理文件不存在时
   跳过；工作目录不存在时进入配置的回退目录。
5. 在容器内运行选定的 `install-vllm-ascend.sh`，并转发 checkout、代理、
   Python、包版本和 Python 镜像源参数。
6. 安装器先以 `--no-deps` 强制安装 vLLM。默认版本为 `0.27.1`，可通过
   `--vllm-version` 覆盖。
7. 如果传入 `--vllm-ascend-version`，从 Python 镜像源安装指定版本；否则先
   同步并初始化选定 checkout 的递归 submodule，再清理 `csrc/output` 和
   `csrc/build_out`，最后执行 editable 安装。
8. 安装结束后打印 vLLM、vLLM-Ascend 版本和 `vllm_ascend` 的实际导入路径。
   安装失败时容器保留运行状态，便于进入容器检查构建环境。
