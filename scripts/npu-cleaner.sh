#!/bin/bash

# 检查是否传入参数
if [ $# -eq 0 ]; then
  echo "用法: $0 <NPU_ID_1> [NPU_ID_2] ... | all"
  exit 1
fi

TARGETS="$@"

# 精准提取目标卡号的去重 PID
pids=$(npu-smi info | awk -F'|' -v targets="$TARGETS" '
BEGIN {
    split(targets, t_arr, " ");
    for (i in t_arr) { target_map[t_arr[i]] = 1; }
}
{
    # 预清洗第2列(卡号)和第3列(PID)
    c2 = $2; gsub(/^[ \t]+|[ \t]+$/, "", c2);
    c3 = $3; gsub(/^[ \t]+|[ \t]+$/, "", c3);

    # 特征双检锁：仅当【第2列是纯数字】且【第3列是纯数字】时，才认定为进程行！
    if (c2 ~ /^[0-9]+$/ && c3 ~ /^[0-9]+$/) {
        if ("all" in target_map || c2 in target_map) {
            print c3
        }
    }
}
')

if [ -z "$pids" ]; then
  echo "NPU ($TARGETS) 上暂无运行进程。"
  exit 0
fi

unique_pids=$(echo "$pids" | sort -u)

# ================= 核心变更：无等待直接强杀 =================
echo "锁定进程: $unique_pids"
echo "正在下达 SIGKILL (9) 强制斩杀指令..."

# 使用 xargs 并发传递 PID，实现毫秒级瞬间清场
echo "$unique_pids" | xargs -r kill -9 2>/dev/null

echo "清理完成！"
