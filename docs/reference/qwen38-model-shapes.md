# Qwen3.8 / Qwen3.5 各尺寸的模型形状

真机跑的这几个模型在 HuggingFace 上都有公开权重仓，`config.json` 直接 `curl` 就能拿，
**不需要权重文件**：

```bash
curl -sfL https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/config.json
```

字段在 `text_config` 下。2026-09-07 实测：

| 仓库 | 层数 | full attn | NQ | NKV | head_dim | hidden |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | 64 | 16 | 24 | 4 | 256 | 5120 |
| `Qwen/Qwen3.8-Flash-Next` | 48 | 12 | 24 | 2 | 256 | 2560 |
| `Qwen/Qwen3.8-2.4T-A95B` | 92 | 23 | 64 | 4 | 256 | 8192 |
| `Qwen/Qwen3.5-397B-A17B` | 60 | 15 | 32 | 2 | 256 | 4096 |
| `Qwen/Qwen3.5-35B-A3B` | 40 | 10 | 16 | 2 | 256 | 2048 |

全系 `full_attention_interval: 4`——3 个 linear attention 配 1 个 full attention，
所以 `full = num_hidden_layers // 4`，只有 full 那些层跑 QFA / FIA。

⚠️ **Qwen3.8 线只有 27B / Flash-Next / 2.4T-A95B 三个尺寸，没有 35B。**
`scripts/` 里 `--model 35b` 那套形状（16/2/256、block 512）对应的是 Qwen3.5-35B-A3B。

## `model_type` 与 `architectures`

量化适配要靠它们选 `packed_modules_mapping`，只能从 `config.json` 拿：

- 397B：顶层 `model_type=qwen3_5_moe`、`text_config.model_type=qwen3_5_moe_text`、
  `architectures=["Qwen3_5MoeForConditionalGeneration"]`——是个多模态 wrapper，
  所以还带 vision tower。
- 397B 其余关键字段：`mtp_num_hidden_layers: 1`、`num_experts: 512`、
  `shared_expert_intermediate_size: 1024`、`mlp_only_layers: []`。

## 注意

**「本机没有 NPU、没有权重」不等于拿不到模型结构。** 需要层数、头数、`layer_types`、
专家数这类信息时先查 HF config，不要因为 `/mnt/share/weight/` 在远端就把它们当作
未知量留在脚本里。

权重侧的格式事实（融合专家布局、MX block 大小、检查点形状对照）见
`../cases/2.4t-weight-history.md`。
