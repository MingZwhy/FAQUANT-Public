# FAQUANT-Public

[English](README.md) | 简体中文

本仓库提供 HiF4 W4A4 量化的复现代码与聚合结果，覆盖 Qwen3-8B（PTQ、QAD、
QAD 后层保护）、Qwen3-32B（带层保护的 PTQ）和 Wan2.2-T2V-A14B（DiT 全部
Linear 与注意力 QK/PV，在完整 VBench 上评测）。

本仓库有意不包含模型权重、检查点、数据集、原始评测样本、模型生成结果和
私有实验历史。

## 主要结果

### Qwen3-8B

| 评测 | BF16 | 候选模型 | 损失 |
|---|---:|---:|---:|
| MMLU 0-shot，14,042 道题 | 72.9597% | **71.9627%** | **0.9970 pp** |
| LongBench v1，21 任务宏平均 | 49.4352 | **48.3529** | **1.0824 pp** |

LongBench 评测中，完整提示词的预填充（prefill）和第一个生成 token 使用量化
模型；后续解码使用 BF16，并复用量化预填充产生的 KV cache。

### Qwen3-32B

只做 PTQ，不做 QAD。BF16：MMLU 80.8859%，LongBench 44.9224。

| HiF4 scale | Linear 输入旋转 | BF16 层 | MMLU 损失 | LongBench 损失（95% 区间） |
|---|---|---|---:|---:|
| 三级 | HiSQ1024，每层 7 次在线旋转 | 35, 41, 42, 45, 46 | 0.7620 pp | −0.20 [−0.75, +0.33] |
| 三级 | `rbhn_r64`，每层 4 次在线分块 Hadamard | 39, 40, 42, 45, 46 | 0.8332 pp | −0.01 [−0.53, +0.52] |
| 两级 | HiSQ1024 | 39, 42, 43, 45, 46, 63 | 1.0896 pp | −0.10 [−0.62, +0.47] |
| 两级 | `rbh_r64`，每层 4 次在线分块 Hadamard | 41, 42, 43, 44, 45, 46 | 1.0113 pp | −0.20 [−0.77, +0.36] |

### Wan2.2-T2V-A14B

完整 VBench 标准 prompt 集，每条 prompt 一个样本，与 BF16 按 prompt 和种子
配对。DiT 矩阵乘法 FLOPs 的 99.99% 以上以 4 比特计算。

| 配置 | VBench Total | 损失（95% 区间） |
|---|---:|---:|
| BF16 | 82.20 | — |
| HiF4 Linear + QK/PV | 78.48 | 3.72 [2.96, 4.60] |
| A：再加 post-RoPE Q/K Hadamard + P-Reordering | 81.60 | 0.60 [−0.03, 1.32] |
| A + GPTQ | 81.66 | 0.54 [−0.12, 1.19] |

详细说明见[方法](docs/METHOD.md)、[复现步骤](docs/REPRODUCE.md)、
[结果](docs/RESULTS.md)、[Qwen3-32B](docs/QWEN3_32B.md)、
[旋转](docs/ROTATION.md)和 [Wan2.2](docs/WAN22.md)（英文）。

## Qwen3-8B 最终候选模型

- 基座模型：`Qwen/Qwen3-8B`
- 量化格式：HiF4 W4A4，group size 64
- PTQ：GPTQ + HiSQ1024 + value-head rotation + Smooth-QK
- 注意力：post-RoPE Q/K Hadamard + tiled HiF4 QK/PV + P-Reordering
- QAD：250 步精确注意力训练 + 125 步部署注意力训练
- QAD 后保护：将第 16、17、18 层恢复为 BF16
- 量化投影层：231 / 252
- QK MXFP8 层：无

评测使用的检查点由
[`results/final_metrics.json`](results/final_metrics.json) 中的 SHA-256
值唯一标识。模型权重单独发布在
[Hugging Face](https://huggingface.co/MingZwhy/qwen3-8b-hif4-l16-17-18)。

## 安装

推荐使用 Python 3.11。请先安装与 CUDA 版本匹配的 PyTorch，然后执行：

```bash
python -m pip install -e ".[test]"
```

仓库提供了一个 80 GB 显存的示例环境：

```bash
bash scripts/setup_env.sh
```

FlashAttention 仅作为环境配置和 BF16 基准评测的可选依赖。Qwen 的量化
QK/PV 模拟器使用 PyTorch 实现。Wan2.2 工作流还需要 Wan2.2、
HiSQRot4_Wan2.2 和 VBench 的代码，见 [Wan2.2](docs/WAN22.md#reproduce)。

## 复现流程

```text
Qwen3-8B BF16
  -> HiF4 PTQ 初始化
  -> Smooth-QK 校准
  -> QAD：精确注意力 250 步 + 部署注意力 125 步
  -> 将第 16/17/18 层的 BF16 保护写入检查点
  -> MMLU 与 LongBench 评测

Qwen3-32B BF16
  -> 构建时排除保护层的 HiF4 PTQ（单卡，逐层卸载）
  -> Smooth-QK 校准
  -> MMLU（16 个分片）与 LongBench 评测

Wan2.2-T2V-A14B
  -> 可选：GPTQ 权重与 Smooth-QK 缩放
  -> VBench 视频生成，每张 GPU 一个进程
  -> VBench 评测，并与 BF16 配对比较
```

所有命令都通过参数显式接收模型、数据、检查点、缩放参数和输出目录，不依赖
任何特定机器的目录结构。

## 仓库结构

```text
configs/   Qwen3-8B 最终配方、Qwen3-32B MMLU 分片、环境示例
docs/      方法、复现步骤、结果、Qwen3-32B、旋转、Wan2.2 与引用
results/   仅包含紧凑的聚合指标
scripts/   精简的命令行入口
src/       Qwen3/Wan2.2 的 HiF4 实现与工作流
tests/     CPU 与合成数值测试
```

## 可复现性范围

检查点哈希用于唯一标识已评测的 Qwen3-8B 模型文件。给定该检查点后，层保护
写入和评测流程可复现；不保证分布式 QAD 训练在相同随机种子下生成逐字节一致
的权重。Qwen3-32B 的构建使用固定种子，在相同软件与硬件上逐位复现。Wan2.2
生成的视频依赖 GPU kernel，量化结果应与同一环境下生成的 BF16 结果比较。

## 许可证

FAQUANT-Public 使用 Apache-2.0 许可证。上游项目许可证与引用信息见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
