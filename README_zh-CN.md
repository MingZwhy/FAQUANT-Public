# FAQUANT-Public

[English](README.md) | 简体中文

本仓库提供 Qwen3-8B HiF4 W4A4 的最小复现代码与聚合结果，覆盖 PTQ、
QAD、QAD 后层保护，以及仅量化预填充阶段的 LongBench 评测。

本仓库有意不包含模型权重、检查点、数据集、原始评测样本、模型生成结果和
私有实验历史。

## 主要结果

| 评测 | BF16 | 候选模型 | 损失 |
|---|---:|---:|---:|
| MMLU 0-shot，14,042 道题 | 72.9597% | **71.9627%** | **0.9970 pp** |
| LongBench v1，21 任务宏平均 | 49.4352 | **48.3529** | **1.0824 pp** |

LongBench 评测中，完整提示词的预填充（prefill）和第一个生成 token 使用量化
模型；后续解码使用 BF16，并复用量化预填充产生的 KV cache。

详细说明见[方法](docs/METHOD.md)、[复现步骤](docs/REPRODUCE.md)和
[结果](docs/RESULTS.md)。

## 最终候选模型

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
值唯一标识。首次发布不包含模型权重。

## 安装

推荐使用 Python 3.11。请先安装与 CUDA 版本匹配的 PyTorch，然后执行：

```bash
python -m pip install -e ".[test]"
```

仓库提供了一个 A800 80GB 示例环境：

```bash
bash scripts/setup_a800.sh
```

FlashAttention 仅作为环境配置和 BF16 基准评测的可选依赖。量化 QK/PV
模拟器使用 PyTorch 实现。

## 复现流程

```text
Qwen3-8B BF16
  -> HiF4 PTQ 初始化
  -> Smooth-QK 校准
  -> QAD：精确注意力 250 步 + 部署注意力 125 步
  -> 将第 16/17/18 层的 BF16 保护写入检查点
  -> MMLU 与 LongBench 评测
```

所有命令都通过参数显式接收模型、数据、检查点、缩放参数和输出目录，不依赖
任何特定机器的目录结构。

## 仓库结构

```text
configs/   最终配方
docs/      方法、复现步骤、结果与引用
results/   仅包含紧凑的聚合指标
scripts/   精简的命令行入口
src/       Qwen/HiF4 实现与工作流
tests/     CPU 与合成数值测试
```

## 可复现性范围

检查点哈希用于唯一标识已评测的模型文件。给定该检查点后，层保护写入和评测
流程可复现；不保证分布式 QAD 训练在相同随机种子下生成逐字节一致的权重。

## 许可证

FAQUANT-Public 使用 Apache-2.0 许可证。上游项目许可证与引用信息见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
