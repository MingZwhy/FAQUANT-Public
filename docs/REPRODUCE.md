# Reproduction

The commands below use explicit paths. No filesystem layout is assumed.

```bash
export MODEL_DIR=/path/to/Qwen3-8B
export DATA_DIR=/path/to/faquant-data
export OUTPUT_DIR=/path/to/faquant-output
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
```

Install the package first:

```bash
python -m pip install -e ".[test]"
```

## 1. Build the QAD corpus

```bash
python scripts/prepare_data.py \
  --tokenizer "$MODEL_DIR" \
  --output-dir "$DATA_DIR/qad_redpajama" \
  --target-records 560000 \
  --validation-records 2000 \
  --max-tokens 300 \
  --target-tokens 256 \
  --target-fraction 0.5 \
  --min-tokens 160 \
  --char-budget 2400 \
  --seed 3407
```

Validate every record with the training collator:

```bash
faquant-validate-data \
  --tokenizer "$MODEL_DIR" \
  --max-length 1024 \
  --max-answer-tokens 256 \
  "$DATA_DIR/qad_redpajama/train.jsonl"
```

## 2. Prepare the PTQ initialization

```bash
python scripts/prepare_ptq.py \
  --model "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR/ptq_init" \
  --rotation hisq1024 \
  --value-head-rotation \
  --weight-quant gptq \
  --gptq-nsamples 128 \
  --gptq-seqlen 2048 \
  --gptq-calibration-seed 0 \
  --gptq-damp 0.01 \
  --gptq-damp-overrides o_proj=0.03,down_proj=0.03 \
  --gptq-sequential-groups
```

## 3. Calibrate Smooth-QK

```bash
faquant-calibrate-smooth-qk \
  --model "$MODEL_DIR" \
  --checkpoint "$OUTPUT_DIR/ptq_init" \
  --sequence-length 4096 \
  --samples 8 \
  --alphas 0.5 \
  --statistic max \
  --scale-clamp 32 \
  --seed 0 \
  --output "$OUTPUT_DIR/smooth_qk_report.json" \
  --scales-output "$OUTPUT_DIR/smooth_qk.pt" \
  --scales-alpha 0.5
```

## 4. Run staged QAD

The example assumes eight GPUs. Change `NPROC_PER_NODE`,
`micro_batch_size`, or gradient accumulation while preserving global batch 256
when using a different layout.

```bash
MODEL_DIR="$MODEL_DIR" \
TRAIN_DATA="$DATA_DIR/qad_redpajama/train.jsonl" \
VALIDATION_DATA="$DATA_DIR/qad_redpajama/validation.jsonl" \
PTQ_INIT="$OUTPUT_DIR/ptq_init" \
SMOOTH_QK="$OUTPUT_DIR/smooth_qk.pt" \
OUTPUT_ROOT="$OUTPUT_DIR/qad" \
NPROC_PER_NODE=8 \
bash scripts/run_qad_staged.sh
```

The selected output is:

```text
$OUTPUT_DIR/qad/deployed/student-step-375
```

Distributed QAD is not claimed to be byte-deterministic across hardware
partitions. Published checkpoint hashes identify the evaluated artifact.

## 5. Obtain or materialize layers 16/17/18

Download the evaluated artifact:

```python
import os

from huggingface_hub import snapshot_download

snapshot_download(
    "MingZwhy/qwen3-8b-hif4-l16-17-18",
    local_dir=f"{os.environ['OUTPUT_DIR']}/qwen3-8b-hif4-l16-17-18",
)
```

Alternatively, materialize the same protection from a reproduced step-375
checkpoint:

```bash
python scripts/materialize_protection.py \
  --model "$MODEL_DIR" \
  --checkpoint "$OUTPUT_DIR/qad/deployed/student-step-375" \
  --output-dir "$OUTPUT_DIR/qwen3-8b-hif4-l16-17-18" \
  --layers 16,17,18 \
  --rotation hisq1024 \
  --device cuda:0
```

## 6. Evaluate MMLU

BF16:

```bash
faquant-eval-bf16 \
  --model "$MODEL_DIR" \
  --tasks mmlu \
  --num-fewshot 0 \
  --batch-size 16 \
  --apply-chat-template \
  --output "$OUTPUT_DIR/mmlu_bf16.json"
```

Candidate:

```bash
python scripts/eval_mmlu.py \
  --model "$MODEL_DIR" \
  --checkpoint "$OUTPUT_DIR/qwen3-8b-hif4-l16-17-18" \
  --rotation hisq1024 \
  --tasks mmlu \
  --num-fewshot 0 \
  --batch-size 16 \
  --apply-chat-template \
  --qk-matmul-quant \
  --pv-matmul-quant \
  --post-rope-qk-rotation \
  --pv-normalizer-mode quantized_same \
  --qk-smooth-scales "$OUTPUT_DIR/smooth_qk.pt" \
  --attention-kernel simulated \
  --output "$OUTPUT_DIR/mmlu_candidate.json"
```

## 7. Evaluate LongBench

The bundled tasks use THUDM/LongBench raw data, official prompts and generation
lengths, and task names prefixed with `faquant_longbench_`.

Run the BF16 task files first. Use chat templates for all tasks except TREC,
TriviaQA, SAMSum, LSHT, LCC, and RepoBench-P:

```bash
faquant-eval-bf16 \
  --model "$MODEL_DIR" \
  --tasks faquant_longbench_qasper \
  --batch-size 1 \
  --middle-truncation \
  --apply-chat-template \
  --output "$OUTPUT_DIR/longbench_bf16/faquant_longbench_qasper_full_bf16_native.json"
```

An agent or scheduler can parallelize the same command over all 21 task names.

Run the candidate with one or more GPUs:

```bash
python scripts/run_longbench.py \
  --model "$MODEL_DIR" \
  --checkpoint "$OUTPUT_DIR/qwen3-8b-hif4-l16-17-18" \
  --scales "$OUTPUT_DIR/smooth_qk.pt" \
  --result-dir "$OUTPUT_DIR/longbench_candidate" \
  --expected-exempt-layers 16,17,18 \
  --gpus 0,1,2,3 \
  --num-shards 4 \
  --truncation-mode middle
```

Audit and summarize:

```bash
python scripts/summarize_longbench.py \
  --result-dir "$OUTPUT_DIR/longbench_candidate" \
  --bf16-dir "$OUTPUT_DIR/longbench_bf16" \
  --num-shards 4 \
  --output "$OUTPUT_DIR/longbench_summary.json" \
  --markdown "$OUTPUT_DIR/longbench_summary.md"
```

## 8. Verify a published checkpoint

```bash
sha256sum "$OUTPUT_DIR/qwen3-8b-hif4-l16-17-18"/*.safetensors
```

Compare the values with `results/final_metrics.json`.
