#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to Qwen3-8B}"
: "${TRAIN_DATA:?set TRAIN_DATA to train.jsonl}"
: "${VALIDATION_DATA:?set VALIDATION_DATA to validation.jsonl}"
: "${PTQ_INIT:?set PTQ_INIT to the prepared QAD init}"
: "${SMOOTH_QK:?set SMOOTH_QK to the calibrated scales file}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON="${PYTHON:-python}"
COMMON=(
  --model "${MODEL_DIR}"
  --train-data "${TRAIN_DATA}"
  --validation-data "${VALIDATION_DATA}"
  --rotation hisq1024
  --loss-mode full
  --task-alpha 0.05
  --logit-alpha 2.0
  --feature-alpha 0.5
  --feature-topk 3
  --kl-mode eakld
  --temperature 1.0
  --scheduler-total-steps 1250
  --learning-rate 4e-5
  --warmup-ratio 0.0104
  --lr-schedule cosine
  --weight-decay 0.01
  --trainable-scope qat-linears
  --metadata-mode fixed
  --microbatch-size 4
  --gradient-accumulation-steps 8
  --max-length 1024
  --max-answer-tokens 256
  --validation-samples 256
  --validation-steps 250
  --seed 3407
  --max-grad-norm 1.0
  --max-shard-size 4GB
)

mkdir -p "${OUTPUT_ROOT}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m faquant.workflows.train_qad \
  "${COMMON[@]}" \
  --student-init "${PTQ_INIT}" \
  --output-dir "${OUTPUT_ROOT}/exact" \
  --max-steps 250 \
  --save-steps 50

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m faquant.workflows.train_qad \
  "${COMMON[@]}" \
  --student-init "${OUTPUT_ROOT}/exact/student-step-250" \
  --output-dir "${OUTPUT_ROOT}/deployed" \
  --completed-steps 250 \
  --max-steps 375 \
  --save-steps 25 \
  --deployed-attention \
  --qk-smooth-scales "${SMOOTH_QK}" \
  --pv-normalizer-mode quantized_same

echo "QAD checkpoint: ${OUTPUT_ROOT}/deployed/student-step-375"
