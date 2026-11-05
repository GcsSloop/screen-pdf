#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_SPLIT_DIR="${INPUT_SPLIT_DIR:-training/runs/global_corner_r139_202604_mall_failure_weighted_from_r138_v1/split}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/runs/global_corner_r143_202604_mall_recap_from_r142_v1}"

python program/engine/global_corner_train.py \
  --dataset-root /Users/gcssloop/WorkSpace/AIGC/展会 \
  --output-dir "$OUTPUT_ROOT/output" \
  --split-dir "$INPUT_SPLIT_DIR" \
  --init-model training/runs/global_corner_r142_202604_guardrail_mall_midpoint_from_r141_v1/output/global_corner_model.pt \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 5e-6 \
  --input-size 256 \
  --output-size 64 \
  --channels 24 \
  --feature-mode rgb_gray_border \
  --training-profile default \
  --merge-focus-train \
  --focus-train-repeat 8 \
  --failure-layer-opencv-boost 2.0 \
  --failure-layer-runtime-boost 1.2 \
  --failure-layer-gain-power 1.0 \
  --teacher-guidance-weight 0.68 \
  --teacher-blend-ratio 1.0 \
  --teacher-target-mode oracle \
  --teacher-candidate-sources teacher,opencv \
  --teacher-opencv-score-min 0.15 \
  --teacher-opencv-agreement-corner-max 0.02 \
  --teacher-opencv-agreement-sample-max 0.01 \
  --teacher-opencv-agreement-penalty-scale 1.4 \
  --teacher-corner-error-max 0.03 \
  --teacher-sample-error-max 0.02 \
  --teacher-activation-min-disagreement 0.003 \
  --inset-weight 1.0 \
  --max-corner-weight 1.15 \
  --edge-weight 0.35 \
  --edge-line-weight 0.9 \
  --edge-length-weight 0.4 \
  --edge-collapse-weight 0.6 \
  --corner-line-weight 0.7 \
  --corner-angle-weight 0.28 \
  --save-epoch-checkpoints
