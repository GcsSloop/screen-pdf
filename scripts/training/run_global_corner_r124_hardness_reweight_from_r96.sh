#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_SPLIT_DIR="${INPUT_SPLIT_DIR:-training/runs/global_corner_r122_202604_runtime_oracle_disagreement_gate_from_r96_v1/split_runtime_oracle}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/runs/global_corner_r124_202604_hardness_reweight_from_r96_v1}"

python program/engine/global_corner_train.py \
  --dataset-root /Users/gcssloop/WorkSpace/AIGC/展会 \
  --output-dir "$OUTPUT_ROOT/output" \
  --split-dir "$INPUT_SPLIT_DIR" \
  --init-model training/runs/global_corner_r96_202603_fuzhoumix_jinyi_rgbgrayborder_antiinset_from_r95_v1/output/global_corner_model.pt \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 8e-5 \
  --input-size 256 \
  --output-size 64 \
  --channels 24 \
  --feature-mode rgb_gray_border \
  --training-profile default \
  --sample-weight-power 0.0 \
  --hardness-sample-weight-power 0.35 \
  --teacher-guidance-weight 0.0 \
  --teacher-blend-ratio 0.0 \
  --inset-weight 1.0 \
  --max-corner-weight 1.15 \
  --edge-weight 0.35 \
  --edge-line-weight 0.9 \
  --edge-length-weight 0.4 \
  --edge-collapse-weight 0.6 \
  --corner-line-weight 0.7 \
  --corner-angle-weight 0.28 \
  --save-epoch-checkpoints
