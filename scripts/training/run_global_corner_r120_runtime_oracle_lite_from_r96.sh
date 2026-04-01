#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_SPLIT_DIR="${INPUT_SPLIT_DIR:-training/runs/global_corner_r107_202603_fuzhoumix_jinyi_teacherblend_from_r96_v1/split}"
OUTPUT_ROOT="${OUTPUT_ROOT:-training/runs/global_corner_r120_202604_runtime_oracle_lite_from_r96_v1}"
ENRICHED_SPLIT_DIR="$OUTPUT_ROOT/split_runtime_oracle"

python program/engine/enrich_global_corner_split_runtime.py \
  --input-split-dir "$INPUT_SPLIT_DIR" \
  --output-split-dir "$ENRICHED_SPLIT_DIR" \
  --global-model models/runtime/global_corner_model.pt \
  --roi-model models/runtime/corner_heatmap_model.pt \
  --local-model models/runtime/local_corner_moe_coord_model.pt \
  --candidate-expand-ratios "0.02,0.04,0.06,0.08,0.10,0.12" \
  --candidate-baseline-gate 0.45 \
  --candidate-min-score-gain 0.03

python program/engine/global_corner_train.py \
  --dataset-root /Users/gcssloop/WorkSpace/AIGC/展会 \
  --output-dir "$OUTPUT_ROOT/output" \
  --split-dir "$ENRICHED_SPLIT_DIR" \
  --init-model training/runs/global_corner_r96_202603_fuzhoumix_jinyi_rgbgrayborder_antiinset_from_r95_v1/output/global_corner_model.pt \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --input-size 256 \
  --output-size 64 \
  --channels 24 \
  --feature-mode rgb_gray_border \
  --training-profile default \
  --sample-weight-power 0.0 \
  --teacher-guidance-weight 0.15 \
  --teacher-blend-ratio 0.35 \
  --teacher-corner-error-max 0.025 \
  --teacher-sample-error-max 0.018 \
  --teacher-target-mode oracle \
  --teacher-candidate-sources teacher,r3,roi \
  --teacher-opencv-score-min 0.2 \
  --inset-weight 1.2 \
  --max-corner-weight 1.2 \
  --edge-weight 0.35 \
  --edge-line-weight 1.1 \
  --edge-length-weight 0.5 \
  --edge-collapse-weight 0.6 \
  --corner-line-weight 0.7 \
  --corner-angle-weight 0.28 \
  --save-epoch-checkpoints
