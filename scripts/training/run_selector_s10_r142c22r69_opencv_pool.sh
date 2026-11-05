#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-training/runs/selector_expand_s10_202604_r142c22r69_opencv_pool_v1}"
mkdir -p "$OUTPUT_ROOT/output"

PYTHONPATH="$REPO_ROOT/program/engine" python program/engine/train_candidate_expand_selector.py \
  --split-dir training/runs/global_corner_r144_202604_mall_curriculum_stage2_from_stage1_v1/split \
  --split train \
  --global-model training/runs/global_corner_r142_202604_guardrail_mall_midpoint_from_r141_v1/output/global_corner_model.pt \
  --roi-model training/runs/roi_corner_c22_202603_jinyi_manualfocus12_from_c21_v1/output/corner_heatmap_model.pt \
  --local-model training/runs/local_corner_r69probe_202603_r92c22_jinyi_fast12k_from_runtime_v1/output/local_corner_moe_coord_model.pt \
  --output "$OUTPUT_ROOT/output/candidate_expand_selector.json" \
  --candidate-expand-ratios 0.02,0.04,0.06,0.08,0.10,0.12 \
  --baseline-expand-ratio 0.08 \
  --epochs 240 \
  --learning-rate 0.02 \
  --weight-decay 1e-4 \
  --balance-power 0.8 \
  --switch-margin 0.08 \
  --preserve-point-margin 0.0015 \
  --preserve-max-margin 0.0025 \
  --objective-mode hybrid \
  --pairwise-weight 0.35
