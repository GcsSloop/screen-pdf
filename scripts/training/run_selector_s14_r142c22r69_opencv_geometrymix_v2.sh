#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/training/build_selector_s11_focus_mix_split.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-training/runs/selector_expand_s14_202604_r142c22r69_opencv_geometrymix_v2}"
mkdir -p "$OUTPUT_ROOT/output"

PYTHONPATH="$REPO_ROOT/program/engine" python program/engine/train_candidate_expand_selector.py \
  --split-dir training/runs/selector_expand_s11_202604_r142c22r69_opencv_focusmix_v1/split \
  --split train \
  --global-model training/runs/global_corner_r142_202604_guardrail_mall_midpoint_from_r141_v1/output/global_corner_model.pt \
  --roi-model training/runs/roi_corner_c22_202603_jinyi_manualfocus12_from_c21_v1/output/corner_heatmap_model.pt \
  --local-model training/runs/local_corner_r69probe_202603_r92c22_jinyi_fast12k_from_runtime_v1/output/local_corner_moe_coord_model.pt \
  --output "$OUTPUT_ROOT/output/candidate_expand_selector.json" \
  --candidate-expand-ratios 0.02,0.04,0.06,0.08,0.10,0.12 \
  --baseline-expand-ratio 0.08 \
  --epochs 260 \
  --learning-rate 0.02 \
  --weight-decay 1e-4 \
  --balance-power 0.5 \
  --switch-margin 0.04 \
  --preserve-point-margin 0.0012 \
  --preserve-max-margin 0.0020 \
  --objective-mode hybrid \
  --pairwise-weight 0.45 \
  --switch-label-boost 1.8 \
  --external-label-boost 1.8
