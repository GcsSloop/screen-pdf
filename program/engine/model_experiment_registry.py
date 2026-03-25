from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOWER_BETTER_METRICS = {
    "screen_relative_error_mean": 0.0015,
    "max_corner_error_mean": 0.0030,
    "perspective_tilt_error_mean": 0.2000,
    "quad_inset_ratio_abs_mean": 0.0100,
}
HIGHER_BETTER_METRICS = {
    "point_le_0_01_ratio": 0.0100,
}


def compare_suite_metrics(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    regressions: list[str] = []
    improvements: list[str] = []
    deltas: dict[str, float] = {}

    for metric, tolerance in LOWER_BETTER_METRICS.items():
        if metric not in candidate or metric not in baseline:
            continue
        delta = float(candidate[metric]) - float(baseline[metric])
        deltas[metric] = round(delta, 6)
        if delta > tolerance:
            regressions.append(metric)
        elif delta < -tolerance:
            improvements.append(metric)

    for metric, tolerance in HIGHER_BETTER_METRICS.items():
        if metric not in candidate or metric not in baseline:
            continue
        delta = float(candidate[metric]) - float(baseline[metric])
        deltas[metric] = round(delta, 6)
        if delta < -tolerance:
            regressions.append(metric)
        elif delta > tolerance:
            improvements.append(metric)

    return {
        "has_regression": bool(regressions),
        "regressions": sorted(regressions),
        "improvements": sorted(improvements),
        "deltas": deltas,
    }


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"experiments": []}
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_registry_entry(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    registry = load_registry(path)
    experiments = [item for item in registry.get("experiments", []) if item.get("experiment_id") != entry.get("experiment_id")]
    experiments.append(entry)
    experiments.sort(key=lambda item: str(item.get("experiment_id", "")))
    out = {"experiments": experiments}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
