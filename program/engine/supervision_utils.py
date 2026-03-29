from __future__ import annotations

from typing import Any


NON_MODEL_METHODS = {
    "contour_quad",
    "document_quad",
    "document_quad_edge",
    "lsd_grabcut_quad_v2",
    "roi_guided_quad",
}
NON_MODEL_SOURCES = {
    "external",
    "human",
    "manual",
    "opencv",
    "review",
}
MODEL_METHOD_PREFIXES = (
    "deep_screen",
    "teacher_",
    "student_",
)
MODEL_SOURCES = {
    "model",
    "runtime_student",
    "runtime_teacher",
}

CURRENT_DATA_STRUCTURE_VERSION = 2


def resolve_selected_candidate(page: dict[str, Any]) -> dict[str, Any] | None:
    candidates = list(page.get("candidates") or [])
    if not candidates:
        return None
    try:
        selected_index = int(page.get("selectedCandidateIndex") or 0)
    except (TypeError, ValueError):
        return None
    if selected_index < 0 or selected_index >= len(candidates):
        return None
    candidate = candidates[selected_index]
    return candidate if isinstance(candidate, dict) else None


def resolve_data_structure_version(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 1
    try:
        version = int(
            payload.get("dataStructureVersion")
            or payload.get("data_structure_version")
            or 1
        )
    except (TypeError, ValueError):
        return 1
    return max(version, 1)


def resolve_manual_quad(page: dict[str, Any]) -> tuple[list[list[float]] | None, str]:
    candidate = resolve_selected_candidate(page)
    candidate_manual_quad = candidate.get("manualQuad") if isinstance(candidate, dict) else None
    if candidate_manual_quad:
        return candidate_manual_quad, "selected_candidate_manual_quad"
    manual_quad = page.get("manualQuad")
    if manual_quad:
        return manual_quad, "manual_quad"
    return None, "missing"


def is_model_generated_candidate(page: dict[str, Any], candidate: dict[str, Any] | None = None) -> bool:
    candidate = candidate or resolve_selected_candidate(page)
    if not candidate:
        return False
    source = str(candidate.get("source") or "").strip().lower()
    method = str(candidate.get("method") or "").strip().lower()
    model_id = str(candidate.get("modelId") or candidate.get("model_id") or "").strip()
    if source in NON_MODEL_SOURCES or method in NON_MODEL_METHODS:
        return False
    if source in MODEL_SOURCES:
        return True
    if model_id:
        return True
    return any(method.startswith(prefix) for prefix in MODEL_METHOD_PREFIXES)


def resolve_supervision_quad(page: dict[str, Any]) -> tuple[list[list[float]] | None, str]:
    manual_quad, manual_source = resolve_manual_quad(page)
    if manual_quad:
        return manual_quad, manual_source
    active_quad = page.get("activeQuad")
    if not active_quad or str(page.get("status") or "").strip().lower() != "reviewed":
        return None, "missing"
    candidate = resolve_selected_candidate(page)
    if candidate is None:
        return None, "missing"
    if is_model_generated_candidate(page, candidate):
        return None, "model_active_quad"
    return active_quad, "accepted_active_quad"
