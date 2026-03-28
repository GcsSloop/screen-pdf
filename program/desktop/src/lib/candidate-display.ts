import type { Candidate } from "./types";

export function buildCandidateTitle(candidate: Candidate): string {
  return candidate.modelId || candidate.method;
}

export function buildCandidateDebugLabel(candidate: Candidate): string {
  const parts = [candidate.source || "opencv"];
  if (candidate.debugOnly) {
    parts.push("debug");
  }
  return parts.join(" · ");
}

export function buildCandidateRowMeta(candidate: Candidate): {
  scoreLabel: string;
  restoreButtonLabel: string;
  restoreDisabled: boolean;
} {
  return {
    scoreLabel: `评分 ${candidate.score.toFixed(4)}${candidate.manualQuad?.length ? " · 已人工调整" : ""}`,
    restoreButtonLabel: "恢复默认",
    restoreDisabled: !candidate.manualQuad?.length
  };
}
