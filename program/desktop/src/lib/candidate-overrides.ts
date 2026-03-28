import type { Candidate, PageRecord, Point } from "./types";

function cloneQuad(quad: Point[]): Point[] {
  return quad.map((point) => [...point]) as Point[];
}

export function getCandidateOriginalQuad(candidate: Candidate): Point[] {
  return cloneQuad(candidate.originalQuad ?? candidate.quad);
}

export function getEffectiveCandidateQuad(candidate: Candidate): Point[] {
  return cloneQuad(candidate.manualQuad ?? candidate.originalQuad ?? candidate.quad);
}

export function withCandidateManualOverride(candidate: Candidate, draftQuad: Point[]): Candidate {
  return {
    ...candidate,
    originalQuad: cloneQuad(candidate.originalQuad ?? candidate.quad),
    manualQuad: cloneQuad(draftQuad)
  };
}

export function clearCandidateManualOverride(candidate: Candidate): Candidate {
  return {
    ...candidate,
    originalQuad: cloneQuad(candidate.originalQuad ?? candidate.quad),
    manualQuad: null
  };
}

export function migrateLegacyManualOverride(page: PageRecord): PageRecord {
  if (!page.manualQuad?.length) {
    return page;
  }
  const baseIndex =
    page.manualBaseCandidateIndex !== undefined && page.manualBaseCandidateIndex !== null
      ? page.manualBaseCandidateIndex
      : page.selectedCandidateIndex;
  const candidate = page.candidates[baseIndex];
  if (!candidate) {
    return {
      ...page,
      manualQuad: null,
      manualBaseCandidateIndex: null
    };
  }
  const candidates = page.candidates.map((entry, index) =>
    index === baseIndex ? withCandidateManualOverride(entry, page.manualQuad as Point[]) : entry
  );
  return {
    ...page,
    candidates,
    activeQuad:
      baseIndex === page.selectedCandidateIndex
        ? getEffectiveCandidateQuad(candidates[baseIndex] as Candidate)
        : cloneQuad(page.activeQuad),
    manualQuad: null,
    manualBaseCandidateIndex: null
  };
}
