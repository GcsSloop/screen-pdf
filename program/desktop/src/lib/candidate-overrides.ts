import type { Candidate, PageRecord, Point, ProjectFile } from "./types";

export const CURRENT_DATA_STRUCTURE_VERSION = 2;

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

export function inferProjectDataStructureVersion(project: ProjectFile): number {
  if (project.dataStructureVersion && project.dataStructureVersion > 0) {
    return project.dataStructureVersion;
  }
  for (const page of project.pages) {
    if (page.manualQuad?.length) {
      return 1;
    }
  }
  for (const page of project.pages) {
    if (page.candidates.some((candidate) => Boolean(candidate.manualQuad?.length))) {
      return CURRENT_DATA_STRUCTURE_VERSION;
    }
  }
  return CURRENT_DATA_STRUCTURE_VERSION;
}

export function normalizeProjectDataStructure(project: ProjectFile): ProjectFile {
  const inferredVersion = inferProjectDataStructureVersion(project);
  const pages =
    inferredVersion <= 1
      ? project.pages.map((page) => migrateLegacyManualOverride(page))
      : project.pages.map((page) => migrateLegacyManualOverride(page));
  return {
    ...project,
    dataStructureVersion: CURRENT_DATA_STRUCTURE_VERSION,
    pages
  };
}
