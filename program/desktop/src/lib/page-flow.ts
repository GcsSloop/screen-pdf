import type { PageRecord, Point } from "./types";

export function movePage(pages: PageRecord[], draggedId: string, targetId: string): PageRecord[] {
  const next = [...pages];
  const fromIndex = next.findIndex((page) => page.id === draggedId);
  const toIndex = next.findIndex((page) => page.id === targetId);
  if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) {
    return next;
  }
  const [dragged] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, dragged);
  return next;
}

export function resolveWorkingQuad(page: PageRecord, draftQuad: Point[] | null): Point[] {
  return draftQuad ?? page.manualQuad ?? page.activeQuad;
}

export function applyCandidateToPage(page: PageRecord, index: number): PageRecord {
  const candidate = page.candidates[index];
  if (!candidate) {
    return page;
  }

  return {
    ...page,
    selectedCandidateIndex: index,
    activeQuad: candidate.quad.map((point) => [...point]) as Point[],
    manualQuad: null,
    status: "reviewed"
  };
}

export function applyDraftQuadToPage(page: PageRecord, draftQuad: Point[]): PageRecord {
  return {
    ...page,
    manualQuad: draftQuad.map((point) => [...point]) as Point[],
    status: "reviewed"
  };
}

export function buildPreviewVersionedPath(path: string, version: number): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}v=${version}`;
}
