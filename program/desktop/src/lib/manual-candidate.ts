import type { Candidate, PageRecord } from "./types";

export type RenderedCandidate = {
  kind: "manual" | "original";
  candidate: Candidate;
  originalIndex: number | null;
  active: boolean;
  baseCandidate: Candidate | null;
};

export function buildRenderedCandidates(page: PageRecord): RenderedCandidate[] {
  const rendered: RenderedCandidate[] = [];

  if (page.manualQuad?.length) {
    const baseIndex =
      page.manualBaseCandidateIndex !== undefined && page.manualBaseCandidateIndex !== null
        ? page.manualBaseCandidateIndex
        : page.selectedCandidateIndex;
    const baseCandidate = page.candidates[baseIndex] ?? null;
    rendered.push({
      kind: "manual",
      candidate: {
        method: "manual_annotation",
        score: page.confidence,
        quad: page.manualQuad.map((point) => [...point]),
        metrics: {},
        source: "manual",
        modelId: "manual_annotation"
      },
      originalIndex: null,
      active: true,
      baseCandidate
    });
  }

  rendered.push(
    ...page.candidates.map((candidate, index) => ({
      kind: "original" as const,
      candidate,
      originalIndex: index,
      active: !page.manualQuad?.length && index === page.selectedCandidateIndex,
      baseCandidate: null
    }))
  );

  return rendered;
}
