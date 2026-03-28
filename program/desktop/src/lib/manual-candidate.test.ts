import { describe, expect, it } from "vitest";

import { buildRenderedCandidates } from "./manual-candidate";
import type { Candidate, PageRecord, Point } from "./types";

function candidate(method: string, score: number, quad: Point[]): Candidate {
  return {
    method,
    score,
    quad,
    metrics: {},
    source: method,
    modelId: method
  };
}

function page(): PageRecord {
  return {
    id: "p1",
    name: "p1.jpg",
    path: "/tmp/p1.jpg",
    createdAt: "0",
    status: "reviewed",
    confidence: 0.92,
    bestMethod: "r3",
    selectedCandidateIndex: 1,
    candidates: [
      candidate("r3", 0.92, [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ]),
      candidate("v28", 0.9, [
        [2, 1],
        [99, 0],
        [98, 98],
        [1, 99]
      ])
    ],
    activeQuad: [
      [2, 1],
      [99, 0],
      [98, 98],
      [1, 99]
    ],
    manualQuad: [
      [3, 2],
      [98, 1],
      [97, 97],
      [2, 98]
    ],
    manualBaseCandidateIndex: 1,
    previewPath: null,
    details: {
      width: 1000,
      height: 800,
      fileSizeBytes: 1024,
      capturedAt: null,
      createdAt: "2026-03-21 10:00:00",
      modifiedAt: "2026-03-21 10:00:00"
    }
  };
}

describe("manual candidate helpers", () => {
  it("prepends one synthetic manual candidate with base candidate metadata", () => {
    const rendered = buildRenderedCandidates(page());

    expect(rendered).toHaveLength(3);
    expect(rendered[0]?.kind).toBe("manual");
    expect(rendered[0]?.candidate.method).toBe("manual_annotation");
    expect(rendered[0]?.baseCandidate?.method).toBe("v28");
    expect(rendered[0]?.active).toBe(true);
  });

  it("does not inject a manual candidate when there is no manual quad", () => {
    const sample = page();
    sample.manualQuad = null;

    expect(buildRenderedCandidates(sample)).toHaveLength(2);
    expect(buildRenderedCandidates(sample).every((item) => item.kind === "original")).toBe(true);
  });
});
