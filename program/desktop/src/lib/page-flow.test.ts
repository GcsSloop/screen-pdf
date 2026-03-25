import { describe, expect, it } from "vitest";

import {
  applyCandidateToPage,
  applyDraftQuadToPage,
  buildPreviewVersionedPath,
  movePage,
  resolveWorkingQuad
} from "./page-flow";
import type { PageRecord, Point } from "./types";

function page(id: string): PageRecord {
  return {
    id,
    name: `${id}.jpg`,
    path: `/tmp/${id}.jpg`,
    createdAt: "2026-03-21 10:00:00",
    status: "auto_ready",
    confidence: 0.9,
    bestMethod: "contour_quad",
    selectedCandidateIndex: 0,
    candidates: [],
    activeQuad: [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100]
    ],
    manualQuad: null,
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

describe("page flow", () => {
  it("moves a dragged page before the target page", () => {
    const pages = [page("a"), page("b"), page("c")];
    expect(movePage(pages, "c", "a").map((item) => item.id)).toEqual(["c", "a", "b"]);
  });

  it("uses draft quad when present", () => {
    const draft: Point[] = [
      [1, 1],
      [9, 1],
      [9, 9],
      [1, 9]
    ];
    expect(resolveWorkingQuad(page("a"), draft)).toEqual(draft);
  });

  it("applies draft quad as reviewed manual quad", () => {
    const draft: Point[] = [
      [2, 2],
      [8, 2],
      [8, 8],
      [2, 8]
    ];
    const updated = applyDraftQuadToPage(page("a"), draft);
    expect(updated.manualQuad).toEqual(draft);
    expect(updated.status).toBe("reviewed");
  });

  it("applies a candidate and confirms the page immediately", () => {
    const candidate = {
      method: "alt_quad",
      score: 0.88,
      quad: [
        [3, 3],
        [93, 4],
        [92, 88],
        [4, 90]
      ] as Point[],
      metrics: {}
    };
    const original = {
      ...page("a"),
      status: "needs_review" as const,
      manualQuad: [
        [1, 1],
        [99, 1],
        [99, 99],
        [1, 99]
      ] as Point[],
      candidates: [candidate]
    };

    const updated = applyCandidateToPage(original, 0);

    expect(updated.selectedCandidateIndex).toBe(0);
    expect(updated.activeQuad).toEqual(candidate.quad);
    expect(updated.manualQuad).toBeNull();
    expect(updated.status).toBe("reviewed");
  });

  it("adds a cache buster after preview regeneration", () => {
    expect(buildPreviewVersionedPath("/tmp/a.png", 12)).toBe("/tmp/a.png?v=12");
  });
});
