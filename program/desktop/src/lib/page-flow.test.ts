import { describe, expect, it } from "vitest";

import {
  applyCandidateToPage,
  applyDraftQuadToPage,
  buildPreviewVersionedPath,
  movePage,
  normalizeProjectDataStructure,
  resolveWorkingQuad
} from "./page-flow";
import type { Candidate, PageRecord, Point, ProjectFile } from "./types";

function candidate(method: string, quad: Point[], overrides: Partial<Candidate> = {}): Candidate {
  return {
    method,
    score: 0.9,
    quad,
    originalQuad: quad,
    manualQuad: null,
    metrics: {},
    ...overrides
  };
}

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
    candidates: [
      candidate("contour_quad", [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ])
    ],
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

function project(pages: PageRecord[]): ProjectFile {
  return {
    version: 1,
    dataStructureVersion: null,
    name: "demo",
    sourceDir: "/tmp/demo",
    projectPath: null,
    selectedPageId: pages[0]?.id ?? null,
    pages
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
    expect(updated.candidates[0]?.manualQuad).toEqual(draft);
    expect(updated.activeQuad).toEqual(draft);
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
      candidates: [
        {
          ...candidate,
          manualQuad: [
            [1, 1],
            [99, 1],
            [99, 99],
            [1, 99]
          ] as Point[]
        }
      ]
    };

    const updated = applyCandidateToPage(original, 0);

    expect(updated.selectedCandidateIndex).toBe(0);
    expect(updated.activeQuad).toEqual(original.candidates[0]?.manualQuad);
    expect(updated.status).toBe("reviewed");
  });

  it("adds a cache buster after preview regeneration", () => {
    expect(buildPreviewVersionedPath("/tmp/a.png", 12)).toBe("/tmp/a.png?v=12");
  });

  it("normalizes a legacy project into structure version 2 on load", () => {
    const legacy = page("a");
    legacy.manualQuad = [
      [2, 2],
      [98, 2],
      [98, 98],
      [2, 98]
    ];
    legacy.manualBaseCandidateIndex = 0;

    const normalized = normalizeProjectDataStructure(project([legacy]));

    expect(normalized.dataStructureVersion).toBe(2);
    expect(normalized.pages[0]?.manualQuad).toBeNull();
    expect(normalized.pages[0]?.manualBaseCandidateIndex).toBeNull();
    expect(normalized.pages[0]?.candidates[0]?.manualQuad).toEqual([
      [2, 2],
      [98, 2],
      [98, 98],
      [2, 98]
    ]);
  });

  it("normalizes a missing-version new-structure project without altering candidate manual data", () => {
    const modern = page("a");
    modern.candidates[0] = {
      ...modern.candidates[0],
      manualQuad: [
        [3, 3],
        [97, 3],
        [97, 97],
        [3, 97]
      ]
    };

    const normalized = normalizeProjectDataStructure(project([modern]));

    expect(normalized.dataStructureVersion).toBe(2);
    expect(normalized.pages[0]?.manualQuad).toBeNull();
    expect(normalized.pages[0]?.candidates[0]?.manualQuad).toEqual([
      [3, 3],
      [97, 3],
      [97, 97],
      [3, 97]
    ]);
  });

  it("normalizes an explicit v1 project into v2 candidate-level manual data", () => {
    const legacy = page("a");
    legacy.manualQuad = [
      [2, 2],
      [98, 2],
      [98, 98],
      [2, 98]
    ];

    const normalized = normalizeProjectDataStructure(
      {
        ...project([legacy]),
        dataStructureVersion: 1
      }
    );

    expect(normalized.dataStructureVersion).toBe(2);
    expect(normalized.pages[0]?.manualQuad).toBeNull();
    expect(normalized.pages[0]?.candidates[0]?.manualQuad).toEqual([
      [2, 2],
      [98, 2],
      [98, 98],
      [2, 98]
    ]);
  });

  it("keeps explicit v2 projects stable during normalization", () => {
    const modern = page("a");
    modern.candidates[0] = {
      ...modern.candidates[0],
      manualQuad: [
        [3, 3],
        [97, 3],
        [97, 97],
        [3, 97]
      ]
    };

    const normalized = normalizeProjectDataStructure(
      {
        ...project([modern]),
        dataStructureVersion: 2
      }
    );

    expect(normalized.dataStructureVersion).toBe(2);
    expect(normalized.pages[0]?.candidates[0]?.manualQuad).toEqual([
      [3, 3],
      [97, 3],
      [97, 97],
      [3, 97]
    ]);
  });
});
