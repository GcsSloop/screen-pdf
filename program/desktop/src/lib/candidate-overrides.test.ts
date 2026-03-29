import { describe, expect, it } from "vitest";

import {
  clearCandidateManualOverride,
  getEffectiveCandidateQuad,
  inferProjectDataStructureVersion,
  migrateLegacyManualOverride
} from "./candidate-overrides";
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
      candidate("r3", [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ]),
      candidate("v28", [
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
    manualQuad: null,
    manualBaseCandidateIndex: null,
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

function project(overrides: Partial<ProjectFile> = {}): ProjectFile {
  return {
    version: 1,
    dataStructureVersion: null,
    name: "demo",
    sourceDir: "/tmp/demo",
    projectPath: null,
    selectedPageId: "p1",
    pages: [page()],
    ...overrides
  };
}

describe("candidate overrides", () => {
  it("prefers the candidate manual quad over the original quad", () => {
    const sample = candidate("v28", [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ], {
      manualQuad: [
        [1, 1],
        [9, 1],
        [9, 9],
        [1, 9]
      ]
    });

    expect(getEffectiveCandidateQuad(sample)).toEqual(sample.manualQuad);
  });

  it("clears only the candidate manual override when restoring default", () => {
    const updated = clearCandidateManualOverride(
      candidate("v28", [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
      ], {
        manualQuad: [
          [1, 1],
          [9, 1],
          [9, 9],
          [1, 9]
        ]
      })
    );

    expect(updated.manualQuad).toBeNull();
    expect(updated.originalQuad).toEqual([
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ]);
  });

  it("migrates legacy page-level manual data into the selected candidate", () => {
    const legacy = page();
    legacy.manualQuad = [
      [3, 2],
      [98, 1],
      [97, 97],
      [2, 98]
    ];
    legacy.manualBaseCandidateIndex = 1;

    const migrated = migrateLegacyManualOverride(legacy);

    expect(migrated.manualQuad).toBeNull();
    expect(migrated.manualBaseCandidateIndex).toBeNull();
    expect(migrated.candidates[1]?.manualQuad).toEqual([
      [3, 2],
      [98, 1],
      [97, 97],
      [2, 98]
    ]);
  });

  it("infers legacy structure when version is missing and page-level manual quad exists", () => {
    const legacyPage = page();
    legacyPage.manualQuad = [
      [3, 2],
      [98, 1],
      [97, 97],
      [2, 98]
    ];

    expect(inferProjectDataStructureVersion(project({ pages: [legacyPage] }))).toBe(1);
  });

  it("infers new structure when version is missing and candidate-level manual quad exists", () => {
    const modernPage = page();
    modernPage.candidates[0] = candidate(
      "r66",
      [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ],
      {
        manualQuad: [
          [4, 4],
          [96, 4],
          [96, 96],
          [4, 96]
        ]
      }
    );

    expect(inferProjectDataStructureVersion(project({ pages: [modernPage] }))).toBe(2);
  });

  it("prefers the explicit data structure version when present", () => {
    const legacyPage = page();
    legacyPage.manualQuad = [
      [3, 2],
      [98, 1],
      [97, 97],
      [2, 98]
    ];

    expect(
      inferProjectDataStructureVersion(
        project({
          dataStructureVersion: 2,
          pages: [legacyPage]
        })
      )
    ).toBe(2);
  });

  it("defaults to the current structure when version is missing and no manual fields exist", () => {
    expect(inferProjectDataStructureVersion(project())).toBe(2);
  });

  it("treats mixed missing-version projects as legacy when page-level manual data still exists", () => {
    const mixedPage = page();
    mixedPage.manualQuad = [
      [3, 2],
      [98, 1],
      [97, 97],
      [2, 98]
    ];
    mixedPage.candidates[1] = candidate(
      "v28",
      [
        [2, 1],
        [99, 0],
        [98, 98],
        [1, 99]
      ],
      {
        manualQuad: [
          [4, 3],
          [97, 2],
          [96, 96],
          [3, 97]
        ]
      }
    );

    expect(inferProjectDataStructureVersion(project({ pages: [mixedPage] }))).toBe(1);
  });
});
