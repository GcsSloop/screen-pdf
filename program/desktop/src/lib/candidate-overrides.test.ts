import { describe, expect, it } from "vitest";

import {
  clearCandidateManualOverride,
  getEffectiveCandidateQuad,
  migrateLegacyManualOverride
} from "./candidate-overrides";
import type { Candidate, PageRecord, Point } from "./types";

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
});
