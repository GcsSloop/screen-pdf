import { describe, expect, it } from "vitest";

import { buildCandidateDebugLabel, buildCandidateRowMeta, buildCandidateTitle } from "./candidate-display";
import type { Candidate } from "./types";

function candidate(overrides: Partial<Candidate> = {}): Candidate {
  return {
    method: "contour_quad",
    score: 0.88,
    quad: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1]
    ],
    metrics: {},
    ...overrides
  };
}

describe("candidate display", () => {
  it("builds a concise title from model id when present", () => {
    expect(buildCandidateTitle(candidate({ modelId: "deep_screen_v1_round_022", method: "deep_screen_v1_best" }))).toBe(
      "deep_screen_v1_round_022"
    );
  });

  it("falls back to method when model id is missing", () => {
    expect(buildCandidateTitle(candidate())).toBe("contour_quad");
  });

  it("includes source and debug flag in the label", () => {
    expect(
      buildCandidateDebugLabel(
        candidate({
          source: "runtime_student",
          debugOnly: true
        })
      )
    ).toBe("runtime_student · debug");
  });

  it("omits the debug suffix for normal candidates", () => {
    expect(
      buildCandidateDebugLabel(
        candidate({
          source: "runtime_teacher",
          debugOnly: false
        })
      )
    ).toBe("runtime_teacher");
  });

  it("builds restore button state without squeezing candidate copy into the title row", () => {
    expect(
      buildCandidateRowMeta(
        candidate({
          source: "runtime_teacher",
          manualQuad: [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
          ]
        })
      )
    ).toEqual({
      scoreLabel: "评分 0.8800 · 已人工调整",
      restoreButtonLabel: "恢复默认",
      restoreDisabled: false
    });
  });
});
