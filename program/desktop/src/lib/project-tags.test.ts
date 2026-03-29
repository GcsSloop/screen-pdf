import { describe, expect, it } from "vitest";

import {
  DEFAULT_REVIEW_TAGS,
  formatBucketLabel,
  formatFailureTags,
  normalizeFailureTags,
  sanitizeBucket
} from "./project-tags";

describe("project tag helpers", () => {
  it("sanitizes unknown buckets to clean", () => {
    expect(sanitizeBucket("weird")).toBe("clean");
    expect(sanitizeBucket("hard")).toBe("hard");
  });

  it("keeps only known failure tags and de-duplicates them", () => {
    expect(normalizeFailureTags(["large_spill", "large_spill", "unknown"])).toEqual(["large_spill"]);
  });

  it("formats readable labels for UI", () => {
    expect(formatBucketLabel("abnormal")).toBe("异常");
    expect(formatFailureTags(["corner_out_of_frame", "black_frame"])).toContain("角点出界");
  });

  it("exposes default review tags for auto labeled pages", () => {
    expect(DEFAULT_REVIEW_TAGS).toEqual(["auto"]);
  });
});
