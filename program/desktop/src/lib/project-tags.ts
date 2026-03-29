export const BUCKET_OPTIONS = ["clean", "hard", "abnormal"] as const;
export type DifficultyBucket = (typeof BUCKET_OPTIONS)[number];

export const FAILURE_TAG_OPTIONS = [
  "corner_out_of_frame",
  "edge_touch_border",
  "heavy_occlusion",
  "edge_only_visible",
  "black_frame",
  "low_contrast",
  "strong_perspective",
  "large_spill",
  "candidate_disagreement"
] as const;
export type FailureTag = (typeof FAILURE_TAG_OPTIONS)[number];

export const DEFAULT_REVIEW_TAGS = ["auto"];

const BUCKET_LABELS: Record<DifficultyBucket, string> = {
  clean: "干净",
  hard: "困难",
  abnormal: "异常"
};

const FAILURE_TAG_LABELS: Record<FailureTag, string> = {
  corner_out_of_frame: "角点出界",
  edge_touch_border: "贴边",
  heavy_occlusion: "明显遮挡",
  edge_only_visible: "仅边缘可见",
  black_frame: "黑边",
  low_contrast: "低对比",
  strong_perspective: "强透视",
  large_spill: "外溢较大",
  candidate_disagreement: "候选分歧"
};

export function sanitizeBucket(value: string | null | undefined): DifficultyBucket {
  return BUCKET_OPTIONS.includes(value as DifficultyBucket) ? (value as DifficultyBucket) : "clean";
}

export function normalizeFailureTags(tags: string[] | null | undefined): FailureTag[] {
  const normalized: FailureTag[] = [];
  for (const tag of tags ?? []) {
    if (FAILURE_TAG_OPTIONS.includes(tag as FailureTag) && !normalized.includes(tag as FailureTag)) {
      normalized.push(tag as FailureTag);
    }
  }
  return normalized;
}

export function formatBucketLabel(bucket: DifficultyBucket): string {
  return BUCKET_LABELS[bucket];
}

export function formatFailureTags(tags: string[] | null | undefined): string {
  const normalized = normalizeFailureTags(tags);
  if (normalized.length === 0) {
    return "无";
  }
  return normalized.map((tag) => FAILURE_TAG_LABELS[tag]).join("、");
}
