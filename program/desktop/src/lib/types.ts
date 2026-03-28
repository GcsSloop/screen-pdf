export type Point = [number, number];
export type DifficultyBucket = "clean" | "hard" | "abnormal";
export type FailureTag =
  | "corner_out_of_frame"
  | "edge_touch_border"
  | "heavy_occlusion"
  | "edge_only_visible"
  | "black_frame"
  | "low_contrast"
  | "strong_perspective"
  | "large_spill"
  | "candidate_disagreement";

export interface Candidate {
  method: string;
  score: number;
  quad: Point[];
  originalQuad?: Point[];
  manualQuad?: Point[] | null;
  metrics: Record<string, number>;
  source?: string;
  modelId?: string;
  debugOnly?: boolean;
}

export interface PageDetails {
  width: number;
  height: number;
  fileSizeBytes: number;
  capturedAt?: string | null;
  createdAt: string;
  modifiedAt: string;
}

export interface PageRecord {
  id: string;
  name: string;
  path: string;
  thumbPath?: string | null;
  createdAt: string;
  status: "new" | "auto_ready" | "needs_review" | "reviewed" | "error";
  confidence: number;
  bestMethod?: string;
  selectedCandidateIndex: number;
  candidates: Candidate[];
  activeQuad: Point[];
  manualQuad?: Point[] | null;
  manualBaseCandidateIndex?: number | null;
  previewPath?: string | null;
  details: PageDetails;
  eventSlug?: string | null;
  difficultyBucket?: DifficultyBucket | null;
  failureTags?: FailureTag[] | null;
  bucketReason?: string[] | null;
  reviewTags?: string[] | null;
  tagVersion?: number | null;
}

export interface ProjectFile {
  version: number;
  name: string;
  sourceDir: string;
  projectPath?: string | null;
  selectedPageId?: string | null;
  eventSlug?: string | null;
  eventName?: string | null;
  tagVersion?: number | null;
  tagSummary?: {
    pages?: number;
    bucketCounts?: Record<string, number>;
    failureTagCounts?: Record<string, number>;
  } | null;
  pages: PageRecord[];
}

export interface ExportOptions {
  outputPath: string;
  exportAllPages: boolean;
  includeAutoReady: boolean;
  minAutoReadyConfidence: number;
  jpegQuality: number;
  maxDimension: number;
  ocrEnabled: boolean;
  ocrLanguages: string;
}

export interface ExportedPage {
  id: string;
  name: string;
  imagePath: string;
  pdfPath: string;
  ocrTextPath?: string | null;
  warning?: string | null;
}

export interface ExportResult {
  outputPath: string;
  reportPath: string;
  pageCount: number;
  effectiveOcrLanguages?: string | null;
  warnings: string[];
  pages: ExportedPage[];
}

export interface ScanProgressEvent {
  scanId: number;
  phase: "started" | "processing" | "completed" | "cancelled";
  processed: number;
  total: number;
  currentName?: string | null;
  message: string;
}
