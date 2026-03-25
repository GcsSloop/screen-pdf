export type Point = [number, number];

export interface Candidate {
  method: string;
  score: number;
  quad: Point[];
  metrics: Record<string, number>;
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
  previewPath?: string | null;
  details: PageDetails;
}

export interface ProjectFile {
  version: number;
  name: string;
  sourceDir: string;
  projectPath?: string | null;
  selectedPageId?: string | null;
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
