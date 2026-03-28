import "./styles.css";

import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import appIconUrl from "../icons/screen-pdf-desktop.png";
import { buildCandidateDebugLabel, buildCandidateRowMeta, buildCandidateTitle } from "./lib/candidate-display";
import {
  clearCandidateManualOverride,
  getEffectiveCandidateQuad,
  migrateLegacyManualOverride
} from "./lib/candidate-overrides";

import {
  buildExportPanelState,
  removePageFromProject,
  resolveActivePage,
  withSelectedPage
} from "./lib/export-flow";
import {
  BUCKET_OPTIONS,
  DEFAULT_REVIEW_TAGS,
  FAILURE_TAG_OPTIONS,
  formatBucketLabel,
  formatFailureTags,
  normalizeFailureTags,
  sanitizeBucket
} from "./lib/project-tags";
import {
  buildPolygonPoints,
  calculateDisplayGeometry,
  clampDisplayPoint,
  clampPointToImage,
  projectQuadToDisplay,
  updateQuadHandle,
  type DisplayGeometry
} from "./lib/editor-geometry";
import {
  MAGNIFIER_SIZE,
  TOOLBAR_ICON_BUTTONS,
  renderToolbarIcon,
  resolveMagnifierPlacement
} from "./lib/editor-ui";
import {
  applyCandidateToPage,
  applyDraftQuadToPage,
  buildPreviewVersionedPath,
  movePage,
  resolveWorkingQuad
} from "./lib/page-flow";
import {
  buildDisplaySourceCandidates,
  buildPreviewSourceCandidates,
  buildThumbnailSourceCandidates,
  canCommitPageRender,
  resolveIntrinsicImageSize
} from "./lib/render-flow";
import { initialState } from "./lib/state";
import type {
  DifficultyBucket,
  ExportOptions,
  ExportResult,
  FailureTag,
  PageDetails,
  PageRecord,
  Point,
  ProjectFile,
  ScanProgressEvent
} from "./lib/types";

const state = initialState();
const toolbarButtonMap = new Map(TOOLBAR_ICON_BUTTONS.map((item) => [item.id, item] as const));

function renderToolbarButton(id: string): string {
  const config = toolbarButtonMap.get(id);
  if (!config) return "";
  return `
    <div class="toolbar-tip-shell">
      <button
        id="${config.id}"
        class="toolbar-action icon-only${config.variant === "primary" ? " primary-action" : ""}"
        aria-label="${config.label}"
        title="${config.tooltip}"
      >
        <span class="toolbar-icon modern-icon" aria-hidden="true">${renderToolbarIcon(config.icon)}</span>
      </button>
      <div class="toolbar-tooltip" role="tooltip">
        <strong>${config.label}</strong>
        <span>${config.tooltip}</span>
      </div>
    </div>
  `;
}

const DEFAULT_EXPORT_OPTIONS = (): ExportOptions => ({
  outputPath: "",
  exportAllPages: true,
  includeAutoReady: true,
  minAutoReadyConfidence: 0.2,
  jpegQuality: 82,
  maxDimension: 2200,
  ocrEnabled: true,
  ocrLanguages: "chi_sim+eng"
});

let exportOptions = DEFAULT_EXPORT_OPTIONS();
let exportState: {
  mode: "idle" | "running" | "success" | "error";
  message?: string;
  result?: ExportResult | null;
} = {
  mode: "idle",
  result: null
};

const app = document.querySelector("#app") as HTMLDivElement;
app.innerHTML = `
  <div class="toolbar">
      <div class="toolbar-brand">
      <img class="brand-mark" src="${appIconUrl}" alt="ScreenPDF 图标" />
      <div class="brand-copy">
        <strong>ScreenPDF</strong>
        <span>投屏照片矫正与可检索 PDF</span>
      </div>
    </div>
    <div class="toolbar-actions">
      ${renderToolbarButton("openFolderBtn")}
      ${renderToolbarButton("loadProjectBtn")}
      ${renderToolbarButton("saveProjectBtn")}
      <div class="toolbar-export">
        <div class="toolbar-popover-shell">
          ${renderToolbarButton("toggleExportSettingsBtn")}
          <div id="exportPopover" class="toolbar-popover hidden">
            <div class="toolbar-popover-head">
              <h3>导出配置</h3>
              <span>点击导出时立即生效</span>
            </div>
            <div class="form-grid">
              <label>输出 PDF</label>
              <div class="inline-field">
                <input id="outputPathInput" class="path-input" readonly />
                <button id="chooseOutputBtn">选择</button>
              </div>
              <label class="check-row">
                <input id="exportAllPagesInput" type="checkbox" checked />
                导出所有页面
              </label>
              <div class="export-tip">默认包含待确认页面；关闭后按页面状态与自动就绪阈值筛选。</div>
              <label class="check-row">
                <input id="includeAutoReadyInput" type="checkbox" checked />
                包含可信的自动就绪页面
              </label>
              <label>自动就绪最低置信度</label>
              <input id="minConfidenceInput" type="number" min="0" max="1" step="0.01" value="0.2" />
              <label>JPEG 质量</label>
              <input id="jpegQualityInput" type="number" min="40" max="100" step="1" value="82" />
              <label>最长边尺寸</label>
              <input id="maxDimensionInput" type="number" min="1200" max="4000" step="100" value="2200" />
              <label class="check-row">
                <input id="ocrEnabledInput" type="checkbox" checked />
                启用 OCR 隐藏文本层
              </label>
              <label>OCR 语言</label>
              <input id="ocrLanguagesInput" type="text" value="chi_sim+eng" />
              <div id="exportStatus" class="status-panel empty">打开文件夹后即可配置导出。</div>
            </div>
          </div>
        </div>
        ${renderToolbarButton("exportProjectBtn")}
      </div>
    </div>
    <div class="meta">
      <span id="projectName" class="meta-pill project-pill">未打开项目</span>
      <span id="pageCount" class="meta-pill">0 页</span>
      <span id="reviewCount" class="meta-pill accent-pill">0 页待确认</span>
    </div>
  </div>
  <div class="layout">
    <aside class="panel panel-pages">
      <div class="section-head">
        <h2>页面</h2>
        <span id="pageOrderHint" class="section-tip">默认按创建时间排序</span>
      </div>
      <div id="pageList" class="page-list"></div>
    </aside>
    <main class="editor">
      <div class="canvas-shell">
        <div id="editorStage" class="editor-stage">
          <img id="editorImage" alt="编辑预览" />
          <svg id="editorOverlay" class="editor-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
            <polygon id="editorPolygon" class="editor-polygon"></polygon>
          </svg>
          <div id="editorMagnifier" class="editor-magnifier hidden" aria-hidden="true">
            <div class="editor-magnifier-crosshair"></div>
          </div>
          <button class="corner-handle" data-handle-index="0" aria-label="左上角锚点"></button>
          <button class="corner-handle" data-handle-index="1" aria-label="右上角锚点"></button>
          <button class="corner-handle" data-handle-index="2" aria-label="右下角锚点"></button>
          <button class="corner-handle" data-handle-index="3" aria-label="左下角锚点"></button>
        </div>
      </div>
      <div class="editor-actions">
        <div class="hintbar">
          <span>四个锚点可独立拖动，松手后自动应用裁剪</span>
          <span>左侧拖拽顺序即导出顺序</span>
          <span>右侧保留一个人工标注候选，并显示 base 来源</span>
        </div>
      </div>
    </main>
    <aside class="panel panel-right">
      <div class="section">
        <h2>预览</h2>
        <div class="preview"><img id="previewImage" alt="preview" /></div>
      </div>
      <div class="section">
        <h3>候选方案</h3>
        <div id="candidateList" class="candidates"></div>
      </div>
      <div class="section">
        <h3>当前页面</h3>
        <div id="pageMeta" class="metrics empty">打开文件夹后开始处理。</div>
      </div>
    </aside>
  </div>
  <div id="infoModal" class="modal hidden">
    <div id="infoModalBackdrop" class="modal-backdrop"></div>
    <div class="modal-card">
      <div class="modal-head">
        <h3>图片详情</h3>
        <button id="closeInfoModalBtn" class="icon-button">关闭</button>
      </div>
      <div id="infoModalBody" class="modal-body"></div>
    </div>
  </div>
  <div id="scanModal" class="modal hidden">
    <div class="modal-backdrop"></div>
    <div class="modal-card scan-modal-card">
      <div class="modal-head">
        <h3>正在打开项目</h3>
        <button id="cancelScanBtn" class="danger-button">中断</button>
      </div>
      <div id="scanStatusText" class="scan-status-text">准备开始...</div>
      <div class="progress-track">
        <div id="scanProgressBar" class="progress-bar"></div>
      </div>
      <div id="scanProgressMeta" class="scan-progress-meta">0 / 0</div>
    </div>
  </div>
`;

const openFolderBtn = document.querySelector("#openFolderBtn") as HTMLButtonElement;
const loadProjectBtn = document.querySelector("#loadProjectBtn") as HTMLButtonElement;
const saveProjectBtn = document.querySelector("#saveProjectBtn") as HTMLButtonElement;
const toggleExportSettingsBtn = document.querySelector("#toggleExportSettingsBtn") as HTMLButtonElement;
const exportPopover = document.querySelector("#exportPopover") as HTMLDivElement;
const chooseOutputBtn = document.querySelector("#chooseOutputBtn") as HTMLButtonElement;
const exportProjectBtn = document.querySelector("#exportProjectBtn") as HTMLButtonElement;
const outputPathInput = document.querySelector("#outputPathInput") as HTMLInputElement;
const exportAllPagesInput = document.querySelector("#exportAllPagesInput") as HTMLInputElement;
const includeAutoReadyInput = document.querySelector("#includeAutoReadyInput") as HTMLInputElement;
const minConfidenceInput = document.querySelector("#minConfidenceInput") as HTMLInputElement;
const jpegQualityInput = document.querySelector("#jpegQualityInput") as HTMLInputElement;
const maxDimensionInput = document.querySelector("#maxDimensionInput") as HTMLInputElement;
const ocrEnabledInput = document.querySelector("#ocrEnabledInput") as HTMLInputElement;
const ocrLanguagesInput = document.querySelector("#ocrLanguagesInput") as HTMLInputElement;
const exportStatus = document.querySelector("#exportStatus") as HTMLDivElement;
const pageList = document.querySelector("#pageList") as HTMLDivElement;
const pageMeta = document.querySelector("#pageMeta") as HTMLDivElement;
const candidateList = document.querySelector("#candidateList") as HTMLDivElement;
const pageOrderHint = document.querySelector("#pageOrderHint") as HTMLSpanElement;
const projectName = document.querySelector("#projectName") as HTMLSpanElement;
const pageCount = document.querySelector("#pageCount") as HTMLSpanElement;
const reviewCount = document.querySelector("#reviewCount") as HTMLSpanElement;
const previewImage = document.querySelector("#previewImage") as HTMLImageElement;
const editorStage = document.querySelector("#editorStage") as HTMLDivElement;
const editorImage = document.querySelector("#editorImage") as HTMLImageElement;
const editorMagnifier = document.querySelector("#editorMagnifier") as HTMLDivElement;
const editorOverlay = document.querySelector("#editorOverlay") as SVGSVGElement;
const editorPolygon = document.querySelector("#editorPolygon") as SVGPolygonElement;
const cornerHandles = Array.from(document.querySelectorAll(".corner-handle")) as HTMLButtonElement[];
const infoModal = document.querySelector("#infoModal") as HTMLDivElement;
const infoModalBackdrop = document.querySelector("#infoModalBackdrop") as HTMLDivElement;
const closeInfoModalBtn = document.querySelector("#closeInfoModalBtn") as HTMLButtonElement;
const infoModalBody = document.querySelector("#infoModalBody") as HTMLDivElement;
const scanModal = document.querySelector("#scanModal") as HTMLDivElement;
const cancelScanBtn = document.querySelector("#cancelScanBtn") as HTMLButtonElement;
const scanStatusText = document.querySelector("#scanStatusText") as HTMLDivElement;
const scanProgressBar = document.querySelector("#scanProgressBar") as HTMLDivElement;
const scanProgressMeta = document.querySelector("#scanProgressMeta") as HTMLDivElement;
let previewSourceState: { sources: string[]; index: number } | null = null;
let currentImage: HTMLImageElement | null = null;
let currentImageUsesFallback = false;
let thumbObserver: IntersectionObserver | null = null;
const thumbSourceState = new WeakMap<HTMLImageElement, { sources: string[]; index: number }>();
let displayGeometry: DisplayGeometry = {
  width: 1000,
  height: 700,
  scale: 1
};
let previewVersion = 0;
let activeViewRequestId = 0;
let exportPopoverOpen = false;
let scanUiState: {
  visible: boolean;
  cancellable: boolean;
  processed: number;
  total: number;
  message: string;
  currentName?: string | null;
} = {
  visible: false,
  cancellable: false,
  processed: 0,
  total: 0,
  message: "准备开始..."
};

function showAppError(message: string) {
  exportState = {
    mode: "error",
    message,
    result: null
  };
  renderExportSection();
}

function setPreviewSource(previewPath: string) {
  previewVersion += 1;
  previewImage.src = buildPreviewVersionedPath(convertFileSrc(previewPath), previewVersion);
}

function updatePreviewImage(page: PageRecord) {
  previewVersion += 1;
  previewSourceState = {
    sources: buildPreviewSourceCandidates(page.path, page.previewPath).map((path) =>
      path === page.previewPath ? buildPreviewVersionedPath(convertFileSrc(path), previewVersion) : convertFileSrc(path)
    ),
    index: 0
  };
  previewImage.src = previewSourceState.sources[0] ?? "";
}

function commitMatchesActivePage(requestId: number, pageId: string): boolean {
  return canCommitPageRender(requestId, activeViewRequestId, pageId, state.activePage?.id);
}

async function loadDisplayImage(
  page: PageRecord,
  preferPreview = false
): Promise<{ image: HTMLImageElement; usedFallback: boolean }> {
  const sources = buildDisplaySourceCandidates(page.path, page.previewPath, preferPreview).map((path) =>
    path === page.previewPath
      ? buildPreviewVersionedPath(convertFileSrc(path), previewVersion)
      : convertFileSrc(path)
  );
  let lastError: unknown = null;

  for (const [index, source] of sources.entries()) {
    const image = new Image();
    image.decoding = "async";

    try {
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error(`failed to load image source: ${source}`));
        image.src = source;
        if (image.complete && image.naturalWidth > 0) {
          resolve();
        }
      });
      return {
        image,
        usedFallback: index > 0
      };
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError ?? new Error(`failed to load image for ${page.name}`);
}

function sanitizeFileName(value: string): string {
  const normalized = value.trim().replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "-");
  return normalized || "screen-pdf-export";
}

function defaultOutputPath(project: ProjectFile): string {
  return `${project.sourceDir}/${sanitizeFileName(project.name)}.pdf`;
}

function getActivePage(): PageRecord | null {
  if (!state.project || !state.activePage) {
    return null;
  }
  return state.project.pages.find((page) => page.id === state.activePage?.id) ?? null;
}

function workingQuad(page: PageRecord): Point[] {
  const draft = state.activePage?.id === page.id ? state.draftQuad : null;
  return resolveWorkingQuad(page, draft);
}

function quadEquals(left: Point[] | null | undefined, right: Point[] | null | undefined): boolean {
  if (!left || !right || left.length !== right.length) return false;
  return left.every((point, index) => point[0] === right[index][0] && point[1] === right[index][1]);
}

function hasPendingDraft(page: PageRecord | null): boolean {
  if (!page || !state.draftQuad) return false;
  return !quadEquals(state.draftQuad, page.activeQuad);
}

function currentMethodLabel(page: PageRecord): string {
  const candidate = page.candidates[page.selectedCandidateIndex];
  if (!candidate) {
    return page.bestMethod ?? "manual";
  }
  return candidate.manualQuad?.length
    ? `${buildCandidateTitle(candidate)} · 已人工调整`
    : buildCandidateTitle(candidate);
}

function currentManualBaseLabel(page: PageRecord): string {
  const candidate = page.candidates[page.selectedCandidateIndex];
  if (!candidate?.manualQuad?.length) {
    return "无";
  }
  return `${buildCandidateTitle(candidate)}（当前候选）`;
}

function statusClass(status: PageRecord["status"]): string {
  if (status === "reviewed") return "reviewed";
  if (status === "needs_review") return "review";
  return "ready";
}

function statusLabel(status: PageRecord["status"]): string {
  if (status === "reviewed") return "已确认";
  if (status === "needs_review") return "待确认";
  if (status === "auto_ready") return "自动就绪";
  if (status === "error") return "异常";
  return "新建";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function detailsHtml(page: PageRecord): string {
  const details: PageDetails = page.details;
  return `
    <div class="details-grid">
      <div><strong>名称</strong><span>${page.name}</span></div>
      <div><strong>场次</strong><span>${page.eventSlug ?? "未设置"}</span></div>
      <div><strong>难度</strong><span>${formatBucketLabel(sanitizeBucket(page.difficultyBucket))}</span></div>
      <div><strong>标签</strong><span>${formatFailureTags(page.failureTags)}</span></div>
      <div><strong>尺寸</strong><span>${details.width} x ${details.height}</span></div>
      <div><strong>拍摄时间</strong><span>${details.capturedAt ?? "无"}</span></div>
      <div><strong>创建时间</strong><span>${details.createdAt}</span></div>
      <div><strong>修改时间</strong><span>${details.modifiedAt}</span></div>
      <div><strong>文件大小</strong><span>${formatFileSize(details.fileSizeBytes)}</span></div>
      <div class="full"><strong>路径</strong><span>${page.path}</span></div>
    </div>
  `;
}

function setProject(project: ProjectFile) {
  project.pages = project.pages.map((page) => migrateLegacyManualOverride(page));
  const activePage = resolveActivePage(project);
  state.activePage = activePage;
  state.project = withSelectedPage(project, activePage?.id ?? null);
  state.activeHandle = null;
  state.dragOrigin = null;
  state.draftQuad = null;
  state.infoPage = null;
  exportState = {
    mode: "idle",
    result: null
  };
  exportPopoverOpen = false;
  exportOptions = {
    ...DEFAULT_EXPORT_OPTIONS(),
    outputPath: defaultOutputPath(project)
  };
  renderPageList();
  renderMeta();
  renderExportSection();
  renderModal();
  void renderActivePage();
}

function recomputeTagSummary(project: ProjectFile) {
  const bucketCounts: Record<string, number> = {};
  const failureTagCounts: Record<string, number> = {};
  for (const page of project.pages) {
    const bucket = sanitizeBucket(page.difficultyBucket);
    bucketCounts[bucket] = (bucketCounts[bucket] ?? 0) + 1;
    for (const tag of normalizeFailureTags(page.failureTags)) {
      failureTagCounts[tag] = (failureTagCounts[tag] ?? 0) + 1;
    }
  }
  project.tagSummary = {
    pages: project.pages.length,
    bucketCounts,
    failureTagCounts
  };
}

function resetAppState() {
  state.project = null;
  state.activePage = null;
  state.activeHandle = null;
  state.dragOrigin = null;
  state.draftQuad = null;
  state.dragPageId = null;
  state.infoPage = null;
  currentImage = null;
  currentImageUsesFallback = false;
  activeViewRequestId += 1;
  previewVersion = 0;
  exportPopoverOpen = false;
  exportState = {
    mode: "idle",
    result: null
  };
  exportOptions = DEFAULT_EXPORT_OPTIONS();
  renderPageList();
  renderMeta();
  renderExportSection();
  renderModal();
  void renderActivePage();
}

function renderScanModal() {
  if (!scanUiState.visible) {
    scanModal.classList.add("hidden");
    return;
  }
  scanModal.classList.remove("hidden");
  scanStatusText.textContent = scanUiState.currentName
    ? `${scanUiState.message} · ${scanUiState.currentName}`
    : scanUiState.message;
  const total = Math.max(0, scanUiState.total);
  const processed = Math.max(0, Math.min(scanUiState.processed, total || scanUiState.processed));
  const progress = total > 0 ? Math.round((processed / total) * 100) : 0;
  scanProgressBar.style.width = `${progress}%`;
  scanProgressMeta.textContent = total > 0 ? `${processed} / ${total}` : "准备中";
  cancelScanBtn.disabled = !scanUiState.cancellable;
  cancelScanBtn.textContent = scanUiState.cancellable ? "中断" : "处理中...";
}

function renderMeta() {
  const project = state.project;
  projectName.textContent = project ? `${project.name}${project.eventName ? ` · ${project.eventName}` : ""}` : "未打开项目";
  pageCount.textContent = project ? `${project.pages.length} 页` : "0 页";
  reviewCount.textContent = project
    ? `${project.pages.filter((page) => page.status === "needs_review").length} 页待确认`
    : "0 页待确认";
}

function updatePageBucket(page: PageRecord, bucket: DifficultyBucket) {
  page.difficultyBucket = bucket;
  page.reviewTags = page.reviewTags?.length ? page.reviewTags : [...DEFAULT_REVIEW_TAGS];
  if (state.project) {
    recomputeTagSummary(state.project);
    renderMeta();
  }
  renderPageDetails(page);
}

function updatePageFailureTags(page: PageRecord, nextTags: FailureTag[]) {
  page.failureTags = normalizeFailureTags(nextTags);
  page.reviewTags = page.reviewTags?.length ? page.reviewTags : [...DEFAULT_REVIEW_TAGS];
  if (state.project) {
    recomputeTagSummary(state.project);
    renderMeta();
  }
  renderPageDetails(page);
}

function setActivePage(pageId: string) {
  if (!state.project) return;
  const page = state.project.pages.find((item) => item.id === pageId) ?? null;
  state.activePage = page;
  state.project = withSelectedPage(state.project, pageId);
  state.activeHandle = null;
  state.dragOrigin = null;
  state.draftQuad = null;
  renderPageList();
  void renderActivePage();
}

function ensureThumbObserver() {
  if (thumbObserver) {
    return thumbObserver;
  }

  thumbObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }
        const thumb = entry.target;
        if (!(thumb instanceof HTMLImageElement) || thumb.getAttribute("src")) {
          continue;
        }
        const state = thumbSourceState.get(thumb);
        if (!state || state.sources.length === 0) {
          continue;
        }
        thumb.src = state.sources[state.index];
      }
    },
    {
      root: pageList,
      rootMargin: "240px 0px"
    }
  );

  return thumbObserver;
}

function attachThumbSource(thumb: HTMLImageElement, page: PageRecord) {
  const sources = buildThumbnailSourceCandidates(page.path, page.thumbPath, page.previewPath).map((path) =>
    convertFileSrc(path)
  );
  thumbSourceState.set(thumb, { sources, index: 0 });
  thumb.removeAttribute("src");
  thumb.classList.add("pending");
  thumb.addEventListener("load", () => {
    thumb.classList.remove("pending");
    thumb.classList.remove("failed");
  });
  thumb.addEventListener("error", () => {
    const state = thumbSourceState.get(thumb);
    if (!state) {
      return;
    }
    state.index += 1;
    if (state.index < state.sources.length) {
      thumb.src = state.sources[state.index];
      return;
    }
    thumb.classList.remove("pending");
    thumb.classList.add("failed");
  });
  ensureThumbObserver().observe(thumb);
}

function renderPageList() {
  const project = state.project;
  thumbObserver?.disconnect();
  pageList.innerHTML = "";
  if (!project) {
    pageOrderHint.textContent = "默认按创建时间排序";
    pageList.innerHTML = `<div class="empty">打开文件夹或加载项目文件。</div>`;
    return;
  }
  pageOrderHint.textContent = "拖拽卡片可调整导出顺序";

  for (const [index, page] of project.pages.entries()) {
    const item = document.createElement("div");
    item.className = `page-item${state.activePage?.id === page.id ? " active" : ""}`;
    item.draggable = true;
    item.dataset.pageId = page.id;
    item.innerHTML = `
      <div class="thumb-shell">
        <img class="thumb pending" alt="${page.name}" loading="lazy" />
        <button class="delete-pill" data-delete-id="${page.id}" title="删除页面">×</button>
        <button class="info-pill" data-info-id="${page.id}" title="图片详情">i</button>
        <div class="page-tooltip">${detailsHtml(page)}</div>
      </div>
      <div class="page-card-body">
        <div class="topline">
          <strong>${index + 1}. ${page.name}</strong>
          <span class="badge ${statusClass(page.status)}">${statusLabel(page.status)}</span>
        </div>
        <div class="confidence">置信度 ${page.confidence.toFixed(3)} · ${page.bestMethod ?? "manual"}</div>
      </div>
    `;
    const thumb = item.querySelector(".thumb") as HTMLImageElement;
    attachThumbSource(thumb, page);
    item.addEventListener("click", () => {
      setActivePage(page.id);
    });
    item.addEventListener("dragstart", () => {
      state.dragPageId = page.id;
      item.classList.add("dragging");
    });
    item.addEventListener("dragend", () => {
      state.dragPageId = null;
      item.classList.remove("dragging");
    });
    item.addEventListener("dragover", (event) => {
      event.preventDefault();
      item.classList.add("drop-target");
    });
    item.addEventListener("dragleave", () => {
      item.classList.remove("drop-target");
    });
    item.addEventListener("drop", (event) => {
      event.preventDefault();
      item.classList.remove("drop-target");
      if (!state.project || !state.dragPageId || state.dragPageId === page.id) return;
      state.project = {
        ...state.project,
        pages: movePage(state.project.pages, state.dragPageId, page.id)
      };
      state.dragPageId = null;
      renderPageList();
      renderMeta();
      renderExportSection();
    });

    const infoButton = item.querySelector(`[data-info-id="${page.id}"]`) as HTMLButtonElement;
    infoButton.addEventListener("click", (event) => {
      event.stopPropagation();
      state.infoPage = page;
      renderModal();
    });

    const deleteButton = item.querySelector(`[data-delete-id="${page.id}"]`) as HTMLButtonElement;
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!state.project) return;
      state.project = removePageFromProject(state.project, page.id);
      state.activePage = resolveActivePage(state.project);
      state.infoPage = state.infoPage?.id === page.id ? null : state.infoPage;
      state.activeHandle = null;
      state.dragOrigin = null;
      state.draftQuad = null;
      currentImage = null;
      currentImageUsesFallback = false;
      activeViewRequestId += 1;
      renderPageList();
      renderMeta();
      renderExportSection();
      renderModal();
      void renderActivePage();
    });
    pageList.appendChild(item);
  }
}

function renderModal() {
  if (!state.infoPage) {
    infoModal.classList.add("hidden");
    infoModalBody.innerHTML = "";
    return;
  }
  infoModal.classList.remove("hidden");
  infoModalBody.innerHTML = detailsHtml(state.infoPage);
}

function beginScanSession() {
  state.scanSessionId += 1;
  scanUiState = {
    visible: true,
    cancellable: true,
    processed: 0,
    total: 0,
    message: "正在准备扫描..."
  };
  renderScanModal();
  return state.scanSessionId;
}

function updateScanSession(scanSessionId: number, payload: ScanProgressEvent) {
  if (scanSessionId !== state.scanSessionId || payload.scanId !== scanSessionId) return;
  scanUiState = {
    ...scanUiState,
    visible: payload.phase !== "completed" && payload.phase !== "cancelled",
    cancellable: payload.phase === "started" || payload.phase === "processing",
    processed: payload.processed,
    total: payload.total,
    message: payload.message,
    currentName: payload.currentName ?? null
  };
  renderScanModal();
}

function finishScanSession(scanSessionId: number) {
  if (scanSessionId !== state.scanSessionId) return;
  scanUiState = {
    ...scanUiState,
    visible: false,
    cancellable: false
  };
  renderScanModal();
}

async function ensurePreview(page: PageRecord, requestId?: number): Promise<string | null> {
  try {
    const previewPath = await invoke<string>("generate_preview", {
      imagePath: page.path,
      quad: page.activeQuad
    });
    page.previewPath = previewPath;
    if (requestId === undefined || commitMatchesActivePage(requestId, page.id)) {
      setPreviewSource(previewPath);
    }
    return previewPath;
  } catch (error) {
    if (requestId === undefined || commitMatchesActivePage(requestId, page.id)) {
      showAppError(String(error));
    }
    return null;
  }
}

function renderCandidateList(page: PageRecord) {
  candidateList.innerHTML = "";
  for (const [index, candidate] of page.candidates.entries()) {
    const hasManualOverride = Boolean(candidate.manualQuad?.length);
    const rowMeta = buildCandidateRowMeta(candidate);
    const row = document.createElement("div");
    row.className = `candidate-row${index === page.selectedCandidateIndex ? " active" : ""}${
      hasManualOverride ? " manual" : ""
    }`;
    row.innerHTML = `
      <div class="candidate-row-head">
        <strong>${buildCandidateTitle(candidate)}</strong>
      </div>
      <div class="meta">${buildCandidateDebugLabel(candidate)}</div>
      <div class="candidate-row-foot">
        <div class="score">${rowMeta.scoreLabel}</div>
        <button class="candidate-reset" ${rowMeta.restoreDisabled ? "disabled" : ""}>${rowMeta.restoreButtonLabel}</button>
      </div>
    `;
    row.addEventListener("click", () => {
      applyCandidate(page, index);
    });
    const resetButton = row.querySelector(".candidate-reset") as HTMLButtonElement;
    resetButton.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (resetButton.disabled || !state.project) {
        return;
      }
      if (!window.confirm(`确认将候选方案“${buildCandidateTitle(candidate)}”恢复为默认计算结果吗？`)) {
        return;
      }
      const updatedCandidate = clearCandidateManualOverride(candidate);
      const updatedPage: PageRecord = {
        ...page,
        candidates: page.candidates.map((entry, entryIndex) => (entryIndex === index ? updatedCandidate : entry)),
        activeQuad:
          index === page.selectedCandidateIndex ? getEffectiveCandidateQuad(updatedCandidate) : page.activeQuad,
        previewPath: null
      };
      state.project = {
        ...state.project,
        pages: state.project.pages.map((entry) => (entry.id === updatedPage.id ? updatedPage : entry))
      };
      state.activePage = updatedPage;
      renderCandidateList(updatedPage);
      renderPageDetails(updatedPage);
      renderPageList();
      drawCanvas(updatedPage);
      await ensurePreview(updatedPage);
    });
    candidateList.appendChild(row);
  }
}

function renderPageDetails(page: PageRecord) {
  const bucket = sanitizeBucket(page.difficultyBucket);
  const failureTags = normalizeFailureTags(page.failureTags);
  const bucketReason = (page.bucketReason ?? []).filter(Boolean);
  pageMeta.innerHTML = `
    <div class="page-meta-title">
      <strong>${page.name}</strong>
      <span class="meta-inline-note">${page.eventSlug ?? state.project?.eventSlug ?? "未设置场次"}</span>
    </div>
    <div>状态：${statusLabel(page.status)}</div>
    <div>置信度：${page.confidence.toFixed(4)}</div>
    <div>当前方案：${currentMethodLabel(page)}</div>
    <div>人工修正：${currentManualBaseLabel(page)}</div>
    <div>创建时间：${page.details.createdAt}</div>
    <div>拍摄时间：${page.details.capturedAt ?? "无"}</div>
    <div>图片尺寸：${page.details.width} x ${page.details.height}</div>
    <div class="tag-editor">
      <label for="difficultyBucketSelect"><strong>粗标签分桶</strong></label>
      <select id="difficultyBucketSelect">
        ${BUCKET_OPTIONS.map((option) => `<option value="${option}" ${option === bucket ? "selected" : ""}>${formatBucketLabel(option)}</option>`).join("")}
      </select>
      <div class="tag-editor-label"><strong>失败标签</strong></div>
      <div class="tag-checklist">
        ${FAILURE_TAG_OPTIONS.map(
          (tag) => `
            <label class="tag-check-row">
              <input type="checkbox" data-failure-tag="${tag}" ${failureTags.includes(tag) ? "checked" : ""} />
              <span>${formatFailureTags([tag])}</span>
            </label>
          `
        ).join("")}
      </div>
      <div class="tag-editor-note">自动原因：${bucketReason.length ? bucketReason.join("；") : "无"}</div>
    </div>
  `;
  const bucketSelect = pageMeta.querySelector("#difficultyBucketSelect") as HTMLSelectElement | null;
  bucketSelect?.addEventListener("change", () => {
    updatePageBucket(page, sanitizeBucket(bucketSelect.value));
  });
  const failureCheckboxes = Array.from(pageMeta.querySelectorAll("[data-failure-tag]")) as HTMLInputElement[];
  for (const checkbox of failureCheckboxes) {
    checkbox.addEventListener("change", () => {
      const nextTags = failureCheckboxes
        .filter((input) => input.checked)
        .map((input) => input.dataset.failureTag as FailureTag);
      updatePageFailureTags(page, nextTags);
    });
  }
}

async function renderActivePage() {
  const page = getActivePage();
  const requestId = ++activeViewRequestId;

  if (!page) {
    currentImage = null;
    currentImageUsesFallback = false;
    pageMeta.innerHTML = `<div class="empty">打开文件夹后开始处理。</div>`;
    candidateList.innerHTML = "";
    previewImage.removeAttribute("src");
    drawCanvas(null);
    return;
  }

  renderPageDetails(page);
  renderCandidateList(page);
  let loaded;
  try {
    loaded = await loadDisplayImage(page);
  } catch (initialError) {
    const previewPath = page.previewPath ?? (await ensurePreview(page, requestId));
    if (!commitMatchesActivePage(requestId, page.id) || !previewPath) {
      if (commitMatchesActivePage(requestId, page.id)) {
        currentImage = null;
        currentImageUsesFallback = false;
        drawCanvas(null);
        showAppError(String(initialError));
      }
      return;
    }

    try {
      loaded = await loadDisplayImage({ ...page, previewPath }, true);
    } catch (recoveryError) {
      if (commitMatchesActivePage(requestId, page.id)) {
        currentImage = null;
        currentImageUsesFallback = false;
        drawCanvas(null);
        showAppError(String(recoveryError));
      }
      return;
    }
  }

  if (!commitMatchesActivePage(requestId, page.id)) {
    return;
  }

  currentImage = loaded.image;
  currentImageUsesFallback = loaded.usedFallback;
  drawCanvas(page);
  if (!commitMatchesActivePage(requestId, page.id)) {
    return;
  }
  if (page.previewPath || page.path) {
    updatePreviewImage(page);
  } else {
    await ensurePreview(page, requestId);
  }
}

function drawCanvas(page: PageRecord | null) {
  if (!page || !currentImage) {
    currentImageUsesFallback = false;
    editorStage.style.width = "1000px";
    editorStage.style.height = "700px";
    editorImage.style.display = "none";
    editorImage.removeAttribute("src");
    editorImage.style.width = "1000px";
    editorImage.style.height = "700px";
    syncEditorOverlay(null);
    return;
  }
  const intrinsic = resolveIntrinsicImageSize(currentImage);
  const maxWidth = editorStage.parentElement?.clientWidth ?? 900;
  const maxHeight = editorStage.parentElement?.clientHeight ?? 700;
  displayGeometry = calculateDisplayGeometry(intrinsic.width, intrinsic.height, maxWidth, maxHeight);
  editorStage.style.width = `${displayGeometry.width}px`;
  editorStage.style.height = `${displayGeometry.height}px`;
  editorImage.src = currentImage.src;
  editorImage.style.display = "block";
  editorImage.style.width = `${displayGeometry.width}px`;
  editorImage.style.height = `${displayGeometry.height}px`;
  syncEditorOverlay(page);
}

function syncEditorOverlay(page: PageRecord | null) {
  if (!page || !currentImage || currentImageUsesFallback) {
    editorOverlay.setAttribute("viewBox", "0 0 1 1");
    editorPolygon.setAttribute("points", "");
    editorMagnifier.classList.add("hidden");
    for (const handle of cornerHandles) {
      handle.style.display = "none";
    }
    return;
  }

  const displayQuad = projectQuadToDisplay(workingQuad(page), displayGeometry);
  editorOverlay.setAttribute("viewBox", `0 0 ${displayGeometry.width} ${displayGeometry.height}`);
  editorPolygon.setAttribute("points", buildPolygonPoints(displayQuad));

  for (const [index, handle] of cornerHandles.entries()) {
    const point = clampDisplayPoint(displayQuad[index], displayGeometry, 14);
    handle.style.display = "block";
    handle.style.left = `${point[0]}px`;
    handle.style.top = `${point[1]}px`;
    handle.classList.toggle("active", state.activeHandle === index);
  }

  if (state.activeHandle === null) {
    editorMagnifier.classList.add("hidden");
    return;
  }

  const handlePoint = displayQuad[state.activeHandle];
  if (!handlePoint) {
    editorMagnifier.classList.add("hidden");
    return;
  }

  const placement = resolveMagnifierPlacement({
    point: handlePoint,
    activeHandle: state.activeHandle,
    quad: displayQuad,
    stage: {
      width: displayGeometry.width,
      height: displayGeometry.height
    }
  });
  const zoom = 3;
  editorMagnifier.classList.remove("hidden");
  editorMagnifier.style.left = `${placement.anchorX}px`;
  editorMagnifier.style.top = `${placement.anchorY}px`;
  editorMagnifier.style.backgroundImage = `url("${editorImage.src}")`;
  editorMagnifier.style.backgroundSize = `${displayGeometry.width * zoom}px ${displayGeometry.height * zoom}px`;
  editorMagnifier.style.backgroundPosition = `${-handlePoint[0] * zoom + MAGNIFIER_SIZE / 2}px ${-handlePoint[1] * zoom + MAGNIFIER_SIZE / 2}px`;
}

function screenToImagePoint(event: PointerEvent): Point {
  const rect = editorStage.getBoundingClientRect();
  if (!currentImage || rect.width <= 0 || rect.height <= 0) {
    return [0, 0];
  }
  const intrinsic = resolveIntrinsicImageSize(currentImage);
  return clampPointToImage(
    [
      ((event.clientX - rect.left) / rect.width) * intrinsic.width,
      ((event.clientY - rect.top) / rect.height) * intrinsic.height
    ],
    { width: intrinsic.width, height: intrinsic.height }
  );
}

function applyCandidate(page: PageRecord, index: number) {
  const updated = applyCandidateToPage(page, index);
  updated.previewPath = null;
  if (state.project) {
    state.project = {
      ...state.project,
      pages: state.project.pages.map((item) => (item.id === updated.id ? updated : item))
    };
  }
  state.activePage = updated;
  state.draftQuad = null;
  renderCandidateList(updated);
  renderMeta();
  renderPageList();
  renderExportSection();
  void renderActivePage();
}

function renderExportSection() {
  const project = state.project;
  const busy = exportState.mode === "running";
  outputPathInput.value = exportOptions.outputPath;
  exportAllPagesInput.checked = exportOptions.exportAllPages;
  includeAutoReadyInput.checked = exportOptions.includeAutoReady;
  minConfidenceInput.value = exportOptions.minAutoReadyConfidence.toFixed(2);
  jpegQualityInput.value = String(exportOptions.jpegQuality);
  maxDimensionInput.value = String(exportOptions.maxDimension);
  ocrEnabledInput.checked = exportOptions.ocrEnabled;
  ocrLanguagesInput.value = exportOptions.ocrLanguages;
  includeAutoReadyInput.disabled = !project || busy || exportOptions.exportAllPages;
  minConfidenceInput.disabled = !project || busy || exportOptions.exportAllPages;
  ocrLanguagesInput.disabled = !project || !exportOptions.ocrEnabled || busy;
  exportPopover.classList.toggle("hidden", !exportPopoverOpen || !project);
  toggleExportSettingsBtn.classList.toggle("active", exportPopoverOpen && Boolean(project));
  exportPopover.setAttribute("aria-hidden", String(!exportPopoverOpen || !project));

  for (const control of [
    toggleExportSettingsBtn,
    chooseOutputBtn,
    exportAllPagesInput,
    includeAutoReadyInput,
    minConfidenceInput,
    jpegQualityInput,
    maxDimensionInput,
    ocrEnabledInput,
    exportProjectBtn
  ]) {
    control.disabled = !project || busy;
  }
  exportProjectBtn.innerHTML = `<span class="toolbar-icon modern-icon" aria-hidden="true">${renderToolbarIcon(
    busy ? "save" : "download"
  )}</span>`;
  exportProjectBtn.setAttribute(
    "title",
    busy ? "正在导出 PDF，请稍候。" : "按当前顺序导出 PDF，并应用压缩与 OCR 文本层。"
  );
  exportProjectBtn.setAttribute("aria-label", busy ? "导出中" : "导出 PDF");

  const panel = buildExportPanelState(project, exportOptions, exportState);
  exportStatus.className = `status-panel ${panel.tone}`;
  exportStatus.innerHTML = panel.body.replace(/\n/g, "<br />");
}

async function commitDraftQuad() {
  const page = getActivePage();
  if (!page || !state.project || !state.draftQuad || !hasPendingDraft(page)) {
    state.draftQuad = null;
    if (page) {
      drawCanvas(page);
    }
    return;
  }
  const requestId = ++activeViewRequestId;
  const updated = applyDraftQuadToPage(page, state.draftQuad);
  updated.previewPath = null;
  state.project = {
    ...state.project,
    pages: state.project.pages.map((item) => (item.id === updated.id ? updated : item))
  };
  state.activePage = updated;
  state.project = withSelectedPage(state.project, updated.id);
  state.draftQuad = null;
  renderPageList();
  renderMeta();
  renderExportSection();
  renderPageDetails(updated);
  renderCandidateList(updated);
  drawCanvas(updated);
  await ensurePreview(updated, requestId);
}

async function chooseOutputPath() {
  if (!state.project) return;
  const file = await save({
    defaultPath: exportOptions.outputPath || defaultOutputPath(state.project),
    filters: [{ name: "PDF", extensions: ["pdf"] }]
  });
  if (!file) return;
  exportOptions.outputPath = file;
  exportState = {
    mode: "idle",
    result: null
  };
  renderExportSection();
}

function syncExportOptionsFromInputs() {
  exportOptions = {
    ...exportOptions,
    exportAllPages: exportAllPagesInput.checked,
    includeAutoReady: includeAutoReadyInput.checked,
    minAutoReadyConfidence: Number.parseFloat(minConfidenceInput.value) || 0,
    jpegQuality: Math.max(40, Math.min(100, Number.parseInt(jpegQualityInput.value, 10) || 82)),
    maxDimension: Math.max(1200, Number.parseInt(maxDimensionInput.value, 10) || 2200),
    ocrEnabled: ocrEnabledInput.checked,
    ocrLanguages: ocrLanguagesInput.value.trim() || "eng"
  };
  if (exportState.mode !== "running") {
    exportState = {
      mode: "idle",
      result: null
    };
  }
  renderExportSection();
}

for (const handle of cornerHandles) {
  handle.addEventListener("pointerdown", (event) => {
    const page = getActivePage();
    if (!page) return;
    const handleIndex = Number(handle.dataset.handleIndex);
    if (Number.isNaN(handleIndex)) return;

    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    state.dragOrigin = screenToImagePoint(event);
    state.activeHandle = handleIndex;
    state.draftQuad = workingQuad(page).map((entry) => [...entry]) as Point[];
    syncEditorOverlay(page);
    renderPageDetails(page);
  });
}

window.addEventListener("pointermove", (event) => {
  const page = getActivePage();
  if (!page || state.activeHandle === null || !state.draftQuad || !currentImage) return;
  const point = screenToImagePoint(event);
  const intrinsic = resolveIntrinsicImageSize(currentImage);
  state.draftQuad = updateQuadHandle(state.draftQuad, state.activeHandle, point, {
    width: intrinsic.width,
    height: intrinsic.height
  });
  state.dragOrigin = point;
  syncEditorOverlay(page);
  renderPageDetails(page);
});

async function stopHandleDrag() {
  state.activeHandle = null;
  state.dragOrigin = null;
  const page = getActivePage();
  if (page) {
    syncEditorOverlay(page);
  }
  await commitDraftQuad();
}

window.addEventListener("pointerup", stopHandleDrag);
window.addEventListener("pointercancel", stopHandleDrag);

window.addEventListener("keydown", async (event) => {
  const page = getActivePage();
  if (!page) return;
  if (event.key === "Enter") {
    page.status = "reviewed";
    renderPageList();
    renderMeta();
    renderPageDetails(page);
    renderExportSection();
    return;
  }
  if (event.key.toLowerCase() === "r") {
    state.draftQuad = null;
    drawCanvas(page);
    renderPageDetails(page);
    return;
  }
  const candidateIndex = Number(event.key) - 1;
  if (candidateIndex >= 0 && candidateIndex < page.candidates.length) {
    applyCandidate(page, candidateIndex);
  }
});

openFolderBtn.addEventListener("click", async () => {
  const folder = await open({
    directory: true,
    multiple: false
  });
  if (!folder || Array.isArray(folder)) return;
  const scanSessionId = beginScanSession();
  try {
    const project = await invoke<ProjectFile>("scan_folder", { folderPath: folder, scanId: scanSessionId });
    if (scanSessionId !== state.scanSessionId) return;
    setProject(project);
    finishScanSession(scanSessionId);
  } catch (error) {
    if (scanSessionId !== state.scanSessionId) return;
    finishScanSession(scanSessionId);
    if (String(error).includes("scan cancelled")) {
      return;
    }
    resetAppState();
    showAppError(String(error));
  }
});

loadProjectBtn.addEventListener("click", async () => {
  const file = await open({
    multiple: false,
    filters: [{ name: "Project", extensions: ["json"] }]
  });
  if (!file || Array.isArray(file)) return;
  try {
    const project = await invoke<ProjectFile>("load_project", { projectPath: file });
    setProject(project);
  } catch (error) {
    showAppError(String(error));
  }
});

saveProjectBtn.addEventListener("click", async () => {
  if (!state.project) return;
  const defaultPath = state.project.projectPath ?? `${state.project.sourceDir}/screen-pdf-project.json`;
  const file = await save({
    defaultPath,
    filters: [{ name: "Project", extensions: ["json"] }]
  });
  if (!file) return;
  try {
    const path = await invoke<string>("save_project", {
      projectPath: file,
      project: withSelectedPage(state.project, state.activePage?.id ?? null)
    });
    state.project.projectPath = path;
    renderMeta();
  } catch (error) {
    showAppError(String(error));
  }
});

chooseOutputBtn.addEventListener("click", () => {
  void chooseOutputPath();
});

toggleExportSettingsBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  if (!state.project || exportState.mode === "running") return;
  exportPopoverOpen = !exportPopoverOpen;
  renderExportSection();
});

exportPopover.addEventListener("click", (event) => {
  event.stopPropagation();
});

for (const input of [
  exportAllPagesInput,
  includeAutoReadyInput,
  minConfidenceInput,
  jpegQualityInput,
  maxDimensionInput,
  ocrEnabledInput,
  ocrLanguagesInput
]) {
  input.addEventListener("change", syncExportOptionsFromInputs);
}

exportProjectBtn.addEventListener("click", async () => {
  if (!state.project) return;
  syncExportOptionsFromInputs();
  if (!exportOptions.outputPath) {
    await chooseOutputPath();
  }
  if (!exportOptions.outputPath) return;

  exportState = {
    mode: "running",
    result: null
  };
  renderExportSection();
  try {
    const result = await invoke<ExportResult>("export_project", {
      project: state.project,
      options: exportOptions
    });
    exportState = {
      mode: "success",
      result
    };
  } catch (error) {
    exportState = {
      mode: "error",
      message: String(error),
      result: null
    };
  }
  renderExportSection();
});

window.addEventListener("pointerdown", (event) => {
  if (!exportPopoverOpen) return;
  const target = event.target;
  if (!(target instanceof Node)) return;
  if (exportPopover.contains(target) || toggleExportSettingsBtn.contains(target)) {
    return;
  }
  exportPopoverOpen = false;
  renderExportSection();
});

closeInfoModalBtn.addEventListener("click", () => {
  state.infoPage = null;
  renderModal();
});

infoModalBackdrop.addEventListener("click", () => {
  state.infoPage = null;
  renderModal();
});

cancelScanBtn.addEventListener("click", async () => {
  const scanSessionId = state.scanSessionId;
  scanUiState = {
    ...scanUiState,
    cancellable: false,
    message: "正在中断并清理临时数据..."
  };
  renderScanModal();
  try {
    await invoke("cancel_scan");
  } catch {
    // ignore cancel failures; UI still resets to default state below
  }
  if (scanSessionId !== state.scanSessionId) return;
  state.scanSessionId += 1;
  scanUiState = {
    visible: false,
    cancellable: false,
    processed: 0,
    total: 0,
    message: "已取消"
  };
  renderScanModal();
  resetAppState();
});

renderExportSection();
renderModal();
renderScanModal();

previewImage.addEventListener("error", () => {
  if (!previewSourceState) {
    return;
  }
  previewSourceState.index += 1;
  if (previewSourceState.index < previewSourceState.sources.length) {
    previewImage.src = previewSourceState.sources[previewSourceState.index] ?? "";
  }
});

void listen<ScanProgressEvent>("scan-progress", (event) => {
  updateScanSession(event.payload.scanId, event.payload);
});
