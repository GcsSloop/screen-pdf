import type { ExportOptions, ExportResult, PageRecord, ProjectFile } from "./types";

export interface ExportUiState {
  mode: "idle" | "running" | "success" | "error";
  message?: string;
  result?: ExportResult | null;
}

export interface ExportPanelViewModel {
  tone: "empty" | "info" | "warn" | "success" | "error";
  body: string;
}

export function eligibleExportCount(project: ProjectFile | null, options: ExportOptions): number {
  if (!project) return 0;
  return project.pages.filter((page) => isExportEligible(page, options)).length;
}

export function isExportEligible(page: PageRecord, options: ExportOptions): boolean {
  if (options.exportAllPages) {
    return page.status !== "error";
  }
  if (page.status === "reviewed") return true;
  if (page.status === "needs_review") return true;
  return (
    page.status === "auto_ready" &&
    options.includeAutoReady &&
    page.confidence >= options.minAutoReadyConfidence
  );
}

export function resolveActivePage(project: ProjectFile | null): PageRecord | null {
  if (!project || project.pages.length === 0) return null;
  if (project.selectedPageId) {
    const selected = project.pages.find((page) => page.id === project.selectedPageId);
    if (selected) return selected;
  }
  return project.pages[0] ?? null;
}

export function withSelectedPage(project: ProjectFile, pageId: string | null): ProjectFile {
  return {
    ...project,
    selectedPageId: pageId
  };
}

export function removePageFromProject(project: ProjectFile, pageId: string): ProjectFile {
  const removeIndex = project.pages.findIndex((page) => page.id === pageId);
  if (removeIndex < 0) {
    return project;
  }

  const pages = project.pages.filter((page) => page.id !== pageId);
  const selectedPageId =
    project.selectedPageId === pageId
      ? (pages[removeIndex] ?? pages[removeIndex - 1] ?? null)?.id ?? null
      : project.selectedPageId;

  return {
    ...project,
    pages,
    selectedPageId
  };
}

export function buildExportPanelState(
  project: ProjectFile | null,
  options: ExportOptions,
  exportState: ExportUiState
): ExportPanelViewModel {
  if (exportState.mode === "error") {
    return {
      tone: "error",
      body: exportState.message ?? "导出失败。"
    };
  }

  if (!project) {
    return {
      tone: "empty",
      body: "打开文件夹后即可配置导出。"
    };
  }

  const eligibleCount = eligibleExportCount(project, options);

  if (exportState.mode === "running") {
    return {
      tone: "info",
      body: `正在导出 ${eligibleCount} 页，OCR 识别可能需要一些时间。`
    };
  }

  if (exportState.mode === "success" && exportState.result) {
    const warnings = exportState.result.warnings.length
      ? `\n${exportState.result.warnings.join("\n")}`
      : "";
    return {
      tone: "success",
      body:
        `已导出 ${exportState.result.pageCount} 页\n` +
        `${exportState.result.outputPath}\n` +
        `报告：${exportState.result.reportPath}\n` +
        `OCR：${exportState.result.effectiveOcrLanguages ?? "未启用"}` +
        warnings
    };
  }

  if (eligibleCount > 0) {
    return {
      tone: "info",
      body: `当前有 ${eligibleCount} 页满足导出条件。`
    };
  }

  return {
    tone: "warn",
    body: "当前没有可导出的页面，请检查页面状态或放宽自动就绪筛选条件。"
  };
}
