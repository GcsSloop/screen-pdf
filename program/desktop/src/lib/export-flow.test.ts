import { describe, expect, it } from "vitest";

import {
  buildExportPanelState,
  eligibleExportCount,
  removePageFromProject,
  resolveActivePage,
  withSelectedPage
} from "./export-flow";
import type { ExportOptions, PageRecord, ProjectFile } from "./types";

function page(id: string, status: PageRecord["status"], confidence: number): PageRecord {
  return {
    id,
    name: `${id}.jpg`,
    path: `/tmp/${id}.jpg`,
    createdAt: "0",
    status,
    confidence,
    bestMethod: "contour_quad",
    selectedCandidateIndex: 0,
    candidates: [],
    activeQuad: [],
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

function project(): ProjectFile {
  return {
    version: 1,
    name: "demo",
    sourceDir: "/tmp/demo",
    projectPath: null,
    selectedPageId: "p2",
    pages: [page("p1", "needs_review", 0.1), page("p2", "reviewed", 0.1)]
  };
}

const options: ExportOptions = {
  outputPath: "/tmp/out.pdf",
  exportAllPages: true,
  includeAutoReady: true,
  minAutoReadyConfidence: 0.2,
  jpegQuality: 82,
  maxDimension: 2200,
  ocrEnabled: true,
  ocrLanguages: "chi_sim+eng"
};

describe("export flow helpers", () => {
  it("respects saved selected page id", () => {
    expect(resolveActivePage(project())?.id).toBe("p2");
  });

  it("counts reviewed, needs review, and trusted auto ready pages", () => {
    const sample = {
      ...project(),
      pages: [
        page("reviewed", "reviewed", 0.01),
        page("needs-review", "needs_review", 0.01),
        page("trusted", "auto_ready", 0.9),
        page("untrusted", "auto_ready", 0.05)
      ]
    };
    expect(eligibleExportCount(sample, options)).toBe(4);
  });

  it("can fall back to filtered export when export all pages is disabled", () => {
    const sample = {
      ...project(),
      pages: [
        page("reviewed", "reviewed", 0.01),
        page("needs-review", "needs_review", 0.01),
        page("trusted", "auto_ready", 0.9),
        page("error", "error", 0.05)
      ]
    };
    expect(
      eligibleExportCount(sample, {
        ...options,
        exportAllPages: false
      })
    ).toBe(3);
  });

  it("keeps explicit error state visible", () => {
    const view = buildExportPanelState(project(), options, {
      mode: "error",
      message: "no pages are eligible",
      result: null
    });
    expect(view.tone).toBe("error");
    expect(view.body).toContain("no pages are eligible");
  });

  it("renders startup failure in the same status area", () => {
    const view = buildExportPanelState(null, options, {
      mode: "error",
      message: "failed to run detection engine: python3 not found",
      result: null
    });
    expect(view.tone).toBe("error");
    expect(view.body).toContain("python3 not found");
  });

  it("updates selected page id in project copy", () => {
    expect(withSelectedPage(project(), "p1").selectedPageId).toBe("p1");
  });

  it("removes a page from the project and keeps neighbors in order", () => {
    const updated = removePageFromProject(
      {
        ...project(),
        pages: [
          page("p1", "reviewed", 0.2),
          page("p2", "reviewed", 0.2),
          page("p3", "reviewed", 0.2)
        ],
        selectedPageId: "p2"
      },
      "p2"
    );

    expect(updated.pages.map((item) => item.id)).toEqual(["p1", "p3"]);
    expect(updated.selectedPageId).toBe("p3");
  });

  it("clears selected page when the last page is deleted", () => {
    const updated = removePageFromProject(
      {
        ...project(),
        pages: [page("p1", "reviewed", 0.2)],
        selectedPageId: "p1"
      },
      "p1"
    );

    expect(updated.pages).toEqual([]);
    expect(updated.selectedPageId).toBeNull();
  });
});
