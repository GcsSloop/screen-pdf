import { describe, expect, it } from "vitest";

import {
  buildDisplaySourceCandidates,
  buildPreviewSourceCandidates,
  buildThumbnailSourceCandidates,
  canCommitPageRender,
  resolveIntrinsicImageSize,
  shouldGeneratePreview
} from "./render-flow";

describe("render flow", () => {
  it("keeps preview as a fallback source after the original image", () => {
    expect(buildDisplaySourceCandidates("/tmp/source.jpeg", "/tmp/preview.png")).toEqual([
      "/tmp/source.jpeg",
      "/tmp/preview.png"
    ]);
  });

  it("can prefer preview when the source image is known to be unavailable", () => {
    expect(buildDisplaySourceCandidates("/tmp/source.heic", "/tmp/preview.png", true)).toEqual([
      "/tmp/preview.png",
      "/tmp/source.heic"
    ]);
  });

  it("deduplicates identical source and preview paths", () => {
    expect(buildDisplaySourceCandidates("/tmp/source.jpeg", "/tmp/source.jpeg", true)).toEqual([
      "/tmp/source.jpeg"
    ]);
  });

  it("prefers natural image dimensions for editor rendering", () => {
    expect(resolveIntrinsicImageSize({ naturalWidth: 1920, naturalHeight: 1080, width: 0, height: 0 })).toEqual({
      width: 1920,
      height: 1080
    });
  });

  it("prefers generated thumbnails for page list rendering", () => {
    expect(
      buildThumbnailSourceCandidates("/tmp/source.jpeg", "/tmp/source-thumb.jpg", "/tmp/source-preview.png")
    ).toEqual(["/tmp/source-thumb.jpg", "/tmp/source-preview.png", "/tmp/source.jpeg"]);
  });

  it("prefers the generated preview first and falls back to the source image", () => {
    expect(buildPreviewSourceCandidates("/tmp/source.jpeg", "/tmp/source-preview.png")).toEqual([
      "/tmp/source-preview.png",
      "/tmp/source.jpeg"
    ]);
  });

  it("requires preview regeneration when the cached preview path is missing", () => {
    expect(shouldGeneratePreview(null)).toBe(true);
    expect(shouldGeneratePreview(undefined)).toBe(true);
    expect(shouldGeneratePreview("/tmp/source-preview.png")).toBe(false);
  });

  it("rejects stale async render results from previous pages", () => {
    expect(canCommitPageRender(2, 3, "page-1", "page-1")).toBe(false);
    expect(canCommitPageRender(3, 3, "page-1", "page-2")).toBe(false);
    expect(canCommitPageRender(3, 3, "page-1", "page-1")).toBe(true);
  });
});
