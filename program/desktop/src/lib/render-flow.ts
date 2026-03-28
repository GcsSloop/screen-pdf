export function buildDisplaySourceCandidates(
  imagePath: string,
  previewPath?: string | null,
  preferPreview = false
): string[] {
  const sources = preferPreview ? [previewPath, imagePath] : [imagePath, previewPath];
  const uniqueSources: string[] = [];

  for (const source of sources) {
    if (!source || uniqueSources.includes(source)) {
      continue;
    }
    uniqueSources.push(source);
  }

  return uniqueSources;
}

export function buildThumbnailSourceCandidates(
  imagePath: string,
  thumbnailPath?: string | null,
  previewPath?: string | null
): string[] {
  const uniqueSources: string[] = [];

  for (const source of [thumbnailPath, previewPath, imagePath]) {
    if (!source || uniqueSources.includes(source)) {
      continue;
    }
    uniqueSources.push(source);
  }

  return uniqueSources;
}

export function buildPreviewSourceCandidates(imagePath: string, previewPath?: string | null): string[] {
  return buildDisplaySourceCandidates(imagePath, previewPath, true);
}

export function canCommitPageRender(
  requestId: number,
  activeRequestId: number,
  requestedPageId: string,
  activePageId: string | null | undefined
): boolean {
  return requestId === activeRequestId && requestedPageId === activePageId;
}

export function resolveIntrinsicImageSize(image: {
  naturalWidth?: number;
  naturalHeight?: number;
  width?: number;
  height?: number;
}): { width: number; height: number } {
  return {
    width: image.naturalWidth || image.width || 0,
    height: image.naturalHeight || image.height || 0
  };
}
