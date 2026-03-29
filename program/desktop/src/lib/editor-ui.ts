import type { Point } from "./types";

export type ToolbarIconButton = {
  id: string;
  label: string;
  tooltip: string;
  icon: string;
  variant?: "primary";
};

const TOOLBAR_ICON_SVGS: Record<string, string> = {
  folder:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.75A2.75 2.75 0 0 1 5.75 5h4.16c.6 0 1.17.24 1.6.66l1.08 1.09h5.66A2.75 2.75 0 0 1 21 9.5v8.75A2.75 2.75 0 0 1 18.25 21H5.75A2.75 2.75 0 0 1 3 18.25V7.75Z" fill="currentColor" opacity=".18"/><path d="M3 9.5A2.75 2.75 0 0 1 5.75 6.75h5.88c.46 0 .9.18 1.22.5l.87.87h4.53A2.75 2.75 0 0 1 21 10.88v7.37A2.75 2.75 0 0 1 18.25 21H5.75A2.75 2.75 0 0 1 3 18.25V9.5Zm2 0v8.75c0 .41.34.75.75.75h12.5c.41 0 .75-.34.75-.75v-7.37a.75.75 0 0 0-.75-.76h-5.36l-1.46-1.46a.75.75 0 0 0-.53-.22H5.75A.75.75 0 0 0 5 9.5Z" fill="currentColor"/></svg>',
  upload:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.75a.75.75 0 0 1 .75.75v8.69l2.47-2.47a.75.75 0 1 1 1.06 1.06l-3.75 3.75a.75.75 0 0 1-1.06 0l-3.75-3.75a.75.75 0 0 1 1.06-1.06l2.47 2.47V4.5a.75.75 0 0 1 .75-.75Z" fill="currentColor"/><path d="M4.75 15a.75.75 0 0 1 .75.75v1.5c0 .69.56 1.25 1.25 1.25h10.5c.69 0 1.25-.56 1.25-1.25v-1.5a.75.75 0 0 1 1.5 0v1.5A2.75 2.75 0 0 1 17.25 20H6.75A2.75 2.75 0 0 1 4 17.25v-1.5a.75.75 0 0 1 .75-.75Z" fill="currentColor"/></svg>',
  save:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5.75 3h9.2c.53 0 1.04.21 1.41.59l3.05 3.05c.38.37.59.88.59 1.41v10.2A2.75 2.75 0 0 1 17.25 21H5.75A2.75 2.75 0 0 1 3 18.25V5.75A2.75 2.75 0 0 1 5.75 3Z" fill="currentColor" opacity=".18"/><path d="M5.75 3A2.75 2.75 0 0 0 3 5.75v12.5A2.75 2.75 0 0 0 5.75 21h11.5A2.75 2.75 0 0 0 20 18.25V8.31c0-.53-.21-1.04-.59-1.41l-3.31-3.31A2 2 0 0 0 14.69 3H5.75Zm0 1.5h8.5v3.75H6.5V4.5h-.75Zm10 14.5H8.25v-4.25h7.5V19Zm2.75-.75a1.25 1.25 0 0 1-1.25 1.25H17.25v-4.75A1.25 1.25 0 0 0 16 13.5H8a1.25 1.25 0 0 0-1.25 1.25v4.75h-1A1.25 1.25 0 0 1 4.5 18.25V5.75A1.25 1.25 0 0 1 5.75 4.5H5v4A1.25 1.25 0 0 0 6.25 9.75h8.5A1.25 1.25 0 0 0 16 8.5V4.56l2.06 2.06c.1.1.19.23.25.36h-1.06a.75.75 0 0 0 0 1.5h1.25v9.77Z" fill="currentColor"/></svg>',
  settings:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.59 3.25a1.5 1.5 0 0 0-3.18 0l-.23 1.4a7.47 7.47 0 0 0-1.75.72L7.2 4.55a1.5 1.5 0 0 0-2.25.66l-.9 1.56a1.5 1.5 0 0 0 .53 2.03l1.2.73a7.6 7.6 0 0 0 0 1.94l-1.2.73a1.5 1.5 0 0 0-.53 2.03l.9 1.56a1.5 1.5 0 0 0 2.25.66l1.23-.82c.55.3 1.13.54 1.75.72l.23 1.4a1.5 1.5 0 0 0 3.18 0l.23-1.4c.62-.18 1.2-.42 1.75-.72l1.23.82a1.5 1.5 0 0 0 2.25-.66l.9-1.56a1.5 1.5 0 0 0-.53-2.03l-1.2-.73a7.6 7.6 0 0 0 0-1.94l1.2-.73a1.5 1.5 0 0 0 .53-2.03l-.9-1.56a1.5 1.5 0 0 0-2.25-.66l-1.23.82a7.47 7.47 0 0 0-1.75-.72l-.23-1.4ZM12 8.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7Z" fill="currentColor" opacity=".18"/><path d="M10.68 3a1.75 1.75 0 0 1 3.45 0l.18 1.15c.46.14.91.32 1.33.55l.94-.62a1.75 1.75 0 0 1 2.62.77l.89 1.54a1.75 1.75 0 0 1-.61 2.39l-.98.6c.05.25.08.51.1.77.01.2.01.4 0 .61l.98.6a1.75 1.75 0 0 1 .61 2.39l-.89 1.54a1.75 1.75 0 0 1-2.62.77l-.94-.62c-.42.23-.87.41-1.33.55L14.13 21a1.75 1.75 0 0 1-3.45 0l-.18-1.15a7.2 7.2 0 0 1-1.33-.55l-.94.62a1.75 1.75 0 0 1-2.62-.77l-.89-1.54a1.75 1.75 0 0 1 .61-2.39l.98-.6A6.95 6.95 0 0 1 5.25 12c0-.41.04-.82.11-1.22l-.98-.6a1.75 1.75 0 0 1-.61-2.39l.89-1.54a1.75 1.75 0 0 1 2.62-.77l.94.62c.42-.23.87-.41 1.33-.55L10.68 3Zm1.97.23a.25.25 0 0 0-.5 0l-.24 1.45a.75.75 0 0 1-.56.61 5.78 5.78 0 0 0-1.66.69.75.75 0 0 1-.83-.02L7.6 5.12a.25.25 0 0 0-.37.1l-.89 1.54a.25.25 0 0 0 .08.34l1.25.77a.75.75 0 0 1 .34.76 5.71 5.71 0 0 0 0 2.74.75.75 0 0 1-.34.76l-1.25.77a.25.25 0 0 0-.08.34l.89 1.54a.25.25 0 0 0 .37.1l1.26-.84a.75.75 0 0 1 .83-.02c.52.31 1.08.54 1.66.69a.75.75 0 0 1 .56.61l.24 1.45a.25.25 0 0 0 .5 0l.24-1.45a.75.75 0 0 1 .56-.61c.58-.15 1.14-.38 1.66-.69a.75.75 0 0 1 .83.02l1.26.84a.25.25 0 0 0 .37-.1l.89-1.54a.25.25 0 0 0-.08-.34l-1.25-.77a.75.75 0 0 1-.34-.76 5.72 5.72 0 0 0 0-2.74.75.75 0 0 1 .34-.76l1.25-.77a.25.25 0 0 0 .08-.34l-.89-1.54a.25.25 0 0 0-.37-.1l-1.26.84a.75.75 0 0 1-.83.02 5.77 5.77 0 0 0-1.66-.69.75.75 0 0 1-.56-.61l-.24-1.45ZM12 8.25a3.75 3.75 0 1 1 0 7.5 3.75 3.75 0 0 1 0-7.5Zm0 1.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5Z" fill="currentColor"/></svg>',
  download:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.75a.75.75 0 0 1 .75.75v8.69l2.47-2.47a.75.75 0 1 1 1.06 1.06l-3.75 3.75a.75.75 0 0 1-1.06 0l-3.75-3.75a.75.75 0 0 1 1.06-1.06l2.47 2.47V4.5a.75.75 0 0 1 .75-.75Z" fill="currentColor"/><path d="M4.75 16.5a.75.75 0 0 1 .75.75v.5c0 .69.56 1.25 1.25 1.25h10.5c.69 0 1.25-.56 1.25-1.25v-.5a.75.75 0 0 1 1.5 0v.5A2.75 2.75 0 0 1 17.25 20H6.75A2.75 2.75 0 0 1 4 17.75v-.5a.75.75 0 0 1 .75-.75Z" fill="currentColor"/></svg>'
};

export const TOOLBAR_ICON_BUTTONS: ToolbarIconButton[] = [
  {
    id: "openFolderBtn",
    label: "打开文件夹",
    tooltip: "打开文件夹并扫描图片，生成当前项目。",
    icon: "folder"
  },
  {
    id: "loadProjectBtn",
    label: "加载项目",
    tooltip: "加载已保存的项目 JSON，继续编辑角点和导出设置。",
    icon: "upload"
  },
  {
    id: "saveProjectBtn",
    label: "保存项目",
    tooltip: "保存当前页面顺序、角点草稿和标签信息。",
    icon: "save"
  },
  {
    id: "toggleExportSettingsBtn",
    label: "导出配置",
    tooltip: "设置 PDF 输出路径、OCR、压缩尺寸和自动导出规则。",
    icon: "settings"
  },
  {
    id: "exportProjectBtn",
    label: "导出 PDF",
    tooltip: "按当前顺序导出 PDF，并应用压缩与 OCR 文本层。",
    icon: "download",
    variant: "primary"
  }
];

export const MAGNIFIER_SIZE = 144;
export const MAGNIFIER_OFFSET = 20;
export const MAGNIFIER_HOVER_DELAY_MS = 500;
const MAGNIFIER_MARGIN = 12;

type StageSize = {
  width: number;
  height: number;
};

type PlacementCandidate = {
  anchorX: number;
  anchorY: number;
  centerX: number;
  centerY: number;
  fitsWithoutClamp: boolean;
  score: number;
};

type ResolveMagnifierPlacementArgs = {
  point: Point;
  activeHandle: number;
  quad: Point[];
  stage: StageSize;
};

const QUADRANT_DIRECTIONS = [
  { dx: 1, dy: -1 },
  { dx: -1, dy: -1 },
  { dx: -1, dy: 1 },
  { dx: 1, dy: 1 }
] as const;

const HANDLE_PREFERRED_QUADRANTS = [
  [3, 2, 1, 0],
  [2, 3, 1, 0],
  [1, 0, 2, 3],
  [0, 1, 3, 2]
] as const;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function distanceToSegment(point: Point, start: Point, end: Point): number {
  const [px, py] = point;
  const [x1, y1] = start;
  const [x2, y2] = end;
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) {
    return Math.hypot(px - x1, py - y1);
  }
  const t = clamp(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy), 0, 1);
  const projX = x1 + dx * t;
  const projY = y1 + dy * t;
  return Math.hypot(px - projX, py - projY);
}

function buildPlacementCandidate(point: Point, stage: StageSize, quadrant: number): PlacementCandidate {
  const { dx, dy } = QUADRANT_DIRECTIONS[quadrant];
  const rawX = point[0] + dx * MAGNIFIER_OFFSET - (dx < 0 ? MAGNIFIER_SIZE : 0);
  const rawY = point[1] + dy * MAGNIFIER_OFFSET - (dy < 0 ? MAGNIFIER_SIZE : 0);
  const anchorX = clamp(rawX, MAGNIFIER_MARGIN, stage.width - MAGNIFIER_MARGIN - MAGNIFIER_SIZE);
  const anchorY = clamp(rawY, MAGNIFIER_MARGIN, stage.height - MAGNIFIER_MARGIN - MAGNIFIER_SIZE);

  return {
    anchorX,
    anchorY,
    centerX: anchorX + MAGNIFIER_SIZE / 2,
    centerY: anchorY + MAGNIFIER_SIZE / 2,
    fitsWithoutClamp: rawX === anchorX && rawY === anchorY,
    score: 0
  };
}

export function resolveMagnifierPlacement({
  point,
  activeHandle,
  quad,
  stage
}: ResolveMagnifierPlacementArgs): { anchorX: number; anchorY: number } {
  const safeQuadrants = HANDLE_PREFERRED_QUADRANTS[activeHandle] ?? HANDLE_PREFERRED_QUADRANTS[0];
  const prev = quad[(activeHandle + quad.length - 1) % quad.length] ?? point;
  const next = quad[(activeHandle + 1) % quad.length] ?? point;
  const rankedCandidates = safeQuadrants.map((quadrant, order) => {
    const candidate = buildPlacementCandidate(point, stage, quadrant);
    const center: Point = [candidate.centerX, candidate.centerY];
    const lineClearance =
      distanceToSegment(center, point, prev) + distanceToSegment(center, point, next);
    const pointClearance = quad.reduce((sum, entry, index) => {
      if (index === activeHandle) {
        return sum;
      }
      return sum + Math.hypot(center[0] - entry[0], center[1] - entry[1]);
    }, 0);
    const overflowPenalty =
      Math.abs(candidate.anchorX - (point[0] + QUADRANT_DIRECTIONS[quadrant].dx * MAGNIFIER_OFFSET)) +
      Math.abs(candidate.anchorY - (point[1] + QUADRANT_DIRECTIONS[quadrant].dy * MAGNIFIER_OFFSET));
    candidate.score =
      lineClearance * 4 +
      pointClearance -
      overflowPenalty * 2 -
      order * 120 +
      (overflowPenalty === 0 ? (safeQuadrants.length - order) * 1000 : 0);
    return candidate;
  });

  const preferredFit = rankedCandidates.find((candidate) => candidate.fitsWithoutClamp);
  if (preferredFit) {
    return {
      anchorX: preferredFit.anchorX,
      anchorY: preferredFit.anchorY
    };
  }

  const candidates = [...rankedCandidates].sort((left, right) => right.score - left.score);
  return {
    anchorX: candidates[0]?.anchorX ?? MAGNIFIER_MARGIN,
    anchorY: candidates[0]?.anchorY ?? MAGNIFIER_MARGIN
  };
}

export function renderToolbarIcon(icon: string): string {
  return TOOLBAR_ICON_SVGS[icon] ?? TOOLBAR_ICON_SVGS.folder;
}

export function resolveMagnifierHandle(activeHandle: number | null, hoverHandle: number | null): number | null {
  return activeHandle ?? hoverHandle;
}
