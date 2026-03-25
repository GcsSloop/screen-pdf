import type { Point } from "./types";

export interface DisplayGeometry {
  width: number;
  height: number;
  scale: number;
}

export function calculateDisplayGeometry(
  imageWidth: number,
  imageHeight: number,
  containerWidth: number,
  containerHeight: number
): DisplayGeometry {
  const safeContainerWidth = Math.max(1, containerWidth);
  const safeContainerHeight = Math.max(1, containerHeight);
  const scale = Math.min(safeContainerWidth / imageWidth, safeContainerHeight / imageHeight);

  return {
    scale,
    width: Math.max(1, Math.round(imageWidth * scale)),
    height: Math.max(1, Math.round(imageHeight * scale))
  };
}

export function projectQuadToDisplay(quad: Point[], geometry: DisplayGeometry): Point[] {
  return quad.map(([x, y]) => [x * geometry.scale, y * geometry.scale]);
}

export function clampDisplayPoint(
  point: Point,
  geometry: Pick<DisplayGeometry, "width" | "height">,
  margin: number
): Point {
  return [
    Math.max(margin, Math.min(geometry.width - margin, point[0])),
    Math.max(margin, Math.min(geometry.height - margin, point[1]))
  ];
}

export function clampPointToImage(point: Point, bounds: { width: number; height: number }): Point {
  return [
    Math.max(0, Math.min(bounds.width, point[0])),
    Math.max(0, Math.min(bounds.height, point[1]))
  ];
}

export function updateQuadHandle(
  quad: Point[],
  handleIndex: number,
  point: Point,
  bounds: { width: number; height: number }
): Point[] {
  return quad.map((entry, index) =>
    index === handleIndex ? clampPointToImage(point, bounds) : ([...entry] as Point)
  ) as Point[];
}

export function buildPolygonPoints(points: Point[]): string {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}
