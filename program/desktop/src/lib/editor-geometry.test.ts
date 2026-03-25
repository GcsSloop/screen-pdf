import { describe, expect, it } from "vitest";

import {
  buildPolygonPoints,
  clampPointToImage,
  clampDisplayPoint,
  projectQuadToDisplay,
  updateQuadHandle,
  type DisplayGeometry
} from "./editor-geometry";
import type { Point } from "./types";

describe("editor geometry", () => {
  const geometry: DisplayGeometry = {
    width: 1200,
    height: 675,
    scale: 0.625
  };

  it("projects every quad point into display coordinates", () => {
    const quad: Point[] = [
      [0, 0],
      [1920, 0],
      [1920, 1080],
      [0, 1080]
    ];

    expect(projectQuadToDisplay(quad, geometry)).toEqual([
      [0, 0],
      [1200, 0],
      [1200, 675],
      [0, 675]
    ]);
  });

  it("updates only the selected handle", () => {
    const quad: Point[] = [
      [10, 10],
      [90, 10],
      [90, 90],
      [10, 90]
    ];

    expect(updateQuadHandle(quad, 2, [60, 75], { width: 100, height: 100 })).toEqual([
      [10, 10],
      [90, 10],
      [60, 75],
      [10, 90]
    ]);
  });

  it("clamps dragged points inside the image bounds", () => {
    expect(clampPointToImage([-20, 140], { width: 100, height: 80 })).toEqual([0, 80]);
  });

  it("builds svg polygon points in display order", () => {
    expect(
      buildPolygonPoints([
        [12.5, 24],
        [50, 24],
        [40, 60],
        [10, 55]
      ])
    ).toBe("12.5,24 50,24 40,60 10,55");
  });

  it("keeps handle display positions inside a clickable safety margin", () => {
    expect(clampDisplayPoint([1200, 675], geometry, 14)).toEqual([1186, 661]);
    expect(clampDisplayPoint([0, 0], geometry, 14)).toEqual([14, 14]);
  });
});
