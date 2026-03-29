import { describe, expect, it } from "vitest";

import {
  MAGNIFIER_HOVER_DELAY_MS,
  MAGNIFIER_SIZE,
  MAGNIFIER_OFFSET,
  resolveMagnifierHandle,
  resolveMagnifierPlacement,
  TOOLBAR_ICON_BUTTONS
} from "./editor-ui";
import type { Point } from "./types";

describe("editor ui helpers", () => {
  it("exposes icon-only toolbar actions with detailed tooltips", () => {
    expect(TOOLBAR_ICON_BUTTONS.map((item) => item.id)).toEqual([
      "openFolderBtn",
      "loadProjectBtn",
      "saveProjectBtn",
      "toggleExportSettingsBtn",
      "exportProjectBtn"
    ]);
    expect(TOOLBAR_ICON_BUTTONS.find((item) => item.id === "openFolderBtn")?.tooltip).toContain("扫描图片");
    expect(TOOLBAR_ICON_BUTTONS.find((item) => item.id === "toggleExportSettingsBtn")?.label).toBe("导出配置");
  });

  it("places the magnifier away from the active corner line when there is room", () => {
    const quad: Point[] = [
      [120, 80],
      [520, 70],
      [500, 380],
      [140, 400]
    ];

    const placement = resolveMagnifierPlacement({
      point: quad[1],
      activeHandle: 1,
      quad,
      stage: { width: 720, height: 520 }
    });

    expect(placement.anchorX).toBeLessThan(quad[1][0]);
    expect(placement.anchorY).toBeGreaterThan(quad[1][1]);
  });

  it("keeps the magnifier inside the stage when the preferred corner would overflow", () => {
    const quad: Point[] = [
      [36, 24],
      [220, 32],
      [210, 240],
      [48, 236]
    ];

    const placement = resolveMagnifierPlacement({
      point: quad[0],
      activeHandle: 0,
      quad,
      stage: { width: 260, height: 260 }
    });

    expect(placement.anchorX).toBeGreaterThanOrEqual(MAGNIFIER_OFFSET);
    expect(placement.anchorY).toBeGreaterThanOrEqual(MAGNIFIER_OFFSET);
    expect(placement.anchorX + MAGNIFIER_SIZE + MAGNIFIER_OFFSET).toBeLessThanOrEqual(260);
    expect(placement.anchorY + MAGNIFIER_SIZE + MAGNIFIER_OFFSET).toBeLessThanOrEqual(260);
  });

  it("uses a delayed hover magnifier and lets dragging take priority", () => {
    expect(MAGNIFIER_HOVER_DELAY_MS).toBe(500);
    expect(resolveMagnifierHandle(2, 1)).toBe(2);
    expect(resolveMagnifierHandle(null, 1)).toBe(1);
    expect(resolveMagnifierHandle(null, null)).toBeNull();
  });
});
