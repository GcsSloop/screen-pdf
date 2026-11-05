import { describe, expect, it } from "vitest";
import { defaultOutputPath, projectDirectoryName, projectSavePath } from "./project-paths";
import type { ProjectFile } from "./types";

const project = (sourceDir: string) => ({ sourceDir, name: "stale-name", projectPath: "/old/export/screen-pdf-project.json" }) as ProjectFile;
describe("portable project paths", () => {
  it("takes current directory name including spaces, dots and Chinese", () => {
    const p = project("/projects/元宇宙 2.0 新名称/");
    expect(projectDirectoryName(p)).toBe("元宇宙 2.0 新名称");
    expect(defaultOutputPath(p)).toBe("/projects/元宇宙 2.0 新名称/元宇宙 2.0 新名称.pdf");
    expect(defaultOutputPath(p, "/out")).toBe("/out/元宇宙 2.0 新名称.pdf");
  });
  it("always saves the canonical file in project root", () => {
    expect(projectSavePath(project("/new/root"))).toBe("/new/root/screen-pdf-project.json");
  });
  it("supports Windows separators after moving from another system", () => {
    const p = project("D:\\新目录\\项目 2.0");
    expect(defaultOutputPath(p)).toBe("D:\\新目录\\项目 2.0\\项目 2.0.pdf");
    expect(projectSavePath(p)).toBe("D:\\新目录\\项目 2.0\\screen-pdf-project.json");
  });
});
