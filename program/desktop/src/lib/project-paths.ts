import type { ProjectFile } from "./types";

export const PROJECT_FILE_NAME = "screen-pdf-project.json";

export function projectDirectoryName(project: ProjectFile): string {
  const parts = project.sourceDir.replace(/\\/g, "/").replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || "ScreenPDF";
}

export function projectSavePath(project: ProjectFile): string {
  const separator = project.sourceDir.includes("\\") ? "\\" : "/";
  return `${project.sourceDir.replace(/[\\/]+$/, "")}${separator}${PROJECT_FILE_NAME}`;
}

export function defaultOutputPath(project: ProjectFile, directory = project.sourceDir): string {
  const separator = directory.includes("\\") ? "\\" : "/";
  return `${directory.replace(/[\\/]+$/, "")}${separator}${projectDirectoryName(project)}.pdf`;
}
