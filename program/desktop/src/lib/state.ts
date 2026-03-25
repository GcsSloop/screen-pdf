import type { PageRecord, Point, ProjectFile } from "./types";

export interface AppState {
  project: ProjectFile | null;
  activePage: PageRecord | null;
  activeHandle: number | null;
  dragOrigin: Point | null;
  draftQuad: Point[] | null;
  dragPageId: string | null;
  infoPage: PageRecord | null;
  scanSessionId: number;
}

export const initialState = (): AppState => ({
  project: null,
  activePage: null,
  activeHandle: null,
  dragOrigin: null,
  draftQuad: null,
  dragPageId: null,
  infoPage: null,
  scanSessionId: 0
});
