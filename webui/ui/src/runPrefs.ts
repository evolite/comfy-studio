import { useSyncExternalStore } from "react";

const KEY = "runOff";  // actions the user hid from the Run menus
const subs = new Set<() => void>();
let cache: string | null = null, set = new Set<string>();

function read(): Set<string> {
  let raw = "[]";
  try { raw = localStorage.getItem(KEY) ?? "[]"; } catch { /* storage unavailable */ }
  if (raw !== cache) { cache = raw; try { set = new Set(JSON.parse(raw)); } catch { set = new Set(); } }
  return set;
}

export function setActionHidden(action: string, hidden: boolean) {
  const n = new Set(read());
  if (hidden) n.add(action); else n.delete(action);
  try { localStorage.setItem(KEY, JSON.stringify([...n])); } catch { /* ignore */ }
  cache = null;
  subs.forEach((f) => f());
}

/** Actions hidden from the Run menus (Settings → Run buttons). Updates every subscriber live. */
export function useHiddenActions(): Set<string> {
  return useSyncExternalStore((f) => { subs.add(f); return () => subs.delete(f); }, read);
}

const TAKES_KEY = "runTakes";  // how many preview takes per stage a Preview run renders (×1 ×2 ×4)
const takeSubs = new Set<() => void>();
const readTakes = () => { try { const n = Number(localStorage.getItem(TAKES_KEY)); return [1, 2, 4].includes(n) ? n : 1; } catch { return 1; } };

export function setPreviewTakes(n: number) {
  try { localStorage.setItem(TAKES_KEY, String(n)); } catch { /* ignore */ }
  takeSubs.forEach((f) => f());
}

/** Preview takes per stage for the next Preview run (global, remembered). */
export function usePreviewTakes(): number {
  return useSyncExternalStore((f) => { takeSubs.add(f); return () => takeSubs.delete(f); }, readTakes);
}
export const previewTakes = readTakes;

const COLS_KEY = "boardCols";  // stage cards per row on the board (Settings → UI)
export const BOARD_COLS = [1, 2, 3, 4];
const colSubs = new Set<() => void>();
const readCols = () => { try { const n = Number(localStorage.getItem(COLS_KEY)); return BOARD_COLS.includes(n) ? n : 2; } catch { return 2; } };

export function setBoardCols(n: number) {
  try { localStorage.setItem(COLS_KEY, String(n)); } catch { /* ignore */ }
  colSubs.forEach((f) => f());
}

/** Board columns (global, this browser; default 2). */
export function useBoardCols(): number {
  return useSyncExternalStore((f) => { colSubs.add(f); return () => colSubs.delete(f); }, readCols);
}

const VOL_KEY = "clipVolume";  // volume clips with audio play at (Settings → UI), 0..1
export const CLIP_VOLUME_DEFAULT = 0.8;
const volSubs = new Set<() => void>();
export const clipVolume = () => { try { const v = localStorage.getItem(VOL_KEY); const n = Number(v); return v !== null && n >= 0 && n <= 1 ? n : CLIP_VOLUME_DEFAULT; } catch { return CLIP_VOLUME_DEFAULT; } };

export function setClipVolume(n: number) {
  try { localStorage.setItem(VOL_KEY, String(n)); } catch { /* ignore */ }
  volSubs.forEach((f) => f());
}

/** Clip preview volume (global, this browser; default 80%). */
export function useClipVolume(): number {
  return useSyncExternalStore((f) => { volSubs.add(f); return () => volSubs.delete(f); }, clipVolume);
}

const BATCH_KEY = "batchArmed";  // the header's "Start batch" toggle (RunPod target): Animate clicks queue into a batch
const batchSubs = new Set<() => void>();
const readBatch = () => { try { return localStorage.getItem(BATCH_KEY) === "1"; } catch { return false; } };

export function setBatchArmed(on: boolean) {
  try { localStorage.setItem(BATCH_KEY, on ? "1" : "0"); } catch { /* ignore */ }
  batchSubs.forEach((f) => f());
}

/** RunPod "Start batch" armed: Animate clicks queue into one batched job instead of starting a job per click. */
export function useBatchArmed(): boolean {
  return useSyncExternalStore((f) => { batchSubs.add(f); return () => batchSubs.delete(f); }, readBatch);
}
