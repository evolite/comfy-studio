import { useEffect, useState } from "react";

export type Stage = {
  id: number; krea_scene?: string; il_scene?: string; il_background?: string; motion?: string; duration?: number;
  done?: boolean | string[];  // folders this stage is completed in (legacy `true` = every folder)
  loras?: Record<string, number>; video_loras?: Record<string, number>; video_engine?: string; seed?: number; video_seed?: number; audio?: boolean;
  voice?: string;  // a voice-library name filling the motion's {voice...} slots (else the project's `voice`)
  [k: string]: unknown;
};
/** One preview render of a stage (pipeline/takes.py); `delta` = what differs from the current take ("seed", "prompt · cfg"). */
export type Take = { id: string; mtime: number; current: boolean; source: "run" | "external" | "legacy"; delta: string };
export type FolderState = { stills: Record<string, number>; clips: Record<string, number>; takes?: Record<string, Take[]>; stale?: number[]; stale_clips?: number[] };
export type Project = {
  name: string; config: { engine: string; oneoff?: boolean; seed?: number; video_megapixels?: number; video_engine?: string; voice?: string; music?: { beat_times_file?: string; audio?: string }; settings: Record<string, unknown>; stages: Stage[] };
  folders: Record<string, FolderState>; edits: { name: string; mtime: number; size: number; folder: string | null }[];
};
export type ProjectRow = { name: string; engine: string; oneoff?: boolean; finished?: boolean; thumb?: number | null; stages: number; stills: number; clips: number; edits: number; mtime: number };
export type Job = {
  id: string; project: string; action: string; cost: "paid" | "free"; lane: string; status: string; returncode: number | null;
  stages: Record<string, string>; req: Record<string, unknown>; argv: string; last: string; created: number; ended?: number | null; lines?: string[];
  progress?: { fraction: number; left: number; exact: boolean } | null;
  per_stage?: number | null;  // wall-clock seconds per finished stage (take), loading included
  took?: Record<string, number>;  // per finished stage (stage id, or "N.tK" for a take): ComfyUI's processed seconds
  mem_freed?: string | null;  // the memory check freed ComfyUI before this local batch (why)
};
export type Registry = {
  settings_keys: Record<string, string[]>; default_settings: Record<string, Record<string, unknown>>;
  model_choices: Record<string, string[]>; default_beats: string;
  template_defaults: Record<string, Record<string, string | number | boolean>>; local_only_unets: string[];
  tiers: number[]; actions: Record<string, { cost: string; lane: string }>;
  video_engines: Record<string, string>;  // i2v model id -> label (animate_pipeline.VIDEO_ENGINES)
  image_lora_fields: Record<string, ImagePass[]>;  // image engine -> the LoRA files it loads (image_loras.ENGINE_FIELDS)
};
export type ImagePass = "krea" | "il" | "moss";  // lora_registry.IMAGE_PASSES
/** One loras.json entry (pipeline/lora_registry.py). Image: krea/il/moss files + triggers + default; video: file + engine (unset = minimax). */
export type LoraSpec = {
  krea?: string; il?: string; moss?: string; krea_trigger?: string; il_trigger?: string; moss_trigger?: string; default?: number; file?: string; engine?: string; trigger?: string;
  category?: LoraCategory; suggested?: number; cloud?: boolean; note?: string; tips?: string; url?: string;
  min?: number; max?: number;  // slider LoRA: the strength range the stage editor's drag slider covers
  disabled?: boolean;  // off in the library: stages using it keep it and run, the stage editor no longer offers it
};
export const LORA_CATEGORIES = ["concept", "style", "character"] as const;
export type LoraCategory = (typeof LORA_CATEGORIES)[number];
/** Entries grouped by category (unset = concept), in LORA_CATEGORIES order, empty groups dropped. */
export const byCategory = <T extends LoraSpec>(entries: [string, T][]) =>
  LORA_CATEGORIES.map((c) => [c, entries.filter(([, e]) => (e.category ?? "concept") === c)] as const).filter(([, g]) => g.length);
export type LoraKind = "image" | "video";
/** + `local`: per image pass, is the file in the local ComfyUI's list (null = server down); `used`: stages with it on. */
export type LoraEntry = LoraSpec & { local: Partial<Record<ImagePass, boolean | null>>; used: { project: string; stage: number }[] };
export type Discovered = { path: string; name: string; pass: ImagePass | null; files: Record<ImagePass, string>; trigger: string | null };
/** A named, tested set of LoRAs + strengths (loras.json `combos`): a preset the stage editor copies into the stage. */
export type LoraCombo = { kind: LoraKind; loras: Record<string, number>; note?: string; tips?: string };
export type LoraLibrary = Record<LoraKind, Record<string, LoraEntry>> & {
  combos: Record<string, LoraCombo>; ignore: string[]; discovered: Discovered[]; literal_video: string[]; local_up: boolean;
};
export type Health = { local: boolean; local_server: string; cloud_key: boolean };

/** A failed API call. One nobody catches (`void api(...)`) reaches the on-screen error in App. */
export class ApiError extends Error {
  constructor(message: string, readonly path: string) { super(message); this.name = "ApiError"; }
}

export async function api<T = unknown>(path: string, method = "GET", body?: unknown): Promise<T> {
  let r: Response;
  try {
    r = await fetch("/api" + path, {
      method, headers: body === undefined ? {} : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error_) { throw new ApiError(`server unreachable (${(error_ as Error).message})`, path); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError((data as { error?: string }).error ?? `${r.status}`, path);
  return data as T;
}

/** URL of a project file; `v` (mtime) busts the cache when a re-roll replaces it. */
export const fileUrl = (project: string, folder: string, kind: "still" | "clip", id: number | string, v: number) =>
  `/api/projects/${project}/files/${folder ? folder + "/" : ""}${kind === "clip" ? "videos/" : ""}stage_${id}.${kind === "clip" ? "mp4" : "png"}?v=${v}`;

export const takeUrl = (project: string, sid: number, id: string) => `/api/projects/${project}/files/preview/takes/stage_${sid}/${id}.png`;

export function useHash(): string {
  const [h, setH] = useState(location.hash.slice(1) || "/");
  useEffect(() => {
    const f = () => setH(location.hash.slice(1) || "/");
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return h;
}

/** Is a stage marked completed in folder `f`? (legacy `done: true` = every folder) */
export const doneIn = (s: Stage, f: string) => (Array.isArray(s.done) ? s.done.includes(f) : !!s.done);

/** `done` entries are folder names for stills; a clip approval is `<folder>/videos`. */
export const clipKey = (f: string) => `${f || "root"}/videos`;
export const anyClipDone = (s: Stage) => (Array.isArray(s.done) ? s.done.some((k) => k.endsWith("/videos")) : !!s.done);

export type RunpodBalance = { balance: number; spend_per_hr: number };
export type Credits = { effective_balance_micros: number; prepaid_balance_micros: number; cloud_credit_balance_micros: number; amount_micros: number; currency: string };

/** voices.json (pipeline/voice_registry.py): the slots a motion's {voice}, {voice.delivery}, {voice.moans}, {voice.sound} take. */
export type VoiceSlot = "voice" | "delivery" | "moans" | "sound";
export type VoiceSample = { project: string; stage: number; folder?: string };
export type VoiceEntry = { label?: string; tags?: string[]; disabled?: boolean; note?: string; tips?: string; sample?: VoiceSample;
  voice: string; delivery?: string; moans?: string; sound?: string;
  used: { project: string; stage: number }[]; sample_mtime: number | null };
export type VoiceLibrary = { voices: Record<string, VoiceEntry>; slots: VoiceSlot[] };

const VOICE_RE = /\{voice(?:\.(delivery|moans|sound))?(?::(S\d+))?\}/g;
/** The motion with its {voice...} slots filled (what animate sends); unfillable slots stay as they are. Same rules as voice_registry.expand. */
export function expandVoice(motion: string, v?: VoiceEntry): string {
  if (!v) return motion;
  return motion.replace(VOICE_RE, (m, slot: VoiceSlot | undefined, sp: string | undefined) => {
    const t = (v[slot ?? "voice"] ?? "").trim();
    return t ? t.replaceAll("(S)", `(${sp ?? "S1"})`) : m;
  });
}
/** The slots a motion uses. */
export const voiceSlotsIn = (motion: string): VoiceSlot[] => [...new Set([...motion.matchAll(VOICE_RE)].map((m) => (m[1] ?? "voice") as VoiceSlot))];
