import { CSSProperties, Fragment, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { anyClipDone, api, clipKey, doneIn, fileUrl, Job, Project, Stage, Take, takeUrl, useHash } from "./api";
import { Active, activeByStage, fmtSecs, pendingTakes, useAnimateBatchAdd, useAnimateTarget, useEstimate, useHealth, useJobs, useLoras, useProject, useRun } from "./hooks";
import { previewTakes, setPreviewTakes, useBatchArmed, useBoardCols, useHiddenActions, usePreviewTakes } from "./runPrefs";
import StageDrawer from "./StageDrawer";
import Versions from "./Versions";
import Lightbox, { LightboxItem } from "./Lightbox";
import { copyRef, RefChip, refOf } from "./controls";

function load(key: string): Set<string> {
  try { return new Set(JSON.parse(localStorage.getItem(key) ?? "[]")); } catch { return new Set(); }
}

type ActionInfo = { label: string; where: "cloud" | "local"; out: string; tip: string };
export const ACTIONS: Record<string, ActionInfo> = {
  preview: { label: "Preview", where: "cloud", out: "preview/", tip: "Krea2 base render in the cloud (cheap pose check) → preview/stage_N.png; every render is kept as a take to pick from" },
  preview_local: { label: "Preview", where: "local", out: "preview/", tip: "Krea2 base render on your local ComfyUI → preview/stage_N.png; every render is kept as a take to pick from" },
  local: { label: "IL + upscale", where: "local", out: "IL/", tip: "Illustrious pass + upscale + finish on your local ComfyUI, starting from preview/stage_N.png → IL/stage_N.png" },
  il_cloud: { label: "IL + upscale", where: "cloud", out: "IL/", tip: "The same IL + upscale + finish pass in the cloud, starting from preview/stage_N.png (no new Krea2 render) → IL/stage_N.png" },
  local_new: { label: "New preview + IL", where: "local", out: "IL/", tip: "Fresh cloud Krea2 preview (becomes the current take; the old one stays as a take), then the local IL pass → IL/" },
  face: { label: "Face detailer", where: "cloud", out: "IL/", tip: "Illustrious FaceDetailer in the cloud on IL/stage_N.png (run after IL + upscale); replaces it; the old one stays under ⋯ › Previous versions" },
  face_local: { label: "Face detailer", where: "local", out: "IL/", tip: "Illustrious FaceDetailer on your local ComfyUI on IL/stage_N.png (run after IL + upscale); replaces it; the old one stays under ⋯ › Previous versions" },
  hires_local: { label: "Latent upscale", where: "local", out: "hires/", tip: "SesquiLSR latent upscale of preview/stage_N.png + a low-denoise Krea2 pass on your local ComfyUI → hires/stage_N.png" },
  finish_local: { label: "Finish (SeedVR)", where: "local", out: "finish/", tip: "Optional SeedVR2 upscale of the upscale result on your local ComfyUI (from the folder you're reviewing, else the ✓'d one, else hires/, else IL/) → finish/stage_N.png" },
  finish: { label: "Finish (SeedVR)", where: "cloud", out: "finish/", tip: "Optional SeedVR2 upscale of the upscale result in the cloud (from the folder you're reviewing, else the ✓'d one, else hires/, else IL/) → finish/stage_N.png" },
  animate: { label: "Animate", where: "cloud", out: "videos/", tip: "Image-to-video from the stills in the chosen folder → <folder>/videos/stage_N.mp4" },
  animate_rife: { label: "Animate smooth", where: "cloud", out: "interp/videos/", tip: "Image-to-video plus frame smoothing (RIFE ×2) in the same cloud job: the 24 fps clip → <folder>/videos/, the 48 fps clip straight into interp/videos/ (⑤ Interpolate)" },
  interpolate: { label: "Interpolate", where: "local", out: "videos/", tip: "RIFE frame interpolation of the folder's clips on your local ComfyUI: 24 → 48 fps, same length; the source clip stays, a redone result's old one stays under ⋯ › Previous versions" },
  generate: { label: "Full cloud run", where: "cloud", out: "root", tip: "Everything in the cloud (Krea2 + IL + upscale) → stage_N.png in the project root" },
  dry_run: { label: "Dry run", where: "local", out: "—", tip: "Build and validate the graphs only; submits nothing" },
  recover: { label: "Recover clips", where: "cloud", out: "videos/", tip: "Saves cloud clips whose animate run was interrupted (started automatically after a restart)" },
};
type Seen = { baseline: number; slots: Record<string, number> };
const slotKey = (id: number, f: string, kind: string) => `${id}/${f}/${kind}`;
const nowSecs = () => Math.floor(Date.now() / 1000);
/** The mtime of each still/clip you last looked at (clicked or opened in the lightbox), per project. The first visit sets a baseline so everything already on disk counts as seen. */
function loadSeen(project: string): Seen {
  try {
    const v = JSON.parse(localStorage.getItem(`seen:${project}`) ?? "null") as { baseline: number; slots?: Record<string, number> } | null;
    if (v) return { baseline: v.baseline, slots: v.slots ?? {} };
  } catch { /* fall through */ }
  const fresh = { baseline: nowSecs(), slots: {} };
  try { localStorage.setItem(`seen:${project}`, JSON.stringify(fresh)); } catch { /* ignore */ }
  return fresh;
}

type Phase = "preview" | "upscale" | "finish" | "animate" | "interp" | "all";
const PHASES: { id: Phase; n: string; label: string; tip: string }[] = [
  { id: "preview", n: "①", label: "Preview", tip: "Check poses in the Krea2 previews and ✓ the ones that are good" },
  { id: "upscale", n: "②", label: "Upscale", tip: "Compare the IL / latent-upscale results against the preview and ✓ the chosen one" },
  { id: "finish", n: "③", label: "Finish", tip: "Optional: SeedVR on the ✓'d upscale; compare it against the upscale and ✓ it (skip straight to Animate if you don't need it)" },
  { id: "animate", n: "④", label: "Animate", tip: "Review the clips: ✓ approves, Re-roll renders a new one (← → move, d ✓, r re-roll, space play)" },
  { id: "interp", n: "⑤", label: "Interpolate", tip: "Optional: RIFE ×2 (24 → 48 fps) of the ✓'d clips into interp/videos/; compare against the source clip and ✓ it (← → move, d ✓, r redo, space play)" },
  { id: "all", n: "", label: "All files", tip: "Every folder's stills and clips at once" },
];
const CIVITAI_FOLDERS = new Set(["preview", "IL", "hires", "finish"]);  // the stills whose generation can be written out for Civitai
const UP_TARGETS = ["IL", "hires"];  // the upscale results (a finish starts from one of them)
const INTERP = "interp";  // where interpolated clips land (project.py interpolate): a clip-only folder, reviewed in the Interpolate phase
const PREVIEW_GATED = new Set(["il_cloud", "local", "face", "face_local", "hires_local", "finish", "finish_local"]);  // steps that build on an approved preview
/** Which still/clip slots a phase shows for each stage row. */
function wantSlot(phase: Phase, folder: string, f: string, kind: "still" | "clip", hidden: Set<string>, hideKind: Set<string>, clipFrom: string) {
  if (phase === "all") return !hidden.has(f) && !hideKind.has(kind);
  if (phase === "preview") return f === "preview" && kind === "still";
  if (phase === "upscale") return kind === "still" && (f === "preview" || f === folder);
  if (phase === "finish") return kind === "still" && (f === "finish" || UP_TARGETS.includes(f));  // the finish, plus the upscales it is made from
  if (phase === "interp") return kind === "clip" && (f === INTERP || f === clipFrom);  // the interpolated clip, plus the clip it was made from
  return f === folder;  // animate: the source folder's clip, plus its still for reference
}
/** Re-running a step makes its ✓ and every ✓ downstream of it stale: returns which `done` entries to drop. */
function staleKeys(action: string, folder: string): (k: string) => boolean {
  const still = (...fs: string[]) => (k: string) => fs.includes(k) || fs.some((f) => k === clipKey(f));
  if (action === "preview" || action === "preview_local") return () => false;  // a new take is only a candidate: picking it (Use this) clears what was made from the old one
  if (action === "local_new") return () => true;  // replaces the preview and the IL result
  if (["local", "il_cloud", "face", "face_local"].includes(action)) return still("IL", "finish");  // a finish may be made from IL/
  if (action === "hires_local") return still("hires", "finish");
  if (action === "finish" || action === "finish_local") return still("finish");
  if (action === "animate" || action === "animate_rife") return (k) => k === clipKey(folder) || k === clipKey(INTERP);  // the interpolation was made from the old clip
  if (action === "interpolate") return (k) => k === clipKey(INTERP);
  if (action === "generate") return still("");
  return () => false;
}

/** Moving on to a next step means you're finished with the one before it: the `done` entry to set when this action is submitted. */
function priorKey(action: string, folder: string): string | null {
  if (["local", "il_cloud", "hires_local"].includes(action)) return "preview";
  if (action === "finish" || action === "finish_local") return UP_TARGETS.includes(folder) ? folder : null;
  if (action === "animate" || action === "animate_rife") return folder;
  if (action === "interpolate") return clipKey(folder);  // interpolating a clip means it is the approved one
  return null;
}
/** Runs offered for a selected result folder: IL shows only the IL runs, hires only the latent ones, each plus the finish (new modes: add a row). Preview and animate stay available.
 * A folder not listed (root, preview) gets every run. */
export const FOLDER_ACTIONS: Record<string, string[]> = {
  preview: ["preview_local", "preview"],
  IL: ["local", "il_cloud", "face_local", "face", "finish_local", "finish"],
  hires: ["hires_local", "finish_local", "finish"],
  finish: ["finish_local", "finish"],
};
export const allowFor = (folder: string): Set<string> | undefined =>
  FOLDER_ACTIONS[folder] ? new Set([...FOLDER_ACTIONS[folder], "preview_local", "preview", "animate", "animate_rife"]) : undefined;

/** Where a stage's finish comes from when no folder is named: the ✓'d upscale, else hires/, else IL/ (as steps.finish_source). */
export function finishSource(p: Project, id: number): string | undefined {
  const s = p.config.stages.find((x) => x.id === id);
  const have = UP_TARGETS.slice().reverse().filter((f) => p.folders[f]?.stills[id]);  // hires first
  return have.find((f) => s && Array.isArray(s.done) && s.done.includes(f)) ?? have[0];
}

const COMPARE_LABELS: Record<string, string> = { hires: "latent upscale", IL: "IL upscale" };
/** The before/after base for a still in the lightbox: a finish is compared with the upscale it was made from, every other result with the preview. */
export function compareFor(p: Project, f: string, id: number): Pick<LightboxItem, "compare" | "compareLabel"> {
  if (f === "preview") return {};
  const base = (f === "finish" && finishSource(p, id)) || "preview";
  const v = p.folders[base]?.stills[id];
  if (!v) return {};
  return { compare: fileUrl(p.name, base, "still", id, v), compareLabel: COMPARE_LABELS[base] ?? "preview" };
}

/** The board phase (and result folder) an action's job is reviewed in. */
function jobPhase(a: string, folder: string): [string, string] {
  if (a.startsWith("preview")) return ["preview", ""];
  if (a === "hires_local") return ["upscale", "hires"];
  if (["local", "il_cloud", "face", "face_local", "local_new"].includes(a)) return ["upscale", "IL"];
  if (a.startsWith("finish")) return ["finish", ""];
  if (a === "animate") return ["animate", folder];
  if (a === "interpolate" || a === "animate_rife") return ["interp", folder];
  return ["all", ""];
}

/** Where a job's result is reviewed on the board: its project, the phase + folder of the step it ran, its first stage. */
export function jobLink(j: Job): string {
  const a = j.action, folder = (j.req.folder as string | undefined) ?? "", stage = ((j.req.stages as number[] | undefined) ?? [])[0];
  if (a === "edit") return `#/p/${j.project}/compose`;
  const [phase, f] = jobPhase(a, folder);
  const withFolder = phase === "upscale" || phase === "animate" || phase === "interp";
  const q = new URLSearchParams({ phase, ...(withFolder ? { folder: f } : {}), ...(stage ? { stage: String(stage) } : {}) });
  return `#/p/${j.project}/board?${q}`;
}

export const ACTION_LABEL: Record<string, string> = Object.fromEntries(Object.entries(ACTIONS).map(([k, v]) => [k, v.label]));
export const LOCAL_ACTIONS = new Set(["local", "face_local", "hires_local", "finish_local", "preview_local", "local_new", "interpolate"]);
// Run menu rows: one operation per row (name + output shown once), the free local run on the left, the paid cloud run on the right
export const ROWS: { title: string; out: string; free?: string; paid?: string; hue: string }[] = [
  { title: "Preview", out: "preview/", free: "preview_local", paid: "preview", hue: "#7c9cff" },
  { title: "IL + upscale", out: "IL/", free: "local", paid: "il_cloud", hue: "#b57cff" },
  { title: "Latent upscale", out: "hires/", free: "hires_local", hue: "#e0a33c" },
  { title: "Finish (SeedVR)", out: "finish/", free: "finish_local", paid: "finish", hue: "#3cc4a0" },
  { title: "Face detailer", out: "IL/ (replaced)", free: "face_local", paid: "face", hue: "#ff6fae" },
];

/** Animate sources for a set of stages: every visible folder with at least one of their stills, and what animating from it would do. */
type AnimSource = { f: string; have: number; missing: number[]; replace: number; thumb?: string };
/** Folders animate can take stills from. Every folder counts: the board's show/hide folder filter is a view setting only. */
export function animSources(p: Project, ids: number[]): AnimSource[] {
  return Object.entries(p.folders).flatMap(([f, st]) => {
    const have = ids.filter((i) => st.stills[i]);
    if (!have.length) return [];
    return [{ f, have: have.length, missing: ids.filter((i) => !st.stills[i]), replace: ids.filter((i) => st.clips[i]).length,
      thumb: ids.length === 1 ? fileUrl(p.name, f, "still", ids[0], st.stills[ids[0]]) : undefined }];
  });
}

function useOutside(open: boolean, close: () => void, sel: string) {
  useEffect(() => {
    if (!open) return;
    const f = (e: PointerEvent) => { if (!(e.target as Element).closest(sel)) close(); };
    addEventListener("pointerdown", f);
    return () => removeEventListener("pointerdown", f);
  }, [open, close, sel]);
}

type RunMenuProps = {
  readonly project: string; readonly ids: number[]; readonly label: string; readonly workFolder: string; readonly why: (a: string) => string;
  readonly onGo: (a: string, folder?: string) => void;
  readonly sources: AnimSource[];
  readonly allow?: Set<string>;  // only these actions are offered (the current phase's applicable runs); undefined = all
  readonly clipFrom?: string;  // the folder whose clips Interpolate reads (the Animate folder); defaults to workFolder
};

type Est = { known: number; of: number; seconds: number; load: number; parallel?: boolean };

/** The expected-time badge on a run button (from past runs). */
function Eta({ e, stages }: { readonly e?: Est; readonly stages: number }) {
  if (!e || e.known === 0) return null;
  const partial = e.known < e.of;
  let title = "expected processing time from past runs";
  if (stages > 1) title += ` (${e.of} stages added up)`;
  if (partial) title += ` — no baseline yet for ${e.of - e.known} of ${e.of}`;
  if (e.load > 20) title += `. includes about ${fmtSecs(e.load)} of model loading, paid once at the start of a local run`;
  else if (e.parallel) title += ". cloud jobs overlap, so the real wait can be shorter";
  return <span className="eta" title={title}>⏱ {partial ? "≥" : "~"}{fmtSecs(e.seconds)}{e.load > 20 && <small> incl. {fmtSecs(e.load)} load</small>}</span>;
}

/** One local/cloud button of a Run menu row. */
function RunCell({ a, col, lbl, hue, solo, blocked, nTakes, est, stages, onPick }: {
  readonly a: string; readonly col: "free" | "paid"; readonly lbl: string; readonly hue: string; readonly solo: boolean; readonly blocked: string; readonly nTakes: number; readonly est?: Est; readonly stages: number; readonly onPick: () => void;
}) {
  const where = col === "free" ? "Local (free)" : "Cloud (paid)";
  const isPreview = a === "preview" || a === "preview_local";
  return (
    <button className={`ritem hued ${col}`} style={{ "--hue": hue, ...(solo ? { gridColumn: "span 2" } : {}) } as CSSProperties} disabled={!!blocked} title={blocked || `${where}: ${ACTIONS[a].tip}`} onClick={onPick}>
      <span className="albl">{lbl}{isPreview && nTakes > 1 ? ` ×${nTakes}` : ""}</span>
      <Eta e={est} stages={stages} />
    </button>
  );
}

function animNote(src: AnimSource): string {
  if (src.missing.length) return `no still for #${src.missing.join(", #")}`;
  return `replaces ${src.replace} clip${src.replace > 1 ? "s" : ""}`;
}

/** The Animate block of a Run menu: pick the source folder, then Animate or Animate smooth. */
function AnimateForm({ sources, src, stages, why, smooth, batchHint, onPickSource, onRun }: {
  readonly sources: AnimSource[]; readonly src?: AnimSource; readonly stages: number; readonly why: (a: string) => string; readonly smooth: boolean; readonly batchHint: boolean;
  readonly onPickSource: (f: string) => void; readonly onRun: (a: string, folder: string) => void;
}) {
  const noStill = !src || src.missing.length > 0;
  return (
    <div className="aform">
      <div className="aftitle" style={{ color: "var(--animate)" }}>Animate</div>
      {sources.length === 0 ? <div className="dim">no stills</div> : <>
        <div className="afrow">
          <span className="afsrcs">
            {sources.map((x) => (
              <button key={x.f} className={`afsrc ${src?.f === x.f ? "on" : ""}`} onClick={() => onPickSource(x.f)}>
                {x.thumb && <img src={x.thumb} alt="" />}
                <span>{x.f || "root"}</span>
                {!x.thumb && <small>{x.have}/{stages} stills</small>}
              </button>
            ))}
          </span>
        </div>
        {src && (src.missing.length > 0 || src.replace > 0) && <div className={`afnote ${src.missing.length ? "warn" : ""}`}
          title={src.replace ? "the old clip moves to Previous versions" : undefined}>{animNote(src)}</div>}
        <div className="afgos">
          <button className="afgo" disabled={noStill || !!why("animate")} title={why("animate") || undefined} onClick={() => onRun("animate", src!.f)}>
            <span className="afgo-main">Animate</span>
            <span className="afgo-play">▶</span>
            {batchHint && <span className="afgo-sub">→ batch</span>}
          </button>
          {smooth && <button className="afgo interp" disabled={noStill || !!why("animate_rife")} title={why("animate_rife") || ACTIONS.animate_rife.tip} onClick={() => onRun("animate_rife", src!.f)}>
            <span className="afgo-main">Animate smooth</span>
            <span className="afgo-play">▶</span>
            <span className="afgo-sub">48 fps{batchHint ? " · → batch" : ""}</span>
          </button>}
        </div>
      </>}
    </div>
  );
}

/** One popover for every run action, grouped by pipeline step; used for the selection and for a single stage. */
export function RunMenu({ project, ids, label, workFolder, why, onGo, sources, allow, clipFrom }: RunMenuProps) {
  const [open, setOpen] = useState(false);
  const [pick, setPick] = useState<string | null>(null);
  const off = useHiddenActions();
  const shown = (a?: string) => !!a && !off.has(a) && (!allow || allow.has(a));
  const src = sources.find((x) => x.f === pick) ?? sources.find((x) => x.f === workFolder) ?? sources[0];
  const batchArmed = useBatchArmed();
  const animTarget = useAnimateTarget();  // both hooks run on every render: skipping one when disarmed crashes the page on arming
  const batchHint = batchArmed && animTarget === "runpod";  // "Start batch" armed: animate clicks queue
  useOutside(open, () => setOpen(false), ".runmenu");
  const none = ids.length === 0;
  const est = useEstimate(project, ids, open).data;
  const nTakes = usePreviewTakes();
  const runIt = (a: string, folder?: string) => { setOpen(false); onGo(a, folder); };
  return (
    <span className="runmenu menuwrap">
      <button className="btn primary" disabled={none} title={none ? "no stages to run" : undefined} onClick={() => setOpen(!open)}>{label} ▾</button>
      {open && (
        <div className="menu runpanel">
          <div className="rgrid">
            {ROWS.filter((r) => shown(r.free) || shown(r.paid)).map((r) => (
              <Fragment key={r.title}>
                <span className="rtitle hued" style={{ "--hue": r.hue } as CSSProperties} title={`→ ${r.out}`}>{r.title}</span>
                {([["free", r.free, "🖥"], ["paid", r.paid, "☁"]] as const).map(([col, a, lbl]) => shown(a) && (
                  <RunCell key={col} a={a!} col={col} lbl={lbl} hue={r.hue} solo={!(shown(r.free) && shown(r.paid))} blocked={why(a!)} nTakes={nTakes} est={est?.[a!]} stages={ids.length} onPick={() => runIt(a!)} />
                ))}
              </Fragment>
            ))}
          </div>
          {(shown("preview_local") || shown("preview")) && <div className="takesrow" title="Render several previews per stage (each with a fresh seed) and pick the one to carry on with">
            <span className="dim">Takes</span>
            <span className="seg">{[1, 2, 4].map((n) => <button key={n} className={nTakes === n ? "on" : ""} onClick={() => setPreviewTakes(n)}>×{n}</button>)}</span>
          </div>}
          {shown("animate") && <AnimateForm sources={sources} src={src} stages={ids.length} why={why} smooth={shown("animate_rife")} batchHint={batchHint} onPickSource={setPick} onRun={runIt} />}
          {shown("interpolate") && <div className="aform">
            <button className="afgo interp" disabled={!!why("interpolate")} title={why("interpolate") || ACTIONS.interpolate.tip}
              onClick={() => runIt("interpolate", clipFrom ?? workFolder)}>
              <span className="afgo-main">Interpolate</span>
              <span className="afgo-play">▶</span>
              <span className="afgo-sub">48 fps · 🖥</span>
            </button>
          </div>}
        </div>
      )}
    </span>
  );
}

type Show = "all" | "todo" | "done";
const store = (key: string, v: string) => { try { localStorage.setItem(key, v); } catch { /* storage unavailable */ } };
const unstore = (...keys: string[]) => { try { keys.forEach((k) => localStorage.removeItem(k)); } catch { /* storage unavailable */ } };
const scrollToStage = (id: number | string) => document.getElementById(`stage-${id}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
const hasModifier = (e: KeyboardEvent) => e.metaKey || e.ctrlKey || e.altKey;
const targetIn = (e: KeyboardEvent, sel: string) => !!(e.target as Element).closest?.(sel);
const isReview = (phase: Phase) => phase === "animate" || phase === "interp";
const isUpscaleStep = (phase: Phase) => phase === "upscale" || phase === "finish";
const isAnimateAction = (a: string) => a === "animate" || a === "animate_rife";

/** The folder a phase works in: what its ✓ applies to and what its Run acts on. */
function phaseFolder(phase: Phase, upTarget: string, animFolder: string): string {
  if (phase === "preview") return "preview";
  if (phase === "upscale") return upTarget;
  if (phase === "finish") return "finish";
  if (phase === "interp") return INTERP;
  return animFolder;
}

/** The Animate folder when the current one holds no stills: finish, else hires, else the first one. */
function defaultAnimFolder(folders: string[]): string {
  return ["finish", "hires"].find((f) => folders.includes(f)) ?? folders[0];
}

/** First visit: the earliest phase that still has unfinished stages. */
function firstOpenPhase(st: Stage[]): Phase {
  if (st.some((s) => !doneIn(s, "preview"))) return "preview";
  if (st.some((s) => !UP_TARGETS.some((f) => doneIn(s, f)))) return "upscale";
  return "animate";
}

type WhyCtx = { p: Project; stages: Stage[]; folder: string; animFolder: string; localDown: boolean };

/** The first stage among `ids` with no motion prompt (animate refuses those). */
function noMotionStage(ids: number[], stages: Stage[]) {
  return ids.find((i) => !stages.find((x) => x.id === i)?.motion?.trim());
}

/** Why a finish can't run: it needs an upscale result (in the folder being reviewed, else in either upscale folder). */
function finishBlock(ids: number[], c: WhyCtx): string {
  const reviewing = UP_TARGETS.includes(c.folder);
  const from = reviewing ? [c.folder] : UP_TARGETS;
  if (!ids.some((i) => !from.some((f) => c.p.folders[f]?.stills[i]))) return "";
  return `needs an upscale result (${reviewing ? c.folder : "IL or hires"}) — run IL + upscale or the latent upscale first`;
}

/** Why an action can't run on these stages; "" = it can. */
function disabledReason(a: string, ids: number[], c: WhyCtx): string {
  if (!ids.length && a !== "dry_run") return "no stages to run";
  if (LOCAL_ACTIONS.has(a) && c.localDown) return "local ComfyUI is down";
  if (a === "interpolate" && ids.some((i) => !c.p.folders[c.animFolder]?.clips[i])) return `needs a clip in ${c.animFolder || "root"}/videos — animate first`;
  const noMotion = isAnimateAction(a) ? noMotionStage(ids, c.stages) : undefined;
  if (noMotion !== undefined) return `stage ${noMotion} has no motion prompt — open its drawer and write one`;
  if (PREVIEW_GATED.has(a) && ids.some((i) => !c.p.folders["preview"]?.stills[i])) return "make a Preview first";
  if ((a === "face" || a === "face_local") && ids.some((i) => !c.p.folders["IL"]?.stills[i])) return "needs IL/stage_N.png — run IL + upscale first";
  return a === "finish" || a === "finish_local" ? finishBlock(ids, c) : "";
}

function togglePlay(id: number) {
  const v = document.querySelector<HTMLVideoElement>(`#stage-${id} video`);
  if (!v) return;
  if (v.paused) void v.play();
  else v.pause();
}

function emptyMessage(notReady: number, show: Show): string {
  if (notReady > 0 && show !== "done") return `${notReady} stage${notReady > 1 ? "s" : ""} haven't reached this step — ✓ them in the previous phase first`;
  return `no ${show === "done" ? "completed" : "to-do"} stages`;
}

function bulkDoneTip(count: number, allDone: boolean, folder: string): string {
  if (!count) return "select stages to ✓ (approve / mark done)";
  return allDone ? `all ${count} completed in ${folder || "root"} — click to undo` : `mark ${count} selected completed in ${folder || "root"}`;
}

const SHOW_TIPS: Record<Show, string> = { all: "all stages", todo: "to do", done: "completed" };

function PhaseBar({ name, phase, stages, notReady, onPhase }: { readonly name: string; readonly phase: Phase; readonly stages: Stage[]; readonly notReady: number; readonly onPhase: (p: Phase) => void }) {
  const t = stages.length;
  return (
    <div className="phasebar">
      {PHASES.map((ph) => {
        const n = phaseStatCount(stages, ph.id);
        const counted = ph.id !== "all";
        return (
          <button key={ph.id} className={`phase ${phase === ph.id ? "on" : ""} ${counted && t > 0 && n === t ? "full" : ""}`} title={ph.tip} onClick={() => onPhase(ph.id)}>
            <span className="pname">{ph.n} {ph.label}</span>
            {counted && <small>{n}/{t}</small>}
            {counted && <i className="pfill" style={{ width: `${t ? (100 * n) / t : 0}%` }} />}
          </button>
        );
      })}
      <a className="phase link" href={`#/p/${name}/compose`} title="Cut the approved stills or clips to the music">⑥ Compose ›</a>
      {notReady > 0 && (isUpscaleStep(phase) || isReview(phase)) && <span className="dim" title={`${notReady} stage${notReady > 1 ? "s" : ""} not at this step yet (previous step not ✓)`}>⏳ {notReady}</span>}
    </div>
  );
}

/** The working-folder switch: which upscale the ✓ approves, or which folder is animated and reviewed. */
function FolderSeg({ phase, upTarget, animFolder, animFolders, onUpTarget, onAnimFolder }: {
  readonly phase: Phase; readonly upTarget: string; readonly animFolder: string; readonly animFolders: string[]; readonly onUpTarget: (f: string) => void; readonly onAnimFolder: (f: string) => void;
}) {
  if (phase === "upscale") return (
    <span className="seg" title="Which result the ✓ approves">
      {UP_TARGETS.map((f) => <button key={f} className={upTarget === f ? "on" : ""} onClick={() => onUpTarget(f)}>{f}</button>)}
    </span>
  );
  if (!isReview(phase) && phase !== "all") return null;
  return (
    <span className="seg" title="The folder whose stills are animated and whose clips are reviewed">
      {animFolders.length === 0 && <button disabled>no stills yet</button>}
      {animFolders.map((f) => <button key={f} className={animFolder === f ? "on" : ""} onClick={() => onAnimFolder(f)}>{f || "root"}</button>)}
    </span>
  );
}

/** Phase "all": which folders and kinds to show. */
function FolderBar({ p, folder, hidden, hideKind, onFlipHidden, onFlipKind, onShowAll, onSolo }: {
  readonly p: Project; readonly folder: string; readonly hidden: Set<string>; readonly hideKind: Set<string>;
  readonly onFlipHidden: (f: string) => void; readonly onFlipKind: (k: string) => void; readonly onShowAll: () => void; readonly onSolo: () => void;
}) {
  return (
    <div className="folderbar">
      {Object.keys(p.folders).map((f) => {
        const st = p.folders[f]; const n = Object.keys(st.stills).length, c = Object.keys(st.clips).length;
        const off = hidden.has(f);
        return (
          <button key={f} className={`chip-f ${off ? "off" : "on"}`} title={`${n} stills, ${c} clips · click to ${off ? "show" : "hide"}`} onClick={() => onFlipHidden(f)}>
            {off ? "○" : "●"} {f || "root"} <small>{n}{c ? ` · ${c}▶` : ""}</small>
          </button>
        );
      })}
      <span className="sep" />
      {(["still", "clip"] as const).map((k) => (
        <button key={k} className={`chip-f ${hideKind.has(k) ? "off" : "on"}`} onClick={() => onFlipKind(k)}>{hideKind.has(k) ? "○" : "●"} {k === "still" ? "stills" : "clips"}</button>
      ))}
      <button className="btn sm" title="show everything" onClick={onShowAll}>all</button>
      <button className="btn sm" title="show only the working folder" onClick={onSolo}>solo {folder || "root"}</button>
    </div>
  );
}

export default function Board({ name }: { readonly name: string }) {
  const proj = useProject(name);
  const qc = useQueryClient();
  const jobs = useJobs().data ?? [];
  const health = useHealth().data;
  const run = useRun(name);
  const batchArmed = useBatchArmed();
  const animTarget = useAnimateTarget();
  const batchAdd = useAnimateBatchAdd(name);
  const [sel, setSel] = useState<Set<number>>(new Set());
  const lastSel = useRef<number | null>(null);
  const [animFolder, setAnimFolder] = useState<string>("");
  const [upTarget, setUpTarget] = useState<string>(() => { const v = localStorage.getItem(`upTarget:${name}`) ?? ""; return UP_TARGETS.includes(v) ? v : "hires"; });
  const [phase, setPhase] = useState<Phase>(() => (localStorage.getItem(`phase:${name}`) as Phase) || "preview");
  const phaseSet = useRef(!!localStorage.getItem(`phase:${name}`));
  const [cur, setCur] = useState<number | null>(null);
  const pickPhase = (v: Phase) => { phaseSet.current = true; setPhase(v); store(`phase:${name}`, v); };
  const pickUpTarget = (f: string) => { setUpTarget(f); store(`upTarget:${name}`, f); };
  const folder = phaseFolder(phase, upTarget, animFolder);
  // what ✓ means here: a still folder name, or `<folder>/videos` when reviewing clips
  const doneKey = isReview(phase) ? clipKey(folder) : folder;
  const [drawer, setDrawer] = useState<number | null>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const cols = useBoardCols();
  const [light, setLight] = useState<{ items: LightboxItem[]; i: number } | null>(null);
  const [takeLight, setTakeLight] = useState<{ sid: number; id: string } | null>(null);
  const [msg, setMsg] = useState("");
  const [show, setShow] = useState<Show>(() => (localStorage.getItem(`show:${name}`) as Show) || "all");
  const [hidden, setHidden] = useState<Set<string>>(() => load(`hidden:${name}`));
  const [hideKind, setHideKind] = useState<Set<string>>(() => load(`hideKind:${name}`));
  const [seen, setSeen] = useState<Seen>(() => loadSeen(name));
  useEffect(() => { setSeen(loadSeen(name)); }, [name]);
  // "new" marks in the Jobs panel clear when a project's board is opened, or when its job finishes while the board is open
  const doneSeen = useRef(new Set<string>());
  useEffect(() => { dispatchEvent(new CustomEvent("project-viewed", { detail: name })); }, [name]);
  useEffect(() => {
    for (const j of jobs) {
      if (j.project === name && j.status === "done" && !doneSeen.current.has(j.id)) {
        doneSeen.current.add(j.id);
        dispatchEvent(new CustomEvent("project-viewed", { detail: name }));
      }
    }
  }, [jobs, name]);
  // a link from a job (jobLink) opens the step it ran: #/p/<name>/board?phase=..&folder=..&stage=..; the query is consumed so later
  // phase changes stick
  const hash = useHash();
  useEffect(() => {
    const [path, query] = hash.split("?");
    if (!query) return;
    const q = new URLSearchParams(query), ph = q.get("phase") as Phase | null, f = q.get("folder");
    if (ph && PHASES.some((x) => x.id === ph)) pickPhase(ph);
    if (ph === "upscale" && f && UP_TARGETS.includes(f)) pickUpTarget(f);
    if ((ph === "animate" || ph === "interp") && f !== null) setAnimFolder(f);
    const st = q.get("stage");
    if (st) setTimeout(() => scrollToStage(st), 400);
    history.replaceState(null, "", `#${path}`);
    dispatchEvent(new HashChangeEvent("hashchange"));
  }, [hash]);  // eslint-disable-line react-hooks/exhaustive-deps
  const markSeen = (...marks: [string, number][]) => setSeen((prev) => {
    const n = { ...prev, slots: { ...prev.slots, ...Object.fromEntries(marks) } };
    store(`seen:${name}`, JSON.stringify(n));
    return n;
  });
  useEffect(() => { setHidden(load(`hidden:${name}`)); setHideKind(load(`hideKind:${name}`)); }, [name]);
  const pickShow = (v: Show) => { setShow(v); store(`show:${name}`, v); };
  const flip = (set: Set<string>, v: string, key: string, put: (s: Set<string>) => void) => {
    const n = new Set(set);
    if (n.has(v)) n.delete(v); else n.add(v);
    put(n);
    store(key, JSON.stringify([...n]));
  };
  const p = proj.data;
  const folders = useMemo(() => Object.keys(p?.folders ?? { "": 1 }), [p]);
  const active = activeByStage(jobs, name);
  const pending = pendingTakes(jobs, name);
  // auto-hide the stage editor on any press outside it (pending edits are flushed when it unmounts)
  useEffect(() => {
    if (drawer === null) return;
    const away = (e: PointerEvent) => {
      if (!(e.target as Element).closest(".drawer, .lightbox, .jobs, .jobsbtn")) setDrawer(null);
    };
    addEventListener("pointerdown", away);
    return () => removeEventListener("pointerdown", away);
  }, [drawer]);
  // Esc: close the editor, else clear the selection (the lightbox handles its own Esc)
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || light || targetIn(e, "input, textarea, select")) return;
      if (drawer !== null) setDrawer(null); else setSel(new Set());
    };
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  }, [drawer, light]);
  // the working folder must hold stills (animate reads <folder>/stage_N.png) and be visible; default to finish / the first one
  const animFolders = useMemo(() => folders.filter((f) => Object.keys(p?.folders[f]?.stills ?? {}).length), [folders, p]);
  useEffect(() => { if (animFolders.length && !animFolders.includes(animFolder)) setAnimFolder(defaultAnimFolder(animFolders)); }, [animFolders, animFolder]);
  // first visit: open the earliest phase that still has unfinished stages
  useEffect(() => {
    if (phaseSet.current || !p?.config.stages.length) return;
    phaseSet.current = true;
    setPhase(firstOpenPhase(p.config.stages));
  }, [p]);

  const reviewKey = useRef<(e: KeyboardEvent) => void>(() => {});
  useEffect(() => {
    const f = (e: KeyboardEvent) => reviewKey.current(e);
    addEventListener("keydown", f);
    return () => removeEventListener("keydown", f);
  }, []);
  if (proj.error) return <div className="page err">{String(proj.error.message)}</div>;
  if (!p) return <div className="page">loading…</div>;
  const allStages = p.config.stages;
  const stageOf = (id: number) => allStages.find((x) => x.id === id)!;
  // completion is per folder: the ✓ and the filter apply to the folder you're working in
  const patchDone = (ids: number[], edit: (cur: string[]) => string[]) => {
    const changed = new Map<number, string[]>();
    for (const id of ids) {
      const s = stageOf(id);
      const have = legacyDone(s, folders);
      const next = edit(have);
      if (Array.isArray(s.done) && next.length === have.length && next.every((k, i) => k === have[i])) continue;  // unchanged: no PATCH
      changed.set(id, next);
    }
    if (!changed.size) return Promise.resolve();
    // optimistic: the ✓ flips at once; the refetch after the PATCHes settles it (or reverts it if one failed)
    qc.setQueryData<Project>(["project", name], (old) => old && { ...old, config: { ...old.config,
      stages: old.config.stages.map((s) => (changed.has(s.id) ? { ...s, done: changed.get(s.id) } : s)) } });
    const calls = [...changed].map(([id, done]) => api(`/projects/${name}/stages/${id}`, "PATCH", { done }));
    return Promise.all(calls).finally(() => void proj.refetch());
  };
  const markDone = (ids: number[], done: boolean) => patchDone(ids, (c) => (done ? [...new Set([...c, doneKey])] : c.filter((f) => f !== doneKey)));
  const nDone = allStages.filter((s) => doneIn(s, doneKey)).length;
  // soft gate: a stage is "ready" for a phase once the step before it is approved (nothing is blocked, just tucked away)
  // a stage has reached a phase once the step before it is ✓, or it already has output there
  const has = (s: Stage, f: string, k: "stills" | "clips") => !!p.folders[f]?.[k][s.id];
  // ...or it already holds this step's output: like Upscale, Finish lists the ✓'d upscales and Animate the ✓'d stills of its folder
  const ready = (s: Stage) => readyFor(s, phase, folder, animFolder, active, has);
  // Animate smooth lands its result in interp/, so it is tracked in the Interpolate phase, not Animate
  const busyIn = (id: number) => phase === "animate" && active[id]?.action === "animate_rife" ? undefined : active[id];
  const notReady = allStages.filter((s) => !ready(s)).length;
  const stages = allStages.filter((s) => ready(s) && (show === "all" || (show === "done") === doneIn(s, doneKey)));

  // an operation on the selection consumes it; one on a single stage's own menu leaves the selection alone
  const clearIfSelection = (ids: number[]) => setSel((prev) => prev.size === ids.length && ids.every((i) => prev.has(i)) ? new Set() : prev);
  // a run started: its earlier approval is stale, and the step before it counts as approved
  const settleDone = (action: string, f: string, ids: number[]) => {
    setMsg("");
    clearIfSelection(ids);
    const stale = staleKeys(action, f), prior = priorKey(action, f);
    void patchDone(ids, (c) => { const n = c.filter((k) => !stale(k)); return prior !== null && !n.includes(prior) ? [...n, prior] : n; });
  };

  const go = async (action: string, ids: number[], extra: Record<string, unknown> = {}) => {
    // picking a later step from an earlier phase approves the preview for you (the server only runs these on an approved preview)
    const unapproved = PREVIEW_GATED.has(action) ? ids.filter((i) => !doneIn(stageOf(i), "preview")) : [];
    if (unapproved.length) await patchDone(unapproved, (c) => [...new Set([...c, "preview"])]);
    // a finish starts from the upscale being reviewed (from the Finish phase: the ✓'d one / hires / IL, decided per stage on the server)
    const sendFolder = isAnimateAction(action) || (action.startsWith("finish") && UP_TARGETS.includes(folder));
    const takes = (action === "preview" || action === "preview_local") && previewTakes() > 1 ? { takes: previewTakes() } : {};
    const src = action === "interpolate" ? { folder: animFolder } : {};  // interpolate always reads the Animate folder's clips
    const f = (extra.folder as string | undefined) ?? folder;
    const onError = (e: Error) => setMsg(`error: ${e.message}`);
    // armed "Start batch" + RunPod target: animate clicks queue on the server and fire as one job at the batch size
    if (sendFolder && batchArmed && animTarget === "runpod") {
      batchAdd.mutate({ action, stages: ids, folder: f }, {
        onSuccess: (r) => {
          if (!r.job) { setMsg(`batch: ${r.pending}/${r.size} queued`); return; }  // still accumulating: nothing renders yet
          settleDone(action, f, ids);
        },
        onError,
      });
      return;
    }
    run.mutate({ action, stages: ids, folder: sendFolder ? folder : undefined, ...takes, ...extra, ...src }, { onSuccess: () => settleDone(action, f, ids), onError });
  };
  const whyCtx: WhyCtx = { p, stages: allStages, folder, animFolder, localDown: !!health && !health.local };
  const disabledWhy = (a: string, ids: number[] = []) => disabledReason(a, ids, whyCtx);
  const toggle = (id: number, range = false) => setSel((prev) => {
    const n = new Set(prev);
    const ids = stages.map((s) => s.id);
    if (range && lastSel.current !== null && ids.includes(lastSel.current)) {
      const [a, b] = [ids.indexOf(lastSel.current), ids.indexOf(id)].sort((x, y) => x - y);
      ids.slice(a, b + 1).forEach((i) => n.add(i));
    } else if (n.has(id)) n.delete(id);
    else n.add(id);
    lastSel.current = id;
    return n;
  });
  const selected = [...sel].sort((a, b) => a - b);
  // the global Run acts on the selection, or on every shown stage when nothing is selected
  const runIds = sel.size ? selected : stages.map((s) => s.id);
  // same toggle as a row's ✓: if every selected stage is already completed here, clicking undoes them, otherwise it completes them all
  const allSelDone = selected.length > 0 && selected.every((id) => doneIn(stageOf(id), doneKey));
  const allShown = stages.length > 0 && stages.every((s) => sel.has(s.id));

  // c copies the current stage's ref; clip review keys: ← → move, d approve, r re-roll, space play/pause
  const curId = cur !== null && stages.some((s) => s.id === cur) ? cur : null;
  const redo = (id: number) => {
    if (phase === "interp") {
      if (!disabledWhy("interpolate", [id])) void go("interpolate", [id]);
    } else if (p.folders[folder]?.stills[id] && !disabledWhy("animate", [id])) void go("animate", [id], { folder });
  };
  const stageKey = (e: KeyboardEvent, id: number) => {
    if (e.key === "d") void markDone([id], !doneIn(stageOf(id), doneKey));
    else if (e.key === "r") redo(id);
    else if (e.key === " ") { e.preventDefault(); togglePlay(id); }
  };
  const moveKey = (e: KeyboardEvent) => {
    const ids = stages.map((s) => s.id);
    const i = curId === null ? -1 : ids.indexOf(curId);
    let to: number;
    if (["ArrowRight", "ArrowDown", "j"].includes(e.key)) to = i + 1;
    else if (["ArrowLeft", "ArrowUp", "k"].includes(e.key)) to = Math.max(0, i - 1);
    else return false;
    e.preventDefault();
    const id = ids[Math.max(0, Math.min(ids.length - 1, to))];
    setCur(id);
    scrollToStage(id);
    return true;
  };
  const canCopyRef = (e: KeyboardEvent) => e.key === "c" && curId !== null && !light && !takeLight && drawer === null && !hasModifier(e) && !targetIn(e, "input, textarea, select");
  reviewKey.current = (e) => {
    if (canCopyRef(e)) {
      const ref = refOf(name, curId!);
      void copyRef(ref).then((ok) => setMsg(ok ? `copied ${ref}` : "clipboard unavailable"));
      return;
    }
    if (!isReview(phase) || light || drawer !== null || hasModifier(e) || targetIn(e, "input, textarea, select, button, video") || !stages.length) return;
    if (!moveKey(e) && curId !== null) stageKey(e, curId);
  };

  const openLight = (sid: number, f: string, kind: "still" | "clip") => {
    const items = stages.flatMap((s) => slotItem(p, s.id, f, kind));
    setLight({ items, i: Math.max(0, items.findIndex((it) => it.id === sid)) });
  };

  // preview takes: every render of a stage's preview, one of them current (preview/stage_N.png)
  const takesOf = (sid: number): Take[] => p.folders["preview"]?.takes?.[sid] ?? [];
  const seenTakes = (sid: number) => { const m = Math.max(0, ...takesOf(sid).map((t) => t.mtime)); if (m) markSeen([slotKey(sid, "takes", "still"), m]); };
  const openTake = (sid: number, id: string) => { seenTakes(sid); setTakeLight({ sid, id }); };
  const showMsgErr = (e: Error) => setMsg(`error: ${e.message}`);
  const useTake = (sid: number, id: string) => api(`/projects/${name}/stages/${sid}/takes/${id}/use`, "POST")
    .then(() => proj.refetch()).catch(showMsgErr);
  const clearTakes = (sid: number) => api(`/projects/${name}/stages/${sid}/takes/clear`, "POST")
    .then(() => proj.refetch()).catch(showMsgErr);
  const takeLightbox = () => {
    if (!takeLight) return null;
    const items = takeItems(p, takeLight.sid, takesOf(takeLight.sid));
    if (!items.length) return null;
    const i = items.findIndex((it) => it.take?.id === takeLight.id);
    return <Lightbox items={items} index={Math.max(0, i)} project={name} onIndex={(j) => setTakeLight({ sid: takeLight.sid, id: items[j].take!.id })}
      onClose={() => setTakeLight(null)} onUse={(it) => void useTake(it.take!.sid, it.take!.id)} />;
  };
  // drag a card's ⠿ onto another: it takes that card's place in project.json (ids, files and refs stay; the edit order changes)
  const moveStage = (from: number, to: number) => {
    setDragging(null);
    const ids = allStages.map((s) => s.id).filter((i) => i !== from);
    const after = allStages.findIndex((s) => s.id === from) < allStages.findIndex((s) => s.id === to);
    ids.splice(ids.indexOf(to) + (after ? 1 : 0), 0, from);
    return api(`/projects/${name}/order`, "PUT", { order: ids }).then(() => proj.refetch()).catch(showMsgErr);
  };
  const showLabels: Record<Show, string> ={ all: `All ${allStages.length}`, todo: `○ ${allStages.length - nDone}`, done: `✓ ${nDone}` };
  const lightIndex = (i: number) => {
    if (!light) return;
    const m = light.items[i]?.seen;
    if (m) markSeen(m);
    setLight({ ...light, i });
  };

  return (
    <div className="page board">
      <div className="pinbar">
      <PhaseBar name={name} phase={phase} stages={allStages} notReady={notReady} onPhase={pickPhase} />
      <div className="cmdbar">
        <div className="cmdrow">
          <label className="chk" title="select / clear all shown stages">
            <input type="checkbox" aria-label="select all shown stages" checked={allShown} ref={(el) => { if (el) el.indeterminate = sel.size > 0 && !allShown; }}
              onChange={(e) => setSel(e.target.checked ? new Set(stages.map((s) => s.id)) : new Set())} />
          </label>
          {sel.size > 0 && <span className="selcount on">{sel.size}</span>}
          {sel.size > 0 && <button className="btn sm" onClick={() => setSel(new Set())} title="clear selection (Esc)">×</button>}
          <span className="vsep" />
          <FolderSeg phase={phase} upTarget={upTarget} animFolder={animFolder} animFolders={animFolders} onUpTarget={pickUpTarget} onAnimFolder={setAnimFolder} />
          <span className="seg" title="Filter by completion in the working folder">
            {(["all", "todo", "done"] as const).map((v) => (
              <button key={v} className={show === v ? "on" : ""} title={SHOW_TIPS[v]} onClick={() => pickShow(v)}>{showLabels[v]}</button>
            ))}
          </span>
          <span className="spacer" />
          <button className={`btn ${allSelDone ? "donebtn" : ""}`} disabled={!sel.size}
            title={bulkDoneTip(sel.size, allSelDone, folder)}
            onClick={() => { void markDone(selected, !allSelDone); clearIfSelection(selected); }}>✓</button>
          <RunMenu project={name} ids={runIds} label="Run" workFolder={folder} allow={isUpscaleStep(phase) ? allowFor(folder) : undefined} sources={animSources(p, runIds)} why={(a) => disabledWhy(a, runIds)} onGo={(a, f) => void go(a, runIds, f === undefined ? {} : { folder: f })} clipFrom={animFolder} />
          <button className="btn" title="add a stage" onClick={() => void api(`/projects/${name}/stages`, "POST").then(() => proj.refetch())}>＋</button>
        </div>
      </div>
      </div>
      {phase === "all" && <FolderBar p={p} folder={folder} hidden={hidden} hideKind={hideKind}
        onFlipHidden={(f) => flip(hidden, f, `hidden:${name}`, setHidden)} onFlipKind={(k) => flip(hideKind, k, `hideKind:${name}`, setHideKind)}
        onShowAll={() => { setHidden(new Set()); setHideKind(new Set()); unstore(`hidden:${name}`, `hideKind:${name}`); }}
        onSolo={() => { const n = new Set(folders.filter((f) => f !== folder)); setHidden(n); store(`hidden:${name}`, JSON.stringify([...n])); }} />}
      {msg && <button type="button" className="msg" onClick={() => setMsg("")}>{msg}</button>}

      {stages.length === 0 && <div className="dim empty">{emptyMessage(notReady, show)}</div>}
      <div className="stages" style={{ "--cols": cols } as CSSProperties}>
        {stages.map((s) => (
          <StageRow key={s.id} p={p} s={s} seen={seen} onSeen={markSeen} hidden={hidden} hideKind={hideKind} sel={sel.has(s.id)} selecting={sel.size > 0} busy={busyIn(s.id)} onToggle={(range) => toggle(s.id, range)}
            folder={folder} clipFrom={animFolder} phase={phase} doneKey={doneKey} cur={curId === s.id} onCur={() => setCur(s.id)} onDone={() => void markDone([s.id], !doneIn(s, doneKey))} onEdit={() => setDrawer(s.id)} onOpen={openLight}
            onGo={(a, f) => void go(a, [s.id], f === undefined ? {} : { folder: f })} disabledWhy={(a) => disabledWhy(a, [s.id])}
            takes={takesOf(s.id)} pendingTakes={pending[s.id] ?? 0} onOpenTake={(id) => openTake(s.id, id)} onUseTake={(id) => void useTake(s.id, id)} onClearTakes={() => void clearTakes(s.id)}
            dragging={dragging} onDrag={setDragging} onDropOn={(to) => void moveStage(dragging!, to)} />
        ))}
      </div>
      {drawer !== null && <StageDrawer project={name} stage={stages.find((s) => s.id === drawer)!} stages={stages} onClose={() => setDrawer(null)} onNav={setDrawer} />}
      {takeLightbox()}
      {light && <Lightbox project={name} items={light.items} index={light.i} onIndex={lightIndex} onClose={() => setLight(null)} />}
    </div>
  );
}

/** A stage's `done` list; the legacy `true` means done everywhere. */
function legacyDone(s: Stage, folders: string[]): string[] {
  if (Array.isArray(s.done)) return s.done;
  return s.done ? folders : [];
}

/** The lightbox entry of one stage's still/clip in a folder (none if it has no such file). */
function slotItem(p: Project, id: number, f: string, kind: "still" | "clip"): LightboxItem[] {
  const fs = p.folders[f]; const v = (kind === "still" ? fs?.stills : fs?.clips)?.[id];
  if (!v) return [];
  const query = `/projects/${p.name}/civitai?folder=${f}&stage=${id}`;
  let civitai: string | undefined;
  if (kind === "clip") civitai = `${query}&kind=clip`;
  else if (CIVITAI_FOLDERS.has(f)) civitai = query;
  return [{ id, label: `stage ${id} · ${f || "root"} ${kind}`, kind, src: fileUrl(p.name, f, kind, id, v), seen: [slotKey(id, f, kind), v],
            ...(kind === "still" ? compareFor(p, f, id) : {}),
            ...(civitai ? { civitai } : {}) }];
}

/** The lightbox entries of a stage's preview takes. */
function takeItems(p: Project, sid: number, rows: Take[]): LightboxItem[] {
  const cur = rows.find((t) => t.current), curV = p.folders["preview"]?.stills[sid];
  return rows.map((t, i) => {
    const source = t.source === "run" ? "" : ` (${t.source})`;
    return {
      id: sid, kind: "still" as const, src: takeUrl(p.name, sid, t.id),
      label: `stage ${sid} · take ${rows.length - i} of ${rows.length}${source} · ${t.current ? "current" : "take"}`,
      take: { project: p.name, sid, id: t.id, current: t.current, delta: t.delta, currentId: cur?.id },
      civitai: `/projects/${p.name}/civitai?folder=preview&stage=${sid}&take=${t.id}`,
      ...(!t.current && curV ? { compare: fileUrl(p.name, "preview", "still", sid, curV), compareLabel: "current" } : {}),
    };
  });
}

type Slot = { f: string; kind: "still" | "clip"; v: number };

/** The results first, the references they were made from after them. */
function sortSlots(slots: Slot[], phase: Phase) {
  const last: Partial<Record<Phase, (x: Slot) => boolean>> = {
    upscale: (x) => x.f === "preview",
    finish: (x) => x.f !== "finish",
    animate: (x) => x.kind !== "clip",
    interp: (x) => x.f !== INTERP,
  };
  const isLast = last[phase];
  if (isLast) slots.sort((a, b) => Number(isLast(a)) - Number(isLast(b)));
}

/** Only this phase's own output can be new; earlier steps' files shown for reference never are. */
function isOwnOutput(phase: Phase, f: string, kind: string) {
  if (phase === "upscale") return f !== "preview";
  if (phase === "finish") return f === "finish";
  if (phase === "animate") return kind === "clip";
  if (phase === "interp") return f === INTERP;
  return true;
}

/** Shown small, for reference: what this phase's result was made from. */
function isRefSlot(phase: Phase, f: string, kind: string) {
  if (phase === "upscale") return f === "preview";
  if (phase === "finish") return f !== "finish";
  if (phase === "interp") return f !== INTERP || kind === "still";
  return phase === "animate" && kind === "still";
}

/** Where a slot's file sits inside its project. */
export function slotPath(f: string, kind: string, id: number) {
  const dir = f ? `${f}/` : "";
  return kind === "clip" ? `${dir}videos/stage_${id}.mp4` : `${dir}stage_${id}.png`;
}

/** Hover-play for a clip inside a media button (children of a button don't get mouse events in every browser). */
export const hoverPlay = {
  onMouseEnter: (e: MouseEvent<HTMLElement>) => { void e.currentTarget.querySelector("video")?.play(); },
  onMouseLeave: (e: MouseEvent<HTMLElement>) => { const v = e.currentTarget.querySelector("video"); if (v) { v.pause(); v.currentTime = 0; } },
};

function SlotFigure({ p, sid, slot, phase, doneKey, isNew, onOpen }: { readonly p: Project; readonly sid: number; readonly slot: Slot; readonly phase: Phase; readonly doneKey: string; readonly isNew: boolean; readonly onOpen: () => void }) {
  const { f, kind, v } = slot;
  const fs = p.folders[f];
  const target = isUpscaleStep(phase) && f === doneKey && kind === "still";
  return (
    <figure className={`${isNew ? "newitem" : ""} ${isRefSlot(phase, f, kind) ? "ref" : ""} ${target ? "target" : ""}`}>
      <button type="button" className="mediabtn" onClick={onOpen} aria-label={`open ${f || "root"} ${kind} of stage ${sid}`} {...(kind === "clip" ? hoverPlay : {})}>
        {kind === "still"
          ? <img loading="lazy" alt="" src={fileUrl(p.name, f, "still", sid, v)} />
          : <video muted loop preload="metadata" src={fileUrl(p.name, f, "clip", sid, v)} />}
      </button>
      <figcaption>{f || "root"}
        {kind === "still" && fs?.stale?.includes(sid) && <span className="pill warn" title="made from a preview take that is no longer current">⚠</span>}
        {kind === "clip" && fs?.stale_clips?.includes(sid) && <span className="pill warn" title={f === INTERP ? "made from an older clip: the source clip was re-rendered since" : "made from an older still: the still was re-rendered since — re-animate to update"}>⚠</span>}
      </figcaption>
      <button className="btn sm reveal" title="open in folder" onClick={(e) => { e.stopPropagation(); void api(`/projects/${p.name}/reveal`, "POST", { path: slotPath(f, kind, sid) }); }}>📂</button>
    </figure>
  );
}

/** Preview takes other than the current one: the 4 newest as thumbs beside the preview, the rest behind "+N". */
function TakeStrip({ p, sid, others, seenAt, pending, onOpen, onUse }: { readonly p: Project; readonly sid: number; readonly others: Take[]; readonly seenAt: number; readonly pending: number; readonly onOpen: (id: string) => void; readonly onUse: (id: string) => void }) {
  return (
    <div className="takes">
      {others.slice(0, 4).map((t) => (
        <figure key={t.id} className={`take ${t.mtime > seenAt ? "newitem" : ""}`} title={t.delta ? `vs current: ${t.delta}` : "no settings recorded for this take — click to view it"}>
          <button type="button" className="mediabtn" onClick={() => onOpen(t.id)} aria-label={`open take ${t.id} of stage ${sid}`}><img loading="lazy" alt="" src={takeUrl(p.name, sid, t.id)} /></button>
          <figcaption>{t.delta || (t.source === "run" ? "take" : t.source)}</figcaption>
          <button className="btn sm use" title="make this the current preview and ✓ it" onClick={(e) => { e.stopPropagation(); onUse(t.id); }}>use</button>
        </figure>
      ))}
      {Array.from({ length: Math.min(pending, 4) }, (_, i) => <figure key={`p${i}`} className="take pending"><div className="ph">rendering…</div></figure>)}
      {others.length > 4 && <button className="btn sm more" title="see every take" onClick={() => onOpen(others[4].id)}>+{others.length - 4}</button>}
    </div>
  );
}

/** Phase "all": this stage's preview, shown on demand. */
function PeekFigure({ p, sid }: { readonly p: Project; readonly sid: number }) {
  const [peek, setPeek] = useState(false);
  const v = p.folders["preview"]?.stills[sid];
  return (
    <figure className="peek" title={v ? "click to show / hide this stage's preview" : "no preview for this stage"}>
      <button type="button" className="mediabtn" onClick={(e) => { e.stopPropagation(); setPeek(!peek); }} aria-label={`show or hide the preview of stage ${sid}`} disabled={!v}>
        {peek && v
          ? <img loading="lazy" alt="" src={fileUrl(p.name, "preview", "still", sid, v)} />
          : <span className={`peekbox ${v ? "" : "none"}`}>{v ? "👁" : "—"}</span>}
      </button>
      <figcaption>preview</figcaption>
    </figure>
  );
}

/** The card's prompt summary; a click opens the editor (keyboard: ⋯ › Edit). */
function PromptPeek({ s, onEdit }: { readonly s: Stage; readonly onEdit: () => void }) {
  const lib = useLoras().data;
  const loras = Object.entries(s.loras ?? {}).filter(([, v]) => v).map(([k, v]) => `${k}:${v}`);
  const vloras = Object.entries(s.video_loras ?? {}).map(([k, v]) => `${k}:${v}`);
  const localOnly = Object.entries(s.loras ?? {}).filter(([k, v]) => v && lib?.image[k]?.cloud === false).map(([k]) => k);
  return (
    <button type="button" className="prompt" onClick={onEdit}>
      <span className="dim">{loras.join(" ")}{loras.length > 0 && vloras.length > 0 && " · "}{vloras.length > 0 && <span title="i2v LoRAs">▶ {vloras.join(" ")}</span>}
        {localOnly.length > 0 && <> <span className="pill local" title={`${localOnly.join(", ")}: local only, cloud runs of this stage are refused`}>local LoRA</span></>}</span>
      <span className="txt">{s.krea_scene || <i className="dim">empty krea_scene</i>}</span>
      <span className="txt motion">{s.motion || <i className="dim">no motion</i>}</span>
    </button>
  );
}

/** The card's ⋯ menu. */
function MoreMenu({ p, s, nOthers, onEdit, onVersions, onClearTakes }: { readonly p: Project; readonly s: Stage; readonly nOthers: number; readonly onEdit: () => void; readonly onVersions: () => void; readonly onClearTakes: () => void }) {
  const qc = useQueryClient();
  const [more, setMore] = useState(false);
  useOutside(more, () => setMore(false), ".morewrap");
  const del = useMutation({ mutationFn: () => api(`/projects/${p.name}/stages/${s.id}`, "DELETE"), onSuccess: () => qc.invalidateQueries({ queryKey: ["project", p.name] }) });
  const dup = useMutation({ mutationFn: () => api(`/projects/${p.name}/stages/${s.id}/duplicate`, "POST"), onSuccess: () => qc.invalidateQueries({ queryKey: ["project", p.name] }) });
  const pick = (f: () => void) => () => { setMore(false); f(); };
  const remove = () => { if (confirm(`Remove stage ${s.id} from project.json? (files stay)`)) del.mutate(); };
  return (
    <span className="morewrap menuwrap">
      <button className="btn sm" title="more" onClick={() => setMore(!more)}>⋯</button>
      {more && (
        <div className="menu morepanel">
          <button className="menuitem" onClick={pick(onEdit)}>✎ Edit</button>
          <button className="menuitem" title="a copy right after this one under a new id: prompts, settings and stills (not the clips), to try another motion" onClick={pick(() => dup.mutate())}>Duplicate stage</button>
          <button className="menuitem" title="earlier results of this stage (re-runs, re-rolls, face passes, removes) — restore one" onClick={pick(onVersions)}>Previous versions…</button>
          {nOthers > 0 && <button className="menuitem" title="move every preview take but the current one to _debug/removed/" onClick={pick(onClearTakes)}>Clear other takes ({nOthers})</button>}
          <button className="menuitem danger" onClick={pick(remove)}>Remove stage…</button>
        </div>
      )}
    </span>
  );
}

function stageBusyClass(busy?: Active): string {
  if (!busy) return "";
  return `${busy.queued ? "busy queued" : "busy"} lane-${busy.lane}`;
}

function doneTip(done: boolean, review: boolean, doneKey: string): string {
  const where = doneKey === "" ? "root" : doneKey;
  if (done) return `${review ? "approved" : "completed"} in ${where} — click to undo`;
  return `${review ? "approve clip" : "mark completed"} in ${where}`;
}

function BusyPill({ busy }: { readonly busy: Active }) {
  const label = ACTION_LABEL[busy.action] ?? busy.action;
  return <span className="pill run" title={`${label}${busy.queued ? " (queued)" : " (running)"}`}>{busy.queued ? "⏳ " : ""}{label}</span>;
}

/** A card's duration + the other folders it is ✓'d in. */
function StageBadges({ s, doneKey }: { readonly s: Stage; readonly doneKey: string }) {
  return <>
    <span className="dim">{s.duration ? `${s.duration}s` : ""}</span>
    {Array.isArray(s.done) && s.done.filter((f) => f !== doneKey).map((f) => <span key={f} className="pill" title={`also completed in ${f}`}>✓ {f || "root"}</span>)}
  </>;
}

function StageRow({ p, s, seen, onSeen, folder, clipFrom, phase, doneKey, cur, onCur, hidden, hideKind, onDone, sel, selecting, busy, onToggle, onEdit, onOpen, onGo, disabledWhy, takes, pendingTakes, onOpenTake, onUseTake, onClearTakes, dragging, onDrag, onDropOn }: {
  readonly p: Project; readonly s: Stage; readonly seen: Seen; readonly onSeen: (...marks: [string, number][]) => void; readonly folder: string; readonly clipFrom: string; readonly phase: Phase; readonly doneKey: string; readonly cur: boolean; readonly onCur: () => void; readonly onDone: () => void; readonly hidden: Set<string>; readonly hideKind: Set<string>; readonly sel: boolean; readonly selecting: boolean; readonly busy?: Active; readonly onToggle: (range: boolean) => void; readonly onEdit: () => void;
  readonly onOpen: (id: number, folder: string, kind: "still" | "clip") => void; readonly onGo: (a: string, folder?: string) => void;
  readonly disabledWhy: (a: string) => string;
  readonly takes: Take[]; readonly pendingTakes: number; readonly onOpenTake: (id: string) => void; readonly onUseTake: (id: string) => void; readonly onClearTakes: () => void;
  readonly dragging: number | null; readonly onDrag: (id: number | null) => void; readonly onDropOn: (id: number) => void;
}) {
  const [versions, setVersions] = useState(false);
  const [over, setOver] = useState(false);
  const slots = Object.entries(p.folders).flatMap(([f, st]) => {
    const row: Slot[] = [];
    if (st.stills[s.id] && wantSlot(phase, folder, f, "still", hidden, hideKind, clipFrom)) row.push({ f, kind: "still", v: st.stills[s.id] });
    if (st.clips[s.id] && wantSlot(phase, folder, f, "clip", hidden, hideKind, clipFrom)) row.push({ f, kind: "clip", v: st.clips[s.id] });
    return row;
  });
  sortSlots(slots, phase);
  const review = isReview(phase);
  const done = doneIn(s, doneKey);
  // something landed for this stage (a new still/clip) since you last clicked it
  const isNew = (x: Slot) => x.v > (seen.slots[slotKey(s.id, x.f, x.kind)] ?? seen.baseline) && isOwnOutput(phase, x.f, x.kind);
  const showTakes = phase === "preview" && (takes.length > 1 || pendingTakes > 0);
  const others = takes.filter((t) => !t.current);
  const takeSeen = seen.slots[slotKey(s.id, "takes", "still")] ?? seen.baseline;
  const fresh = slots.some(isNew) || (phase === "preview" && others.some((t) => t.mtime > takeSeen));
  // selection mode (something is selected): a click anywhere on the card toggles it instead of acting; the checkbox handles its own
  const selectClick = (e: MouseEvent) => {
    if ((e.target as Element).closest(".chk")) return;
    e.preventDefault();
    e.stopPropagation();
    onToggle(e.shiftKey);
  };
  return (
    <div id={`stage-${s.id}`} onPointerDownCapture={() => onCur()}
      onClickCapture={selecting ? selectClick : undefined}
      onDragOver={dragging !== null && dragging !== s.id ? (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; if (!over) setOver(true); } : undefined}
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOver(false); }}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (dragging !== null && dragging !== s.id) onDropOn(s.id); }}
      className={`stage ${selecting ? "selecting" : ""} ${cur && review ? "cur" : ""} ${fresh ? "fresh" : ""} ${sel ? "sel" : ""} ${stageBusyClass(busy)} ${done ? "isdone" : ""} ${dragging === s.id ? "dragging" : ""} ${over && dragging !== null ? "dropover" : ""}`}>
      <div className="head">
        <span className="grab" draggable title="drag to move this stage (order = edit order)" aria-label={`move stage ${s.id}`}
          onDragStart={(e) => { e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/x-stage", String(s.id)); e.dataTransfer.setDragImage(e.currentTarget.closest(".stage")!, 20, 20); onDrag(s.id); }}
          onDragEnd={() => onDrag(null)}>⠿</span>
        <label className="chk" title="select (shift-click for a range)">
          <input type="checkbox" aria-label={`select stage ${s.id}`} checked={sel} readOnly onClick={(e) => onToggle(e.shiftKey)} />
        </label>
        <RefChip project={p.name} stage={s.id} />
        {busy && <BusyPill busy={busy} />}
        <StageBadges s={s} doneKey={doneKey} />
        <span className="spacer" />
        <button className={`btn sm ${done ? "donebtn" : ""}`} title={doneTip(done, review, doneKey)} onClick={onDone}>✓</button>

        <RunMenu project={p.name} ids={[s.id]} label="Run" workFolder={folder} allow={isUpscaleStep(phase) ? allowFor(folder) : undefined} why={disabledWhy} onGo={onGo} sources={animSources(p, [s.id])} clipFrom={clipFrom} />
        <MoreMenu p={p} s={s} nOthers={others.length} onEdit={onEdit} onVersions={() => setVersions(true)} onClearTakes={onClearTakes} />
      </div>
      <div className="slots">
        {slots.length === 0 && <div className="empty dim">{busy && !busy.queued ? "rendering…" : "—"}</div>}
        {slots.map((x) => (
          <SlotFigure key={x.f + x.kind} p={p} sid={s.id} slot={x} phase={phase} doneKey={doneKey} isNew={isNew(x)} onOpen={() => { onSeen([slotKey(s.id, x.f, x.kind), x.v]); onOpen(s.id, x.f, x.kind); }} />
        ))}
        {showTakes && <TakeStrip p={p} sid={s.id} others={others} seenAt={takeSeen} pending={pendingTakes} onOpen={onOpenTake} onUse={onUseTake} />}
        {phase === "all" && <PeekFigure p={p} sid={s.id} />}
      </div>
      {versions && <Versions p={p} sid={s.id} first={review ? clipFrom : folder} onClose={() => setVersions(false)} />}
      <PromptPeek s={s} onEdit={onEdit} />
    </div>
  );
}


/** A stage has reached the phase being viewed once the step before it is ✓, or it already holds output there (Upscale lists
 * the preview-✓'d, Finish the upscale-✓'d, Animate the still-✓'d of its folder). */
function readyFor(s: Stage, phase: Phase, folder: string, animFolder: string, active: Record<number, { action?: string }>,
                  has: (s: Stage, f: string, k: "stills" | "clips") => boolean) {
  if (phase === "upscale") return doneIn(s, "preview") || has(s, folder, "stills");
  if (phase === "finish") return UP_TARGETS.some((f) => doneIn(s, f)) || has(s, "finish", "stills");
  if (phase === "animate") return doneIn(s, folder) || has(s, folder, "clips");
  if (phase === "interp") return doneIn(s, clipKey(animFolder)) || has(s, INTERP, "clips") || active[s.id]?.action === "animate_rife";
  return true;
}

/** The phase tabs' done-count: how many stages carry the phase's ✓. */
function phaseStatCount(allStages: Stage[], ph: Phase) {
  if (ph === "preview") return allStages.filter((s) => doneIn(s, "preview")).length;
  if (ph === "upscale") return allStages.filter((s) => UP_TARGETS.some((f) => doneIn(s, f))).length;
  if (ph === "finish") return allStages.filter((s) => doneIn(s, "finish")).length;
  if (ph === "animate") return allStages.filter(anyClipDone).length;
  if (ph === "interp") return allStages.filter((s) => doneIn(s, clipKey(INTERP))).length;
  return 0;
}
