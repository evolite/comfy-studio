import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, clipKey, doneIn, fileUrl, Job, Project } from "./api";
import { hoverPlay } from "./Board";
import Lightbox from "./Lightbox";
import { useJobs, useProject, useRun } from "./hooks";
import { ScenarioControls, StartSlider, useSaveMusic, useDefaultBeats, type Fit, type Music } from "./Music";

type Audio = { tracks: string[]; beat_files: string[] };
type Item = { key: number; id: number };
type Mode = "animate" | "stills" | "mix";
type ClipKind = "clip" | "still";
type Progress = { done: number; total: number };
type EditFile = { name: string; mtime: number; size: number; folder?: string | null };
let keySeq = 0;

const store = (key: string, v: unknown) => { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage unavailable */ } };
function loadJson<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) ?? "null") ?? fallback; } catch { return fallback; }
}
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
const baseName = (path: string) => path.split("/").pop()!;
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const isActive = (j: Job) => ["running", "queued"].includes(j.status);

type SeenEdits = { baseline: number; files: Record<string, number> };
/** The mtime of each edit you last opened, per project. The first visit sets a baseline so edits already on disk count as seen. */
function loadSeenEdits(project: string): SeenEdits {
  const key = `seenedits:${project}`;
  const v = loadJson<SeenEdits | null>(key, null);
  if (v) return { baseline: v.baseline, files: v.files ?? {} };
  const fresh = { baseline: Math.floor(Date.now() / 1000), files: {} };
  store(key, fresh);
  return fresh;
}

/** The folder picked when the current one can't be cut. Mix: the one with the most approved clips (interp/ over a stray approved clip elsewhere). */
function bestFolder(mode: Mode, eligible: string[], progress: (f: string) => Progress): string {
  // the folder with the most approved; among equals finish, then hires
  const bonus: Record<string, number> = { finish: 2, hires: 1 };
  const rank = (f: string) => progress(f).done * 4 + (bonus[f] ?? 0);
  return [...eligible].sort((a, b) => rank(b) - rank(a))[0];
}

/** Where a mix's still stages come from: the picked folder, else the clip folder, else the best still folder. */
function stillsFolder(p: Project | undefined, stillsFrom: string | null, folder: string, stillFolders: string[]): string {
  if (stillsFrom !== null && stillFolders.includes(stillsFrom)) return stillsFrom;
  const hasStills = (f: string) => Object.keys(p?.folders[f]?.stills ?? {}).length > 0;
  if (hasStills(folder)) return folder;
  return ["finish", "hires", "IL"].find(hasStills) ?? stillFolders[0] ?? folder;
}

/** Stages in the timeline not ✓'d for the key they are cut under. */
function unapprovedStages(p: Project, items: Item[], keyOf: (id: number) => string): number[] {
  return [...new Set(items.map((i) => i.id))].filter((id) => {
    const st = p.config.stages.find((s) => s.id === id);
    return !st || !doneIn(st, keyOf(id));
  });
}

/** Confirm and delete a project's old versions; returns the message to show ("" = you declined). */
async function cleanupProject(name: string, onCleaned: () => void): Promise<string> {
  try {
    const c = await api<{ files: number; bytes: number }>(`/projects/${name}/cleanup`);
    if (!c.files) return "nothing to clean up: no previous versions, extra takes or build folders";
    if (!confirm(`Permanently delete ${c.files} files (${(c.bytes / 1e6).toFixed(0)} MB)?\n\nEvery previous version, every preview take except the current one, and the edits' build folders. `
      + "The current stills and clips, approvals (✓) and finished edits stay. This can't be undone.")) return "";
    const r = await api<{ files: number; bytes: number }>(`/projects/${name}/cleanup`, "POST");
    onCleaned();
    return `cleaned up: ${r.files} files, ${(r.bytes / 1e6).toFixed(0)} MB freed`;
  } catch (e) { return `error: ${(e as Error).message}`; }
}

/** The project's active edit job of this kind (beat-synced / no-music), and of the other kind. */
function editJobsOf(jobs: Job[], project: string, noMusic: boolean): [Job | undefined, Job | undefined] {
  const active = jobs.filter((j) => j.project === project && j.action === "edit" && isActive(j));
  return [active.find((j) => !!j.req.no_music === noMusic), active.find((j) => !!j.req.no_music !== noMusic)];
}

/** What a stage is cut as: fixed by the mode, or in a mix what you picked, else its clip when it has one. */
function kindOf(id: number, mode: Mode, kinds: Record<number, ClipKind>, hasClip: boolean, hasStill: boolean): ClipKind {
  if (mode !== "mix") return mode === "animate" ? "clip" : "still";
  const wanted = kinds[id];
  if (wanted === "clip" && hasClip) return "clip";
  if (wanted === "still" && hasStill) return "still";
  return hasClip ? "clip" : "still";
}

/** The stage ids a mode can cut from these folders (removed stages keep their files but are not part of the project any more). */
function availableStages(mode: Mode, fs: Project["folders"][string] | undefined, sfs: Project["folders"][string] | undefined, stageIds: string, approvedIds: string): number[] {
  let ids: string[];
  if (mode === "mix") ids = [...Object.keys(fs?.clips ?? {}), ...Object.keys(sfs?.stills ?? {})];
  else ids = Object.keys((mode === "animate" ? fs?.clips : fs?.stills) ?? {});
  const ok = new Set(approvedIds.split(",").map(Number));  // clips / stills mode: only the approved ones (mix checks each chip itself)
  const have = new Set(ids.map(Number).filter((id) => mode === "mix" || ok.has(id)));
  return stageIds.split(",").map(Number).filter((id) => have.has(id));  // in stage order (the Board's ⠿ order)
}

/** The strip in stage order: each stage's first chip takes the next stage in project order; repeats keep their slots. */
function followStageOrder(seq: number[], order: number[]): number[] {
  const firsts = new Set(seq);
  const sorted = order.filter((id) => firsts.has(id));
  const seen = new Set<number>();
  let k = 0;
  return seq.map((id) => {
    if (seen.has(id)) return id;
    seen.add(id);
    return sorted[k++];
  });
}

/** The project's full stage order with the timeline's stages put in the timeline's order (the others keep their places). */
function mergeOrder(all: number[], seq: number[]): number[] {
  const firsts = [...new Set(seq)], inSeq = new Set(firsts);
  let k = 0;
  return all.map((id) => (inSeq.has(id) ? firsts[k++] : id));
}

type EditReqArgs = {
  folder: string; mode: Mode; items: Item[]; isClip: (id: number) => boolean; sfolder: string; zoom: boolean; noMusic: boolean; clipAudio: boolean;
  trims: Record<number, string>; trimsEnd: Record<number, string>; muted: number[]; music: Music; opts: Record<string, string>;
  pol: Polaroid; cfolder: string;
};

/** Polaroid look (per project, in browser storage): stills as cards, the collage outro's stages (picked: null = auto, the
 * approved stills that have no clip), card width, beats per landing. `collage` is the effective list, filled in by the page. */
type Polaroid = { on: boolean; picked: number[] | null; size: string; beats: string; sounds: boolean; svol: string; collage: number[] };
const NO_POLAROID: Polaroid = { on: false, picked: null, size: "", beats: "", sounds: false, svol: "", collage: [] };

/** The collage's default: every still in cfolder that is approved there (✓ on the Board) and has no clip in any folder. */
function autoCollage(p: Project | undefined, cfolder: string, stills: number[]): number[] {
  if (!p) return [];
  const animated = (id: number) => Object.values(p.folders).some((f) => !!f.clips?.[id]);
  return stills.filter((id) => {
    const st = p.config.stages.find((s) => s.id === id);
    return !!st && doneIn(st, cfolder) && !animated(id);
  });
}

/** The polaroid fields of an edit request. */
function polaroidFields(a: EditReqArgs): Record<string, unknown> {
  const { pol } = a;
  const req: Record<string, unknown> = {};
  if (pol.on) req.polaroid = true;
  if (pol.collage.length) { req.collage = pol.collage; req.collage_from = a.cfolder; if (pol.beats && !a.noMusic) req.collage_beats = Number(pol.beats); }
  if ((pol.on || pol.collage.length) && pol.size) req.card_size = Number(pol.size);
  if ((pol.on || pol.collage.length) && pol.sounds) { req.card_sounds = true; if (pol.svol) req.card_sound_volume = Number(pol.svol); }
  return req;
}

const STRING_OPTS = new Set(["audio", "beat_times_file", "out_name"]);

/** The no-music fields of an edit request: scenario settings + clip audio mutes. */
function scenarioFields(a: EditReqArgs, mutedHere: number[]) {
  const req: Record<string, unknown> = { no_music: true, mute_clip_audio: mutedHere };
  for (const k of ["transition", "transition_seconds", "still_seconds", "fade_in", "fade_out"] as const) if (a.music[k] != null) req[k] = a.music[k];
  if (a.opts.out_name) req.out_name = a.opts.out_name;
  return req;
}

/** The beat-synced fields of an edit request: music + cut options, numbers as numbers. */
function beatFields(a: EditReqArgs, mutedHere: number[]) {
  const req: Record<string, unknown> = {};
  if (a.mode !== "stills" && a.clipAudio) { req.clip_audio = true; req.mute_clip_audio = mutedHere; }
  for (const [k, v] of Object.entries({ audio: a.music.audio, beat_times_file: a.music.beat_times_file, ...a.opts })) {
    if (v !== "" && v != null) req[k] = STRING_OPTS.has(k) ? v : Number(v);
  }
  return req;
}

/** The POST body of an edit run. */
function editRequest(a: EditReqArgs): Record<string, unknown> {
  const { items, isClip, mode, noMusic } = a;
  const req: Record<string, unknown> = { action: "edit", folder: a.folder, mode: mode === "mix" ? "animate" : mode, stages: items.map((i) => i.id), zoom: (!noMusic && a.zoom) || undefined };
  if (mode === "mix") { req.still_stages = [...new Set(items.filter((i) => !isClip(i.id)).map((i) => i.id))]; req.stills_from = a.sfolder; }
  const forClips = (t: Record<number, string>) => Object.fromEntries(Object.entries(t).filter(([id, v]) => v !== "" && items.some((i) => i.id === Number(id) && isClip(i.id))));
  if (mode !== "stills") { req.trims = forClips(a.trims); req.trims_end = forClips(a.trimsEnd); }
  const mutedHere = a.muted.filter((id) => items.some((i) => i.id === id));
  return { ...req, ...(noMusic ? scenarioFields(a, mutedHere) : beatFields(a, mutedHere)), ...polaroidFields(a) };
}

const NO_ELIGIBLE: Record<Mode, string> = {
  mix: "No folder has an approved clip yet. Review the clips in the Board's Animate phase and ✓ them to cut them here.",
  animate: "No folder has an approved clip yet. Review the clips in the Board's Animate phase and ✓ them to cut them here.",
  stills: "No folder has an approved still yet. ✓ stills on the Board to cut them here.",
};
const EMPTY_KIND: Record<Mode, string> = { animate: "clips", mix: "clips or stills", stills: "stills" };

function folderOption(f: string, g: Progress, mode: Mode, ok: boolean): string {
  if (mode === "mix") return `${clipKey(f)} — ${g.done} approved`;
  const name = f || "root";
  return !ok || g.done < g.total ? `${name} — ${g.done}/${g.total} approved` : name;
}

function SourceCard({ p, mode, folder, sfolder, eligible, stillFolders, progress, onMode, onFolder, onStillsFrom }: {
  readonly p: Project; readonly mode: Mode; readonly folder: string; readonly sfolder: string; readonly eligible: string[]; readonly stillFolders: string[]; readonly progress: (f: string) => Progress;
  readonly onMode: (m: Mode) => void; readonly onFolder: (f: string) => void; readonly onStillsFrom: (f: string) => void;
}) {
  return (
    <section className="card cmp-card">
      <div className="cmp-head"><h3>Source</h3><span className="dim">what the edit is cut from</span></div>
      <div className="row wrap">
        <div className="field"><span>cut from</span>
          <div className="seg">
            {([["animate", "Clips"], ["stills", "Stills"], ["mix", "Clips + stills"]] as const).map(([m, l]) => <button key={m} className={mode === m ? "on" : ""} onClick={() => onMode(m)}>{l}</button>)}
          </div>
        </div>
        <label className="field"><span>{mode === "mix" ? "clips from" : "folder"}</span>
          <select value={folder} onChange={(e) => onFolder(e.target.value)}>
            {Object.keys(p.folders).map((f) => <option key={f} value={f} disabled={!eligible.includes(f)}>{folderOption(f, progress(f), mode, eligible.includes(f))}</option>)}
          </select>
        </label>
        {mode === "mix" && <label className="field" title="the folder the stages shown as stills come from"><span>stills from</span>
          <select value={sfolder} onChange={(e) => onStillsFrom(e.target.value)}>
            {stillFolders.map((f) => <option key={f} value={f}>{f || "root"}</option>)}
          </select>
        </label>}
      </div>
      {eligible.length === 0 && <div className="msg">{NO_ELIGIBLE[mode]}</div>}
    </section>
  );
}

/** One shot in the timeline: thumb, clip/still switch, audio mute, trims. */
function Chip({ it, i, dragging, clip, thumb, canSwitch, audible, muted, trim, trimEnd, onDragStart, onDrop, onDrop_, onSwitch, onMute, onTrim, onTrimEnd }: {
  readonly it: Item; readonly i: number; readonly dragging: boolean; readonly clip: boolean; readonly thumb: string; readonly canSwitch: boolean; readonly audible: boolean; readonly muted: boolean; readonly trim: string; readonly trimEnd: string;
  readonly onDragStart: () => void; readonly onDrop: () => void; readonly onDrop_: () => void; readonly onSwitch: () => void; readonly onMute: () => void; readonly onTrim: (v: string) => void; readonly onTrimEnd: (v: string) => void;
}) {
  const icon = clip ? "🎞" : "🖼";
  const stop = (f: () => void) => (e: { stopPropagation: () => void }) => { e.stopPropagation(); f(); };
  return (
    <div className={`chip${dragging ? " dragging" : ""}`} draggable onDragStart={onDragStart} onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <div className="chip-media">
        {clip ? <video muted preload="metadata" src={thumb} /> : <img alt={`stage ${it.id}`} src={thumb} />}
        <span className="chip-n">{i + 1}</span>
        <button className="chip-x" title="drop from the edit" onClick={onDrop_}>✕</button>
      </div>
      <div className="chip-bar">
        <b>#{it.id}</b>
        {canSwitch
          ? <button className="chip-t" title={clip ? "cut as its clip — click to show its still" : "shown as its still — click to cut its clip"} onClick={stop(onSwitch)}>{icon}</button>
          : <span className="chip-kind">{icon}</span>}
        {audible && <button className={`chip-t${muted ? " off" : ""}`} title={muted ? "clip audio muted — click to include" : "clip audio on — click to mute"} onClick={stop(onMute)}>{muted ? "🔇" : "🔊"}</button>}
      </div>
      {clip && <div className="chip-trim" title="seconds cut from the start (blank = 0.2) and the end (blank = 0) of this clip">
        <label>in<input type="number" min={0} step={0.05} placeholder="0.2" value={trim} onChange={(e) => onTrim(e.target.value)} onClick={(e) => e.stopPropagation()} draggable={false} /></label>
        <label>out<input type="number" min={0} step={0.05} placeholder="0" value={trimEnd} onChange={(e) => onTrimEnd(e.target.value)} onClick={(e) => e.stopPropagation()} draggable={false} /></label>
      </div>}
    </div>
  );
}

/** One finished edit: hover plays it, a click opens the viewer. */
function EditTile({ project, e, isNew, onOpen, onDelete }: { readonly project: string; readonly e: EditFile; readonly isNew: boolean; readonly onOpen: () => void; readonly onDelete: () => void }) {
  const noMusic = /_nomusic_/.test(e.name);
  return (
    <figure className={isNew ? "newitem" : ""}>
      <button type="button" className="mediabtn" onClick={onOpen} aria-label={`play ${baseName(e.name)}`} {...hoverPlay}>
        <video muted preload="metadata" src={`/api/projects/${project}/files/${e.name}?v=${e.mtime}`} />
      </button>
      <figcaption><span className={`kindtag${noMusic ? " nm" : ""}`}>{noMusic ? "no music" : "beat"}</span> {baseName(e.name).replace(`${project}_`, "")} ▶{isNew && <b className="newtag"> · new</b>}<br />{(e.size / 1e6).toFixed(1)} MB · {new Date(e.mtime * 1000).toLocaleString()}</figcaption>
      <button className="btn sm reveal" title="open in folder" onClick={(ev) => { ev.stopPropagation(); void api(`/projects/${project}/reveal`, "POST", { path: e.name }); }}>📂</button>
      <button className="btn sm danger del" onClick={(ev) => { ev.stopPropagation(); onDelete(); }}>✕</button>
    </figure>
  );
}

/** Beat-synced sound + cut options. */
function BeatControls({ music, audio, opts, setOpts, setO, num, beatsFile, fit, mode, zoom, setZoom, clipAudio, setClipAudio }: {
  readonly music: Music; readonly audio?: Audio; readonly opts: Record<string, string>; readonly setOpts: (o: Record<string, string>) => void; readonly setO: (k: string) => (e: { target: { value: string } }) => void;
  readonly num: (k: string, label: string, ph: string) => React.ReactNode; readonly beatsFile: string; readonly fit?: Fit; readonly mode: Mode; readonly zoom: boolean; readonly setZoom: (v: boolean) => void; readonly clipAudio: boolean; readonly setClipAudio: (v: boolean) => void;
}) {
  const extra = (have: string[] | undefined, v?: string) => (v && !have?.includes(v) ? [v] : []);
  const resetStart = () => setOpts(Object.fromEntries(Object.entries(opts).filter(([k]) => k !== "start_time")));
  const withClips = mode !== "stills";
  return <>
    <div className="row wrap">
      <label className="field"><span>track</span>
        <select value={opts.audio ?? music.audio ?? ""} onChange={setO("audio")}><option value="">the beat file's track (else house)</option>{[...(audio?.tracks ?? []), ...extra(audio?.tracks, music.audio)].map((t) => <option key={t} value={t}>{baseName(t)}</option>)}</select>
      </label>
      <label className="field"><span>beat times</span>
        <select value={opts.beat_times_file ?? music.beat_times_file ?? ""} onChange={setO("beat_times_file")}><option value="">default</option>{[...(audio?.beat_files ?? []), ...extra(audio?.beat_files, music.beat_times_file)].map((t) => <option key={t}>{t}</option>)}</select>
      </label>
      <StartSlider label="start time (this edit)" file={opts.beat_times_file || beatsFile} audio={opts.audio || music.audio}
        value={opts.start_time ? Number(opts.start_time) : fit?.start_time} onCommit={(s) => setOpts({ ...opts, start_time: String(s) })} />
      {opts.start_time && <button className="btn sm" title="back to the track's start time from Settings" onClick={resetStart}>reset</button>}
    </div>
    <div className="row wrap">
      <label className="chk inline"><input type="checkbox" checked={zoom} onChange={(e) => setZoom(e.target.checked)} /> punch-in zoom</label>
      {zoom && num("zoom", "zoom ×", "1.5")}
      {withClips && <label className="chk inline" title="layer each clip's own generated sound (dialogue, breathing, ambience) over the music; plays once at the clip's start, music ducks under it"><input type="checkbox" checked={clipAudio} onChange={(e) => setClipAudio(e.target.checked)} /> clip audio over music</label>}
      {withClips && clipAudio && <>{num("clip_volume", "voice ×", "1.0")}{num("music_volume", "music ×", "1.0")}{num("duck", "duck (0–1)", "0")}</>}
    </div>
    <details className="advanced">
      <summary>Cut timing <span className="dim">blank = defaults: lead-in 2s, 2 beats per cut, 2 subcuts per still, last cut ×2, 0.5s fade-out</span></summary>
      <div className="row wrap">
        {num("lead_in_seconds", "lead-in (s)", "2.0")}
        {num("beats_per_cut", "beats / cut", "2")}
        {num("subcuts_per_image", "subcuts / still", "2")}
        {num("last_weight", "last cut ×", "2.0")}
        {num("fade_out", "fade-out (s)", "0.5")}
      </div>
    </details>
  </>;
}

/** Polaroid stills + the collage outro: which stages pile up at the end (stills from cfolder), card size, landing spacing. */
function PolaroidControls({ pol, setPol, stills, cfolder, noMusic }: {
  readonly pol: Polaroid; readonly setPol: (p: Polaroid) => void; readonly stills: number[]; readonly cfolder: string; readonly noMusic: boolean;
}) {
  const toggle = (id: number) => setPol({ ...pol, picked: pol.collage.includes(id) ? pol.collage.filter((x) => x !== id) : [...pol.collage, id] });
  const size = Number(pol.size) || 0;
  return (
    <details className="advanced" open={pol.on || pol.collage.length > 0}>
      <summary>Polaroid <span className="dim">{pol.on ? "stills as cards" : "off"}{pol.collage.length ? ` · collage outro of ${plural(pol.collage.length, "card")}` : ""}</span></summary>
      <div className="row wrap">
        <label className="chk inline" title="every still in the edit shows as an Instax card (film grade, paper frame, handwritten caption) dropped onto a blurred copy of itself, one cut per still">
          <input type="checkbox" checked={pol.on} onChange={(e) => setPol({ ...pol, on: e.target.checked })} /> stills as polaroids</label>
        <label className="field slider cardsize" title="card width in px of the 1080-wide frame; auto = by card count (one card 900, a 6-card pile 700)"><span>card size {size ? `${size}px` : "auto"}</span>
          <input type="range" min={400} max={1060} step={20} value={size || 760} onChange={(e) => setPol({ ...pol, size: e.target.value })} /></label>
        {size > 0 && <button className="btn sm" title="back to the size by card count" onClick={() => setPol({ ...pol, size: "" })}>auto</button>}
        <label className="chk inline" title="a camera sound (shutter, flash, eject...) on every card landing, varied: picked from the files in data/audio/sfx/polaroid/">
          <input type="checkbox" checked={pol.sounds} onChange={(e) => setPol({ ...pol, sounds: e.target.checked })} /> card sounds</label>
        {pol.sounds && <label className="field"><span>sound ×</span>
          <input type="number" min={0} step={0.1} placeholder="0.8" value={pol.svol} onChange={(e) => setPol({ ...pol, svol: e.target.value })} /></label>}
        {!noMusic && pol.collage.length > 0 && <label className="field"><span>beats / card</span>
          <input type="number" min={1} step={1} placeholder="2" value={pol.beats} onChange={(e) => setPol({ ...pol, beats: e.target.value })} /></label>}
      </div>
      <div className="pool" title={`the edit ends on these stills (from ${cfolder || "root"}) piling up as polaroids, in the order picked`}>
        <span className="dim">collage outro</span>
        {stills.map((id) => {
          const k = pol.collage.indexOf(id);
          return <button key={id} className={`btn sm${k >= 0 ? " on" : ""}`} onClick={() => toggle(id)}>{k >= 0 ? `${k + 1}· ` : ""}#{id}</button>;
        })}
        {stills.length === 0 && <span className="dim">no stills in {cfolder || "root"}</span>}
        {pol.collage.length > 0 && <button className="btn sm" onClick={() => setPol({ ...pol, picked: [] })}>clear</button>}
        {pol.picked !== null ? <button className="btn sm" title="back to the default: the approved stills that have no clip" onClick={() => setPol({ ...pol, picked: null })}>auto</button>
          : <span className="dim">auto: approved stills without a clip</span>}
      </div>
    </details>
  );
}

/** The Sound & style header: beat-synced or no music, saved to the project (same switch as Settings → Music). */
function SoundHead({ noMusic, onSwitch }: { readonly noMusic: boolean; readonly onSwitch: (none: boolean) => void }) {
  return (
    <div className="cmp-head"><h3>Sound &amp; style</h3>
      <div className="seg" title="saved to the project (same switch as Settings → Music)">
        <button className={noMusic ? "" : "on"} onClick={() => { if (noMusic) onSwitch(false); }}>♪ Beat-synced</button>
        <button className={noMusic ? "on" : ""} onClick={() => { if (!noMusic) onSwitch(true); }}>No music</button>
      </div>
      <span className="dim">{noMusic ? "every stage in full, flowing into the next; the clips' own audio is the soundtrack" : "cut to the beats of the project's track"}</span>
    </div>
  );
}

function OtherEditNote({ job, noMusic }: { readonly job: Job; readonly noMusic: boolean }) {
  return <div className="afnote">the {noMusic ? "beat-synced" : "no-music"} edit is {job.status} too — it renders separately and keeps its own file</div>;
}

function renderTiming(job: Job | undefined, lastSeconds?: number): string {
  if (job?.progress) return `~${clock(job.progress.left)} left (the last render took ${clock(lastSeconds ?? 0)})`;
  return lastSeconds ? `last render took ${clock(lastSeconds)}` : "no render timed yet";
}

/** The render button's label: the running job's status + elapsed time, else the shot count. */
function RenderLabel({ job, shots, elapsed }: { readonly job?: Job; readonly shots: number; readonly elapsed: number }) {
  if (!job) return <>Render {plural(shots, "shot")}</>;
  return <><span className="spin" />{job.status === "queued" ? "Queued" : "Rendering"} {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</>;
}

/** The finished edits of the current folder (optionally every folder) + project cleanup. */
function EditsCard({ folder, list, nOther, showOther, onShowOther, tile, projBusy, onCleanup }: {
  readonly folder: string; readonly list: EditFile[]; readonly nOther: number; readonly showOther: boolean; readonly onShowOther: (v: boolean) => void; readonly tile: (e: EditFile) => React.ReactNode; readonly projBusy: boolean; readonly onCleanup: () => void;
}) {
  return (
    <section className="card cmp-card">
      <div className="cmp-head"><h3>Edits</h3><span className="dim">from {folder || "root"}</span><span className="spacer" />
        {nOther > 0 && <label className="chk dim"><input type="checkbox" checked={showOther} onChange={(e) => onShowOther(e.target.checked)} /> also show {nOther} from other folders</label>}</div>
      {list.length === 0 && <div className="dim">no edits from {folder || "root"} yet</div>}
      {list.length > 0 && <>
        <div className="dim editlabel">Latest</div>
        <div className="editgrid">{tile(list[0])}</div>
        {list.length > 1 && (
          <details className="older">
            <summary className="dim editlabel">Older ({list.length - 1})</summary>
            <div className="editgrid">{list.slice(1).map(tile)}</div>
          </details>
        )}
      </>}
      <div className="row cleanup">
        <button className="btn sm danger" disabled={projBusy} title={projBusy ? "a job is running on this project" : "delete every previous version and every preview take but the current one; keeps the current results, approvals and edits"} onClick={onCleanup}>Clean up project…</button>
        <span className="dim">once you're happy with the edit: frees the disk space of every earlier version and extra take</span>
      </div>
    </section>
  );
}

/** Per-project UI state kept in browser storage: cut-from mode + folders, per-stage kinds, trims and mutes, the clip-audio switch + levels. */
function useComposePrefs(name: string) {
  const vkey = `composeView:${name}`;
  const savedView = useMemo(() => loadJson<{ mode?: Mode; folder?: string; stillsFrom?: string | null }>(vkey, {}), [vkey]);
  const [folder, setFolder] = useState(savedView.folder ?? "");
  const [mode, setMode] = useState<Mode>(savedView.mode ?? "animate");
  const [stillsFrom, setStillsFrom] = useState<string | null>(savedView.stillsFrom ?? null);  // mix: the folder the still stages come from (null = auto)
  useEffect(() => store(vkey, { mode, folder, stillsFrom }), [vkey, mode, folder, stillsFrom]);
  const kkey = `composeKind:${name}`;  // mix: clip or still per stage
  const [kinds, setKinds] = useState<Record<number, ClipKind>>(() => loadJson(kkey, {}));
  const setKind = (id: number, k: ClipKind) => setKinds((m) => { const n = { ...m, [id]: k }; store(kkey, n); return n; });
  const tkey = `composeTrim:${name}`;  // per-stage trim (s cut from the clip start)
  const [trims, setTrims] = useState<Record<number, string>>(() => loadJson(tkey, {}));
  const setTrim = (id: number, v: string) => setTrims((t) => { const n = { ...t, [id]: v }; store(tkey, n); return n; });
  const ekey = `composeTrimEnd:${name}`;  // per-stage end trim (s cut from the clip's end)
  const [trimsEnd, setTrimsEnd] = useState<Record<number, string>>(() => loadJson(ekey, {}));
  const setTrimEnd = (id: number, v: string) => setTrimsEnd((t) => { const n = { ...t, [id]: v }; store(ekey, n); return n; });
  const mkey = `composeMute:${name}`;  // stages whose clip audio is left out of the mix
  const [muted, setMuted] = useState<number[]>(() => loadJson(mkey, []));
  const toggleMute = (id: number) => setMuted((m) => { const n = m.includes(id) ? m.filter((x) => x !== id) : [...m, id]; store(mkey, n); return n; });
  const akey = `composeAudio:${name}`;  // clip-audio switch + its levels
  const savedAudio = useMemo(() => loadJson<{ on?: boolean; clip_volume?: string; music_volume?: string; duck?: string }>(akey, {}), [akey]);
  const [opts, setOpts] = useState<Record<string, string>>(() => Object.fromEntries((["clip_volume", "music_volume", "duck"] as const).filter((k) => savedAudio[k]).map((k) => [k, savedAudio[k]!])));
  const [clipAudio, setClipAudio] = useState(!!savedAudio.on);
  useEffect(() => store(akey, { on: clipAudio, clip_volume: opts.clip_volume, music_volume: opts.music_volume, duck: opts.duck }), [akey, clipAudio, opts.clip_volume, opts.music_volume, opts.duck]);
  return { folder, setFolder, mode, setMode, stillsFrom, setStillsFrom, kinds, setKind, trims, setTrim, trimsEnd, setTrimEnd, muted, toggleMute, opts, setOpts, clipAudio, setClipAudio };
}

/** The timeline: the stages in project order (the Board's ⠿ order), synced both ways — a drag here reorders the project's
 * stages (onOrder). Which stages are dropped and any repeats are remembered per project + folder + mode (browser storage);
 * stages added since are appended. */
function useStrip(skey: string, avail: number[], onOrder: (seq: number[]) => void) {
  const [strip, setStrip] = useState<number[] | null>(null);
  const [ver, setVer] = useState(0);
  const seq = useMemo(() => {
    if (strip) return strip.filter((id) => avail.includes(id));
    const v = loadJson<{ order: number[]; known: number[] } | null>(skey, null);
    if (!v) return avail;
    const live = new Set(avail), known = new Set(v.known);
    return [...v.order.filter((id) => live.has(id)), ...avail.filter((id) => !known.has(id))];
  }, [skey, avail, strip, ver]);  // eslint-disable-line react-hooks/exhaustive-deps
  const items = useMemo(() => followStageOrder(seq, avail).map((id) => ({ key: ++keySeq, id })), [seq, avail]);
  const commit = (a: Item[] | null, reordered = false) => {
    const ids = a?.map((i) => i.id) ?? null;
    setStrip(ids); setVer((v) => v + 1);
    if (ids) store(skey, { order: ids, known: avail });
    else { try { localStorage.removeItem(skey); } catch { /* storage unavailable */ } }
    if (ids && reordered) onOrder(ids);
  };
  return { items, commit, resetStrip: () => setStrip(null) };
}

export default function Compose({ name }: { readonly name: string }) {
  const qc = useQueryClient();
  const proj = useProject(name);
  const audio = useQuery({ queryKey: ["audio"], queryFn: () => api<Audio>("/audio") }).data;
  const run = useRun(name);
  const jobs = useJobs().data ?? [];
  const { folder, setFolder, mode, setMode, stillsFrom, setStillsFrom, kinds, setKind, trims, setTrim, trimsEnd, setTrimEnd, muted, toggleMute, opts, setOpts, clipAudio, setClipAudio } = useComposePrefs(name);
  const [seen, setSeen] = useState<SeenEdits>(() => loadSeenEdits(name));
  useEffect(() => { setSeen(loadSeenEdits(name)); }, [name]);
  const markSeen = (file: string, mtime: number) => setSeen((prev) => {
    const n = { ...prev, files: { ...prev.files, [file]: mtime } };
    store(`seenedits:${name}`, n);
    return n;
  });
  const [zoom, setZoom] = useState(false);
  const pkey = `composePolaroid:${name}`;
  const [polSaved, setPolSaved] = useState<Polaroid>(() => ({ ...NO_POLAROID, ...loadJson<Partial<Polaroid>>(pkey, {}), collage: [] }));
  useEffect(() => store(pkey, { on: polSaved.on, picked: polSaved.picked, size: polSaved.size, beats: polSaved.beats, sounds: polSaved.sounds, svol: polSaved.svol }), [pkey, polSaved]);
  const [play, setPlay] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const [drag, setDrag] = useState<number | null>(null);
  const p = proj.data;
  const music = (p?.config.music ?? {}) as Music;
  const noMusic = !!music.none;  // scenario edit: no beats, every stage in full with transitions (Settings → Music)
  const saveMusic = useSaveMusic(name);
  const defaultBeats = useDefaultBeats();
  const beatsFile = opts.beat_times_file || music.beat_times_file || defaultBeats;
  const fit = useQuery({ queryKey: ["durations", beatsFile], queryFn: () => api<Fit>(`/durations?beat_times_file=${encodeURIComponent(beatsFile)}`) }).data;
  const fnames = Object.keys(p?.folders ?? {});
  // a folder can be cut once something in it is approved: its stills (✓ in the Board's Upscale phase) or, when cutting clips, its clips
  // (✓ in Animate); the timeline then holds only the approved ones
  const progress = (f: string) => { const st = p?.config.stages ?? []; const key = mode === "stills" ? f : clipKey(f); return { done: st.filter((s) => doneIn(s, key)).length, total: st.length }; };
  // mix: any folder with an approved clip; each stage is then checked on its own (its clip here, or its still in the stills folder)
  const eligible = fnames.filter((f) => progress(f).done > 0);
  if (eligible.length && !eligible.includes(folder)) setFolder(bestFolder(mode, eligible, progress));
  const fs = p?.folders[folder];
  const stillFolders = fnames.filter((f) => Object.keys(p?.folders[f]?.stills ?? {}).length > 0);
  const sfolder = mode === "mix" ? stillsFolder(p, stillsFrom, folder, stillFolders) : folder;
  const sfs = p?.folders[sfolder];
  const cfolder = mode === "stills" ? folder : stillsFolder(p, stillsFrom, folder, stillFolders);  // where the collage's stills come from
  const collageStills = Object.keys(p?.folders[cfolder]?.stills ?? {}).map(Number).sort((a, b) => a - b);
  const pol: Polaroid = { ...polSaved, collage: polSaved.picked ?? autoCollage(p, cfolder, collageStills) };
  const stageIds = (p?.config.stages ?? []).map((s) => s.id).join(",");
  const approvedIds = (p?.config.stages ?? []).filter((s) => doneIn(s, mode === "stills" ? folder : clipKey(folder))).map((s) => s.id).join(",");
  const avail = useMemo(() => availableStages(mode, fs, sfs, stageIds, approvedIds), [JSON.stringify(fs), JSON.stringify(sfs), mode, stageIds, approvedIds]);  // eslint-disable-line react-hooks/exhaustive-deps
  const { items, commit, resetStrip } = useStrip(`compose:${name}:${folder}:${mode}`, avail, (seq) => {
    const all = stageIds.split(",").map(Number), next = mergeOrder(all, seq);
    if (next.join(",") !== stageIds) void api(`/projects/${name}/order`, "PUT", { order: next }).then(() => proj.refetch()).catch((e: Error) => setMsg(`error: ${e.message}`));
  });
  const [showOther, setShowOther] = useState(false);
  const shown = (p?.edits ?? []).filter((e) => e.folder === folder);
  const other = (p?.edits ?? []).filter((e) => e.folder !== folder);
  const list: EditFile[] = showOther ? [...shown, ...other] : shown;
  const deleteEdit = (file: string) => {
    if (confirm(`Delete ${baseName(file)}?`)) void api(`/projects/${name}/files/${file}`, "DELETE").then(() => proj.refetch());
  };
  const tile = (e: EditFile) => (
    <EditTile key={e.name} project={name} e={e} isNew={e.mtime > (seen.files[e.name] ?? seen.baseline)} onOpen={() => { markSeen(e.name, e.mtime); setPlay(e.name); }} onDelete={() => deleteEdit(e.name)} />
  );
  const projBusy = jobs.some((j) => j.project === name && isActive(j));
  const cleanup = () => cleanupProject(name, () => void qc.invalidateQueries({ queryKey: ["project", name] })).then((m) => { if (m) setMsg(m); });
  // a beat-synced and a no-music edit are separate jobs (and files): each can render while the other does
  const kind = noMusic ? "nomusic" : "beat";
  const [editJob, otherJob] = editJobsOf(jobs, name, noMusic);
  const editing = !!editJob;
  const last = useQuery({ queryKey: ["edit-timing", kind, editing], queryFn: () => api<{ seconds?: number }>(`/edit-timing?kind=${kind}`) }).data;  // refetched when a render starts/ends
  const [tick, setTick] = useState(Date.now());
  useEffect(() => {
    if (!editing) return;
    const t = setInterval(() => setTick(Date.now()), 1000);
    return () => clearInterval(t);
  }, [editing]);
  const elapsed = editJob ? Math.max(0, Math.floor(tick / 1000 - editJob.created)) : 0;
  if (!p) return <div className="page">loading…</div>;
  const isClip = (id: number) => kindOf(id, mode, kinds, !!fs?.clips?.[id], !!sfs?.stills?.[id]) === "clip";
  const thumb = (id: number) => isClip(id) ? fileUrl(name, folder, "clip", id, fs?.clips?.[id] ?? 0) : fileUrl(name, sfolder, "still", id, sfs?.stills?.[id] ?? 0);
  // mix: every chip needs its own approval (✓ on the Board) for the kind it is cut as
  const unapproved = mode === "mix" ? unapprovedStages(p, items, (id) => (isClip(id) ? clipKey(folder) : sfolder)) : [];
  const setO = (k: string) => (e: { target: { value: string } }) => setOpts({ ...opts, [k]: e.target.value });
  const num = (k: string, label: string, ph: string) => (
    <label className="field"><span>{label}</span><input type="number" step="any" placeholder={ph} value={opts[k] ?? ""} onChange={setO(k)} /></label>
  );
  const reorder = (to: number) => {
    if (drag === null) return;
    const a = [...items]; const [m] = a.splice(drag, 1); a.splice(to, 0, m); commit(a, true); setDrag(null);
  };
  const render = () => {
    run.mutate(editRequest({ folder, mode, items, isClip, sfolder, zoom, noMusic, clipAudio, trims, trimsEnd, muted, music, opts, pol, cfolder }), {
      onSuccess: () => { const t = setInterval(() => qc.invalidateQueries({ queryKey: ["project", name] }), 3000); setTimeout(() => clearInterval(t), 600000); },
      onError: (e) => setMsg(`error: ${e.message}`),
    });
  };
  const pickMode = (m: Mode) => { setMode(m); resetStrip(); if (m === "mix") setFolder(""); };
  const clipsIn = items.filter((i) => isClip(i.id)).length;
  const canRender = !editing && items.length > 0 && eligible.includes(folder) && unapproved.length === 0;
  const mixCount = mode === "mix" ? ` · ${plural(clipsIn, "clip")}, ${plural(items.length - clipsIn, "still")}` : "";
  const canSwitch = (id: number) => mode === "mix" && !!fs?.clips?.[id] && !!sfs?.stills?.[id];
  const unapprovedLabel = (id: number) => (isClip(id) ? `#${id} (clip)` : `#${id} (still in ${sfolder || "root"})`);
  const playIndex = play ? list.findIndex((x) => x.name === play) : -1;
  return (
    <div className="page compose">
      <SourceCard p={p} mode={mode} folder={folder} sfolder={sfolder} eligible={eligible} stillFolders={stillFolders} progress={progress} onMode={pickMode}
        onFolder={(f) => { setFolder(f); resetStrip(); }} onStillsFrom={setStillsFrom} />

      <section className="card cmp-card">
        <div className="cmp-head"><h3>Timeline</h3>
          <span className="dim">{plural(items.length, "shot")}{mixCount} · drag to reorder</span>
          <span className="spacer" />
          <button className="btn sm" title="back to every stage in stage order" onClick={() => commit(null)}>Reset order</button>
        </div>
        <div className="strip">
          {items.map((it, i) => (
            <Chip key={it.key} it={it} i={i} dragging={drag === i} clip={isClip(it.id)} thumb={thumb(it.id)} canSwitch={canSwitch(it.id)}
              audible={isClip(it.id) && (clipAudio || noMusic)} muted={muted.includes(it.id)} trim={trims[it.id] ?? ""} trimEnd={trimsEnd[it.id] ?? ""}
              onDragStart={() => setDrag(i)} onDrop={() => reorder(i)} onDrop_={() => commit(items.filter((x) => x.key !== it.key))}
              onSwitch={() => setKind(it.id, isClip(it.id) ? "still" : "clip")} onMute={() => toggleMute(it.id)} onTrim={(v) => setTrim(it.id, v)} onTrimEnd={(v) => setTrimEnd(it.id, v)} />
          ))}
          {items.length === 0 && <div className="dim cmp-empty">no {mode === "mix" ? "" : "approved "}{EMPTY_KIND[mode]} in this folder</div>}
        </div>
        <div className="pool"><span className="dim">add</span>
          {avail.map((id) => <button key={id} className="btn sm" title="append (repeats allowed)" onClick={() => commit([...items, { key: ++keySeq, id }])}>+ #{id}</button>)}
        </div>
      </section>

      <section className="card cmp-card">
        <SoundHead noMusic={noMusic} onSwitch={(none) => void saveMusic({ none: none || undefined })} />
        {noMusic ? <ScenarioControls name={name} /> : <BeatControls music={music} audio={audio} opts={opts} setOpts={setOpts} setO={setO} num={num} beatsFile={beatsFile} fit={fit} mode={mode} zoom={zoom} setZoom={setZoom} clipAudio={clipAudio} setClipAudio={setClipAudio} />}
        <PolaroidControls pol={pol} setPol={setPolSaved} stills={collageStills} cfolder={cfolder} noMusic={noMusic} />
      </section>

      <section className="card cmp-card cmp-render">
        <div className="row wrap">
          <label className="field cmp-name"><span>output name</span><input placeholder="auto (timestamped)" value={opts.out_name ?? ""} onChange={setO("out_name")} /></label>
          <button className={`btn primary cmp-go${editing ? " rendering" : ""}`} disabled={!canRender} onClick={render}><RenderLabel job={editJob} shots={items.length} elapsed={elapsed} /></button>
          <span className="dim">{renderTiming(editJob, last?.seconds)}</span>
        </div>
        {editJob?.progress && <div className="jbar" title="estimated from the last render"><i style={{ width: `${editJob.progress.fraction * 100}%` }} /></div>}
        {otherJob && <OtherEditNote job={otherJob} noMusic={noMusic} />}
        {unapproved.length > 0 && <div className="afnote warn">not approved yet: {unapproved.map(unapprovedLabel).join(", ")} — ✓ them on the Board or drop them</div>}
        {msg && <button type="button" className="msg" onClick={() => setMsg("")}>{msg}</button>}
      </section>

      <EditsCard folder={folder} list={list} nOther={other.length} showOther={showOther} onShowOther={setShowOther} tile={tile} projBusy={projBusy} onCleanup={() => void cleanup()} />
      {playIndex >= 0 && <Lightbox items={list.map((e) => ({ id: 0, label: baseName(e.name), kind: "clip" as const, src: `/api/projects/${name}/files/${e.name}?v=${e.mtime}` }))} index={playIndex} onIndex={(n) => setPlay(list[n].name)} onClose={() => setPlay(null)} />}
    </div>
  );
}
