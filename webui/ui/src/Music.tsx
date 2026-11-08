import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { useProject, useRegistry } from "./hooks";

export type Music = { beat_times_file?: string; audio?: string; none?: boolean; transition?: string;
  transition_seconds?: number; still_seconds?: number; fade_in?: number; fade_out?: number };
export type FitOption = { request: number; frames: number; seconds: number; cuts: number; coverage: number; recommended: boolean };
export type Fit = { beat_times_file: string; bpm: number; beat: number; cut: number; start_time: number | null; options: FitOption[] };
type Audio = { tracks: string[]; beat_files: string[] };

/** The house track's beat file (config.DEFAULT_BEATS, served by /api/registry). */
export const useDefaultBeats = () => useRegistry().data?.default_beats ?? "";
export const beatLabel = (f: string) => f.split("/").pop()!.replace(/^beat_times_/, "").replace(/\.json$/, "");

/** Clip lengths that fit this project's music (its chosen beat file, else the house track's) on an i2v engine's frame grid
 *  (default: the project's video_engine). */
export function useFit(project: string, engine?: string) {
  const cfg = useProject(project).data?.config;
  const music = cfg?.music as Music | undefined;
  const def = useDefaultBeats();
  const file = music?.beat_times_file ?? def;
  const eng = engine ?? cfg?.video_engine ?? "minimax";
  return { music, file, fit: useQuery({ queryKey: ["durations", file, eng], queryFn: () => api<Fit>(`/durations?beat_times_file=${encodeURIComponent(file)}&engine=${eng}`) }).data };
}

export const fitLabel = (o: FitOption) => `${o.seconds}s · ${o.cuts} cuts · ${Math.round(o.coverage * 100)}%${o.recommended ? " ★" : ""}`;
const fitNote = (o: FitOption) => {
  if (o.recommended) return "★ fits";
  if (o.coverage < 0.9) return "tail unused";
  if (o.coverage > 1.07) return "freezes";
  return "";
};

/** Plays the server-cut preview clip (a few seconds from the beat file's start time) of the track an edit would use.
 *  Web Audio rather than an <audio> element: decoded in memory, nothing to abort mid-load. */
function PreviewButton({ file, audio, start }: { readonly file: string; readonly audio?: string; readonly start: number | null | undefined }) {
  const ctx = useRef<AudioContext | null>(null);
  const src = useRef<AudioBufferSourceNode | null>(null);
  const token = useRef(0);  // bumped by every stop, so a clip still downloading when stopped never starts
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState("");
  const stop = () => {
    token.current++;
    const s = src.current;
    src.current = null;
    if (s) { s.onended = null; try { s.stop(); } catch { /* already stopped */ } }
    setPlaying(false);
  };
  const sel = `${file}|${audio}|${start}`;
  const prev = useRef(sel);
  useEffect(() => { if (prev.current !== sel) { prev.current = sel; stop(); } }, [sel]);
  useEffect(() => () => { token.current++; src.current?.stop(); }, []);
  const toggle = async () => {
    if (playing) return stop();
    setErr("");
    setPlaying(true);
    const mine = ++token.current;
    try {
      const c = (ctx.current ??= new AudioContext());
      const resumed = c.resume();  // called inside the click, so autoplay rules allow it
      const res = await fetch(`/api/audio/clip?beat_times_file=${encodeURIComponent(file)}&audio=${encodeURIComponent(audio ?? "")}`);
      if (!res.ok) throw new Error((await res.json().catch(() => null))?.error ?? `HTTP ${res.status}`);
      const buf = await c.decodeAudioData(await res.arrayBuffer());
      await resumed;
      if (token.current !== mine) return;
      const s = c.createBufferSource();
      s.buffer = buf;
      const g = c.createGain(); g.gain.value = 0.05;  // preview at 5% volume
      s.connect(g).connect(c.destination);
      s.onended = () => { src.current = null; setPlaying(false); };
      src.current = s;
      s.start();
    } catch (e) {
      if (token.current === mine) { setErr(e instanceof Error ? `${e.name}: ${e.message}` : String(e)); stop(); }
    }
  };
  return <>
    <button type="button" className="previewbtn" onClick={toggle} title={playing ? "stop" : "play 5s from the start time"}>{playing ? "■" : "▶"}</button>
    {err && <span className="dim musicinfo">{err}</span>}
  </>;
}

const PREVIEW_VOLUME = 0.05;  // previews play at 5% volume
const SCRUB_SECONDS = 5;  // how long a spot plays after you let go of the slider
const mmss = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

/** Where in the song the video starts: drag to hear that spot, let go to set it (onCommit). */
export function StartSlider({ file, audio, value, onCommit, label = "start time" }: { readonly file: string; readonly audio?: string; readonly value: number | null | undefined; readonly onCommit: (s: number) => void; readonly label?: string }) {
  const el = useRef<HTMLAudioElement>(null);
  const timer = useRef<number | undefined>(undefined);
  const [dur, setDur] = useState(0);
  const [pos, setPos] = useState(value ?? 0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => setPos(value ?? 0), [value]);
  const stop = () => { clearTimeout(timer.current); el.current?.pause(); };
  useEffect(() => stop, []);
  const playAt = (s: number) => {
    const a = el.current;
    if (!a) return;
    clearTimeout(timer.current);
    a.volume = PREVIEW_VOLUME;
    a.currentTime = s;
    a.play().catch(() => { /* autoplay blocked until the first click */ });
    timer.current = window.setTimeout(() => a.pause(), SCRUB_SECONDS * 1000);
  };
  const commit = () => { if (pos !== value) onCommit(pos); };
  return (
    <label className="field startslider"><span>{label} <b>{mmss(pos)}</b>{dur > 0 && <span className="dim"> / {mmss(dur)}</span>}</span>
      <span className="row">
        <button type="button" className="previewbtn" title={playing ? "stop" : `play ${SCRUB_SECONDS}s from here`} onClick={() => (playing ? stop() : playAt(pos))}>{playing ? "■" : "▶"}</button>
        <input type="range" min={0} max={dur || Math.max(pos, 1)} step={0.05} value={pos}
          onChange={(e) => { const s = Number(e.target.value); setPos(s); playAt(s); }}
          onPointerUp={commit} onKeyUp={commit} />
      </span>
      <audio /* NOSONAR(S4084): the user's own songs, no captions exist */ ref={el} preload="metadata" src={`/api/audio/track?beat_times_file=${encodeURIComponent(file)}&audio=${encodeURIComponent(audio ?? "")}`}
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}></audio>
    </label>
  );
}

// xfade transitions for a no-music edit (make_edit.TRANSITIONS); "cut" = hard cuts
export const TRANSITIONS = ["fade", "fadeblack", "fadewhite", "dissolve", "fadeslow", "smoothleft", "smoothup", "slideleft", "wipeleft",
  "circleopen", "radial", "zoomin", "hblur", "pixelize", "cut"];
const SCENARIO_NUMS: [keyof Music, string, number][] = [["transition_seconds", "transition (s)", 0.6], ["still_seconds", "still holds (s)", 3],
  ["fade_in", "fade in (s)", 0], ["fade_out", "fade out (s)", 0.5]];

/** Save a patch of the project's music settings (project.json "music"). */
export function useSaveMusic(name: string) {
  const qc = useQueryClient();
  const music = useProject(name).data?.config.music as Music | undefined;
  return async (patch: Music) => {
    await api(`/projects/${name}/music`, "PUT", { ...music, ...patch });
    void qc.invalidateQueries({ queryKey: ["project", name] });
  };
}

/** No-music ("scenario") edit settings: every stage plays in full and flows into the next through a transition. Saved to the project. */
export function ScenarioControls({ name }: { readonly name: string }) {
  const music = (useProject(name).data?.config.music ?? {}) as Music;
  const save = useSaveMusic(name);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const commit = (k: keyof Music) => {
    if (!(k in draft)) return;
    const v = draft[k];
    setDraft(({ [k]: _, ...rest }) => rest);
    void save({ [k]: v === "" ? undefined : Number(v) });
  };
  return (
    <div className="row wrap">
      <label className="field"><span>transition</span>
        <select value={music.transition ?? "fade"} onChange={(e) => save({ transition: e.target.value })}>
          {TRANSITIONS.map((t) => <option key={t} value={t}>{t === "cut" ? "cut (no transition)" : t}</option>)}
        </select>
      </label>
      {SCENARIO_NUMS.map(([k, label, def]) => (
        <label key={k} className="field"><span>{label}</span>
          <input type="number" min={0} max={30} step={0.1} placeholder={String(def)} disabled={k === "transition_seconds" && music.transition === "cut"}
            value={draft[k] ?? (music[k] as number | undefined) ?? ""} onChange={(e) => setDraft({ ...draft, [k]: e.target.value })}
            onBlur={() => commit(k)} onKeyDown={(e) => e.key === "Enter" && commit(k)} />
        </label>
      ))}
    </div>
  );
}

/** Pick the music the project will be cut to; the stage editor's duration list is computed from it. */
export default function MusicPicker({ name, full = false }: { readonly name: string; readonly full?: boolean }) {
  const qc = useQueryClient();
  const { music, file, fit } = useFit(name);
  const audio = useQuery({ queryKey: ["audio"], queryFn: () => api<Audio>("/audio") }).data;
  const save = useSaveMusic(name);
  const noMusic = <label className="chk inline" title="no song: every stage plays in full and fades into the next, like a scenario; the clips' own audio is the soundtrack">
    <input type="checkbox" checked={!!music?.none} onChange={(e) => save({ none: e.target.checked || undefined })} /> no music</label>;
  if (music?.none) return (
    <div className="musicpicker">
      {noMusic}
      {full ? <ScenarioControls name={name} /> : <span className="dim musicinfo">scenario edit · {music.transition ?? "fade"} transitions</span>}
    </div>
  );
  return (
    <div className="musicpicker">
      {noMusic}
      {!full && <PreviewButton file={file} audio={music?.audio} start={fit?.start_time} />}
      <label className="field"><span>♪ music (beats)</span>
        <select value={file} onChange={(e) => save({ beat_times_file: e.target.value })}>
          {(audio?.beat_files ?? [file]).map((f) => <option key={f} value={f}>{beatLabel(f)}</option>)}
        </select>
      </label>
      {full && <>
        <label className="field"><span>audio file</span>
          <select value={music?.audio ?? ""} onChange={(e) => save({ audio: e.target.value || undefined })}>
            <option value="">the beat file's track (else house)</option>
            {audio?.tracks.map((t) => <option key={t} value={t}>{t.split("/").pop()}</option>)}
          </select>
        </label>
        <StartSlider file={file} audio={music?.audio} value={fit?.start_time} onCommit={async (s) => {
          await api("/audio/start", "PUT", { beat_times_file: file, start_time: s });
          void qc.invalidateQueries({ queryKey: ["durations", file] });  // every engine's
        }} />
      </>}
      {fit && <span className="dim musicinfo">{fit.bpm} BPM · a 2-beat cut = {fit.cut}s</span>}
    </div>
  );
}

/** Table of the lengths the model can render and how they fill this music's cuts. */
export function FitTable({ name }: { readonly name: string }) {
  const { fit } = useFit(name);
  if (!fit) return null;
  return (
    <table className="fittable">
      <thead><tr><th>clip</th><th>cuts</th><th>fills</th><th></th></tr></thead>
      <tbody>{fit.options.map((o) => (
        <tr key={o.frames} className={o.recommended ? "rec" : ""}><td>{o.seconds}s <small>({o.frames}f)</small></td><td>{o.cuts}</td><td>{Math.round(o.coverage * 100)}%</td><td>{fitNote(o)}</td></tr>
      ))}</tbody>
    </table>
  );
}
