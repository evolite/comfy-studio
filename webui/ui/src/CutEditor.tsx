import { KeyboardEvent, PointerEvent, useEffect, useRef, useState } from "react";
import { api, fileUrl } from "./api";

type CutInfo = { folder?: string; start?: number; end?: number; loop?: { in: number; out: number; seconds: number } };
type Grip = "in" | "out" | "head";

/** The board's Cut phase, one per stage: the source clip with in/out handles to drag; Save cut writes cut/videos/stage_N.mp4
 * (video/cut_clip.py); Seamless loop finds the best loop inside the in/out range and bridges its seam with RIFE (video/loop_clip.py). Keys while the bar is focused (click it or the video): i / o = in / out at the playhead, ← → = one frame, space = play the kept span. */
export default function CutEditor({ project, sid, folder, v, fps, onSaved }: {
  readonly project: string; readonly sid: number; readonly folder: string; readonly v: number; readonly fps: number; readonly onSaved: () => void;
}) {
  const vid = useRef<HTMLVideoElement>(null);
  const bar = useRef<HTMLDivElement>(null);
  const grip = useRef<Grip | null>(null);
  const [dur, setDur] = useState(0);
  const [span, setSpan] = useState<[number, number]>([0, 0]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [saved, setSaved] = useState<CutInfo | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [minLoop, setMinLoop] = useState(3);  // Seamless loop: the shortest loop it may pick, seconds
  const [inP, outP] = span;
  const snap = (s: number) => Math.max(0, Math.min(dur, Math.round(s * fps) / fps));

  useEffect(() => { void api<CutInfo>(`/projects/${project}/stages/${sid}/cut`).then(setSaved).catch(() => setSaved({})); }, [project, sid]);
  // start from the last cut of this same source, else the whole clip
  useEffect(() => {
    if (!dur || saved === null) return;
    const same = saved.folder === folder && saved.end !== undefined;
    setSpan(same ? [saved.start ?? 0, Math.min(saved.end!, dur)] : [0, dur]);
  }, [dur, saved, folder]);
  // while playing: follow the playhead every frame and loop the kept span
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const el = vid.current;
      if (el) {
        if (el.currentTime >= outP || el.currentTime < inP - 0.05) el.currentTime = inP;
        setT(el.currentTime);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, inP, outP]);

  const seek = (s: number) => {
    const el = vid.current;
    if (el) el.currentTime = s;
    setT(s);
  };
  const setIn = (s: number) => { const x = Math.min(snap(s), outP - 1 / fps); setSpan([Math.max(0, x), outP]); seek(Math.max(0, x)); };
  const setOut = (s: number) => { const x = Math.max(snap(s), inP + 1 / fps); setSpan([inP, Math.min(dur, x)]); seek(Math.min(dur, x)); };
  const play = () => {
    const el = vid.current;
    if (!el) return;
    if (playing) { el.pause(); setPlaying(false); return; }
    if (el.currentTime < inP || el.currentTime >= outP) el.currentTime = inP;
    void el.play();
    setPlaying(true);
  };
  const stop = () => { vid.current?.pause(); setPlaying(false); };
  const at = (e: PointerEvent) => {
    const r = bar.current!.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * dur;
  };
  const down = (e: PointerEvent, g: Grip) => {
    e.stopPropagation();
    stop();
    grip.current = g;
    bar.current!.setPointerCapture(e.pointerId);
    if (g === "head") seek(snap(at(e)));
  };
  const move = (e: PointerEvent) => {
    if (!grip.current) return;
    const s = at(e);
    if (grip.current === "in") setIn(s);
    else if (grip.current === "out") setOut(s);
    else seek(snap(s));
  };
  const key = (e: KeyboardEvent) => {
    const k = e.key;
    if (k === "i") setIn(t);
    else if (k === "o") setOut(t);
    else if (k === "ArrowLeft" || k === "ArrowRight") { stop(); seek(snap(t + (k === "ArrowLeft" ? -1 : 1) / fps)); }
    else if (k === " ") play();
    else return;
    e.preventDefault();
    e.stopPropagation();
  };
  const save = () => {
    setBusy(true);
    setMsg("cutting…");
    api<CutInfo>(`/projects/${project}/stages/${sid}/cut`, "POST", { folder, start: inP, end: outP })
      .then((r) => { setSaved(r); setMsg("saved → cut/videos ✓"); onSaved(); })
      .catch((e: Error) => setMsg(`error: ${e.message}`))
      .finally(() => setBusy(false));
  };
  const makeLoop = () => {
    setBusy(true);
    setMsg("finding the loop and bridging its seam with RIFE…");
    api<CutInfo>(`/projects/${project}/stages/${sid}/loop`, "POST", { folder, start: inP, end: outP, min_seconds: minLoop })
      .then((r) => { setSaved(r); setMsg(`loop ${r.loop?.seconds.toFixed(2)}s (${r.loop?.in.toFixed(2)}–${r.loop?.out.toFixed(2)}s) → cut/videos ✓`); onSaved(); })
      .catch((e: Error) => setMsg(`error: ${e.message}`))
      .finally(() => setBusy(false));
  };
  const pct = (s: number) => `${dur ? (s / dur) * 100 : 0}%`;
  const changed = saved?.folder !== folder || Math.abs((saved?.start ?? 0) - inP) > 1e-3 || Math.abs((saved?.end ?? -1) - outP) > 1e-3;

  return (
    <div className="cutter">
      <video ref={vid} preload="auto" src={fileUrl(project, folder, "clip", sid, v)} onClick={() => { play(); bar.current?.focus(); }}
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration)} onPause={() => setPlaying(false)}
        onSeeked={(e) => setT(e.currentTarget.currentTime)}>
        <track kind="captions" />
      </video>
      <div ref={bar} className="cut-bar" role="slider" tabIndex={0} aria-label="playhead" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={t}
        title="i / o = set in / out at the playhead · ← → = one frame · space = play the kept part"
        onKeyDown={key} onPointerDown={(e) => down(e, "head")} onPointerMove={move} onPointerUp={() => { grip.current = null; }}>
        <div className="cut-keep" style={{ left: pct(inP), width: pct(outP - inP) }} />
        <div className="cut-head" style={{ left: pct(t) }} />
        <button type="button" className="cut-grip in" style={{ left: pct(inP) }} aria-label="in point" onPointerDown={(e) => down(e, "in")} />
        <button type="button" className="cut-grip out" style={{ left: pct(outP) }} aria-label="out point" onPointerDown={(e) => down(e, "out")} />
      </div>
      <div className="cut-row">
        <button className="btn sm" onClick={play}>{playing ? "❚❚" : "▶"}</button>
        <span className="dim">in {inP.toFixed(2)}s · out {outP.toFixed(2)}s · keeps {(outP - inP).toFixed(2)}s of {dur.toFixed(2)}s · from {folder || "root"}</span>
        <span className="spacer" />
        <button className="btn sm" onClick={() => { stop(); setSpan([0, dur]); seek(0); }}>Reset</button>
        <button className="btn sm primary" disabled={busy || !dur || !changed} onClick={save}>Save cut</button>
        <label className="dim" title="the shortest loop Seamless loop may pick: longer loops repeat less often but their seam matches a little less well">
          min <input type="number" min={0.5} max={Math.max(0.5, Math.floor(dur * 10) / 10)} step={0.5} value={minLoop} style={{ width: "4em" }}
            onChange={(e) => setMinLoop(Number(e.target.value) || 3)} /> s
        </label>
        <button className="btn sm" disabled={busy || !dur} onClick={makeLoop}
          title="find the best seamless loop inside the in/out range (keep anything that shouldn't repeat out of it) and bridge its seam with RIFE on the local ComfyUI">Seamless loop</button>
      </div>
      {msg ? <div className="dim">{msg}</div> : saved?.loop && saved.folder === folder && <div className="dim">cut/videos holds a seamless loop of {saved.loop.seconds.toFixed(2)}s ({saved.loop.in.toFixed(2)}–{saved.loop.out.toFixed(2)}s)</div>}
    </div>
  );
}
