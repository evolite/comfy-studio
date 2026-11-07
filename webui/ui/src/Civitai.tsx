import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, clipKey, doneIn, fileUrl, Project } from "./api";
import { useProject } from "./hooks";

/** Civitai: pick stages, export each one's best still (with Civitai metadata) next to its clip into one fresh folder = one post. */
const STILL_ORDER = ["finish", "hires", "IL"];  // best upscale first; a ✓'d preview only gates the upscale, it isn't a pick

type Pair = { id: number; still?: string; clip?: string };

/** The approved (✓ on the Board) still and clip a stage would be exported with: the best approved upscale, and the approved
 *  interpolated clip, else the approved clip made from that still, else any approved clip. */
function pairOf(p: Project, id: number): Pair {
  const s = p.config.stages.find((st) => st.id === id)!;
  const still = STILL_ORDER.find((f) => p.folders[f]?.stills[id] && doneIn(s, f));
  const withClip = (f?: string) => (f !== undefined && p.folders[f]?.clips[id] && doneIn(s, clipKey(f)) ? f : undefined);
  const clip = withClip("interp") ?? withClip(still) ?? Object.keys(p.folders).find((f) => withClip(f) !== undefined);
  return { id, still, clip };
}

type Result = { path: string; pairs: number; clips: number; missing_params: number[] };
type Mode = "pairs" | "stills";
type Existing = { path?: string; stages?: number[]; mtime?: number };

const plural = (n: number, word: string) => `${n} ${word}${n > 1 ? "s" : ""}`;

function resultLine(res: Result): string {
  return `${plural(res.pairs, res.clips ? "pair" : "still")} → ${res.path.split("/").pop()}`;
}

function clipCaption(clip?: string): string {
  if (clip === undefined) return " · no clip";
  return ` · ${clip || "root"}`;
}

/** One stage: its still next to its clip; a click (or its checkbox) picks it. */
function PairCard({ p, x, ok, picked, withClips, onToggle }: { readonly p: Project; readonly x: Pair; readonly ok: boolean; readonly picked: boolean; readonly withClips: boolean; readonly onToggle: () => void }) {
  const on = ok && picked;
  const offHint = withClips ? "needs a still and a clip" : "needs a still";
  return (
    <label className={`cvpair ${on ? "on" : ""} ${ok ? "" : "off"}`} title={ok ? undefined : offHint}>
      <span className={`cvmedia ${withClips ? "" : "solo"}`}
        onMouseEnter={(e) => { const v = e.currentTarget.querySelector("video"); if (v) void v.play(); }}
        onMouseLeave={(e) => { const v = e.currentTarget.querySelector("video"); if (v) { v.pause(); v.currentTime = 0; } }}>
        {x.still ? <img loading="lazy" alt="" src={fileUrl(p.name, x.still, "still", x.id, p.folders[x.still].stills[x.id])} /> : <span className="cvnone">—</span>}
        {withClips && (x.clip === undefined
          ? <span className="cvnone">—</span>
          : <video muted loop preload="metadata" src={fileUrl(p.name, x.clip, "clip", x.id, p.folders[x.clip].clips[x.id])} />)}
      </span>
      <span className="cvcap"><input type="checkbox" aria-label={`pick stage ${x.id}`} checked={on} disabled={!ok} onChange={onToggle} /> <b>#{x.id}</b> <span className="dim">{x.still ?? "no still"}{withClips && clipCaption(x.clip)}</span></span>
    </label>
  );
}

export default function Civitai({ name }: { readonly name: string }) {
  const p = useProject(name).data;
  const qc = useQueryClient();
  const existing = useQuery({ queryKey: ["civitai-post", name], queryFn: () => api<Existing>(`/projects/${name}/civitai/post`) }).data;
  const [sel, setSel] = useState<Set<number> | null>(null);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<Result | null>(null);
  const [err, setErr] = useState("");
  const mkey = `civitaiMode:${name}`;  // stills + clips, or stills only; remembered per project in browser storage
  const [mode, setMode] = useState<Mode>(() => { try { return localStorage.getItem(mkey) === "stills" ? "stills" : "pairs"; } catch { return "pairs"; } });
  const pickMode = (m: Mode) => { setMode(m); try { localStorage.setItem(mkey, m); } catch { /* ignore */ } };
  const withClips = mode === "pairs";
  const pairs = p ? p.config.stages.map((s) => pairOf(p, s.id)) : [];
  const usable = (x: Pair) => !!x.still && (!withClips || x.clip !== undefined);
  const ready = pairs.filter(usable);
  // first visit: every stage shown (only approved stills and clips are)
  useEffect(() => {
    if (!p || sel) return;
    setSel(new Set(pairs.filter((x) => x.still).map((x) => x.id)));
  }, [p]);  // eslint-disable-line react-hooks/exhaustive-deps
  if (!p) return <div className="page">loading…</div>;
  const picked = ready.filter((x) => sel?.has(x.id));
  const toggle = (id: number) => setSel((cur) => {
    const n = new Set(cur);
    if (n.has(id)) n.delete(id); else n.add(id);
    return n;
  });
  const exportPost = async (overwrite = false) => {
    if (overwrite && !confirm(`Replace the exported stills, clips and clips.txt in ${existing?.path?.split("/").pop()} with the ${picked.length} picked stage${picked.length > 1 ? "s" : ""}?`)) return;
    setBusy(true); setErr(""); setRes(null);
    const items = picked.map((x) => ({ stage: x.id, still: x.still, ...(withClips ? { clip: x.clip } : {}) }));
    try { setRes(await api<Result>(`/projects/${name}/civitai/post`, "POST", { items, overwrite })); }
    catch (e) { setErr((e as Error).message); }
    void qc.invalidateQueries({ queryKey: ["civitai-post", name] });
    setBusy(false);
  };
  const openFolder = () => api(`/projects/${name}/civitai/open`, "POST").catch((e) => setErr((e as Error).message));
  const done = !!existing?.path;
  return (
    <div className="page">
      <div className="cmdbar"><div className="cmdrow">
        <label className="chk" title="select / clear all">
          <input type="checkbox" aria-label="select all" checked={ready.length > 0 && picked.length === ready.length}
            ref={(el) => { if (el) el.indeterminate = picked.length > 0 && picked.length < ready.length; }}
            onChange={(e) => setSel(new Set(e.target.checked ? ready.map((x) => x.id) : []))} />
        </label>
        {picked.length > 0 && <span className="selcount on">{picked.length}</span>}
        <span className="vsep" />
        <span className="seg" title="what each picked stage exports">
          <button className={withClips ? "on" : ""} title="each still next to its clip" onClick={() => pickMode("pairs")}>🖼 + 🎞</button>
          <button className={withClips ? "" : "on"} title="stills only" onClick={() => pickMode("stills")}>🖼</button>
        </span>
        <span className="spacer" />
        {res && <span className="dim" title={res.path}>✓ {resultLine(res)}</span>}
        {done && !res && <span className="dim" title={existing!.path}>exported {existing!.stages!.length ? `#${existing!.stages!.join(", #")}` : ""} · {new Date(existing!.mtime! * 1000).toLocaleString()}</span>}
        {done
          ? <>
            <button className="btn primary" onClick={() => void openFolder()} title={existing!.path}>📂 Open folder</button>
            <button className="btn danger" disabled={!picked.length || busy} title="replace the exported files with the picked stages (anything else you put in the folder stays)"
              onClick={() => exportPost(true)}>{busy ? "exporting…" : "Overwrite"}</button>
          </>
          : <button className="btn primary" disabled={!picked.length || busy} title={withClips ? "one folder: each still (with Civitai metadata) next to its clip, plus clips.txt with the clip prompts to paste" : "one folder: the picked stills with Civitai metadata"}
              onClick={() => exportPost()}>{busy ? "exporting…" : "Export"}</button>}
      </div></div>
      {err && <button type="button" className="msg err" onClick={() => setErr("")}>{err}</button>}
      {res && res.missing_params.length > 0 && <div className="msg">⚠ no generation metadata for #{res.missing_params.join(", #")} (no recorded graph): those stills are plain copies</div>}
      {!ready.length && <div className="msg">nothing approved yet: ✓ {withClips ? "a still and its clip" : "a still"} on the Board to see it here</div>}
      <div className={`cvgrid ${withClips ? "" : "solo"}`}>
        {ready.map((x) => <PairCard key={x.id} p={p} x={x} ok={usable(x)} picked={!!sel?.has(x.id)} withClips={withClips} onToggle={() => toggle(x.id)} />)}
      </div>
    </div>
  );
}
