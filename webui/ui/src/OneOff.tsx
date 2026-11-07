import { MouseEvent, useState } from "react";
import { api, fileUrl } from "./api";
import { ACTIONS, animSources, compareFor, hoverPlay, LOCAL_ACTIONS, RunMenu, slotPath } from "./Board";
import { activeByStage, useHealth, useJobs, useProject, useRun } from "./hooks";
import Lightbox, { LightboxItem } from "./Lightbox";
import StageDrawer from "./StageDrawer";
import { clipVolume } from "./runPrefs";

const ORDER = ["import", "preview", "IL", "hires", "finish", ""];  // rows top to bottom; any other folder lands above these

/** Hover-play for a one-off's clip, at the viewer's clip volume. */
function hoverStart(e: MouseEvent<HTMLElement>) {
  const v = e.currentTarget.querySelector("video");
  if (!v) return;
  v.volume = clipVolume();
  void v.play();
}

/** Why an action can't run yet; "" = it can. */
function oneOffBlock(a: string, localDown: boolean, hasStill: (f: string) => boolean): string {
  if (LOCAL_ACTIONS.has(a) && localDown) return "local ComfyUI is down";
  if (["local", "il_cloud", "hires_local", "face", "face_local"].includes(a) && !hasStill("preview")) return "make a Preview first";
  if ((a === "face" || a === "face_local") && !hasStill("IL")) return "needs IL/stage_N.png — run IL + upscale first";
  if ((a === "finish" || a === "finish_local") && !hasStill("hires") && !hasStill("IL")) return "needs an upscale result (IL or hires) — run IL + upscale or the latent upscale first";
  return "";
}

function busyClass(busy?: { queued?: boolean; lane: string }): string {
  if (!busy) return "";
  return `busy ${busy.queued ? "queued" : ""} lane-${busy.lane}`;
}

function clipTag(kind: string, audioOn: boolean): string {
  if (kind !== "clip") return "";
  return audioOn ? " ▶ 🔊" : " ▶";
}

/** A one-off is a single image: results big on the left, the prompt editor always open on the right, one Run menu. No approvals, phases or stage list. */
export default function OneOff({ name }: { readonly name: string }) {
  const proj = useProject(name);
  const jobs = useJobs().data ?? [];
  const health = useHealth().data;
  const run = useRun(name);
  const [light, setLight] = useState<number | null>(null);
  const [msg, setMsg] = useState("");
  const [pick, setPick] = useState<string | null>(null);  // the selected result's folder
  const [size, setSize] = useState(() => { try { return Number(localStorage.getItem("oneoffSize")) || 160; } catch { return 260; } });
  const pickSize = (n: number) => { setSize(n); try { localStorage.setItem("oneoffSize", String(n)); } catch { /* ignore */ } };
  const [url, setUrl] = useState("");
  const [importing, setImporting] = useState(false);
  const p = proj.data;
  const seenKey = `oneoffSeen:${name}`;
  // per-tile "last opened" mtimes; __since = when this one-off was first opened here (anything older counts as seen)
  const [seen, setSeen] = useState<Record<string, number>>(() => {
    let m: Record<string, number> = {};
    try { m = JSON.parse(localStorage.getItem(seenKey) ?? "null") ?? {}; } catch { /* ignore */ }
    if (!("__since" in m)) { m.__since = Math.floor(Date.now() / 1000); try { localStorage.setItem(seenKey, JSON.stringify(m)); } catch { /* ignore */ } }
    return m;
  });
  const markSeen = (m: Record<string, number>) => setSeen((cur) => { const n = { ...cur, ...m }; try { localStorage.setItem(seenKey, JSON.stringify(n)); } catch { /* ignore */ } return n; });
  if (proj.error) return <div className="page err">{String(proj.error.message)}</div>;
  if (!p) return <div className="page">loading…</div>;
  const stage = p.config.stages[0];
  if (!stage) return <div className="page dim">this one-off has no stage</div>;
  const id = stage.id;
  const audioOn = !!stage.audio;
  const busy = activeByStage(jobs, name)[id];
  // oldest on the left, newest on the right (v = file mtime)
  const slots = Object.entries(p.folders).flatMap(([f, st]) => [
    ...(st.stills[id] ? [{ f, kind: "still" as const, v: st.stills[id] }] : []),
    ...(st.clips[id] ? [{ f, kind: "clip" as const, v: st.clips[id] }] : []),
  ]).sort((a, b) => a.v - b.v);
  const skey = (s: { f: string; kind: string }) => `${s.f}|${s.kind}`;
  // a result is "new" until you open it; the results from before you first opened this one-off count as seen
  const isNew = (s: { f: string; kind: string; v: number }) => s.v > (seen[skey(s)] ?? seen.__since);
  const hasStill = (f: string) => !!p.folders[f]?.stills[id];
  const latest = slots.at(-1)?.f;  // nothing picked yet: work on the newest result
  const folder = pick !== null && p.folders[pick] ? pick : latest ?? "preview";
  const why = (a: string) => oneOffBlock(a, !!health && !health.local, hasStill);
  const go = (action: string, f?: string) =>
    run.mutate({ action, stages: [id], folder: action === "animate" ? f ?? folder : undefined }, { onSuccess: () => setMsg(""), onError: (e) => setMsg(`error: ${e.message}`) });
  const doImport = (u: string) => {
    if (!u.trim()) return;
    setImporting(true); setMsg("");
    api(`/projects/${name}/stages/${id}/import`, "POST", { url: u }).then(() => { setUrl(""); setPick("import"); proj.refetch(); }, (e) => setMsg(`error: ${e.message}`)).finally(() => setImporting(false));
  };
  const items: LightboxItem[] = slots.map((s) => ({ id, label: `${s.f || "root"} ${s.kind}`, kind: s.kind, src: fileUrl(name, s.f, s.kind, id, s.v), audio: s.kind === "clip" && audioOn,
                                                    ...(s.kind === "still" ? compareFor(p, s.f, id) : {}) }));
  return (
    <div className="page oneoffpage">
      <div className={`oneoffmain ${busyClass(busy)}`} style={{ ["--sz" as string]: `${size}px` }}>
        <div className="ootools">
          <span className="title" title={name}>{name}</span>
          {busy && <span className="pill run">{ACTIONS[busy.action]?.label ?? busy.action}{busy.queued ? " (queued)" : ""}</span>}
          <input className="importurl" placeholder="paste a civitai image link to import" value={url} spellCheck={false} disabled={importing}
            onChange={(e) => setUrl(e.target.value)} onPaste={(e) => { const t = e.clipboardData.getData("text").trim(); if (t.startsWith("http")) { e.preventDefault(); setUrl(t); doImport(t); } }}
            onKeyDown={(e) => { if (e.key === "Enter") doImport(url); }} />
          <span className="spacer" />
          <label className="size" title="thumbnail size">size <input type="range" min={80} max={640} step={20} value={size} onChange={(e) => pickSize(Number(e.target.value))} /></label>
          <RunMenu project={p.name} ids={[id]} label="Run" workFolder={folder} why={why} onGo={go} sources={animSources(p, [id])} />
        </div>
        {msg && <button type="button" className="msg" onClick={() => setMsg("")}>{msg}</button>}
        {slots.length === 0 && !busy && <div className="dim empty">nothing yet — open Run ▾ and start with a Preview</div>}
        {[...new Set(slots.map((s) => s.f))].sort((x, y) => ORDER.indexOf(x) - ORDER.indexOf(y)).map((f) => (
        <div key={f} className="oorow">
          <div className="oolabel">{f || "root"}</div>
          <div className="oneoffgrid">
          {slots.map((s, i) => s.f === f && (
            <figure key={s.f + s.kind} className={`tile ${s.kind} ${isNew(s) ? "new" : ""} ${s.f === folder ? "sel" : ""}`} title="click to enlarge">
              <button className="expand rm" title="remove (moved to _debug/removed)" onClick={(e) => { e.stopPropagation(); api(`/projects/${name}/stages/${id}/remove`, "POST", { folder: s.f, kind: s.kind }).then(() => { setPick(null); void proj.refetch(); }, (error_) => setMsg(`error: ${error_.message}`)); }}>✕</button>
              <button className="expand" title="open in folder" onClick={(e) => { e.stopPropagation(); void api(`/projects/${name}/reveal`, "POST", { path: slotPath(s.f, s.kind, id) }); }}>📂</button>
              <button type="button" className="mediabtn" onClick={() => { setPick(s.f); setLight(i); markSeen({ [skey(s)]: s.v }); }} aria-label={`open ${s.f || "root"} ${s.kind}`} onMouseEnter={hoverStart} onMouseLeave={hoverPlay.onMouseLeave}>
                {s.kind === "still"
                  ? <img loading="lazy" alt="" src={fileUrl(name, s.f, "still", id, s.v)} />
                  : <video /* NOSONAR(S4084): generated clips, no captions exist */ muted={!audioOn} loop preload="metadata" src={fileUrl(name, s.f, "clip", id, s.v)}></video>}
              </button>
              <span className="tag">{isNew(s) && "● new · "}{s.f || "root"}{clipTag(s.kind, audioOn)}</span>
            </figure>
          ))}
          </div>
        </div>
        ))}
      </div>
      <StageDrawer inline project={name} stage={stage} stages={p.config.stages} onClose={() => {}} onNav={() => {}} />
      {light !== null && <Lightbox project={name} items={items} index={light} onIndex={setLight} onClose={() => setLight(null)} />}
    </div>
  );
}
