import { useEffect, useState } from "react";
import TakeDiff from "./TakeDiff";
import { RefChip } from "./controls";
import { useClipVolume } from "./runPrefs";

/** `compare`: the still this one was made from (preview, or hires/ for finish/); a still that has one opens as a before/after slider
 * (that one left, labelled `compareLabel`, default "preview"; this one right). */
export type LightboxItem = { id: number; label: string; kind: "still" | "clip"; src: string; audio?: boolean; seen?: [string, number]; compare?: string; compareLabel?: string;
  take?: { project: string; sid: number; id: string; current: boolean; delta: string; currentId?: string };  // a preview take (Use this / Changes vs current)
  civitai?: string };  // query of /api/projects/<p>/civitai for this still: a copy with Civitai's generation metadata

const CMP_KEY = "lightbox:compare";
const DIFF_KEY = "lightbox:diff";
/** Save a copy with Civitai's generation metadata; `query` = the item's `civitai` path + query. Returns the note to show. */
async function exportCivitaiPng(query: string): Promise<string> {
  const [path, qs] = query.split("?");
  const q = new URLSearchParams(qs);
  try {
    const r = await fetch(`/api${path}/export`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ folder: q.get("folder"), stage: Number(q.get("stage")), take: q.get("take") ?? undefined }) });
    if (!r.ok) throw new Error(`export failed: ${r.status}`);
    const j = await r.json() as { params: boolean };
    return j.params ? "saved — drag it from the file manager onto Civitai" : "saved, but no settings were recorded for this image";
  } catch { return "couldn't save the Civitai copy"; }
}

/** Copy the generation parameters to the clipboard. Returns the note to show. */
async function copyCivitaiParams(query: string): Promise<string> {
  let params: string | null;
  try {
    const r = await fetch(`/api${query}&format=text`);
    params = (await r.json() as { params: string | null }).params;
  } catch { return "couldn't read the parameters"; }
  if (!params) return "no settings recorded for this image";
  try {
    await navigator.clipboard.writeText(params);
    return "parameters copied";
  } catch { return "clipboard unavailable"; }
}

const loadCmp = () => { try { return localStorage.getItem(CMP_KEY) !== "0"; } catch { return true; } };

export default function Lightbox({ items, index, onIndex, onClose, onUse, project }: { readonly items: LightboxItem[]; readonly index: number; readonly onIndex: (i: number) => void; readonly onClose: () => void;
  readonly onUse?: (it: LightboxItem) => void; readonly project?: string }) {  // project: show the stage's RefChip (items' `id` = stage id)
  const vol = useClipVolume();
  const [cmp, setCmp] = useState(loadCmp);
  const [diff, setDiff] = useState(() => { try { return localStorage.getItem(DIFF_KEY) === "1"; } catch { return false; } });
  const pickDiff = (v: boolean) => { setDiff(v); try { localStorage.setItem(DIFF_KEY, v ? "1" : "0"); } catch { /* ignore */ } };
  const [pos, setPos] = useState(50);
  const [note, setNote] = useState("");  // split position, % of the image width from the left
  const pickCmp = (v: boolean) => { setCmp(v); try { localStorage.setItem(CMP_KEY, v ? "1" : "0"); } catch { /* ignore */ } };
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") onIndex(Math.min(items.length - 1, index + 1));
      if (e.key === "ArrowLeft") onIndex(Math.max(0, index - 1));
      if (e.key === "c") pickCmp(!cmp);
      const t = items[index]?.take;
      if (t && e.key === "d") pickDiff(!diff);
      if (t && e.key === "Enter" && !t.current && onUse) { e.preventDefault(); onUse(items[index]); }
    };
    addEventListener("keydown", k);
    return () => removeEventListener("keydown", k);
  }, [index, items, onIndex, onClose, cmp, diff, onUse]);
  const it = items[index];
  if (!it) return null;
  const comparing = it.kind === "still" && !!it.compare && cmp;
  const follow = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)));
  };
  return (
    <div className="lightbox">
      <button type="button" className="backdrop" aria-label="close" tabIndex={-1} onClick={onClose} />
      <div className="lb-bar">
        <button className="btn" onClick={() => onIndex(Math.max(0, index - 1))}>←</button>
        {project && <RefChip project={project} stage={it.id} />}
        <span>{it.label} ({index + 1}/{items.length})</span>
        <button className="btn" onClick={() => onIndex(Math.min(items.length - 1, index + 1))}>→</button>
        <a className="btn" href={it.src} target="_blank">open</a>
        {it.kind === "still" && it.compare && (
          <button className={`btn ${cmp ? "primary" : ""}`} title={`before/after slider against the ${it.compareLabel ?? "preview"} (c)`} onClick={() => pickCmp(!cmp)}>compare {it.compareLabel ?? "preview"}</button>
        )}
        {it.take && (it.take.current
          ? <span className="pill ok" title="later steps (IL, latent upscale, ...) use this take">current</span>
          : <>
            {it.take.delta && <span className="pill" title="what differs from the current take">{it.take.delta}</span>}
            {onUse && <button className="btn primary" title="make this the current preview and ✓ it (Enter)" onClick={() => onUse(it)}>Use this</button>}
          </>)}
        {it.civitai && <>
          {it.kind === "still" && <button className="btn" title="save a copy with the prompt, sampler, seed, model and LoRA hashes embedded (data/civitai/) and show it in the file manager: drag it onto Civitai's upload" onClick={() => {
            void exportCivitaiPng(it.civitai!).then(setNote);
            setTimeout(() => setNote(""), 4000);
          }}>Civitai PNG</button>}
          <button className="btn" title={it.kind === "clip" ? "copy the motion prompt, LoRAs and seed to paste into Civitai (it doesn't read them from video files)" : "copy the generation parameters (Civitai's manual fields)"} onClick={() => {
            void copyCivitaiParams(it.civitai!).then(setNote);
            setTimeout(() => setNote(""), 2500);
          }}>copy params</button>
          {note && <span className="dim">{note}</span>}
        </>}
        {it.take && <button className={`btn ${diff ? "primary" : ""}`} title="what changed between this take and the current one (d)" onClick={() => pickDiff(!diff)}>changes vs current</button>}
      </div>
      <div className="lb-body">
        {comparing && (
          <div className="lb-cmp" onPointerMove={follow} onPointerDown={follow}>
            <img src={it.src} alt={it.label} draggable={false} />
            <img className="lb-cmp-before" src={it.compare} alt={it.compareLabel ?? "preview"} draggable={false} style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
            <div className="lb-cmp-line" style={{ left: `${pos}%` }} />
            <span className="lb-cmp-tag" style={{ left: 8 }}>{it.compareLabel ?? "preview"}</span>
            <span className="lb-cmp-tag" style={{ right: 8 }}>{it.label.split(" · ").pop()}</span>
          </div>
        )}
        {!comparing && it.kind === "still" && <img src={it.src} alt={it.label} />}
        {it.kind === "clip" && <video /* NOSONAR(S4084): generated clips, no captions exist */ key={it.src} src={it.src} controls autoPlay loop ref={(v) => { if (v) v.volume = it.audio === false ? 0.01 : vol; }}></video>}
        {it.take && diff && <TakeDiff project={it.take.project} sid={it.take.sid} id={it.take.id} currentId={it.take.currentId} />}
      </div>
    </div>
  );
}
