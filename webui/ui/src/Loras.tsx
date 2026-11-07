import { MouseEvent, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, byCategory, Discovered, ImagePass, LORA_CATEGORIES, LoraCategory, LoraCombo, LoraEntry, LoraKind, LoraSpec } from "./api";
import { saveStateClass } from "./controls";
import { useLoras, useRegistry } from "./hooks";

/** The LoRA library (loras.json): every image + i2v LoRA with notes, tips and availability; local files to register. */
const NAME_RE = /^[a-z0-9_]+$/;
const SPEC_KEYS: Record<LoraKind, (keyof LoraSpec)[]> = {
  image: ["category", "krea", "il", "moss", "krea_trigger", "il_trigger", "moss_trigger", "default", "suggested", "min", "max", "cloud", "disabled", "note", "tips", "url"],
  video: ["category", "file", "engine", "trigger", "suggested", "min", "max", "cloud", "disabled", "note", "tips", "url"],
};
const specOf = (kind: LoraKind, e: LoraSpec): LoraSpec =>
  Object.fromEntries(SPEC_KEYS[kind].filter((k) => e[k] !== undefined).map((k) => [k, e[k]]));

/** One image pass's local-availability pill: found, missing, or unknown while the local ComfyUI is down. */
function localPill(v: boolean | null | undefined): { cls: string; tip: string; mark: string } {
  if (v == null) return { cls: "", tip: "local ComfyUI down", mark: "?" };
  return v ? { cls: "ok", tip: "file found locally", mark: "✓" } : { cls: "bad", tip: "not in the local loras folder", mark: "✗" };
}

/** Availability badges: per image pass local ✓/✗/? + the manual cloud flag. */
export function Avail({ kind, e }: { readonly kind: LoraKind; readonly e: LoraEntry }) {
  return (
    <span className="lavail">
      {kind === "image" && (["krea", "il", "moss"] as const).filter((p) => e[p]).map((p) => {
        const { cls, tip, mark } = localPill(e.local[p]);
        return <span key={p} className={`pill ${cls}`} title={tip}>{p} local {mark}</span>;
      })}
      <span className={`pill ${e.cloud === false ? "local" : ""}`} title="set by hand: the cloud catalog doesn't list your own uploads">
        {e.cloud === false ? "local only" : "cloud: uploaded"}</span>
    </span>
  );
}

function Editor({ kind, name, e, onDone }: { readonly kind: LoraKind; readonly name: string; readonly e: LoraEntry; readonly onDone: () => void }) {
  const qc = useQueryClient();
  const [d, setD] = useState<LoraSpec>(specOf(kind, e));
  const engines = useRegistry().data?.video_engines ?? { minimax: "MiniMax H3" };
  const [state, setState] = useState<"saved" | "saving" | "error">("saved");
  const [err, setErr] = useState("");
  const timer = useRef<number>(0);
  const latest = useRef(d);
  const save = async () => {
    try {
      await api(`/loras/${kind}/${name}`, "PUT", latest.current);
      setState("saved"); setErr(""); qc.invalidateQueries({ queryKey: ["loras"] });
    } catch (error_) { setState("error"); setErr((error_ as Error).message); }
  };
  useEffect(() => () => { if (timer.current) { clearTimeout(timer.current); void save(); } }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (k: keyof LoraSpec, v: unknown) => {
    const next = { ...latest.current, [k]: v === "" || v === null ? undefined : v } as LoraSpec;
    latest.current = next; setD(next); setState("saving");
    clearTimeout(timer.current); timer.current = window.setTimeout(() => { timer.current = 0; void save(); }, 600);
  };
  const txt = (k: keyof LoraSpec, label: string, ph = "") => (
    <label className="field"><span>{label}</span>
      <input type="text" value={(d[k] as string) ?? ""} placeholder={ph} onChange={(x) => set(k, x.target.value)} spellCheck={false} /></label>
  );
  const num = (k: keyof LoraSpec, label: string, ph = "") => (
    <label className="field"><span>{label}</span>
      <input type="number" step="0.1" value={(d[k] as number) ?? ""} placeholder={ph} onChange={(x) => set(k, x.target.value === "" ? undefined : Number(x.target.value))} /></label>
  );
  const del = async () => {
    const n = e.used.length;
    if (!confirm(`Remove ${name} from the library? (the file itself stays)` + (n ? `\n\nIt is on ${n} stage(s); it will be taken off them.` : ""))) return;
    try { await api(`/loras/${kind}/${name}?strip=true`, "DELETE"); qc.invalidateQueries({ queryKey: ["loras"] }); qc.invalidateQueries({ queryKey: ["project"] }); onDone(); }
    catch (error_) { setErr((error_ as Error).message); }
  };
  return (
    <div className="lform">
      <div className="lgrid">
        <label className="field"><span>category</span>
          <select value={d.category ?? "concept"} onChange={(x) => set("category", x.target.value)}>
            {LORA_CATEGORIES.map((c) => <option key={c}>{c}</option>)}
          </select></label>
        {kind === "image" ? <>
          {txt("krea", "Krea2 file", "name.safetensors (blank = no Krea2 pass)")}
          {txt("il", "Illustrious file (full engine)", "blank = no IL pass")}
          {txt("moss", "MOSS-Image file (moss engine)", "moss/name.safetensors (blank = none)")}
          {txt("krea_trigger", "Krea2 trigger (appended while on)")}
          {txt("il_trigger", "IL trigger tags (appended while on)")}
          {txt("moss_trigger", "MOSS trigger (appended while on)")}
        </> : <>
          {txt("file", "file", "name.safetensors as uploaded")}
          {txt("trigger", "trigger (put in front of the motion while on)")}
          <label className="field"><span>i2v model it was trained for</span>
            <select value={d.engine ?? "minimax"} onChange={(x) => set("engine", x.target.value === "minimax" ? undefined : x.target.value)}>
              {Object.entries(engines).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
            </select></label>
        </>}
        {num("suggested", "suggested strength (the picker adds it at this)", "1.0")}
        {kind === "image" && num("default", "default (on in every stage unless set; leave blank)")}
        {num("min", "slider min (slider LoRAs only: strength range the stage editor's slider covers)")}
        {num("max", "slider max (set both, or leave both blank)")}
      </div>
      {txt("note", "note (one line, shown in the picker)")}
      <label className="field"><span>tips (prompt phrasing, tested strengths: shown beside the prompt while the LoRA is on)</span>
        <textarea rows={5} value={d.tips ?? ""} onChange={(x) => set("tips", x.target.value)} spellCheck={false} /></label>
      {txt("url", "source url")}
      <div className="row">
        <label className="chk" title="Comfy Cloud can't list your own uploads, so this is by hand. Off = cloud runs refuse stages using it.">
          <input type="checkbox" checked={d.cloud !== false} onChange={(x) => set("cloud", x.target.checked)} /> uploaded to Comfy Cloud</label>
        <span className={`pill ${saveStateClass(state)}`}>{state}</span>
        <span className="spacer" />
        <button className="btn sm danger" title={e.used.length ? `remove from the library and take it off the ${e.used.length} stage(s) using it` : "remove from the library"} onClick={del}>remove</button>
      </div>
      {err && <div className="err">{err}</div>}
    </div>
  );
}

function Row({ kind, name, e, open, onToggle }: { readonly kind: LoraKind; readonly name: string; readonly e: LoraEntry; readonly open: boolean; readonly onToggle: () => void }) {
  const qc = useQueryClient();
  const [showUse, setShowUse] = useState(false);
  const toggle = (x: MouseEvent) => {
    x.stopPropagation();
    void api(`/loras/${kind}/${name}`, "PUT", { ...specOf(kind, e), disabled: e.disabled ? undefined : true }).then(() => qc.invalidateQueries({ queryKey: ["loras"] }));
  };
  return (
    <div className={`lrow ${open ? "on" : ""} ${e.disabled ? "loff" : ""}`} id={`lora-${kind}-${name}`}>
      <div className="lhead">
        <button className={`btn sm ${e.disabled ? "" : "primary"}`} onClick={toggle}
          title={e.disabled ? "off: stages already using it keep it, the stage editor doesn't offer it. Click to turn on" : "on: offered in the stage editor. Click to turn off (stages already using it keep it)"}>
          {e.disabled ? "off" : "on"}</button>
        <button type="button" className="namebtn" aria-expanded={open} onClick={onToggle}><b>{name}</b></button>
        <span className="dim">{e.suggested ?? 1}{e.default != null && ` · default ${e.default}`}{e.min != null && ` · slider ${e.min}…${e.max}`}</span>
        <Avail kind={kind} e={e} />
        {kind === "video" && e.engine && <span className="pill" title="i2v model this LoRA is for">{e.engine}</span>}
        <span className="lnote">{e.note || <i className="dim">no note yet</i>}</span>
        <button className="btn sm" disabled={!e.used.length} onClick={(x) => { x.stopPropagation(); setShowUse(!showUse); }}>used {e.used.length}</button>
        {e.url && <a href={e.url} target="_blank" rel="noreferrer" onClick={(x) => x.stopPropagation()}>source ↗</a>}
      </div>
      {showUse && <div className="luse">{e.used.map((u) => <a key={`${u.project}/${u.stage}`} href={`#/p/${u.project}/board`}>{u.project} #{u.stage}</a>)}</div>}
      {open && <Editor kind={kind} name={name} e={e} onDone={onToggle} />}
    </div>
  );
}

/** A combo: its LoRAs + strengths, note and tips, edited in place (saved on blur). Made from a stage ("save as combo…"). */
function ComboRow({ name, c, entries }: { readonly name: string; readonly c: LoraCombo; readonly entries: Record<string, LoraEntry> }) {
  const qc = useQueryClient();
  const [d, setD] = useState(c);
  const [err, setErr] = useState("");
  useEffect(() => setD(c), [c]);
  const save = async (next: LoraCombo) => {
    try { await api(`/lora-combos/${name}`, "PUT", next); setErr(""); qc.invalidateQueries({ queryKey: ["loras"] }); }
    catch (error_) { setErr((error_ as Error).message); }
  };
  const del = async () => {
    if (!confirm(`Delete the combo ${name}? (stages that used it keep their LoRAs)`)) return;
    await api(`/lora-combos/${name}`, "DELETE"); qc.invalidateQueries({ queryKey: ["loras"] });
  };
  const without = (n: string) => Object.fromEntries(Object.entries(d.loras).filter(([k]) => k !== n));
  return (
    <div className="lrow">
      <div className="lhead">
        <b>{name}</b>
        {Object.entries(d.loras).map(([n, v]) => (
          <span key={n} className="pill" title={entries[n]?.note}>{n}{" "}
            <input type="number" step="0.1" value={v} style={{ width: 56 }} onChange={(x) => setD({ ...d, loras: { ...d.loras, [n]: Number(x.target.value) } })} onBlur={() => save(d)} />
            {Object.keys(d.loras).length > 1 && <button className="btn sm" title="take out of the combo" onClick={() => save({ ...d, loras: without(n) })}>✕</button>}
          </span>
        ))}
        <input placeholder="note" value={d.note ?? ""} onChange={(x) => setD({ ...d, note: x.target.value })} onBlur={() => save(d)} style={{ flex: 1 }} spellCheck={false} />
        <button className="btn sm danger" onClick={del}>delete</button>
      </div>
      <textarea rows={2} placeholder="tips (shown beside the motion/prompt while the combo is applied)" value={d.tips ?? ""}
        onChange={(x) => setD({ ...d, tips: x.target.value })} onBlur={() => save(d)} spellCheck={false} style={{ width: "100%" }} />
      {err && <div className="err">{err}</div>}
    </div>
  );
}

function DiscoveredRow({ f, image, onCreated }: { readonly f: Discovered; readonly image: Record<string, LoraEntry>; readonly onCreated: (n: string) => void }) {
  const qc = useQueryClient();
  const [pass, setPass] = useState<ImagePass>(f.pass ?? "krea");
  const [name, setName] = useState(f.name);
  const [into, setInto] = useState("");
  const [cat, setCat] = useState<LoraCategory>("concept");
  const [err, setErr] = useState("");
  const put = async (n: string, spec: LoraSpec) => {
    try { await api(`/loras/image/${n}`, "PUT", spec); qc.invalidateQueries({ queryKey: ["loras"] }); onCreated(n); }
    catch (error_) { setErr((error_ as Error).message); }
  };
  const register = () => put(name, { category: cat, [pass]: f.files[pass], ...(f.trigger ? { [`${pass}_trigger`]: f.trigger } : {}), suggested: 1, cloud: false });
  const attach = () => put(into, { ...specOf("image", image[into]), [pass]: f.files[pass] });
  const ignore = async () => { await api("/loras/ignore", "POST", { file: f.path, on: true }); qc.invalidateQueries({ queryKey: ["loras"] }); };
  const free = Object.keys(image).filter((n) => !image[n][pass]);
  return (
    <div className="ldisc">
      <code title={f.path}>{f.path}</code>
      <select value={pass} onChange={(x) => setPass(x.target.value as ImagePass)} title={f.pass ? "from the folder" : "root folder: pick the pass it belongs to"}>
        <option value="krea">Krea2</option><option value="il">Illustrious</option><option value="moss">MOSS</option>
      </select>
      <input value={name} onChange={(x) => setName(x.target.value)} spellCheck={false} style={{ width: 150 }} />
      <select value={cat} onChange={(x) => setCat(x.target.value as LoraCategory)}>{LORA_CATEGORIES.map((c) => <option key={c}>{c}</option>)}</select>
      <button className="btn sm primary" disabled={!NAME_RE.test(name) || name in image} title={name in image ? "name taken: use 'add to'" : ""} onClick={register}>register</button>
      <select value={into} onChange={(x) => setInto(x.target.value)} title={`add as the ${pass} file of an existing LoRA`}>
        <option value="">add to…</option>{free.map((n) => <option key={n}>{n}</option>)}
      </select>
      {into && <button className="btn sm" onClick={attach}>add</button>}
      <button className="btn sm" title="not a LoRA for these passes (ControlNet, Pony/SDXL, ...): stop listing it" onClick={ignore}>ignore</button>
      {err && <span className="err">{err}</span>}
    </div>
  );
}

export default function Loras({ hash }: { readonly hash: string }) {
  const qc = useQueryClient();
  const lib = useLoras();
  const m = /^\/loras(?:\/(image|video))?(?:\/(\w+))?/.exec(hash);
  const kind = (m?.[1] as LoraKind) ?? "image";
  const [open, setOpen] = useState<string | null>(m?.[2] ?? null);
  const [newName, setNewName] = useState("");
  const [newFile, setNewFile] = useState("");
  const [newCat, setNewCat] = useState<LoraCategory>("concept");
  const [err, setErr] = useState("");
  useEffect(() => { if (m?.[2]) { setOpen(m[2]); document.getElementById(`lora-${kind}-${m[2]}`)?.scrollIntoView({ block: "center" }); } }, [hash, lib.isSuccess]); // eslint-disable-line react-hooks/exhaustive-deps
  const d = lib.data;
  if (!d) return <div className="empty dim">{lib.isError ? String(lib.error) : "loading…"}</div>;
  const entries = d[kind];
  const create = async () => {
    try {
      await api(`/loras/${kind}/${newName}`, "PUT", { category: newCat, [kind === "image" ? "krea" : "file"]: newFile, suggested: 1, cloud: true });
      qc.invalidateQueries({ queryKey: ["loras"] }); setOpen(newName); setNewName(""); setNewFile(""); setErr("");
    } catch (error_) { setErr((error_ as Error).message); }
  };
  const fromFile = (f: string) => { setNewFile(f); setNewName(f.replace(/\.safetensors$/, "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 40)); };
  return (
    <div className="lpage">
      <div className="toolbar">
        <div className="seg">
          {(["image", "video"] as const).map((k) => <button key={k} className={kind === k ? "on" : ""} onClick={() => { location.hash = `#/loras/${k}`; setOpen(null); }}>
            {k === "image" ? "Image (Krea2 / IL)" : "Video (i2v)"} · {Object.keys(d[k]).length}</button>)}
        </div>
        <span className="vsep" />
        <input placeholder="new: name" value={newName} onChange={(x) => setNewName(x.target.value)} style={{ width: 130 }} spellCheck={false} />
        <input placeholder={kind === "image" ? "Krea2 file.safetensors (set IL in the form)" : "uploaded file.safetensors"} value={newFile}
          onChange={(x) => { if (!newName) fromFile(x.target.value); else setNewFile(x.target.value); }} style={{ width: 300 }} spellCheck={false} />
        <select value={newCat} onChange={(x) => setNewCat(x.target.value as LoraCategory)}>{LORA_CATEGORIES.map((c) => <option key={c}>{c}</option>)}</select>
        <button className="btn primary" disabled={!NAME_RE.test(newName) || newName in entries || !newFile.endsWith(".safetensors")} onClick={create}>+ add LoRA</button>
        {err && <span className="err">{err}</span>}
        <span className="spacer" />
        <button className="btn" title="re-read loras.json and the local ComfyUI's loras folder" onClick={() => lib.refetch()}>{lib.isFetching ? "…" : "rescan"}</button>
      </div>
      {byCategory(Object.entries(entries)).map(([cat, group]) => (
        <div key={cat} className="lpage">
          <div className="rhead dim">{cat} · {group.length}</div>
          {group.map(([n, e]) => <Row key={n} kind={kind} name={n} e={e} open={open === n} onToggle={() => setOpen(open === n ? null : n)} />)}
        </div>
      ))}
      {Object.entries(d.combos ?? {}).some(([, c]) => c.kind === kind) && (
        <div className="lpage">
          <div className="rhead dim">combos · tested LoRA sets, applied to a stage in one click (make one with "save as combo…" in the stage editor)</div>
          {Object.entries(d.combos).filter(([, c]) => c.kind === kind).map(([n, c]) => <ComboRow key={n} name={n} c={c} entries={entries} />)}
        </div>
      )}
      {kind === "video" && d.literal_video.length > 0 && (
        <div className="card lsec"><b>Used as raw file names in stages</b> <span className="dim">register to give them a name, note and tips</span>
          {d.literal_video.map((f) => <div key={f} className="ldisc"><code>{f}</code><button className="btn sm" onClick={() => fromFile(f)}>register…</button></div>)}
        </div>
      )}
      {kind === "image" && (
        <div className="card lsec">
          <b>In the local loras folder, not registered</b> <span className="dim">{d.local_up ? `${d.discovered.length} file(s); registered as local only until you tick "uploaded to Comfy Cloud"` : "local ComfyUI is down: can't list its files"}</span>
          {d.discovered.map((f) => <DiscoveredRow key={f.path} f={f} image={d.image} onCreated={(n) => setOpen(n)} />)}
          {d.ignore.length > 0 && <details><summary className="dim">ignored ({d.ignore.length})</summary>
            {d.ignore.map((f) => <div key={f} className="ldisc"><code>{f}</code>
              <button className="btn sm" onClick={async () => { await api("/loras/ignore", "POST", { file: f, on: false }); qc.invalidateQueries({ queryKey: ["loras"] }); }}>un-ignore</button></div>)}
          </details>}
        </div>
      )}
    </div>
  );
}
