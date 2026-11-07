import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, VoiceEntry, VoiceSlot } from "./api";
import { RefChip, saveStateClass } from "./controls";
import { useAnimateTarget, useVoices } from "./hooks";
import { useBatchArmed } from "./runPrefs";

const NAME_RE = /^[a-z0-9_]+$/;
/** What each slot is and where it goes in a MiniMax H3 motion (pipeline/voice_registry.py). */
const SLOT_HELP: Record<VoiceSlot, string> = {
  voice: 'the voice itself — "Her voice (S1) is {voice}."',
  delivery: 'verb + how a line is said, (S) = the speaker tag — "... and {voice.delivery}, <d>[English] ...</d>"',
  moans: 'verb + her moans, (S) = the speaker tag — "She {voice.moans}, at the same pace ..."',
  sound: 'the moans as heard in overall_soundscape — "... and her {voice.sound}, over ..."',
};
const SLOTS: VoiceSlot[] = ["voice", "delivery", "moans", "sound"];

export const sampleUrl = (e: VoiceEntry) => e.sample && e.sample_mtime
  ? `/api/projects/${e.sample.project}/files/${e.sample.folder ? `${e.sample.folder}/` : ""}videos/stage_${e.sample.stage}.mp4?v=${e.sample_mtime}` : null;

/** Plays the voice's sample clip (sound on); a second click stops it. */
export function SampleButton({ e }: { readonly e: VoiceEntry }) {
  const url = sampleUrl(e);
  const [playing, setPlaying] = useState<HTMLAudioElement | null>(null);
  if (!url) return <span className="dim" title={e.sample ? `no clip yet at ${e.sample.project} #${e.sample.stage}` : "no sample set"}>no sample</span>;
  const toggle = () => {
    if (playing) { playing.pause(); setPlaying(null); return; }
    const a = new Audio(url);
    a.onended = () => setPlaying(null);
    void a.play();
    setPlaying(a);
  };
  return <button type="button" className="btn sm" title={`listen: ${e.sample!.project} #${e.sample!.stage}`} onClick={toggle}>{playing ? "■" : "▶"} sample</button>;
}

function Editor({ name, e, onDone }: { readonly name: string; readonly e: VoiceEntry; readonly onDone: () => void }) {
  const qc = useQueryClient();
  const [d, setD] = useState<Record<string, string>>(() => ({
    label: e.label ?? "", tags: (e.tags ?? []).join(", "), note: e.note ?? "", tips: e.tips ?? "",
    voice: e.voice ?? "", delivery: e.delivery ?? "", moans: e.moans ?? "", sound: e.sound ?? "",
    sproject: e.sample?.project ?? "", sstage: e.sample ? String(e.sample.stage) : "", sfolder: e.sample?.folder ?? "",
  }));
  const [state, setState] = useState("saved");
  const [err, setErr] = useState("");
  const set = (k: string, v: string) => { setD({ ...d, [k]: v }); setState("unsaved"); };
  const save = async () => {
    const spec = {
      label: d.label, note: d.note, tips: d.tips, voice: d.voice, delivery: d.delivery, moans: d.moans, sound: d.sound,
      tags: d.tags.split(",").map((t) => t.trim()).filter(Boolean),
      disabled: e.disabled || undefined,
      sample: d.sproject && d.sstage ? { project: d.sproject, stage: Number(d.sstage), folder: d.sfolder } : undefined,
    };
    try { await api(`/voices/${name}`, "PUT", spec); setState("saved"); setErr(""); void qc.invalidateQueries({ queryKey: ["voices"] }); }
    catch (error_) { setState("error"); setErr((error_ as Error).message); }
  };
  const remove = async () => {
    if (!confirm(`Delete the voice ${name}?`)) return;
    try { await api(`/voices/${name}`, "DELETE"); void qc.invalidateQueries({ queryKey: ["voices"] }); onDone(); }
    catch (error_) { setErr((error_ as Error).message); }
  };
  const line = (k: string, label: string, placeholder = "") => (
    <label className="field"><span>{label}</span><input value={d[k]} placeholder={placeholder} onChange={(x) => set(k, x.target.value)} spellCheck={false} /></label>
  );
  return (
    <div className="lform">
      <div className="row wrap">{line("label", "label")}{line("tags", "tags", "accent, french")}</div>
      {SLOTS.map((k) => (
        <label key={k} className="field"><span>{k === "voice" ? "{voice}" : `{voice.${k}}`} <small className="dim">{SLOT_HELP[k]}</small></span>
          <textarea rows={2} value={d[k]} onChange={(x) => set(k, x.target.value)} spellCheck={false} /></label>
      ))}
      {line("note", "note")}
      <label className="field"><span>tips</span><textarea rows={2} value={d.tips} onChange={(x) => set("tips", x.target.value)} /></label>
      <div className="row wrap">{line("sproject", "sample project")}{line("sstage", "stage")}{line("sfolder", "clip folder", "interp / hires / blank = root")}</div>
      <div className="row">
        <button className="btn primary" disabled={!d.voice.trim()} onClick={() => void save()}>save</button>
        <span className={`pill ${saveStateClass(state)}`}>{state}</span>
        {err && <span className="err">{err}</span>}
        <span className="spacer" />
        <button className="btn danger" disabled={e.used.length > 0} title={e.used.length ? "in use by stages: pick another voice there first" : "delete"} onClick={() => void remove()}>delete</button>
      </div>
    </div>
  );
}

/** Render this voice's sample: (re)make its stage in the samples project, then Animate smooth it (queued into the batch
 *  when "Batch" is armed on RunPod, like an Animate click on the board). Paid: one clip. */
function MakeSample({ name, e }: { readonly name: string; readonly e: VoiceEntry }) {
  const qc = useQueryClient();
  const armed = useBatchArmed();
  const target = useAnimateTarget();  // both hooks run every render (a skipped hook crashes the page)
  const batch = armed && target === "runpod";
  const [msg, setMsg] = useState("");
  const go = async () => {
    if (!confirm(`Render a sample clip for ${e.label || name}? (one paid Animate smooth${batch ? ", queued into the batch" : ""})`)) return;
    try {
      const t = await api<{ project: string; stage: number; folder: string }>(`/voices/${name}/sample`, "POST");
      const req = { action: "animate_rife", stages: [t.stage], folder: t.folder };
      if (batch) {
        const r = await api<{ job: unknown; pending: number; size: number }>(`/projects/${t.project}/animate-batch`, "POST", req);
        setMsg(r.job ? "sent with the batch" : `batch ${r.pending}/${r.size}`);
        void qc.invalidateQueries({ queryKey: ["animate-batch"] });
      } else {
        await api(`/projects/${t.project}/run`, "POST", req);
        setMsg(`rendering (${t.project} #${t.stage})`);
      }
      void qc.invalidateQueries({ queryKey: ["jobs"] });
      void qc.invalidateQueries({ queryKey: ["voices"] });
    } catch (error_) { setMsg(`error: ${(error_ as Error).message}`); }
  };
  return <>
    <button type="button" className="btn sm" title="render this voice saying the sample line (the samples project's first stage) as its sample" onClick={() => void go()}>
      {e.sample_mtime ? "re-make sample" : "make sample"}</button>
    {msg && <span className="dim">{msg}</span>}
  </>;
}

/** The entry as saved (what the API returns minus its computed fields). */
const specOf = (e: VoiceEntry) => Object.fromEntries(Object.entries(e).filter(([k]) => k !== "used" && k !== "sample_mtime"));

function Row({ name, e, open, onToggle }: { readonly name: string; readonly e: VoiceEntry; readonly open: boolean; readonly onToggle: () => void }) {
  const qc = useQueryClient();
  const projects = [...new Set(e.used.map((u) => u.project))];
  const flip = () => void api(`/voices/${name}`, "PUT", { ...specOf(e), disabled: e.disabled ? undefined : true }).then(() => qc.invalidateQueries({ queryKey: ["voices"] }));
  return (
    <div className={`lrow ${open ? "on" : ""} ${e.disabled ? "loff" : ""}`}>
      <div className="lhead">
        <button className={`btn sm ${e.disabled ? "" : "primary"}`} onClick={flip}
          title={e.disabled ? "off: the voice pickers don't offer it; stages already using it keep it. Click to turn on" : "on: offered in the voice pickers. Click to turn off"}>
          {e.disabled ? "off" : "on"}</button>
        <button type="button" className="namebtn" aria-expanded={open} onClick={onToggle}><b>{e.label || name}</b></button>
        <RefChip value={`@voice:${name}`} label={name} />
        {(e.tags ?? []).map((t) => <span key={t} className="pill">{t}</span>)}
        <SampleButton e={e} />
        <MakeSample name={name} e={e} />
        <span className="lnote dim">{e.note || e.voice}</span>
        {e.used.length > 0 && <span className="dim" title={e.used.map((u) => `${u.project} #${u.stage}`).join("\n")}>used by {e.used.length} stage{e.used.length > 1 ? "s" : ""} in {projects.length} project{projects.length > 1 ? "s" : ""}</span>}
      </div>
      {open && <Editor key={name} name={name} e={e} onDone={onToggle} />}
    </div>
  );
}

/** The voice library (voices.json): named voices whose slots fill a stage's {voice...} placeholders, each with a clip to listen to. */
export default function Voices({ hash }: { readonly hash: string }) {
  const qc = useQueryClient();
  const lib = useVoices();
  const m = /^\/voices(?:\/(\w+))?/.exec(hash);
  const [open, setOpen] = useState<string | null>(m?.[1] ?? null);
  const [q, setQ] = useState("");
  const [newName, setNewName] = useState("");
  const [err, setErr] = useState("");
  const voices = lib.data?.voices ?? {};
  const create = async () => {
    try {
      await api(`/voices/${newName}`, "PUT", { voice: "describe the voice", label: newName.replaceAll("_", " ") });
      void qc.invalidateQueries({ queryKey: ["voices"] }); setOpen(newName); setNewName(""); setErr("");
    } catch (error_) { setErr((error_ as Error).message); }
  };
  const [hideOff, setHideOff] = useState(() => { try { return localStorage.getItem("voices:hideOff") === "1"; } catch { return false; } });
  const pickHideOff = (v: boolean) => { setHideOff(v); try { localStorage.setItem("voices:hideOff", v ? "1" : "0"); } catch { /* storage unavailable */ } };
  const nOff = Object.values(voices).filter((e) => e.disabled).length;
  const ql = q.toLowerCase();
  const rows = Object.entries(voices).filter(([n, e]) => !(hideOff && e.disabled) && (!ql || [n, e.label, e.voice, ...(e.tags ?? [])].some((x) => x?.toLowerCase().includes(ql))));
  const tags = [...new Set(Object.values(voices).flatMap((e) => e.tags ?? []))].sort((a, b) => a.localeCompare(b));
  return (
    <div className="lpage">
      <div className="toolbar">
        <b>Voices · {Object.keys(voices).length}</b>
        <span className="vsep" />
        <input placeholder="filter: name, tag, words…" value={q} onChange={(x) => setQ(x.target.value)} style={{ width: 200 }} spellCheck={false} />
        <span className="seg">{tags.slice(0, 12).map((t) => <button key={t} className={q === t ? "on" : ""} onClick={() => setQ(q === t ? "" : t)}>{t}</button>)}</span>
        <label className="chk" title="leave the voices turned off out of this list"><input type="checkbox" checked={hideOff} onChange={(x) => pickHideOff(x.target.checked)} /> hide off{nOff ? ` (${nOff})` : ""}</label>
        <span className="spacer" />
        <input placeholder="new: name" value={newName} onChange={(x) => setNewName(x.target.value)} style={{ width: 150 }} spellCheck={false} />
        <button className="btn primary" disabled={!NAME_RE.test(newName) || newName in voices} onClick={() => void create()}>+ add voice</button>
        {err && <span className="err">{err}</span>}
      </div>
      <div className="dim">A stage's motion uses <code>{"{voice}"}</code>, <code>{"{voice.delivery}"}</code>, <code>{"{voice.moans}"}</code>, <code>{"{voice.sound}"}</code>; the stage's voice (or the project's, ⚙ Settings → Animate) fills them when it animates.</div>
      {rows.map(([n, e]) => <Row key={n} name={n} e={e} open={open === n} onToggle={() => setOpen(open === n ? null : n)} />)}
      {lib.isSuccess && rows.length === 0 && <div className="dim">no voice matches</div>}
    </div>
  );
}
