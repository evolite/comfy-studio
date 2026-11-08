import { ReactNode, RefObject, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, byCategory, expandVoice, LoraEntry, LoraKind, Stage, VoiceCast, voiceSlotsIn, VoiceSlot, voiceUsesIn } from "./api";
import { useLoras, useProject, useRegistry, useVoices } from "./hooks";
import { SampleButton } from "./Voices";
import MusicPicker, { fitLabel, useFit, beatLabel } from "./Music";
import { RefChip, saveStateClass } from "./controls";

/** A typed strength; blank = back to the default when it has one, else forced off. */
function strengthOf(raw: string, e?: LoraEntry): number | null {
  if (raw !== "") return Number(raw);
  return e?.default == null ? 0 : null;
}

function comboTitle(c: { note?: string; loras: Record<string, number>; tips?: string }): string {
  const list = Object.entries(c.loras).map(([l, v]) => `${l} ${v}`).join(" · ");
  return `${c.note ?? ""}\n${list}${c.tips ? "\n\n" + c.tips : ""}`;
}

export default function StageDrawer({ project, stage, stages, onClose, onNav, inline }: {
  readonly project: string; readonly stage: Stage; readonly stages: Stage[]; readonly onClose: () => void; readonly onNav: (id: number) => void;
  readonly inline?: boolean;  // a one-off's editor: a plain panel in the page, no stage navigation or close
}) {
  const qc = useQueryClient();
  const lib = useLoras().data;
  const voiceLib = useVoices().data;
  const motionRef = useRef<HTMLTextAreaElement>(null);
  const projCfg = useProject(project).data?.config;
  const projHires = projCfg?.settings?.hires_prompt as string | undefined;  // the project-wide value, shown as the placeholder
  const reg = useRegistry().data;
  const engines = reg?.video_engines ?? {};
  const imgFields = reg?.image_lora_fields?.[projCfg?.engine ?? "full"];  // the LoRA files this project's image engine loads
  const [draft, setDraft] = useState<Stage>(stage);
  const [state, setState] = useState<"saved" | "saving" | "error">("saved");
  const [err, setErr] = useState("");
  const timer = useRef<number>(0);
  const pending = useRef<Record<string, unknown>>({});
  const engine = draft.video_engine ?? projCfg?.video_engine ?? "minimax";  // the i2v model this stage animates with
  const { fit, file } = useFit(project, engine);

  useEffect(() => { setDraft(stage); pending.current = {}; }, [stage.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const flush = async () => {
    const patch = pending.current; pending.current = {};
    if (!Object.keys(patch).length) return;
    setState("saving");
    try {
      await api(`/projects/${project}/stages/${stage.id}`, "PATCH", patch);
      setState("saved"); setErr("");
      qc.invalidateQueries({ queryKey: ["project", project] });
    } catch (e) { setState("error"); setErr(String((e as Error).message)); }
  };
  useEffect(() => () => { clearTimeout(timer.current); void flush(); }, [stage.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = (k: string, v: unknown) => {
    setDraft((d) => ({ ...d, [k]: v }));
    pending.current[k] = v; setState("saving");
    clearTimeout(timer.current); timer.current = window.setTimeout(flush, 700);
  };
  /** Copy a combo's LoRAs + strengths into the stage (on top of what's there). */
  const applyCombo = (field: "loras" | "video_loras", loras: Record<string, number>) =>
    set(field, { ...(draft[field] as Record<string, number> | undefined), ...loras });
  const saveCombo = async (kind: LoraKind, loras: Record<string, number>) => {
    const list = Object.entries(loras).map(([n, v]) => `${n} ${v}`).join(", ");
    const name = prompt(`Save these ${kind} LoRAs as a combo:\n${list}\n\nname (lowercase, digits, _):`)?.trim();
    if (!name) return;
    if (lib?.combos?.[name] && !confirm(`Replace the combo ${name}?`)) return;
    try { await api(`/lora-combos/${name}`, "PUT", { kind, loras, note: lib?.combos?.[name]?.note }); qc.invalidateQueries({ queryKey: ["loras"] }); }
    catch (e) { setErr((e as Error).message); }
  };
  const setLora = (field: "loras" | "video_loras", name: string, val: number | null) => {
    const cur = { ...(draft[field] as Record<string, number> | undefined) };
    if (val === null) delete cur[name]; else cur[name] = val;
    set(field, cur);
  };
  const idx = stages.findIndex((s) => s.id === stage.id);
  const text = (k: string, label: string, rows = 6, hint?: ReactNode, ref?: RefObject<HTMLTextAreaElement | null>) => (
    <div className="field">
      <label className="field" style={{ margin: 0 }}><span>{label}</span>
        <textarea ref={ref} rows={rows} value={(draft[k] as string) ?? ""} onChange={(e) => set(k, e.target.value)} spellCheck={false} />
      </label>
      {hint}
    </div>
  );
  const [info, setInfo] = useState<string | null>(null);  // "<kind>:<name>" of the LoRA whose info box is open
  /** The stage's LoRAs with their strength (0 = forced off); default-on ones (a `default` in loras.json) count as on unless set. */
  const activeOf = (kind: LoraKind) => {
    const val = (draft[kind === "image" ? "loras" : "video_loras"] as Record<string, number> | undefined) ?? {};
    const out: [string, number | undefined, LoraEntry | undefined][] = [];
    for (const [n, e] of Object.entries(lib?.[kind] ?? {})) if (n in val || e.default != null) out.push([n, val[n], e]);
    for (const n of Object.keys(val)) if (!lib?.[kind]?.[n]) out.push([n, val[n], undefined]);  // literal .safetensors / unknown names
    return out;
  };
  const isOn = (v: number | undefined, e?: LoraEntry) => (v ?? e?.default ?? 0) !== 0;
  /** Where a LoRA can run, for its chip outline: the local file's presence (unknown while ComfyUI is down: assume local) + the cloud flag. */
  const where = (e: LoraEntry) => {
    const local = Object.values(e.local ?? {}).some((x) => x !== false) && Object.keys(e.local ?? {}).length > 0;
    if (e.cloud === false) return "local";
    return local ? "both" : "cloud";
  };
  const picker = (kind: LoraKind) => {
    const field = kind === "image" ? "loras" : "video_loras";
    const active = activeOf(kind);
    const shown = new Set(active.map(([n]) => n));
    const open = info?.startsWith(kind + ":") ? info.slice(kind.length + 1) : null;
    const o = open ? lib?.[kind]?.[open] : undefined;
    const localOnly = active.filter(([, v, e]) => e?.cloud === false && isOn(v, e)).map(([n]) => n);
    const forEngine = (e?: LoraEntry) => kind === "video" ? (e?.engine ?? "minimax") === engine
      : !imgFields || imgFields.some((f) => e?.[f]);
    const otherEngine = active.filter(([, v, e]) => e && isOn(v, e) && !forEngine(e)).map(([n]) => n);
    const cur = (draft[field] as Record<string, number> | undefined) ?? {};
    const combos = Object.entries(lib?.combos ?? {}).filter(([, c]) => c.kind === kind && Object.keys(c.loras).every((n) => forEngine(lib?.[kind]?.[n])));
    const applied = (c: Record<string, number>) => Object.entries(c).every(([n, v]) => cur[n] === v);
    const savable = Object.fromEntries(Object.entries(cur).filter(([n, v]) => v !== 0 && lib?.[kind]?.[n]));
    return (
      <>
        {active.map(([n, v, e]) => (
          <div key={n} className={`larow ${isOn(v, e) ? "" : "off"}`}>
            <span className="lname" title={e?.note}>{n}</span>
            {e?.min != null && e.max != null && (
              <input type="range" className="lslider" min={e.min} max={e.max} step="0.05" value={v ?? e.default ?? 0}
                title={`${e.min} … ${e.max}`} onChange={(x) => setLora(field, n, Number(x.target.value))} />
            )}
            <input type="number" step="0.1" value={v ?? ""} placeholder={e?.default != null ? `default ${e.default}` : ""}
              onChange={(x) => setLora(field, n, strengthOf(x.target.value, e))} />
            {e?.cloud === false && <span className="pill local" title="local only: cloud runs of this stage are refused">local</span>}
            <span className="dim lnote">{e?.note ?? (e ? "" : "not in the library")}</span>
            {e && <button type="button" className={`loinfo ${open === n ? "on" : ""}`} title="note, tips, source" onClick={() => setInfo(open === n ? null : `${kind}:${n}`)}>ⓘ</button>}
            {e?.default != null
              ? <button type="button" className="btn sm" title={isOn(v, e) ? "turn off for this stage" : "back to the default"} onClick={() => setLora(field, n, isOn(v, e) ? 0 : null)}>{isOn(v, e) ? "off" : "on"}</button>
              : <button type="button" className="btn sm" title="remove from this stage" onClick={() => setLora(field, n, null)}>✕</button>}
          </div>
        ))}
        {(combos.length > 0 || Object.keys(savable).length > 1) && (
          <div className="lchips"><span className="dim lcat">combos</span>
            {combos.map(([n, c]) => (
              <button key={n} type="button" className={`chip-f ${applied(c.loras) ? "on" : ""}`} disabled={applied(c.loras)}
                title={comboTitle(c)}
                onClick={() => applyCombo(field, c.loras)}>{applied(c.loras) ? "✓" : "+"} {n}<small>{Object.keys(c.loras).length}</small></button>
            ))}
            {Object.keys(savable).length > 1 && <button type="button" className="btn sm" title="save this stage's LoRAs + strengths as a named combo" onClick={() => saveCombo(kind, savable)}>save as combo…</button>}
          </div>
        )}
        {byCategory(Object.entries(lib?.[kind] ?? {}).filter(([n, e]) => !shown.has(n) && !e.disabled && forEngine(e))).map(([cat, group]) => (
          <div key={cat} className="lchips"><span className="dim lcat">{cat}</span>
            {group.map(([n, e]) => (
              <button key={n} type="button" className={`chip-f lane-${where(e)}`}
                title={`${e.note ?? ""}${e.cloud === false ? "\nlocal only: cloud runs of this stage would be refused" : ""}`}
                onClick={() => setLora(field, n, e.suggested ?? 1)}>+ {n}<small>{e.suggested ?? 1}</small></button>
            ))}
          </div>
        ))}
        <a className="dim" href={`#/loras/${kind}`} title="add a LoRA, edit notes and tips">manage LoRAs…</a>
        {otherEngine.length > 0 && (kind === "video"
          ? <div className="warnline">⚠ {otherEngine.join(", ")}: made for another i2v model than {engines[engine] ?? engine}; animate refuses it (set it to 0 or remove it)</div>
          : <div className="warnline">⚠ {otherEngine.join(", ")}: no file for the {projCfg?.engine} engine's model, so it does nothing here (remove it)</div>)}
        {localOnly.length > 0 && <div className="warnline">⚠ {localOnly.join(", ")} {localOnly.length > 1 ? "are" : "is"} local only: cloud runs of this stage are refused (use --local / local preview, or turn {localOnly.length > 1 ? "them" : "it"} off)</div>}
        {open && o && <div className="lorabox"><b>{open}</b>{[o.krea, o.il, o.moss, o.file].filter(Boolean).map((f) => <code key={f}>{f}</code>)}
          <div>{o.note || "no note yet"}</div>
          {o.tips && <div className="ltips">{o.tips}</div>}
          <div className="row">{o.url ? <a href={o.url} target="_blank" rel="noreferrer">source ↗</a> : <span className="dim">no source link</span>}
            <a href={`#/loras/${kind}/${open}`}>edit in the library</a></div></div>}
      </>
    );
  };
  /** Triggers that get appended to a prompt pass while its LoRA is on, so they needn't be typed. */
  const triggers = (key: "krea_trigger" | "il_trigger") => activeOf("image").filter(([, v, e]) => isOn(v, e) && e?.[key]).map(([n, , e]) => `${n}: ${e![key]}`);
  const trigHint = (key: "krea_trigger" | "il_trigger") => {
    const t = triggers(key);
    return t.length > 0 && <span className="dim">auto-appended: {t.join(" · ")}</span>;
  };
  const motionCombos = Object.entries(lib?.combos ?? {}).filter(([, c]) => c.kind === "video" && c.tips
    && Object.entries(c.loras).every(([n, v]) => draft.video_loras?.[n] === v));
  const motionTips = activeOf("video").filter(([, v, e]) => isOn(v, e) && (e?.tips || e?.note));
  const vtrig = activeOf("video").filter(([, v, e]) => isOn(v, e) && e?.trigger).map(([n, , e]) => `${n}: ${e!.trigger}`);
  // the voice library: the stage's voice (else the project's) fills the motion's {voice...} slots when it animates
  const voiceName = draft.voice || projCfg?.voice || "";
  const voice = voiceName ? voiceLib?.voices[voiceName] : undefined;
  const motion = draft.motion ?? "";
  const usedSlots = voiceSlotsIn(motion);
  // other speakers' voices ({"S2": name}, stage else project); a speaker without one uses S1's
  const castNames: Record<string, string> = { ...projCfg?.voices, ...draft.voices, ...(voiceName ? { S1: voiceName } : undefined) };
  const cast: VoiceCast = Object.fromEntries(Object.entries(castNames).map(([sp, n]) => [sp, voiceLib?.voices[n]]));
  const s2Name = castNames.S2 ?? "";
  const nameOf = (sp: string) => castNames[sp] ?? voiceName;
  const voiceGaps = [...new Set(voiceUsesIn(motion).filter(({ slot, speaker }) => !(cast[speaker] ?? cast.S1)?.[slot]?.trim())
    .map(({ slot, speaker }) => `${nameOf(speaker) || speaker}: ${slot}`))];
  const setS2 = (n: string) => {
    const rest = Object.fromEntries(Object.entries(draft.voices ?? {}).filter(([k]) => k !== "S2"));
    const next = n ? { ...rest, S2: n } : rest;
    set("voices", Object.keys(next).length ? next : null);
  };
  const insertSlot = (k: VoiceSlot) => {
    const t = motionRef.current, ph = k === "voice" ? "{voice}" : `{voice.${k}}`;
    const at = t?.selectionStart ?? motion.length, end = t?.selectionEnd ?? at;
    set("motion", motion.slice(0, at) + ph + motion.slice(end));
    requestAnimationFrame(() => { t?.focus(); t?.setSelectionRange(at + ph.length, at + ph.length); });
  };
  const voiceHint = (
    <div className="lhints">
      <span className="dim">voice slots:</span>
      {(["voice", "delivery", "moans", "sound"] as const).map((k) => (
        <button key={k} type="button" className={`chip-f ${usedSlots.includes(k) ? "on" : ""}`} title={voice?.[k] ? `${voiceName}: ${voice[k]}` : "insert at the cursor"} onClick={() => insertSlot(k)}>
          {k === "voice" ? "{voice}" : `{voice.${k}}`}</button>
      ))}
      {usedSlots.length > 0 && !voiceName && <div className="warnline">⚠ the motion uses voice slots but no voice is set (pick one below, or the project's in ⚙ Settings → Animate); animate refuses it</div>}
      {voiceName && !voice && voiceLib && <div className="warnline">⚠ no voice {voiceName} in the library</div>}
      {voice && voiceGaps.length > 0 && <div className="warnline">⚠ missing voice slots ({voiceGaps.join(", ")}): fill them in the voice library or drop the slot</div>}
      {voice && usedSlots.length > 0 && voiceGaps.length === 0 && <details><summary className="dim">as animated ({voice.label || voiceName}{s2Name ? ` + S2 ${voiceLib?.voices[s2Name]?.label ?? s2Name}` : ""})</summary><div className="ltips">{expandVoice(motion, cast)}</div></details>}
    </div>
  );
  const motionHint = (motionTips.length > 0 || vtrig.length > 0 || motionCombos.length > 0) && (
    <div className="lhints">{vtrig.length > 0 && <span className="dim">auto-prepended: {vtrig.join(" · ")}</span>}{motionCombos.map(([n, c]) => (
      <details key={"combo:" + n}><summary><b>{n}</b> <span className="dim">combo · {c.note}</span></summary><div className="ltips">{c.tips}</div></details>
    ))}{motionTips.map(([n, , e]) => (
      <details key={n}><summary><b>{n}</b> <span className="dim">{e!.note}</span></summary>{e!.tips && <div className="ltips">{e!.tips}</div>}</details>
    ))}</div>
  );

  return (
    <aside className={inline ? "drawer inline" : "drawer"}>
      <div className="row">
        {!inline && <button className="btn sm" disabled={idx <= 0} onClick={() => onNav(stages[idx - 1].id)}>←</button>}
        <b>{inline ? "Prompts" : "Stage"}</b>
        <RefChip project={project} stage={stage.id} label={inline ? "@" : undefined} />
        {!inline && <button className="btn sm" disabled={idx >= stages.length - 1} onClick={() => onNav(stages[idx + 1].id)}>→</button>}
        <span className={`pill ${saveStateClass(state)}`}>{state}</span>
        <span className="spacer" />
        {!inline && <button className="btn sm" onClick={onClose}>close</button>}
      </div>
      {err && <div className="err">{err}</div>}
      <label className="chk" title="i2v also generates a soundtrack for the clip (voice, sound/ambience from the motion prompt)">
        <input type="checkbox" checked={!!draft.audio} onChange={(e) => set("audio", e.target.checked)} /> 🔊 audio
      </label>
      {text("krea_scene", "krea_scene (full Krea2 prompt, restate clothing state)", 16, trigHint("krea_trigger"))}
      {text("il_scene", "il_scene (verified Danbooru tags)", 8, trigHint("il_trigger"))}
      {text("il_background", "il_background (bgswap only)", 2)}
      {text("motion", "motion", 5, <>{voiceHint}{motionHint}</>, motionRef)}
      {!inline && <MusicPicker name={project} />}
      <div className="row">
        <label className="field"><span>duration</span>
          {inline ? (
            <select value={String(draft.duration ?? 5)} onChange={(e) => set("duration", Number(e.target.value))}>
              {Array.from({ length: 10 }, (_, i) => <option key={i} value={i + 1}>{i + 1}s</option>)}
            </select>
          ) : <select value={String(draft.duration ?? "")} onChange={(e) => set("duration", e.target.value ? Number(e.target.value) : null)}>
            <option value="">default</option>
            <optgroup label={`fits ${beatLabel(file)} (★ = best)`}>
              {fit?.options.map((o) => <option key={o.frames} value={o.request}>{fitLabel(o)}</option>)}
            </optgroup>
            {draft.duration != null && !fit?.options.some((o) => o.request === draft.duration) && <option value={String(draft.duration)}>{String(draft.duration)}s (custom)</option>}
          </select>}
        </label>
      </div>
      <div className="dsec">Image</div>
      <div className="field">{picker("image")}</div>
      <div className="row">
        <label className="field"><span>seed</span>
          <input type="number" value={draft.seed ?? ""} onChange={(e) => set("seed", e.target.value ? Number(e.target.value) : null)} />
        </label>
      </div>
      <details className="fold" title="per-stage overrides; blank = project setting">
        <summary>upscale + face overrides</summary>
        <div className="row">
        <label className="field"><span>il_denoise</span>
          <input type="number" step="0.05" min="0" max="1" value={(draft.il_denoise as number | undefined) ?? ""} onChange={(e) => set("il_denoise", e.target.value ? Number(e.target.value) : null)} />
        </label>
        <label className="field"><span>face_wildcard</span>
          <input type="text" value={(draft.face_wildcard as string) ?? ""} onChange={(e) => set("face_wildcard", e.target.value)} spellCheck={false} />
        </label>
        </div>
      <label className="field"><span>hires_prompt</span>
        <input type="text" value={(draft.hires_prompt as string) ?? ""} placeholder={projHires ? `project: ${projHires}` : "e.g. smooth flawless skin, soft even skin texture"}
          onChange={(e) => set("hires_prompt", e.target.value)} spellCheck={false} />
      </label>
      <label className="chk" title="IL FaceDetailer pass on this stage (full engine)">
        <input type="checkbox" checked={!!draft.face_detailer_enabled} onChange={(e) => set("face_detailer_enabled", e.target.checked)} /> face detailer
      </label>
      </details>
      <div className="dsec">Video</div>
      <div className="field">{picker("video")}</div>
      <div className="row">
        {!inline && <label className="field"><span>i2v model</span>
          <select value={draft.video_engine ?? ""} onChange={(e) => set("video_engine", e.target.value || null)}>
            <option value="">project ({engines[projCfg?.video_engine ?? "minimax"] ?? "minimax"})</option>
            {Object.entries(engines).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
          </select>
        </label>}
        <label className="field"><span>video_seed</span>
          <input type="number" value={draft.video_seed ?? ""} onChange={(e) => set("video_seed", e.target.value ? Number(e.target.value) : null)} />
        </label>
      </div>
      <div className="row">
        <label className="field"><span>voice</span>
          <select value={draft.voice ?? ""} onChange={(e) => set("voice", e.target.value || null)}>
            <option value="">{projCfg?.voice ? `project (${voiceLib?.voices[projCfg.voice]?.label ?? projCfg.voice})` : "none"}</option>
            {Object.entries(voiceLib?.voices ?? {}).filter(([k, v]) => !v.disabled || k === draft.voice).map(([k, v]) => <option key={k} value={k}>{v.label || k}</option>)}
          </select>
        </label>
        <label className="field" title="a second speaker's voice: fills the motion's {voice...:S2} slots (empty = S1's voice)"><span>S2 voice</span>
          <select value={draft.voices?.S2 ?? ""} onChange={(e) => setS2(e.target.value)}>
            <option value="">{projCfg?.voices?.S2 ? `project (${voiceLib?.voices[projCfg.voices.S2]?.label ?? projCfg.voices.S2})` : "same as voice"}</option>
            {Object.entries(voiceLib?.voices ?? {}).filter(([k, v]) => !v.disabled || k === draft.voices?.S2).map(([k, v]) => <option key={k} value={k}>{v.label || k}</option>)}
          </select>
        </label>
        {voice && <SampleButton e={voice} />}
        <a className="dim" href={`#/voices/${voiceName}`} title="add a voice, edit its slots">manage voices…</a>
      </div>
    </aside>
  );
}
