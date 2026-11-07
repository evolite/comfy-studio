import { ChangeEvent, CSSProperties, ReactNode, useEffect, useState } from "react";
import { Ctl, Lora, Val } from "./settingsSchema";
import { Icon } from "./icons";

/** A number box that lets "0." / "0.0" survive while typing; commits only finished numbers. */
export function NumBox({ value, onChange, step, min, max }: { readonly value: number; readonly onChange: (v: number) => void; readonly step?: number; readonly min?: number; readonly max?: number }) {
  const [raw, setRaw] = useState(String(value));
  useEffect(() => { if (Number(raw) !== value) setRaw(String(value)); }, [value]);  // eslint-disable-line react-hooks/exhaustive-deps
  return <input className="numbox" inputMode="decimal" value={raw} step={step} min={min} max={max}
    onChange={(e) => { setRaw(e.target.value); const n = Number(e.target.value); if (e.target.value.trim() !== "" && Number.isFinite(n)) onChange(n); }} />;
}

function Slider({ c, v, set }: { readonly c: Ctl; readonly v: number; readonly set: (v: number) => void }) {
  const min = c.min ?? 0, max = c.max ?? 1;
  const pct = (x: number) => `${((Math.min(max, Math.max(min, x)) - min) / (max - min)) * 100}%`;
  return (
    <div className="slider">
      <div className="track">
        {c.hint && <span className="hint" style={{ left: pct(c.hint[0]), width: `calc(${pct(c.hint[1])} - ${pct(c.hint[0])})` }} title={`sweet spot ${c.hint[0]}–${c.hint[1]}`} />}
        <input type="range" min={min} max={max} step={c.step ?? 0.01} value={v} style={{ "--fill": pct(v) } as CSSProperties} onChange={(e) => set(Number(e.target.value))} />
      </div>
      <NumBox value={v} onChange={set} step={c.step} />
      {c.unit && <span className="unit">{c.unit}</span>}
    </div>
  );
}

/** Small box drawn in the ratio's shape, for the aspect picker. */
const Glyph = ({ aspect }: { readonly aspect: string }) => {
  const m = /^(\d+):(\d+)/.exec(aspect); const r = m ? Number(m[1]) / Number(m[2]) : 1;
  const w = r >= 1 ? 18 : 18 * r, h = r >= 1 ? 18 / r : 18;
  return <span className="glyph" style={{ width: w, height: h }} />;
};

export type FieldProps = {
  readonly c: Ctl; readonly value: Val | Lora | undefined; readonly def: Val | Lora | undefined; readonly set: (v: Val | Lora) => void; readonly reset: () => void;
  readonly choices?: string[]; readonly localOnly?: Set<string>; readonly extra?: ReactNode;
};

/** A setting's value as control text; objects (a LoRA) have no text form. */
const textOf = (v: FieldProps["value"]) => (typeof v === "object" ? "" : String(v ?? ""));

function resetTip(def: FieldProps["def"]): string {
  if (def === undefined) return "unset";
  return `back to the default (${typeof def === "object" ? "none" : String(def)})`;
}

/** The status pill's colour for an autosaving form. */
export function saveStateClass(state: string): string {
  if (state === "error") return "bad";
  return state === "saved" ? "ok" : "";
}

/** One setting: label, control, a dimmed default when unset (with a "default" tag), ↺ when set, and the help line. */
export function Field({ c, value, def, set, reset, choices, localOnly, extra }: FieldProps) {
  const isSet = value !== undefined;
  const v = (isSet ? value : def) as Val | Lora | undefined;
  let ctl: ReactNode;
  if (c.kind === "slider") ctl = <Slider c={c} v={Number(v ?? c.min ?? 0)} set={set} />;
  else if (c.kind === "seg") ctl = (
    <div className="seg">{(c.options ?? []).map((o) => <button key={o} className={v === o ? "on" : ""} onClick={() => set(o)}>{o}</button>)}</div>);
  else if (c.kind === "aspect") ctl = (
    <div className="aspects">{(c.options ?? []).map((o) => (
      <button key={o} className={`aspect ${v === o ? "on" : ""}`} title={o} onClick={() => set(o)}><Glyph aspect={o} /><small>{o.split(" ")[0]}</small></button>))}
    </div>);
  else if (c.kind === "toggle") {
    const onVal = c.on ?? true;
    const on = v === onVal;
    const onSwitch = (e: ChangeEvent<HTMLInputElement>) => {
      if (e.target.checked) set(onVal);
      else if (c.on === undefined) set(false);
      else reset();
    };
    ctl = <label className="switch"><input type="checkbox" aria-label={c.label} checked={on} onChange={onSwitch} /><span /></label>;
  } else if (c.kind === "select") {
    const opts = [...(choices ?? c.options ?? [])];
    if (typeof v === "string" && v && !opts.includes(v)) opts.push(v);
    ctl = <select value={textOf(v)} onChange={(e) => set(e.target.value)}>
      {v === undefined && <option value="">—</option>}
      {opts.map((o) => <option key={o} value={o}>{o}{localOnly?.has(o) ? "  (local only)" : ""}</option>)}
    </select>;
  } else ctl = <input className="text" value={textOf(v)} onChange={(e) => set(e.target.value)} spellCheck={false} />;
  return (
    <div className={`ctl ${isSet ? "isset" : "isdef"}`}>
      <div className="ctlhead">
        <span className="ctllabel">{c.label}</span>
        {c.help && <span className="hint" title={c.help}><Icon n="info" size={12} /></span>}
        {!isSet && def !== undefined && <span className="deftag">default</span>}
        {extra && <span className="ctlextra">{extra}</span>}
        <span className="spacer" />
        {isSet && <button className="reset" title={resetTip(def)} onClick={reset}>↺</button>}
      </div>
      {ctl}
    </div>
  );
}

/** "@<project>" or "@<project>#<id>": what you paste to Claude to point at a project or stage (an agent resolves it with `project.py <name> show`). */
export const refOf = (project: string, stage?: number) => (stage == null ? `@${project}` : `@${project}#${stage}`);

/** Copy a ref to the clipboard; the promise says how it went, for the caller's feedback. */
export const copyRef = (ref: string) => navigator.clipboard.writeText(ref).then(() => true, () => false);

/** The one chip every project/stage/voice reference uses (header, project cards, stage cards, editor, lightbox, voices): click = copy
 *  the ref. `value` = a ready-made ref (e.g. `@voice:<name>`) instead of a project/stage one. */
export function RefChip({ project = "", stage, label, value }: { readonly project?: string; readonly stage?: number; readonly label?: string; readonly value?: string }) {
  const [state, setState] = useState<"" | "ok" | "fail">("");
  const ref = value ?? refOf(project, stage);
  let body: ReactNode;
  if (state === "ok") body = <><Icon n="check" size={12} /> copied</>;
  else if (state === "fail") body = "clipboard unavailable";
  else body = <><Icon n="at" size={12} />{label ?? (stage != null ? `#${stage}` : null)}</>;
  useEffect(() => {
    if (!state) return;
    const t = setTimeout(() => setState(""), 1200);
    return () => clearTimeout(t);
  }, [state]);
  return (
    <button type="button" className={`refchip ${state}`} title={`${ref}\nclick to copy for Agent reference${stage != null ? " (c on the current stage)" : ""}`}
      onClick={(e) => { e.preventDefault(); e.stopPropagation(); void copyRef(ref).then((ok) => setState(ok ? "ok" : "fail")); }}>
      {body}
    </button>
  );
}
