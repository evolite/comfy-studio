import { Fragment, ReactNode, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, Registry } from "./api";
import { useProject, useRegistry, useVoices } from "./hooks";
import { SampleButton } from "./Voices";
import { ROWS } from "./Board";
import { BOARD_COLS, CLIP_VOLUME_DEFAULT, setActionHidden, setBoardCols, setClipVolume, useBoardCols, useClipVolume, useHiddenActions } from "./runPrefs";
import MusicPicker, { FitTable } from "./Music";
import { Field } from "./controls";
import { Icon } from "./icons";
import { Ctl, Lora, resolution, SCHEMA_KEYS, SECTIONS, Val } from "./settingsSchema";

type Settings = Record<string, Val | Lora>;
type Status = { s: "idle" | "saving" | "saved" | "error"; msg?: string };

const VIDEO_SIZES: [string, string][] = [["0.4", "0.4 MP ≈ 544×800"], ["0.6", "0.6 MP ≈ 640×960"], ["0.8", "0.8 MP ≈ 736×1088"], ["1.5", "1.5 MP ≈ 1000×1504"]];
const EXTRA_SECTIONS = [["animate", "Animate"], ["music", "Music"], ["runs", "Run buttons"], ["ui", "UI"], ["advanced", "Advanced"]];

function saveLabel(status: Status): string {
  if (status.s === "error") return `error: ${status.msg}`;
  return { idle: "auto-saves", saving: "saving…", saved: "saved ✓" }[status.s];
}

/** Project settings as a control hub: one card per pipeline step, a rail to jump between them, autosave. Same project.json keys as before. */
export default function Settings({ name }: { readonly name: string }) {
  const qc = useQueryClient();
  const proj = useProject(name);
  const reg = useRegistry().data;
  const voices = useVoices().data?.voices;
  const cfg = proj.data?.config;
  const [vals, setVals] = useState<Settings>({});
  const [seed, setSeed] = useState("");
  const [vmp, setVmp] = useState("");
  const [rev, setRev] = useState(0);  // bumps on every local edit; the autosave effect keys off it
  const [status, setStatus] = useState<Status>({ s: "idle" });
  const [active, setActive] = useState("preview");
  const pending = useRef(false);
  const loaded = useRef<string | null>(null);

  // adopt the server's config on first load / project switch, and after refetches while nothing local is waiting to save
  useEffect(() => {
    if (!cfg || (pending.current && loaded.current === name)) return;
    loaded.current = name;
    setVals((cfg.settings ?? {}) as Settings); setSeed(cfg.seed?.toString() ?? ""); setVmp(cfg.video_megapixels?.toString() ?? "");
  }, [cfg, name]);

  useEffect(() => {
    if (!rev) return;
    pending.current = true;
    const t = setTimeout(async () => {
      setStatus({ s: "saving" });
      try {
        await api(`/projects/${name}/settings`, "PUT", vals);
        await api(`/projects/${name}/seed`, "PUT", { seed: seed === "" ? null : Number(seed) });
        await api(`/projects/${name}/video-megapixels`, "PUT", { megapixels: vmp === "" ? null : Number(vmp) });
        pending.current = false;
        setStatus({ s: "saved" });
        qc.invalidateQueries({ queryKey: ["project", name] });
      } catch (e) { pending.current = false; setStatus({ s: "error", msg: (e as Error).message }); }
    }, 500);
    return () => clearTimeout(t);
  }, [rev]);  // eslint-disable-line react-hooks/exhaustive-deps

  // scroll-spy: the rail highlights the section nearest the top of the viewport
  useEffect(() => {
    const els = [...document.querySelectorAll<HTMLElement>(".hub section[id]")];
    const io = new IntersectionObserver((es) => {
      const top = es.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (top) setActive(top.target.id.replace(/^s-/, ""));
    }, { rootMargin: "-60px 0px -60% 0px" });
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [cfg?.engine, !!reg]);  // eslint-disable-line react-hooks/exhaustive-deps

  if (!cfg || !reg) return <div className="page">loading…</div>;
  const engine = cfg.engine;
  const keys = new Set(reg.settings_keys[engine] ?? []);
  const tdef = reg.template_defaults?.[engine] ?? {};
  const edit = (next: Settings) => { setVals(next); setRev((r) => r + 1); };
  const setKey = (k: string, v: Val | Lora) => edit({ ...vals, [k]: v });
  const unset = (k: string) => { const n = { ...vals }; delete n[k]; edit(n); };
  const eff = (k: string, fallback?: Val | Lora) => (vals[k] ?? tdef[k] ?? fallback) as Val | Lora | undefined;

  const aspect = eff("aspect_ratio");
  const [pw, ph] = resolution(typeof aspect === "string" ? aspect : "2:3", Number(eff("megapixels") ?? 1));
  const times = (f: number) => `${Math.round(pw * f / 8) * 8}×${Math.round(ph * f / 8) * 8}`;
  const extraFor = (k: string): ReactNode => {
    if (k === "megapixels") return `${pw}×${ph}`;
    if (k === "hires_scale") return `→ ${times(Number(eff(k, 1.5)))}`;
    if (k === "upscale_factor") return `→ ${times(Number(eff(k, 1)))}`;
    if (k === "seedvr_resolution") { const f = Number(eff(k, 1680)) / Math.min(pw, ph); return `→ ${times(f)}`; }
    if (k === "finish_resolution") { const f = Number(eff(k, 1440)) / Math.min(pw, ph); return `→ ${times(f)}`; }
    return null;
  };
  const field = (c: Ctl) => (
    <Field key={c.key} c={c} value={vals[c.key]} def={(tdef[c.key] as Val | undefined) ?? c.fallback} set={(v) => setKey(c.key, v)} reset={() => unset(c.key)}
      choices={c.models ? reg.model_choices[c.key] : undefined} localOnly={new Set(reg.local_only_unets ?? [])} extra={extraFor(c.key)} />
  );
  /** Controls in order; consecutive ones sharing a `fold` go into one collapsed group. */
  const controls = (ctls: Ctl[]) => {
    const shown = ctls.filter((c) => keys.has(c.key));
    const out: ReactNode[] = [];
    for (let i = 0; i < shown.length;) {
      const fold = shown[i].fold;
      if (!fold) { out.push(field(shown[i])); i++; continue; }
      const group: Ctl[] = [];
      while (i < shown.length && shown[i].fold === fold) group.push(shown[i++]);
      const nSet = group.filter((c) => c.key in vals).length;
      out.push(<details key={`fold-${fold}-${i}`} className="fold"><summary>{fold}{nSet > 0 && <small> · {nSet} set</small>}</summary>{group.map(field)}</details>);
    }
    return out;
  };
  const sections = SECTIONS.filter((s) => !s.engines || s.engines.includes(engine))
    .map((s) => ({ ...s, panels: s.panels.filter((p) => (!p.engines || p.engines.includes(engine)) && p.ctls.some((c) => keys.has(c.key))) }))
    .filter((s) => s.panels.length);
  const rail = [...sections.map((s) => [s.id, s.title]), ...EXTRA_SECTIONS];
  const go = (id: string) => document.getElementById(`s-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <div className="page hub">
      <nav className="hubrail">
        <div className="hubeng"><b>{engine}</b> engine</div>
        {rail.map(([id, t]) => <button type="button" key={id} className={`railitem ${active === id ? "on" : ""}`} onClick={() => go(id)}>{t}</button>)}
        <a className="raillink" href="#/loras" title="the LoRA library: notes, tips, ranges, local/cloud, register local files">LoRA library ↗</a>
        <a className="raillink" href="#/voices" title="the voice library: named voices for the motion's {voice} slots">Voice library ↗</a>
        <div className={`savepill ${status.s}`} title={status.msg}>
          {saveLabel(status)}
        </div>
      </nav>
      <div className="hubmain">
        {sections.map((s) => (
          <section key={s.id} id={`s-${s.id}`} className="card hubcard">
            <div className="hubtitle"><h3>{s.title}</h3>{s.sub && <span className="dim">{s.sub}</span>}</div>
            <div className={s.panels.length > 1 ? "panels" : ""}>
              {s.panels.map((p) => (
                <div key={p.ctls[0].key} className={s.panels.length > 1 ? "panel" : ""}>
                  {(typeof p.title === "string" ? p.title : p.title[engine]) && <div className="paneltitle">{typeof p.title === "string" ? p.title : p.title[engine]}{p.sub && <small>{p.sub}</small>}</div>}
                  {controls(p.ctls)}
                  {s.id === "preview" && (
                    <div className="ctl">
                      <div className="ctlhead"><span className="ctllabel">Project seed</span><span className="hint" title="pins every generation (the same pose again); blank = a fresh seed each run"><Icon n="info" size={12} /></span>{seed === "" && <span className="deftag">random</span>}<span className="spacer" />
                        {seed !== "" && <button className="reset" title="back to a fresh random seed every run" onClick={() => { setSeed(""); setRev((r) => r + 1); }}>↺</button>}</div>
                      <input className="text" type="number" placeholder="random each run" value={seed} onChange={(e) => { setSeed(e.target.value); setRev((r) => r + 1); }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        ))}

        <section id="s-animate" className="card hubcard">
          <div className="hubtitle"><h3>Animate</h3><span className="dim">image-to-video clips</span></div>
          <div className="ctl">
            <div className="ctlhead"><span className="ctllabel">i2v model</span><span className="hint" title="Every stage animates with this unless its editor picks another. Video LoRAs and prompt style are per model (MiniMax: minimax-h3-prompt; LTX: ltx2-prompt)."><Icon n="info" size={12} /></span>{!cfg.video_engine && <span className="deftag">default</span>}</div>
            <div className="seg wrapseg">
              {Object.entries(reg.video_engines ?? {}).map(([k, label]) => (
                <button key={k} className={(cfg.video_engine ?? "minimax") === k ? "on" : ""} onClick={async () => {
                  try { await api(`/projects/${name}/video-engine`, "PUT", { engine: k }); qc.invalidateQueries({ queryKey: ["project", name] }); }
                  catch (e) { setStatus({ s: "error", msg: (e as Error).message }); }
                }}>{label}</button>))}
            </div>
          </div>
          <div className="ctl">
            <div className="ctlhead"><span className="ctllabel">Voice</span><span className="hint" title="The character's voice: fills the {voice}, {voice.delivery}, {voice.moans}, {voice.sound} slots of every stage's motion; a stage can pick another in its editor."><Icon n="info" size={12} /></span>{!cfg.voice && <span className="deftag">none</span>}</div>
            <div className="row">
              <select value={cfg.voice ?? ""} onChange={async (e) => {
                try { await api(`/projects/${name}/voice`, "PUT", { voice: e.target.value || null }); void qc.invalidateQueries({ queryKey: ["project", name] }); void qc.invalidateQueries({ queryKey: ["voices"] }); }
                catch (error_) { setStatus({ s: "error", msg: (error_ as Error).message }); }
              }}>
                <option value="">none (stages set their own)</option>
                {Object.entries(voices ?? {}).filter(([k, v]) => !v.disabled || k === cfg.voice).map(([k, v]) => <option key={k} value={k}>{v.label || k}</option>)}
              </select>
              {cfg.voice && voices?.[cfg.voice] && <SampleButton e={voices[cfg.voice]} />}
              <a className="dim" href="#/voices">manage voices…</a>
            </div>
          </div>
          <div className="ctl">
            <div className="ctlhead"><span className="ctllabel">Clip size</span><span className="hint" title="2:3; applies to clips animated after the change. Bigger clips cost more credits."><Icon n="info" size={12} /></span>{vmp === "" && <span className="deftag">default</span>}</div>
            <div className="seg wrapseg">
              {[["", "1 MP ≈ 832×1248 (default)"], ...VIDEO_SIZES].map(([mp, label]) => <button key={mp} className={vmp === mp ? "on" : ""} onClick={() => { setVmp(mp); setRev((r) => r + 1); }}>{label}</button>)}
              {vmp !== "" && !VIDEO_SIZES.some(([mp]) => mp === vmp) && <button className="on">{vmp} MP (custom)</button>}
            </div>
          </div>
        </section>

        <section id="s-music" className="card hubcard musiccard">
          <div className="hubtitle"><h3>Music</h3><span className="dim">the track the edit is cut to; clip durations come from its beats</span></div>
          <MusicPicker name={name} full />
          <FitTable name={name} />
        </section>

        <section id="s-runs" className="card hubcard">
          <div className="hubtitle"><h3>Run buttons</h3><span className="dim">which runs the Run menus offer (every project, this browser)</span></div>
          <RunButtons />
        </section>

        <section id="s-ui" className="card hubcard">
          <div className="hubtitle"><h3>UI</h3><span className="dim">layout (every project, this browser)</span></div>
          <UiPrefs />
        </section>

        <section id="s-advanced" className="card hubcard">
          <div className="hubtitle"><h3>Advanced</h3><span className="dim">raw keys without a control above</span></div>
          <Advanced vals={vals} keys={[...keys]} defaults={{ ...reg.default_settings.full, ...reg.default_settings[engine], ...tdef }} setKey={setKey} unset={unset} />
        </section>
      </div>
    </div>
  );
}

function UiPrefs() {
  const cols = useBoardCols(), vol = useClipVolume();
  return (
    <div className="ctl">
      <div className="ctlhead"><span className="ctllabel">Board columns</span><span className="hint" title="stage cards per row on the board; windows under 1100px wide always show one"><Icon n="info" size={12} /></span>{cols === 2 && <span className="deftag">default</span>}</div>
      <div className="seg">{BOARD_COLS.map((n) => <button key={n} className={cols === n ? "on" : ""} onClick={() => setBoardCols(n)}>{n}</button>)}</div>
      <div className="ctlhead" style={{ marginTop: 12 }}><span className="ctllabel">Clip volume</span><span className="hint" title="how loud clips with audio play in the viewer and one-off previews (this browser)"><Icon n="info" size={12} /></span>{vol === CLIP_VOLUME_DEFAULT && <span className="deftag">default</span>}</div>
      <div className="row"><input type="range" min={0} max={100} step={5} value={Math.round(vol * 100)} onChange={(e) => setClipVolume(Number(e.target.value) / 100)} /> <span>{Math.round(vol * 100)}%</span></div>
    </div>
  );
}

function RunButtons() {
  const off = useHiddenActions();
  return (
    <div className="rgrid" style={{ marginTop: 8 }}>
      <span /><span className="dim">Local</span><span className="dim">Cloud</span>
      {ROWS.map((r) => (
        <Fragment key={r.title}>
          <span className="rtitle">{r.title}</span>
          {([["local", r.free], ["cloud", r.paid]] as const).map(([col, a]) => a
            ? <label key={col}><input type="checkbox" aria-label={`${r.title} ${col}`} checked={!off.has(a)} onChange={(e) => setActionHidden(a, !e.target.checked)} /></label>
            : <span key={col} />)}
        </Fragment>
      ))}
      <span className="rtitle">Animate</span><span /><label><input type="checkbox" aria-label="Animate" checked={!off.has("animate")} onChange={(e) => setActionHidden("animate", !e.target.checked)} /></label>
      <span className="rtitle">Animate smooth</span><span /><label><input type="checkbox" aria-label="Animate smooth" checked={!off.has("animate_rife")} onChange={(e) => setActionHidden("animate_rife", !e.target.checked)} /></label>
      <span className="rtitle">Interpolate clips</span><label><input type="checkbox" aria-label="Interpolate clips" checked={!off.has("interpolate")} onChange={(e) => setActionHidden("interpolate", !e.target.checked)} /></label><span />
    </div>
  );
}

/** The old raw editor, for keys the schema has no control for (a new graph_settings key shows up here until it gets one). */
function Advanced({ vals, keys, defaults, setKey, unset }: { readonly vals: Settings; readonly keys: string[]; readonly defaults: Registry["default_settings"][string]; readonly setKey: (k: string, v: Val) => void; readonly unset: (k: string) => void }) {
  const [txt, setTxt] = useState<Record<string, string>>({});
  const extra = Object.keys(vals).filter((k) => !SCHEMA_KEYS.has(k));
  const addable = keys.filter((k) => !SCHEMA_KEYS.has(k) && !(k in vals));
  return (
    <>
      {extra.length === 0 && <div className="dim">nothing here: every setting this project uses has a control above</div>}
      <div className="settings">
        {extra.map((k) => {
          const v = vals[k] as Val; const like = v ?? defaults[k];
          return (
            <label key={k} className="setrow"><span>{k}</span>
              {typeof like === "boolean"
                ? <select value={String(v)} onChange={(e) => setKey(k, e.target.value === "true")}><option>true</option><option>false</option></select>
                : <input inputMode={typeof like === "number" ? "decimal" : undefined} value={txt[k] ?? (typeof v === "object" ? JSON.stringify(v) : String(v))} onChange={(e) => {
                    const raw = e.target.value; setTxt({ ...txt, [k]: raw });
                    if (typeof like !== "number") setKey(k, raw);
                    else if (raw.trim() !== "" && Number.isFinite(Number(raw))) setKey(k, Number(raw));
                  }} />}
              <button className="btn sm" onClick={() => unset(k)}>✕</button>
            </label>
          );
        })}
      </div>
      {addable.length > 0 && (
        <select value="" onChange={(e) => e.target.value && setKey(e.target.value, (defaults[e.target.value] as Val) ?? 0)}>
          <option value="">+ add setting…</option>
          {addable.map((k) => <option key={k}>{k}</option>)}
        </select>
      )}
    </>
  );
}
