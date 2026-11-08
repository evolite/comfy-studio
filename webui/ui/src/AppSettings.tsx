import { ReactNode, useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { Icon } from "./icons";

type AppSettings = { animate_target: "cloud" | "runpod"; animate_batch_size: number; runpod_h3_unet: string; runpod_h3_choices: string[]; pushover_user: string; pushover_token_set: boolean; pushover_from: "settings" | ".env" | null };
type Pod = { up: boolean; id?: string; state?: string; ready?: boolean; uptime?: number | null; cost_per_hr?: number | null; gpu?: string; error?: string };

const hm = (sec: number) => `${Math.floor(sec / 3600)}h ${String(Math.floor(sec / 60) % 60).padStart(2, "0")}m`;

/** The helper text pattern for the chrome: an ⓘ icon whose hover (title) carries the explanation. */
function Hint({ children }: { readonly children: ReactNode }) {
  return <span className="hint" title={typeof children === "string" ? children : undefined}><Icon n="info" size={13} /></span>;
}

/** One pod status line: checking / error / down / stopped / up with uptime and $/hr. */
function podLine(p: Pod | undefined) {
  if (!p) return "checking…";
  if (p.error) return `✗ ${p.error}`;
  if (!p.id) return `pod down (${p.gpu})`;
  if (!p.up) return `pod ${p.id} ${p.state?.toLowerCase()} (not billing GPU time)`;
  const ready = p.ready ? "up" : "booting";
  const upFor = p.uptime != null ? ` for ${hm(p.uptime)}` : "";
  const cost = p.cost_per_hr != null ? ` · $${p.cost_per_hr.toFixed(2)}/hr` : "";
  return `pod ${p.id} ${ready}${upFor}${cost} · ${p.gpu}`;
}

/** Where `animate` renders, and the RunPod pod's line: up/down, uptime, $/hr, "Stop pod now" (pipeline/runpod_pod.py). */
function AnimateTarget({ target, batchSize, h3, h3Choices, onSaved }: { readonly target: AppSettings["animate_target"]; readonly batchSize: number; readonly h3: string; readonly h3Choices: string[]; readonly onSaved: () => void }) {
  const pod = useQuery({ queryKey: ["runpod"], queryFn: () => api<Pod>("/runpod"), refetchInterval: 15000 });
  const [msg, setMsg] = useState("");
  const set = (t: AppSettings["animate_target"]) => api("/app-settings", "PUT", { animate_target: t }).then(onSaved, (e: Error) => setMsg(`✗ ${e.message}`));
  const stop = () => {
    if (!confirm("Terminate the RunPod pod now? Animate jobs on it are cancelled.")) return;
    setMsg("stopping…");
    api<{ terminated: string | null }>("/runpod/stop", "POST")
      .then((r) => { setMsg(r.terminated ? `pod ${r.terminated} terminated` : "no pod was up"); void pod.refetch(); }, (e: Error) => setMsg(`✗ ${e.message}`));
  };
  const line = podLine(pod.data);
  return (
    <section className="panel">
      <h3>Animate target <Hint>Where MiniMax H3 clips render. RunPod starts our pod on demand (a few minutes the first time), bills per hour
        while it is up, and the pod deletes itself after it has been idle a while; Comfy Cloud spends credits per clip.</Hint></h3>
      <div className="row">
        {(["cloud", "runpod"] as const).map((t) => (
          <button key={t} className={`btn ${target === t ? "primary" : ""}`} onClick={() => target !== t && set(t)}>{t === "cloud" ? "Comfy Cloud" : "RunPod"}</button>
        ))}
      </div>
      <div className="row podline">
        <span className="dim">{line}</span>
        {pod.data?.id && <button className="btn danger" onClick={stop}>Stop pod now</button>}
        {msg && <span className="dim">{msg}</span>}
      </div>
      <PodH3 unet={h3} choices={h3Choices} onSaved={onSaved} />
      <BatchSize size={batchSize} onSaved={onSaved} />
    </section>
  );
}

/** The H3 UNet the RunPod pod loads: stock MiniMax H3 (+ its turbo LoRA) or a finetune on the volume (Comfy Cloud is always stock). */
function PodH3({ unet, choices, onSaved }: { readonly unet: string; readonly choices: string[]; readonly onSaved: () => void }) {
  const [msg, setMsg] = useState("");
  const set = (v: string) => api("/app-settings", "PUT", { runpod_h3_unet: v }).then(() => { setMsg("saved ✓"); onSaved(); }, (e: Error) => setMsg(`✗ ${e.message}`));
  return (
    <div className="row podline">
      <label title="The MiniMax H3 model RunPod clips load. A finetune is distilled: the pod skips the stock turbo LoRA and samples with euler. Comfy Cloud always uses the stock model.">
        <span>Pod H3 model</span>
        <select value={unet} onChange={(e) => set(e.target.value)}>
          <option value="">stock MiniMax H3</option>
          {choices.map((c) => <option key={c} value={c}>{c.replace(/\.safetensors$/, "")}</option>)}
        </select>
      </label>
      {msg && <span className="dim">{msg}</span>}
    </div>
  );
}

/** The RunPod "Start batch" target: how many queued Animate stages fire as one job (the header's "Batch"). */
function BatchSize({ size, onSaved }: { readonly size: number; readonly onSaved: () => void }) {
  const [v, setV] = useState(String(size));
  const [msg, setMsg] = useState("");
  useEffect(() => setV(String(size)), [size]);
  const save = () => {
    if (Number(v) === size) return;
    api("/app-settings", "PUT", { animate_batch_size: Number(v) }).then(() => { setMsg("saved ✓"); onSaved(); }, (e: Error) => setMsg(`✗ ${e.message}`));
  };
  return (
    <div className="row podline">
      <label title={'With the header\'s "Batch" armed (RunPod target), the server waits for this many queued Animate stages before starting the pod — one pod start and model load per batch'}>
        <span>Animate batch size</span>
        <input type="number" min={1} max={100} value={v} onChange={(e) => setV(e.target.value)} onBlur={save} style={{ width: "4.5em" }} />
      </label>
      {msg && <span className="dim">{msg}</span>}
    </div>
  );
}

/** The Pushover keys behind the 🔔. Saved to data/app_settings.json; the token is never shown back. */
function Pushover({ s, onSaved }: { readonly s: AppSettings; readonly onSaved: () => void }) {
  const [user, setUser] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [msg, setMsg] = useState("");
  const save = () => api("/app-settings", "PUT", { ...(user !== null ? { pushover_user: user } : {}), ...(token ? { pushover_token: token } : {}) })
    .then(() => { setUser(null); setToken(""); setMsg("saved ✓"); onSaved(); }, (e: Error) => setMsg(`✗ ${e.message}`));
  const clear = () => api("/app-settings", "PUT", { pushover_user: "", pushover_token: "" }).then(() => { setMsg("cleared"); onSaved(); }, (e: Error) => setMsg(`✗ ${e.message}`));
  const test = () => { setMsg("sending…"); api("/app-settings/test-push", "POST").then(() => setMsg("test push sent ✓ check your phone"), (e: Error) => setMsg(`✗ ${e.message}`)); };
  return (
    <section className="panel">
      <h3>Pushover notifications <Hint>The bell in the header pushes to your phone when the job queue empties, and when the local ComfyUI goes down
        or restarts. Get a user key and create an application token at pushover.net. Messages carry job ids, never project names.</Hint></h3>
      <label><span>User key</span>
        <input type="text" autoComplete="off" spellCheck={false} value={user ?? s.pushover_user} placeholder="uQiRzpo4DXghDmr9QzzfQu27cmVRsG"
          onChange={(e) => setUser(e.target.value)} />
      </label>
      <label><span>App token</span>
        <input type="password" autoComplete="new-password" value={token} onChange={(e) => setToken(e.target.value)}
          placeholder={s.pushover_token_set ? "•••••••• set (type to replace)" : "azGDORePK8gMaC0QOYAMyEEuzJnyUi"} />
      </label>
      <div className="row">
        <button className="btn primary" disabled={user === null && !token} onClick={save}>Save</button>
        <button className="btn" disabled={!s.pushover_user || !s.pushover_token_set} onClick={test}>Send test</button>
        {s.pushover_from === "settings" && <button className="btn danger" onClick={clear}>Clear</button>}
        <span className="dim">{msg || (s.pushover_from === ".env" ? "currently from .env; saving here overrides it" : "")}</span>
      </div>
    </section>
  );
}

/** ⚙ header button: the app-wide settings popover (animate target + RunPod line, Pushover). A project's own settings live in its Settings tab. */
export default function SettingsButton() {
  const [open, setOpen] = useState(false);
  const qc = useQueryClient();
  const s = useQuery({ queryKey: ["app-settings"], queryFn: () => api<AppSettings>("/app-settings"), enabled: open }).data;
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => { if (!(e.target as Element).closest(".settingswrap")) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    addEventListener("pointerdown", away); addEventListener("keydown", esc);
    return () => { removeEventListener("pointerdown", away); removeEventListener("keydown", esc); };
  }, [open]);
  const saved = () => {
    qc.invalidateQueries({ queryKey: ["app-settings"] });
    qc.invalidateQueries({ queryKey: ["notify"] });
  };
  return (
    <span className="settingswrap menuwrap">
      <button className={`qbtn ${open ? "on" : ""}`} title="settings: animate target, RunPod, Pushover notifications" onClick={() => setOpen(!open)}>
        <Icon n="sliders" size={16} />
      </button>
      {open && (
        <div className="menu settingspanel">
          {!s ? <div className="dim pad">loading…</div> : (
            <>
              <AnimateTarget target={s.animate_target} batchSize={s.animate_batch_size} h3={s.runpod_h3_unet} h3Choices={s.runpod_h3_choices} onSaved={() => saved()} />
              <Pushover s={s} onSaved={saved} />
            </>
          )}
        </div>
      )}
    </span>
  );
}
