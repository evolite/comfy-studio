import SettingsButton from "./AppSettings";
import { ReactNode, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api, fileUrl, ProjectRow, useHash } from "./api";
import { useAnimateBatch, useAnimateTarget, useCredits, useHealth, useJobs, useProject, useRunpodBalance, useUnseen } from "./hooks";
import Projects from "./Projects";
import Board from "./Board";
import OneOff from "./OneOff";
import Compose from "./Compose";
import Settings from "./Settings";
import Loras from "./Loras";
import Voices from "./Voices";
import Civitai from "./Civitai";
import JobsPanel from "./JobsPanel";
import { RefChip } from "./controls";
import { Icon } from "./icons";
import { setBatchArmed, useBatchArmed } from "./runPrefs";

/** A header chip that opens its menu while hovered (or on click, for touch); the menu holds labeled actions. */
function HoverMenu({ chip, lane, title, live, children }: { readonly chip: ReactNode; readonly lane: "local" | "cloud"; readonly title: string; readonly live?: boolean; readonly children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const timer = useRef<number>(0);
  const enter = () => { clearTimeout(timer.current); setOpen(true); };
  const leave = () => { timer.current = window.setTimeout(() => setOpen(false), 180); };  // a short grace period to reach the menu
  return (
    <span className={`hmenu menuwrap lane-${lane} ${open ? "open" : ""} ${live ? "live" : ""}`} onMouseEnter={enter} onMouseLeave={leave}>
      <button className="hchip" title={title} onClick={() => setOpen(!open)}>{chip}</button>
      {open && <div className="menu hmenupanel">{children}</div>}
    </span>
  );
}

/** One header action: confirm, call, flash the result inside the menu. */
function useFlash() {
  const [msg, setMsg] = useState("");
  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(""), 3000); };
  const act = (question: string, call: () => Promise<string>) => async () => {
    if (!confirm(question)) return;
    try { flash(await call()); } catch (e) { flash(String((e as Error).message)); }
  };
  return { msg, act };
}

/** Local ComfyUI: status chip; menu = free memory, restart, cancel local jobs. */
function LocalMenu({ up, server }: { readonly up: boolean; readonly server?: string }) {
  const { msg, act } = useFlash();
  return (
    <HoverMenu lane="local" title={up ? `local ComfyUI up: ${server}` : "local ComfyUI down; local actions disabled"}
      chip={<><i className={`sdot ${up ? "ok" : "bad"}`} />Local</>}>
      <div className="hmhead dim">{up ? `up · ${server}` : "down"}</div>
      <button className="menuitem row" disabled={!up} onClick={act("Unload all models and clear the cache on the local ComfyUI? The next run has to reload them.",
        async () => { await api("/local/free", "POST", { unload_models: true, free_memory: true }); return "memory freed ✓"; })}>
        <Icon n="arrow-down" size={15} />Free memory</button>
      <button className="menuitem row" onClick={act("Restart the local ComfyUI? Running and queued local jobs are cancelled.",
        async () => { await api("/local/restart", "POST"); return "restarting… (~30 s)"; })}>
        <Icon n="restart" size={15} />Restart ComfyUI</button>
      <button className="menuitem row danger" onClick={act("Cancel all local jobs and clear the local ComfyUI queue?",
        async () => { const r = await api<{ jobs: number }>("/local/cancel_all", "POST"); return `cancelled ${r.jobs} job(s) ✓`; })}>
        <Icon n="stop" size={15} />Stop all local jobs</button>
      {msg && <div className="dim hmmsg">{msg}</div>}
    </HoverMenu>
  );
}

// the billing API reports 822.42 where the Comfy site shows 1735 credits
const CREDITS_PER_UNIT = 2.11;

/** Comfy Cloud: credits chip; menu = balance details, stop cloud jobs. */
function CloudMenu({ noKey }: { readonly noKey: boolean }) {
  const { msg, act } = useFlash();
  const credits = useCredits();
  const jobs = useJobs().data ?? [];
  const n = jobs.filter((j) => j.lane === "cloud" && ["queued", "running"].includes(j.status)).length;
  let value: string | number = "…";
  if (credits.data) value = Math.round(credits.data.effective_balance_micros * CREDITS_PER_UNIT);
  else if (credits.isError) value = "n/a";
  let head = "loading balance…";
  if (credits.data) head = `${value} credits (API ${credits.data.effective_balance_micros.toFixed(2)} ×${CREDITS_PER_UNIT})`;
  else if (credits.error) head = String((credits.error as Error).message);
  return (
    <HoverMenu lane="cloud" title="Comfy Cloud" chip={<><Icon n="cloud" size={15} /><b className="credits">{value}</b>{noKey && <span className="pill bad">no key</span>}</>}>
      <div className="hmhead dim">{head}</div>
      <button className="menuitem row danger" disabled={!n} onClick={act(`Cancel all ${n} cloud job(s)? Work already submitted is removed from the cloud queue (credits for finished renders are spent).`,
        async () => { const r = await api<{ jobs: number }>("/cloud/cancel_all", "POST"); return `cancelled ${r.jobs} job(s) ✓`; })}>
        <Icon n="stop" size={15} />{n ? `Stop all cloud jobs (${n})` : "No cloud jobs running"}</button>
      {msg && <div className="dim hmmsg">{msg}</div>}
    </HoverMenu>
  );
}

type ColdStart = { t: number; gpu: string; ready_s: number; first_clip_s?: number; warm_clip_s?: number };
const mmss = (s: number) => `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;
/** "last cold start Oct 30 18:30 · RTX PRO 4500: ready 2m56s · first clip 5m42s · next 3m40s" (clips appear once they are done). */
const coldLine = (c: ColdStart) => `last cold start ${new Date(c.t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
  + ` · ${c.gpu.replace(/^NVIDIA (GeForce )?/, "")}: ready ${mmss(c.ready_s)}`
  + (c.first_clip_s != null ? ` · first clip ${mmss(c.first_clip_s)}` : "") + (c.warm_clip_s != null ? ` · next ${mmss(c.warm_clip_s)}` : "");

type PodStatus = { up: boolean; id?: string; ready?: boolean; gpu?: string; cost_per_hr?: number | null; error?: string; last_start?: ColdStart | null; session_cost?: number | null;
  last_run?: { cost: number; minutes: number; gpu: string; ended: number } | null };

function podStatusLine(pod?: PodStatus): string {
  if (!pod) return "checking the pod…";
  if (pod.error) return pod.error;
  if (!pod.id) return "pod down";
  const session = pod.session_cost == null ? "" : ` · this session $${pod.session_cost.toFixed(2)}`;
  return `pod ${pod.id} ${pod.ready ? "up" : "booting"} · ${pod.gpu}${session}`;
}

/** The chip's small print: this session's cost, else the last run's, else the current spend rate. */
function podChipSub(pod?: PodStatus, spendPerHr?: number): string | null {
  if (pod?.session_cost != null) return ` · session $${pod.session_cost.toFixed(2)}`;
  if (pod?.last_run) return ` · last run $${pod.last_run.cost.toFixed(2)}`;
  return spendPerHr == null ? null : ` · $${spendPerHr.toFixed(2)}/h`;
}

function lastRunLine(r: NonNullable<PodStatus["last_run"]>): string {
  const ended = new Date(r.ended * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `last run $${r.cost.toFixed(2)} · ${Math.round(r.minutes)} min on ${r.gpu.replace(/^NVIDIA (GeForce )?/, "")}, ended ${ended} (≈, to the minute)`;
}

/** RunPod (⚙ animate target = runpod): balance chip with the current pod's cost so far (else what the last run cost), glowing while a pod exists; menu = balance, the pod, stop it. */
function RunpodMenu() {
  const { msg, act } = useFlash();
  const bal = useRunpodBalance(true);
  const pod = useQuery({ queryKey: ["runpod"], queryFn: () => api<PodStatus>("/runpod"),
    refetchInterval: 15000 }).data;
  const b = bal.data;
  let value = "…";
  if (b) value = `$${b.balance.toFixed(2)}`;
  else if (bal.isError) value = "n/a";
  const sub = podChipSub(pod, b?.spend_per_hr);
  let head = "loading balance…";
  if (b) head = `RunPod balance ${value} · spending $${b.spend_per_hr.toFixed(2)}/hr now (pods + volume)`;
  else if (bal.error) head = String((bal.error as Error).message);
  return (
    <HoverMenu lane="cloud" title="RunPod" live={!!pod?.id} chip={<><Icon n="cpu" size={15} /><b className="credits">{value}</b>{sub && <span className="sub">{sub}</span>}</>}>
      <div className="hmhead dim">{head}</div>
      <div className="hmhead dim">{podStatusLine(pod)}</div>
      {!pod?.id && pod?.last_run && <div className="hmhead dim">{lastRunLine(pod.last_run)}</div>}
      {pod?.last_start && <div className="hmhead dim" title="the newest pod the studio started: until ComfyUI answered, then its first clip (models loaded from the volume) and its second (warm)">
        {coldLine(pod.last_start)}</div>}
      <button className="menuitem row danger" disabled={!pod?.id} onClick={act("Stop the RunPod pod now? Its running and waiting animate jobs are cancelled; the volume stays.",
        async () => { const r = await api<{ jobs: number }>("/runpod/stop", "POST"); return `pod stopped, ${r.jobs} job(s) cancelled ✓`; })}>
        <Icon n="stop" size={15} />{pod?.id ? "Stop pod now" : "No pod running"}</button>
      {msg && <div className="dim hmmsg">{msg}</div>}
    </HoverMenu>
  );
}

function noMatchText(nRows: number, doneHidden: boolean): string {
  if (!nRows) return "loading…";
  return doneHidden ? "no match (done projects hidden)" : "no match";
}

/** The current project's name, clickable: a searchable dropdown of every project that has a project.json. */
function ProjectSwitcher({ current, tab }: { readonly current: string; readonly tab: string }) {
  const [open, setOpen] = useState(false);
  const [remix, setRemix] = useState(false);
  const [q, setQ] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const rows = useQuery({ queryKey: ["projects"], queryFn: () => api<ProjectRow[]>("/projects"), enabled: open }).data ?? [];
  const [hideDone, setHideDone] = useState(() => { try { return localStorage.getItem("switcher:hideDone") === "1"; } catch { return false; } });
  const toggleHideDone = (on: boolean) => { setHideDone(on); try { localStorage.setItem("switcher:hideDone", on ? "1" : "0"); } catch { /* ignore */ } };
  const nDone = rows.filter((r) => r.finished).length;
  // done projects drop out when hidden, except the one you're on
  const list = rows.filter((r) => r.name.toLowerCase().includes(q.toLowerCase()) && !(hideDone && r.finished && r.name !== current));
  useEffect(() => {
    if (!open) return;
    setQ(""); setRemix(false); input.current?.focus();
    const away = (e: PointerEvent) => { if (!(e.target as Element).closest(".switcher")) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    addEventListener("pointerdown", away); addEventListener("keydown", esc);
    return () => { removeEventListener("pointerdown", away); removeEventListener("keydown", esc); };
  }, [open]);
  const pick = (name: string) => { setOpen(false); location.hash = `#/p/${name}/${tab}`; };
  return (
    <span className="switcher menuwrap">
      <button className="crumb" title={`${current} — switch project`} onClick={() => setOpen(!open)}><span className="n">{current}</span><Icon n="chevron-down" size={13} /></button>
      {open && (
        <div className="menu switchpanel">
          {remix ? (
            <RemixPanel current={current} onClose={() => setRemix(false)}
              onCreated={(name) => { setOpen(false); location.hash = `#/p/${name}/board`; }} />
          ) : (<>
          <input ref={input} placeholder="find project…" value={q} onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && list[0]) pick(list[0].name); }} />
          <div className="switchhead">
            <label className="switchfilter" title="leave projects marked ✓ Done out of this list">
              <span className="switch"><input type="checkbox" checked={hideDone} onChange={(e) => toggleHideDone(e.target.checked)} /><span /></span>
              <small>Hide done{nDone ? ` (${nDone})` : ""}</small>
            </label>
            <span className="spacer" />
            <a className="qbtn" href="#/new" title="new project" onClick={() => setOpen(false)}><Icon n="plus" size={13} />New</a>
            <button className="qbtn" title="remix the current project into a new one" onClick={() => setRemix(true)}>
              <Icon n="shuffle" size={13} />Remix</button>
          </div>
          <div className="switchlist">
            {list.map((r) => (
              <button key={r.name} className={`switchitem ${r.name === current ? "on" : ""} ${r.finished ? "finished" : ""}`} onClick={() => pick(r.name)}>
                <span className="n">{r.finished && <span className="donemark" title="done">✓</span>}{r.name}{r.oneoff && <span className="oneoff" title="one-off (single stage)"><Icon n="zap" size={12} /></span>}</span>
                <small>{r.engine} · {r.stages} stages · {r.stills} stills · {r.clips} clips</small>
              </button>
            ))}
            {list.length === 0 && <div className="dim pad">{noMatchText(rows.length, hideDone && nDone > 0)}</div>}
          </div>
          </>)}
        </div>
      )}
    </span>
  );
}

/** 🔔: while on, a Pushover message goes out every time the queue empties; it stays on until clicked off. */
function NotifyButton() {
  const qc = useQueryClient();
  const n = useQuery({ queryKey: ["notify"], queryFn: () => api<{ armed: boolean; configured: boolean }>("/notify"), refetchInterval: 3000 }).data;
  if (!n?.configured) return null;
  const toggle = () => api("/notify", "POST", { armed: !n.armed }).then(() => void qc.invalidateQueries({ queryKey: ["notify"] }), (e: Error) => alert(e.message));
  return <button className={`qbtn notifybtn ${n.armed ? "on" : ""}`} onClick={toggle}
    title={n.armed ? "pushes to your phone every time the queue empties: click to turn off" : "push to my phone every time the queue empties"}>
    <Icon n={n.armed ? "bell" : "bell-off"} size={16} /></button>;
}

/** Mark the project done: cleans it up (only the current stills, clips and edits stay), then flags it finished. */
function DoneButton({ name, finished }: { readonly name: string; readonly finished: boolean }) {
  const qc = useQueryClient();
  const [busy, setBusy] = useState(false);
  const toggle = async () => {
    try {
      if (finished) {
        await api(`/projects/${name}/finished`, "POST", { finished: false });
      } else {
        const c = await api<{ files: number; bytes: number }>(`/projects/${name}/cleanup`);
        const cleanupNote = c.files ? `This permanently deletes ${c.files} files (${(c.bytes / 1e6).toFixed(0)} MB): every previous version and every preview take except the current one.` : "Nothing to clean up.";
        if (!confirm(`Mark ${name} done?\n\n${cleanupNote} The current stills, clips, approvals and edits stay.`)) return;
        setBusy(true);
        await api(`/projects/${name}/finished`, "POST", { finished: true });
      }
      void qc.invalidateQueries({ queryKey: ["projects"] });
      void qc.invalidateQueries({ queryKey: ["project", name] });
    } catch (e) { alert((e as Error).message); }
    setBusy(false);
  };
  return (
    <button className={`qbtn donebtn ${finished ? "on" : ""}`} disabled={busy} onClick={toggle}
      title={finished ? "done — click to reopen (cleaned-up versions don't come back)" : "mark done: clean up every previous version and extra take, keep the current results"}>
      <Icon n="check-circle" size={16} />{busy && "cleaning…"}{!busy && finished && "Done"}
    </button>
  );
}

/** Remix: a new project <current>_remix_<text> holding the picked stages with their previews, takes, stills and clips.
 *  Lives inside the project switcher's dropdown (its "Remix" button), not a standalone menu. */
function RemixPanel({ current, onClose, onCreated }: { readonly current: string; readonly onClose: () => void; readonly onCreated: (name: string) => void }) {
  const qc = useQueryClient();
  const [suffix, setSuffix] = useState("");
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const nameInput = useRef<HTMLInputElement>(null);
  useEffect(() => { nameInput.current?.focus(); }, []);
  const p = useProject(current).data;
  const stages = p?.config.stages ?? [];
  const previews = p?.folders.preview?.stills ?? {};
  const stageIds = stages.map((s) => s.id).join(",");  // a refetch gives a new array: only a changed stage list resets the picks
  useEffect(() => { setPicked(new Set(stageIds ? stageIds.split(",").map(Number) : [])); }, [stageIds]);
  const toggle = (id: number) => { const n = new Set(picked); n.has(id) ? n.delete(id) : n.add(id); setPicked(n); };
  const create = async () => {
    setBusy(true); setErr("");
    try {
      const r = await api<{ name: string }>(`/projects/${current}/remix`, "POST", { suffix, stages: stages.map((s) => s.id).filter((id) => picked.has(id)) });
      void qc.invalidateQueries({ queryKey: ["projects"] });
      onCreated(r.name);
    } catch (e) { setErr(String((e as Error).message)); }
    setBusy(false);
  };
  return (
    <div className="remixbody">
      <div className="row">
        <b>Remix <span className="dim">{current}</span></b>
        <span className="spacer" />
        <button className="qbtn" title="back to the project list" onClick={onClose}><Icon n="x" size={14} /></button>
      </div>
      <div className="row">
        <span className="spacer" />
        <button className="btn sm" onClick={() => setPicked(new Set(stages.map((s) => s.id)))}>all</button>
        <button className="btn sm" onClick={() => setPicked(new Set())}>none</button>
      </div>
          <div className="remixlist">
            {stages.map((s) => (
              <label key={s.id} className={`remixitem ${picked.has(s.id) ? "on" : ""}`}>
                <input type="checkbox" checked={picked.has(s.id)} onChange={() => toggle(s.id)} />
                {previews[s.id] ? <img src={fileUrl(current, "preview", "still", s.id, previews[s.id])} alt="" /> : <span className="noimg" />}
                <span className="n">#{s.id}</span>
                <small>{(s.krea_scene || "").slice(0, 60)}</small>
              </label>
            ))}
          </div>
          <div className="row remixname">
            <span className="dim">{current}_remix_</span>
            <input ref={nameInput} placeholder="name" value={suffix} onChange={(e) => setSuffix(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && suffix && picked.size && !busy) void create(); }} />
          </div>
      <button className="btn primary" disabled={!suffix || !picked.size || busy} onClick={create}>{busy ? "copying…" : "Create"}</button>
      {err && <span className="err">{err}</span>}
    </div>
  );
}

/** RunPod "Start batch": while armed, Animate clicks from any project queue on the server and fire together at the
 * ⚙ batch size (one pod start for the whole batch), or immediately via "Send now". */
function BatchButton() {
  const qc = useQueryClient();
  const armed = useBatchArmed();
  const batch = useAnimateBatch().data;
  const entries = batch?.pending ?? [];
  const pending = entries.reduce((n, b) => n + b.stages.length, 0);
  const queued = entries.map((b) => `${b.project} (${b.folder || "root"}): ${b.stages.join(", ")}`).join("\n");
  const size = batch?.size ?? 4;
  const act = (p: Promise<unknown>) => p.then(
    () => { void qc.invalidateQueries({ queryKey: ["animate-batch"] }); void qc.invalidateQueries({ queryKey: ["jobs"] }); },
    (e: Error) => alert(e.message));
  if (!armed)
    return <button className="qbtn" title="Arm batching: Animate clicks from any project queue up and fire together on RunPod at the ⚙ batch size, sharing one pod start and model load"
      onClick={() => setBatchArmed(true)}><Icon n="layers" size={15} /><span className="btext">Batch</span></button>;
  return (
    <span className="batchctl">
      <button className={`qbtn ${pending ? "on" : ""}`} title={pending ? `${pending} stage(s) queued; Animate clicks keep queueing until the size is reached\n${queued}` : "batch armed"}
        onClick={() => { if (!pending) setBatchArmed(false); }}><Icon n="layers" size={15} />{pending}/{size}</button>
      {pending > 0 && <button className="qbtn" title="Send the queued batch now, even below the size" onClick={() => act(api("/animate-batch/flush", "POST"))}><Icon n="send" size={14} />Send</button>}
      <button className="qbtn danger" title={pending ? "Disarm and drop the queued batch" : "Disarm"} onClick={() => act(api("/animate-batch", "DELETE").then(() => setBatchArmed(false)))}><Icon n="x" size={15} /></button>
    </span>
  );
}

/** Shows API calls that failed with nobody handling them (cancel, delete, open in folder...), so a failure never looks like success. */
function ApiErrorToast() {
  const [failure, setFailure] = useState("");
  useEffect(() => {
    const onRejection = (e: PromiseRejectionEvent) => {
      if (!(e.reason instanceof ApiError)) return;  // e.g. a hover video's interrupted play()
      e.preventDefault();
      setFailure(`${e.reason.path} failed: ${e.reason.message}`);
    };
    window.addEventListener("unhandledrejection", onRejection);
    return () => window.removeEventListener("unhandledrejection", onRejection);
  }, []);
  return failure ? <button type="button" className="msg err toast" title="dismiss" onClick={() => setFailure("")}>{failure}</button> : null;
}

export default function App() {
  const hash = useHash();
  const qc = useQueryClient();
  const [showJobs, setShowJobs] = useState(false);
  const health = useHealth().data;
  const target = useAnimateTarget();
  const jobs = useJobs().data ?? [];
  const active = jobs.filter((j) => j.status === "running" || j.status === "queued").length;
  // auto-hide: any press outside the panel (and outside its toggle button) closes it
  useEffect(() => {
    if (!showJobs) return;
    const away = (e: PointerEvent) => {
      if (!(e.target as Element).closest(".jobs, .jobsbtn")) setShowJobs(false);
    };
    addEventListener("pointerdown", away);
    return () => removeEventListener("pointerdown", away);
  }, [showJobs]);
  const m = /^\/p\/([^/]+)(?:\/(\w+))?/.exec(hash);
  const project = m?.[1];
  const { unseen, dismiss } = useUnseen(jobs, project);
  const list = useQuery({ queryKey: ["projects"], queryFn: () => api<ProjectRow[]>("/projects"), enabled: !!project }).data;
  const oneoff = !!list?.find((r) => r.name === project)?.oneoff;  // a one-off has its own single-frame view and no Compose
  const tab = oneoff && m?.[2] === "compose" ? "board" : m?.[2] ?? "board";

  return (
    <div className="app">
      <header className="top">
        <a href="#/" className="brand"><span className="brandmark">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">
            <circle cx="6" cy="6" r="2.6" /><circle cx="18" cy="9" r="2.6" /><circle cx="9" cy="18" r="2.6" />
            <path d="M8.4 6.6 15.4 8.4M16.3 11.3 11.3 16.3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
          </svg></span>Comfy Studio</a>
        {project && (
          <span className="navpill">
            <ProjectSwitcher current={project} tab={tab} />
            <RefChip project={project} />
            <i className="sep" />
            <nav className="tabs">
              {(oneoff ? ["board", "settings"] : ["board", "compose", "civitai", "settings"]).map((t) => (
                <a key={t} href={`#/p/${project}/${t}`} className={tab === t ? "on" : ""}>{t}</a>
              ))}
            </nav>
            <i className="sep" />
            <DoneButton name={project} finished={!!list?.find((r) => r.name === project)?.finished} />
            <button className="qbtn danger" title="delete this project and its folder" onClick={() => {
              if (!confirm(`Delete ${project} and its folder permanently?`)) return;
              api(`/projects/${project}`, "DELETE").then(() => { void qc.invalidateQueries({ queryKey: ["projects"] }); location.hash = "#/"; }, (e: Error) => alert(e.message));
            }}><Icon n="trash" size={15} /></button>
          </span>
        )}
        <span className="spacer" />
        <span className="hact">
          <LocalMenu up={!!health?.local} server={health?.local_server} />
          {target === "runpod" ? <RunpodMenu /> : <CloudMenu noKey={!!health && !health.cloud_key} />}
          {target === "runpod" && <BatchButton />}
        </span>
        <i className="vsep" />
        <span className="hact">
          <NotifyButton />
          <SettingsButton />
          <button className={`hchip jobsbtn ${active ? "on" : ""}`} onClick={() => setShowJobs(!showJobs)}>
            <Icon n="list" size={15} />Jobs{active > 0 && <b className="jcount">· {active}</b>}{unseen.length > 0 && <span className="jbadge" title="jobs finished in other projects"><Icon n="check" size={11} />{unseen.length}</span>}
          </button>
        </span>
      </header>
      <main>
        {!project && hash.startsWith("/loras") && <Loras hash={hash} />}
        {!project && hash.startsWith("/voices") && <Voices hash={hash} />}
        {!project && !hash.startsWith("/loras") && !hash.startsWith("/voices") && <Projects forceNew={hash === "/new"} />}
        {project && list && tab === "board" && (oneoff ? <OneOff key={project} name={project} /> : <Board key={project} name={project} />)}
        {project && list && tab === "compose" && <Compose name={project} />}
        {project && list && tab === "civitai" && !oneoff && <Civitai name={project} />}
        {project && list && tab === "settings" && <Settings name={project} />}
      </main>
      {showJobs && <JobsPanel jobs={jobs} unseen={unseen} dismiss={dismiss} onClose={() => setShowJobs(false)} />}
      <ApiErrorToast />
    </div>
  );
}
