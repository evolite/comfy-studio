import { useEffect, useRef, useState } from "react";
import { api, Job } from "./api";
import { ACTION_LABEL, jobLink } from "./Board";
import { useNewJobs } from "./hooks";

// "1,2,3,5" -> "1–3, 5"
const ranges = (ns: number[]) => {
  const s = [...ns].sort((a, b) => a - b), out: string[] = [];
  for (let i = 0; i < s.length;) {
    let j = i;
    while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
    out.push(j - i >= 2 ? `${s[i]}–${s[j]}` : s.slice(i, j + 1).join(", "));
    i = j + 1;
  }
  return out.join(", ");
};
const ago = (t: number) => {
  const d = Math.max(0, Date.now() / 1000 - (t > 1e12 ? t / 1000 : t));
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};

const secs = (s: number) => s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`;
const fmtLeft = (sec: number) => {
  const s = Math.round(sec);
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
};
const statusPill = (j: Job) => {
  if (j.status === "done") return "ok";
  if (j.status === "failed") return "bad";
  if (j.status === "running") return `run lane-${j.lane === "local" ? "local" : "cloud"}`;
  return "";
};

export default function JobsPanel({ jobs, unseen, dismiss, onClose }: { readonly jobs: Job[]; readonly unseen: Job[]; readonly dismiss: (ids: string[]) => void; readonly onClose: () => void }) {
  const [sel, setSel] = useState<string | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const box = useRef<HTMLPreElement>(null);
  const stick = useRef(true);
  const cur = sel ?? jobs[0]?.id ?? null;
  const job = jobs.find((j) => j.id === cur);
  useEffect(() => {
    if (!cur) return;
    setLines([]); stick.current = true;
    const es = new EventSource(`/api/jobs/${cur}/events`);
    es.onmessage = (e) => setLines((l) => [...l, JSON.parse(e.data).line]);
    es.addEventListener("end", () => es.close());
    return () => es.close();
  }, [cur]);
  // follow the tail only while the reader is at the bottom
  useEffect(() => { if (stick.current) box.current?.scrollTo(0, box.current.scrollHeight); }, [lines]);
  const active = (j: Job) => j.status === "running" || j.status === "queued";
  const news = useNewJobs(jobs);
  return (
    <div className="jobs">
      <div className="row"><b>Jobs</b>{jobs.some(active) && <span className="dim">{jobs.filter(active).length} active</span>}<span className="spacer" /><button className="btn sm" title="close" onClick={onClose}>×</button></div>
      {unseen.length > 0 && (
        <div className="finished">
          <div className="row dim"><span>Finished in other projects</span><span className="spacer" /><button className="btn sm" title="dismiss all" onClick={() => dismiss(unseen.map((j) => j.id))}>×</button></div>
          {unseen.map((j) => (
            <div key={j.id} className={`fin ${j.status}`}>
              <span className={`pill ${j.status === "done" ? "ok" : "bad"}`}>{j.status}</span>
              <a href={jobLink(j)} onClick={() => { dismiss(unseen.filter((u) => u.project === j.project).map((u) => u.id)); onClose(); }} title={`go to ${j.project}`}>
                <b>{ACTION_LABEL[j.action] ?? j.action}</b> · {j.project} →
              </a>
              <span className="spacer" /><span className="dim">{j.ended ? ago(j.ended) : ""}</span>
              <button className="btn sm" title="dismiss" onClick={() => dismiss([j.id])}>×</button>
            </div>
          ))}
        </div>
      )}
      <div className="joblist">
        {jobs.map((j) => {
          const st = (j.req.stages as number[] | undefined) ?? [];
          return (
            <div key={j.id} className={`job ${j.id === cur ? "on" : ""}`}>
              <div className="jtop">
                <span className={`pill ${statusPill(j)}`}>{j.status}</span>
                {news.has(j.id) && <i className="badge new" title="finished since you last opened its project — open it (open →) to see the renders">new</i>}
                <button type="button" className="namebtn" aria-pressed={j.id === cur} onClick={() => setSel(j.id)}><b>{ACTION_LABEL[j.action] ?? j.action}</b></button>
                {j.per_stage != null && <span className="pill" title="time per stage: wall clock from start to the last finished stage, divided by the stages (takes) finished, model loading included">⏱ {j.per_stage >= 60 ? `${Math.floor(j.per_stage / 60)}m ${Math.round(j.per_stage % 60)}s` : `${Math.round(j.per_stage)}s`}</span>}
                {j.mem_freed && <span className="memwarn" title={`memory check: ${j.mem_freed} at the start, so ComfyUI was freed and the models reloaded`}>⚠</span>}
                <i className={`badge ${j.cost}`}>{j.cost}</i>
                <span className="spacer" />
                {active(j) && <button className="btn sm danger" title="cancel this job" onClick={(e) => { e.stopPropagation(); void api(`/jobs/${j.id}/cancel`, "POST"); }}>⏹</button>}
                {(j.status === "failed" || j.status === "cancelled") && <button className="btn sm" title={`retry: submit this job again${st.length && j.action !== "edit" ? " (only the stages that didn't finish)" : ""}${j.cost === "paid" ? " — spends credits" : ""}`} onClick={(e) => { e.stopPropagation(); void api<Job>(`/jobs/${j.id}/retry`, "POST").then((n) => setSel(n.id)); }}>↻</button>}
              </div>
              {active(j) && (() => {
                const total = st.length * Math.max(1, Number(j.req.takes) || 1) || Object.keys(j.stages).length, done = Object.values(j.stages).filter((v) => v === "done" || v === "failed").length;
                const pr = j.progress;
                const approx = pr && !pr.exact ? "~" : "";
                const title = pr ? `${done}/${total} stages · ${approx}${fmtLeft(pr.left)} left (from past runs)` : `${done}/${total} stages`;
                return total > 0 && <div className={`jbar ${j.status === "queued" ? "queued" : ""}`} title={title}><i style={{ width: `${(pr ? pr.fraction : done / total) * 100}%` }} /></div>;
              })()}
              {j.took && Object.keys(j.took).length > 0 && <div className="jtook dim" title="time per stage: ComfyUI's processing time (model loading included for the first)">
                {Object.entries(j.took).map(([k, s]) => <span key={k}>#{k.replace(".t", " take ")} {secs(s)}</span>)}
              </div>}
              <div className="jsub dim">
                <span className="jproj" title={j.project}>{j.project}</span>
                {st.length > 0 && <span>· stage{st.length > 1 ? "s" : ""} {ranges(st)}</span>}
                {active(j) && j.progress && j.progress.left > 0 && <span>· {j.progress.exact ? "" : "~"}{fmtLeft(j.progress.left)} left</span>}
                <span className="spacer" /><span>{ago(j.created)}</span>
              </div>
            </div>
          );
        })}
        {jobs.length === 0 && <div className="dim">no jobs yet</div>}
      </div>
      {job && <div className="dim jlogh" title={job.argv}>
        <a href={jobLink(job)} onClick={onClose} title={`go to ${job.project}'s ${job.action === "edit" ? "Compose (the edit is under Edits)" : "Board"}`}>open →</a> · {job.argv}
      </div>}
      <pre ref={box} className="log" onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24; }}>{lines.length ? lines.join("\n") : <span className="dim">{job ? "no output yet" : ""}</span>}</pre>
    </div>
  );
}
