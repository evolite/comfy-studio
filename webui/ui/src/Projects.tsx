import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ProjectRow } from "./api";
import { useRegistry } from "./hooks";
import { RefChip } from "./controls";

export default function Projects({ forceNew = false }: { readonly forceNew?: boolean }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["projects"], queryFn: () => api<ProjectRow[]>("/projects") });
  const sets = useQuery({ queryKey: ["stage_sets"], queryFn: () => api<string[]>("/stage_sets") });
  const reg = useRegistry().data;
  const [filter, setFilter] = useState("");
  const [creating, setCreating] = useState(forceNew);
  useEffect(() => { setCreating(forceNew); }, [forceNew]);
  useEffect(() => {
    if (!creating) return;
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") { setCreating(false); location.hash = "#/"; } };
    addEventListener("keydown", esc);
    return () => removeEventListener("keydown", esc);
  }, [creating]);
  const [form, setForm] = useState({ name: "", engine: "full", from_stage_set: "", oneoff: false });
  const create = useMutation({
    mutationFn: () => api<{ name: string }>("/projects", "POST", { ...form, from_stage_set: form.from_stage_set || undefined, oneoff: form.oneoff || undefined }),
    onSuccess: (r: { name: string }) => { qc.invalidateQueries({ queryKey: ["projects"] }); location.hash = `#/p/${r.name}/board`; },
  });
  const del = useMutation({
    mutationFn: (name: string) => api(`/projects/${name}`, "DELETE"),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
    onError: (e: Error) => alert(e.message),
  });
  const [showDone, setShowDone] = useState(() => { try { return localStorage.getItem("projects:showDone") === "1"; } catch { return false; } });
  const nDone = (q.data ?? []).filter((r) => r.finished).length;
  // open projects first; done ones (dimmed) only when shown
  const rows = (q.data ?? []).filter((r) => r.name.includes(filter) && (showDone || !r.finished)).sort((a, b) => Number(!!a.finished) - Number(!!b.finished));
  return (
    <div className="page">
      <div className="row">
        <input placeholder="filter projects…" value={filter} onChange={(e) => setFilter(e.target.value)} />
        <button className="btn primary" onClick={() => { const n = !creating; setCreating(n); if (!n) location.hash = "#/"; }}>＋ New project</button>
        {nDone > 0 && <button className={`btn ${showDone ? "primary" : ""}`} title="show the projects marked done" onClick={() => {
          setShowDone(!showDone); try { localStorage.setItem("projects:showDone", showDone ? "0" : "1"); } catch { /* ignore */ } }}>✓ {nDone}</button>}
        <span className="spacer" />
        <a className="btn" href="#/loras" title="the LoRA library: notes, tips, ranges, local/cloud, register local files">⚙ LoRAs</a>
        <a className="btn" href="#/voices" title="the voice library: named voices a stage's motion fills its {voice} slots from, each with a sample to listen to">🎙 Voices</a>
      </div>
      {creating && (
        <div className="card form">
          <input placeholder={form.oneoff ? "name (optional, auto-dated)" : "20261002_my_project"} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <select value={form.engine} onChange={(e) => setForm({ ...form, engine: e.target.value })}>
            {Object.keys(reg?.settings_keys ?? { full: 1, krea2: 1, bgswap: 1 }).map((e) => <option key={e}>{e}</option>)}
          </select>
          {!form.oneoff && <select value={form.from_stage_set} onChange={(e) => setForm({ ...form, from_stage_set: e.target.value })}>
            <option value="">blank (2 stages)</option>
            {sets.data?.map((s) => <option key={s}>{s}</option>)}
          </select>}
          <button type="button" className={`btn oneoffbtn ${form.oneoff ? "on" : ""}`} aria-pressed={form.oneoff} title="one-off: a single-stage project" onClick={() => setForm({ ...form, oneoff: !form.oneoff })}>⚡</button>
          <button className="btn primary" disabled={(!form.oneoff && !form.name) || create.isPending} onClick={() => create.mutate()}>Create</button>
          <button className="btn" onClick={() => { setCreating(false); location.hash = "#/"; }}>Cancel</button>
          {create.error && <span className="err">{String(create.error.message)}</span>}
        </div>
      )}
      <div className="grid projects">
        {rows.map((r) => (
          <a key={r.name} className={`card project ${r.finished ? "finished" : ""}`} href={`#/p/${r.name}/board`}>
            <div className="name">{r.finished && <span className="donemark" title="done">✓</span>}{r.name}{r.oneoff && <span className="oneoff" title="one-off (single stage)">⚡</span>}</div>
            <span className="refproj"><RefChip project={r.name} /></span>
            <button className="btn sm danger delproj" title="delete project and its folder" onClick={(e) => {
              e.preventDefault(); e.stopPropagation();
              if (confirm(`Delete ${r.name}?\n${r.stills} stills, ${r.clips} clips and ${r.edits} edits are deleted from disk permanently.`)) del.mutate(r.name);
            }}>🗑</button>
            <div className="meta">{r.engine} · {r.stages} stages · {r.stills} stills · {r.clips} clips · {r.edits} edits</div>
          </a>
        ))}
      </div>
    </div>
  );
}
