import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, fileUrl, type Project } from "./api";

type Version = { folder: string; kind: "still" | "clip"; file: string; mtime: number };

/** A stage's earlier results (pipeline/history.py), grouped by folder with the current one first; Restore swaps one back in and
 *  the result it replaces becomes a previous version itself, so a roll-back can be undone. Preview renders are takes, not here. */
export default function Versions({ p, sid, first, onClose }: { readonly p: Project; readonly sid: number; readonly first?: string; readonly onClose: () => void }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["history", p.name, sid], queryFn: () => api<Version[]>(`/projects/${p.name}/stages/${sid}/history`) });
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  useEffect(() => { const k = (e: KeyboardEvent) => e.key === "Escape" && onClose(); window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k); }, [onClose]);
  const rows = q.data ?? [];
  const groups = [...new Set(rows.map((r) => `${r.folder}|${r.kind}`))]
    .sort((a, b) => (b.startsWith(`${first ?? "\0"}|`) ? 1 : 0) - (a.startsWith(`${first ?? "\0"}|`) ? 1 : 0));
  const restore = async (v: Version) => {
    setBusy(v.file); setErr("");
    try {
      await api(`/projects/${p.name}/stages/${sid}/restore`, "POST", { file: v.file });
      await Promise.all([qc.invalidateQueries({ queryKey: ["project", p.name] }), q.refetch()]);
    } catch (e) { setErr((e as Error).message); }
    setBusy("");
  };
  const media = (kind: string, src: string) => kind === "clip"
    ? <video muted loop preload="metadata" src={src} onMouseEnter={(e) => void e.currentTarget.play()} onMouseLeave={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0; }} />
    : <img loading="lazy" alt="" src={src} />;
  return createPortal(
    // the backdrop click closes; Escape is on the effect above and the panel has a real close button
    <div className="versions">
      <button type="button" className="backdrop" aria-label="close" tabIndex={-1} onClick={onClose} />
      <div className="versionspanel">
        <div className="row"><h3>Stage #{sid} — previous versions</h3><span className="spacer" /><button className="btn sm" onClick={onClose}>✕</button></div>
        <div className="dim">Restore puts a version back as the stage's result; the one it replaces moves in here, so you can switch back. Approvals (✓) stay as they are. Preview renders are under the preview's takes instead.</div>
        {err && <div className="msg">{err}</div>}
        {q.isLoading && <div className="dim">loading…</div>}
        {!q.isLoading && rows.length === 0 && <div className="dim">No previous versions — a re-run, re-roll, face pass or remove keeps the result it replaces here.</div>}
        {groups.map((g) => {
          const [folder, kind] = g.split("|") as [string, "still" | "clip"];
          const curV = kind === "clip" ? p.folders[folder]?.clips[sid] : p.folders[folder]?.stills[sid];
          return (
            <div key={g} className="vgroup">
              <h4>{folder || "root"}{kind === "clip" ? " · clip" : ""}</h4>
              <div className="vgrid">
                {curV ? <figure className="vcur">{media(kind, fileUrl(p.name, folder, kind, sid, curV))}<figcaption>current · {new Date(curV * 1000).toLocaleString()}</figcaption></figure>
                  : <figure className="vcur"><div className="ph">no current {kind}</div><figcaption>current</figcaption></figure>}
                {rows.filter((r) => r.folder === folder && r.kind === kind).map((v) => (
                  <figure key={v.file}>
                    {media(kind, `/api/projects/${p.name}/files/_debug/${v.file}?v=${v.mtime}`)}
                    <figcaption>{new Date(v.mtime * 1000).toLocaleString()}</figcaption>
                    <button className="btn sm primary" disabled={!!busy} onClick={() => restore(v)}>{busy === v.file ? "restoring…" : "Restore"}</button>
                  </figure>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>,
    document.body,  // not inside the stage card: its click handlers and styles stay out
  );
}
