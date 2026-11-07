import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { SECTIONS } from "./settingsSchema";

type Row = { key?: string; label?: string; name?: string; this: unknown; current: unknown };
type Diff = { unknown?: true; fields: Row[]; loras: Row[]; settings: Row[]; prompt: [string, string][] | null };
type Info = { id: string; current: string | null; record: Record<string, unknown> | null; current_record: Record<string, unknown> | null; diff: Diff | null };

/** Settings key -> its label in the Settings page (walks the schema, whatever its nesting). */
function labels(): Record<string, string> {
  const out: Record<string, string> = {};
  const walk = (x: unknown) => {
    if (Array.isArray(x)) x.forEach(walk);
    else if (x && typeof x === "object") {
      const o = x as Record<string, unknown>;
      if (typeof o.key === "string" && typeof o.label === "string") out[o.key] = o.label;
      Object.values(o).forEach(walk);
    }
  };
  walk(SECTIONS);
  return out;
}
const LABELS = labels();
const show = (v: unknown) => {
  if (v === undefined || v === null || v === "") return "—";
  if (typeof v === "number") return String(+v.toFixed(4));
  if (typeof v === "string" || typeof v === "boolean") return String(v);
  return JSON.stringify(v) ?? "—";
};

/** "Changes vs current": the fields where a preview take differs from the current one, and the prompt as a word diff. */
let counts: Record<string, number> = {};

export default function TakeDiff({ project, sid, id, currentId }: { readonly project: string; readonly sid: number; readonly id: string; readonly currentId?: string }) {
  const q = useQuery({ queryKey: ["take", project, sid, id, currentId], queryFn: () => api<Info>(`/projects/${project}/stages/${sid}/takes/${id}`) });
  if (q.isLoading) return <div className="tdiff dim">loading…</div>;
  if (q.error) return <div className="tdiff err">{String((q.error as Error).message)}</div>;
  const info = q.data!;
  if (info.id === info.current) return <div className="tdiff dim">This is the current take.</div>;
  const d = info.diff;
  if (!d || d.unknown) return <div className="tdiff dim">{!info.record ? "No settings recorded for this take (made before takes were kept, or the file was replaced by hand)." : "No settings recorded for the current take, so there is nothing to compare against."}</div>;
  const rows: { label: string; a: unknown; b: unknown }[] = [
    ...d.fields.map((f) => ({ label: f.label ?? f.key ?? "", a: f.this, b: f.current })),
    ...d.loras.map((l) => ({ label: `LoRA ${l.name}`, a: l.this ?? "off", b: l.current ?? "off" })),
    ...d.settings.map((s) => ({ label: LABELS[s.key ?? ""] ?? s.key ?? "", a: s.this, b: s.current })),
  ];
  return (
    <div className="tdiff">
      <div className="paneltitle">Changes vs current</div>
      {rows.length === 0 && !d.prompt && <div className="dim">Same prompt and settings; only the noise differs.</div>}
      {rows.length > 0 && <table>
        <thead><tr><th /><th>this take</th><th>current</th></tr></thead>
        <tbody>{rows.map((r) => <tr key={r.label}><td>{r.label}</td><td className="this">{show(r.a)}</td><td>{show(r.b)}</td></tr>)}</tbody>
      </table>}
      {d.prompt && <>
        {(() => { counts = {}; return null; })()}
        <div className="ctlhead">Prompt <small className="dim"><ins>added in this take</ins> · <del>only in current</del></small></div>
        <div className="tdiff-prompt">{d.prompt.map(([op, s]) => {
          const k = `${op}:${s}`;
          counts[k] = (counts[k] ?? 0) + 1;
          const key = `${k}#${counts[k]}`;
          if (op === "+") return <ins key={key}>{s}</ins>;
          if (op === "-") return <del key={key}>{s}</del>;
          return <span key={key}>{s}</span>;
        })}</div>
      </>}
    </div>
  );
}
