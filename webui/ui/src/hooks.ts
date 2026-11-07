import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, Credits, Health, Job, LoraLibrary, Project, Registry, RunpodBalance, VoiceLibrary } from "./api";

export const useRegistry = () => useQuery({ queryKey: ["registry"], queryFn: () => api<Registry>("/registry"), staleTime: Infinity });
export const useLoras = () => useQuery({ queryKey: ["loras"], queryFn: () => api<LoraLibrary>("/loras"), staleTime: 30000 });
export const useVoices = () => useQuery({ queryKey: ["voices"], queryFn: () => api<VoiceLibrary>("/voices"), staleTime: 30000 });
/** The ⚙ animate target: which paid lane the header's money chip shows (Comfy credits or the RunPod balance). */
export const useAnimateTarget = () =>
  useQuery({ queryKey: ["app-settings"], queryFn: () => api<{ animate_target: "cloud" | "runpod" }>("/app-settings") }).data?.animate_target ?? "cloud";
export const useRunpodBalance = (on: boolean) =>
  useQuery({ queryKey: ["runpod-balance"], queryFn: () => api<RunpodBalance>("/runpod/balance"), enabled: on, refetchInterval: 60000, staleTime: 20000 });
export const useCredits = () => useQuery({ queryKey: ["credits"], queryFn: () => api<Credits>("/credits"), refetchInterval: 60000, staleTime: 20000 });
export const useHealth = () => useQuery({ queryKey: ["health"], queryFn: () => api<Health>("/health"), refetchInterval: 10000 });
/** While a job is queued/running for this project, re-read it every 3s so new stills/clips appear as each one lands. */
export function useProject(name: string) {
  const qc = useQueryClient();
  return useQuery({
    queryKey: ["project", name], queryFn: () => api<Project>(`/projects/${name}`),
    refetchInterval: () => ((qc.getQueryData<Job[]>(["jobs"]) ?? []).some((j) => j.project === name && ["queued", "running"].includes(j.status)) ? 3000 : false),
  });
}

export function useJobs() {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["jobs"], queryFn: () => api<Job[]>("/jobs"),
    refetchInterval: (query) => ((query.state.data ?? []).some((j) => j.status === "running" || j.status === "queued") ? 1200 : 5000),
  });
  // when a job leaves running/queued, refresh the project it touched so new images/clips appear
  const seen = useRef<Record<string, string>>({});
  useEffect(() => {
    for (const j of q.data ?? []) {
      const prev = seen.current[j.id];
      if (prev && prev !== j.status && !["running", "queued"].includes(j.status)) {
        void qc.invalidateQueries({ queryKey: ["project", j.project] });
        if (j.cost === "paid") { void qc.invalidateQueries({ queryKey: ["credits"] }); void qc.invalidateQueries({ queryKey: ["runpod-balance"] }); }
      }
      seen.current[j.id] = j.status;
    }
  }, [q.data, qc]);
  return q;
}

export function useRun(project: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: Record<string, unknown>) => api<Job>(`/projects/${project}/run`, "POST", req),
    onSettled: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export type BatchEntry = { project: string; action: string; folder: string; stages: number[]; added: number };
export type BatchAddResult = { job: Job | null; pending: number; size: number };

/** The RunPod "Start batch" queue, across all projects: animate clicks the server holds until the batch size is reached. */
export function useAnimateBatch() {
  return useQuery({
    queryKey: ["animate-batch"],
    queryFn: () => api<{ size: number; pending: BatchEntry[] }>("/animate-batch"),
    refetchInterval: 4000,
  });
}

export function useAnimateBatchAdd(project: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: Record<string, unknown>) => api<BatchAddResult>(`/projects/${project}/animate-batch`, "POST", req),
    onSuccess: (r) => {
      void qc.invalidateQueries({ queryKey: ["animate-batch"] });
      if (!r.pending) void qc.invalidateQueries({ queryKey: ["jobs"] });  // the batch reached its size and fired
    },
  });
}

export type Active = { lane: "local" | "cloud"; action: string; queued: boolean; folder: string };

/** Preview takes per stage a job renders (1 for everything but a multi-take Preview). */
export const jobTakes = (j: Job) => Math.max(1, Number(j.req.takes) || 1);
/** How many of a stage's renders in this job have finished (done or failed); keys are "N", or "N.tK" for takes. */
export const finishedTakes = (j: Job, id: number) =>
  Object.entries(j.stages ?? {}).filter(([k, v]) => (k === String(id) || k.startsWith(`${id}.t`)) && (v === "done" || v === "failed")).length;

/** Stage ids -> takes still to come from this project's running/queued multi-take Preview jobs (grey placeholders on the board). */
export function pendingTakes(jobs: Job[], project: string): Record<number, number> {
  const out: Record<number, number> = {};
  for (const j of jobs) {
    if (j.project !== project || !["queued", "running"].includes(j.status) || jobTakes(j) < 2) continue;
    for (const id of (j.req.stages as number[] | undefined) ?? []) out[id] = (out[id] ?? 0) + jobTakes(j) - finishedTakes(j, id);
  }
  return out;
}

/** Stage ids with an active (queued/running) job for this project -> that job's lane/action/target folder. */
export function activeByStage(jobs: Job[], project: string): Record<number, Active> {
  const out: Record<number, Active> = {};
  for (const j of jobs) {
    if (j.project !== project || !["queued", "running"].includes(j.status)) continue;
    const ids = (j.req.stages as number[] | undefined) ?? [];
    // the script reports per-stage progress: stages already done/failed stop pulsing; with no explicit stage list, use the reported ones
    const st = j.stages ?? {};
    const all = ids.length ? ids : [...new Set(Object.keys(st).map((k) => Number(k.split(".")[0])))];
    for (const id of all) {
      if (finishedTakes(j, id) >= jobTakes(j)) continue;
      out[id] = { lane: j.lane === "local" ? "local" : "cloud", action: j.action, queued: j.status === "queued", folder: (j.req.folder as string | undefined) ?? "" };
    }
  }
  return out;
}

const FINISHED = new Set(["done", "failed"]);

/** Jobs that finished in a project other than the one on screen and that you haven't looked at yet.
 * Anything already finished when the page loads, or finishing while you're in its project, counts as seen;
 * opening a project (or dismissing) clears its entries. */
export function useUnseen(jobs: Job[], current: string | undefined) {
  const [seen, setSeen] = useState<Set<string>>(new Set());
  const primed = useRef(false);
  const live = useRef(false);
  useEffect(() => {
    if (!live.current && !jobs.length) return;  // wait for the first real list
    const mark = jobs.filter((j) => FINISHED.has(j.status) && (!primed.current || j.project === current)).map((j) => j.id);
    primed.current = true; live.current = true;
    if (mark.some((id) => !seen.has(id))) setSeen((s) => new Set([...s, ...mark]));
  }, [jobs, current]); // eslint-disable-line react-hooks/exhaustive-deps
  const unseen = jobs.filter((j) => FINISHED.has(j.status) && !seen.has(j.id) && j.project !== current);
  const dismiss = (ids: string[]) => setSeen((s) => new Set([...s, ...ids]));
  return { unseen, dismiss };
}

/** Expected seconds of each local action over these stages, from past runs' timings (fetched while the run menu is open). */
export type Estimate = Record<string, { seconds: number; known: number; of: number; parallel: boolean; load: number }>;
export const useEstimate = (project: string, ids: number[], enabled: boolean) =>
  useQuery({ queryKey: ["estimate", project, ids.join(",")], queryFn: () => api<Estimate>(`/projects/${project}/estimate?stages=${ids.join(",")}`), enabled: enabled && ids.length > 0, staleTime: 15000 });

export const fmtSecs = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${Math.round(s)}s`);


const VIEWED_KEY = "jobs-viewed-at";  // project -> when its board was last opened (jobs ended before that are not "new")
const readViewed = (): Record<string, number> => {
  try { return JSON.parse(localStorage.getItem(VIEWED_KEY) ?? "{}"); } catch { return {}; }
};

/** Ids of done jobs whose renders you haven't looked at yet: a job is "new" from when it finishes until you open its
 * project (the board is where renders are reviewed -- Board.tsx dispatches "project-viewed"). The first load primes
 * every already-finished job as viewed, so history doesn't all flash new. */
export function useNewJobs(jobs: Job[]) {
  const [viewedAt, setViewedAt] = useState(readViewed);
  const primed = useRef(false);
  useEffect(() => {
    const bump = (e: Event) => {
      const project = (e as CustomEvent<string>).detail;
      setViewedAt((cur) => {
        const next = { ...cur, [project]: Date.now() / 1000 };
        try { localStorage.setItem(VIEWED_KEY, JSON.stringify(next)); } catch { /* ignore */ }
        return next;
      });
    };
    addEventListener("project-viewed", bump);
    return () => removeEventListener("project-viewed", bump);
  }, []);
  useEffect(() => {
    if (primed.current) return;
    primed.current = true;
    if (localStorage.getItem(VIEWED_KEY)) return;
    const fresh = Object.fromEntries([...new Set(jobs.filter((j) => j.status === "done").map((j) => j.project))].map((p) => [p, Date.now() / 1000]));
    setViewedAt(fresh);
    try { localStorage.setItem(VIEWED_KEY, JSON.stringify(fresh)); } catch { /* ignore */ }
  }, [jobs]);
  return new Set(jobs.filter((j) => j.status === "done" && j.ended != null && (viewedAt[j.project] ?? 0) < j.ended).map((j) => j.id));
}
