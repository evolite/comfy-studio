/** The Settings hub as data: sections in pipeline order, each a list of controls over `project.json` settings keys.
 * A control shows only when the project's engine has its key (registry settings_keys). Defaults come from the registry's
 * template_defaults; `fallback` covers keys handled in code (no template value). Keys not listed here land in Advanced. */
export type Val = string | number | boolean;
export type Lora = Record<string, number>;

export type Ctl = {
  key: string; label: string; help?: string;
  kind: "slider" | "seg" | "select" | "toggle" | "text" | "aspect";
  min?: number; max?: number; step?: number; unit?: string;
  options?: string[];               // seg / select; "models" = reg.model_choices[key]
  models?: boolean;
  on?: Val;                         // toggle: the value written when on (off removes the key), default true/false
  fallback?: Val | Lora;
  hint?: [number, number];          // slider: a highlighted "good range" band
  fold?: string;                    // grouped under a collapsed sub-heading
};
export type Panel = { title: string | Record<string, string>; sub?: string; ctls: Ctl[]; engines?: string[] };
export type Section = { id: string; title: string; sub?: string; engines?: string[]; panels: Panel[] };

export const SAMPLERS = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "dpmpp_3m_sde", "res_multistep", "lcm", "ddim", "uni_pc"];
export const SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform", "beta", "kl_optimal"];
export const ASPECTS = ["2:3 (Portrait Photo)", "3:4 (Portrait Standard)", "9:16 (Portrait Widescreen)", "1:1 (Square)",
  "3:2 (Photo)", "4:3 (Standard)", "16:9 (Widescreen)", "21:9 (Ultrawide)"];

/** ResolutionSelector's output size for a ratio label ("2:3 (...)") and megapixels (1 MP = 1024², multiple of 8). */
export function resolution(aspect: string, mp: number): [number, number] {
  const m = /^(\d+):(\d+)/.exec(aspect) ?? ["", "1", "1"];
  const r = Number(m[1]) / Number(m[2]);
  const px = mp * 1024 * 1024;
  const r8 = (v: number) => Math.max(8, Math.round(v / 8) * 8);
  return [r8(Math.sqrt(px * r)), r8(Math.sqrt(px / r))];
}

const sampler = (prefix: string, fold = "Sampler"): Ctl[] => [
  { key: `${prefix}_sampler`, label: "Sampler", kind: "select", options: SAMPLERS, fold },
  { key: `${prefix}_scheduler`, label: "Scheduler", kind: "select", options: SCHEDULERS, fold },
];

export const SECTIONS: Section[] = [
  { id: "preview", title: "Preview", sub: "Krea2 base render: the pose and composition everything else builds on", panels: [{ title: "", ctls: [
    { key: "aspect_ratio", label: "Shape", kind: "aspect", options: ASPECTS },
    { key: "megapixels", label: "Size", kind: "slider", min: 0.5, max: 2, step: 0.05, unit: "MP",
      help: "Krea2 is trained around 1 MP; bigger previews cost more time and can repeat limbs" },
    { key: "krea_unet", label: "Model", kind: "select", models: true },
    { key: "krea_vae", label: "VAE", kind: "select", models: true, help: "default: this machine's Krea2 VAE; the HDR VAE is local only" },
    { key: "krea_steps", label: "Steps", kind: "slider", min: 4, max: 30, step: 1, fold: "Sampler", help: "turbo models: 8" },
    { key: "krea_cfg", label: "CFG", kind: "slider", min: 1, max: 6, step: 0.1, fold: "Sampler", help: "turbo models: 1 (above 1 the negative starts to count)" },
    ...sampler("krea"),
  ] }] },

  { id: "upscale", title: "Upscale", sub: "IL + upscale or the latent upscale, side by side: ✓ the one you keep", panels: [
    { title: { full: "IL + upscale", krea2: "Krea2 upscale", bgswap: "Upscale" }, sub: "→ IL/  (full engine: Illustrious over ControlNets, then iterative upscale)", engines: ["full", "krea2"], ctls: [
      { key: "il_checkpoint", label: "Checkpoint", kind: "select", models: true },
      { key: "il_denoise", label: "IL denoise", kind: "slider", min: 0, max: 1, step: 0.01, help: "how much the anime pass repaints Krea2's image; 0.2 keeps the pose and face" },
      { key: "il_mode", label: "Skip the 1 MP IL pass", kind: "toggle", on: "upscale_only", help: "ESRGAN + one IL resample on Krea2's image (cheaper, closer to the preview)" },
      { key: "hed_strength", label: "HED edges", kind: "slider", min: 0, max: 1, step: 0.01, fold: "ControlNets" },
      { key: "normal_strength", label: "Normal map", kind: "slider", min: 0, max: 1, step: 0.01, fold: "ControlNets" },
      { key: "canny_strength", label: "Canny", kind: "slider", min: 0, max: 1, step: 0.01, fold: "ControlNets" },
      { key: "edge_controlnet_strength", label: "All three", kind: "slider", min: 0, max: 1, step: 0.01, fold: "ControlNets",
        help: "overrides the three above; 0 drops the ControlNets entirely" },
      { key: "il_steps", label: "IL steps", kind: "slider", min: 10, max: 60, step: 1, fold: "IL sampler" },
      { key: "il_cfg", label: "IL CFG", kind: "slider", min: 1, max: 10, step: 0.1, fold: "IL sampler" },
      ...sampler("il", "IL sampler"),
      { key: "upscale_model", label: "ESRGAN model", kind: "select", models: true },
      { key: "upscale_factor", label: "Scale", kind: "slider", min: 1, max: 2.5, step: 0.05, unit: "×" },
      { key: "upscale_iterations", label: "Hops", kind: "slider", min: 1, max: 4, step: 1, help: "the scale is reached in this many steps" },
      { key: "upscale_denoise", label: "Upscale denoise", kind: "slider", min: 0, max: 0.6, step: 0.01, help: "0 = plain ESRGAN, no re-diffusion" },
      { key: "upscale_step_mode", label: "Hop sizes", kind: "seg", options: ["simple", "geometric"], fold: "Upscale sampler" },
      { key: "upscale_steps", label: "Steps", kind: "slider", min: 4, max: 60, step: 1, fold: "Upscale sampler", help: "runs every step even at low denoise: the main cost lever" },
      { key: "upscale_cfg", label: "CFG", kind: "slider", min: 1, max: 10, step: 0.1, fold: "Upscale sampler" },
      ...sampler("upscale", "Upscale sampler"),
    ] },
    { title: "Latent (SesquiLSR)", sub: "→ hires/  (local: latent upscale + Krea2 re-diffusion, adds real texture)", ctls: [
      { key: "hires_scale", label: "Scale", kind: "slider", min: 1, max: 2, step: 0.05, unit: "×", fallback: 1.5, help: "2× samples ~4 MP: more VRAM, ~3× the time per step" },
      { key: "hires_denoise", label: "Denoise", kind: "slider", min: 0.01, max: 0.7, step: 0.01, fallback: 0.05, hint: [0.03, 0.12],
        help: "0.05 mostly sharpens the upscale; higher re-draws more skin/hair detail (Krea2 turbo models redraw quickly)" },
      { key: "hires_vae", label: "VAE", kind: "select", models: true, help: "encodes the preview and decodes the result; unset = the preview's VAE" },
      { key: "hires_prompt", label: "Extra prompt for this pass", kind: "text", fallback: "",
        help: "added to the end of each stage's prompt, only on the latent pass, e.g. \"smooth flawless skin, soft even texture\" (describe what you want, no negations)" },
      { key: "hires_lora_scale", label: "LoRA strength", kind: "slider", min: 0, max: 1, step: 0.05, unit: "×", fallback: 0.5,
        help: "each stage's own LoRAs at this fraction of their strength; 1 = full strength (more skin texture), 0 = none" },
    ] },
  ] },

  { id: "face", title: "Face detailer", sub: "Illustrious FaceDetailer on IL/ stills", engines: ["full"], panels: [{ title: "", ctls: [
    { key: "face_detailer_enabled", label: "Run it inside the IL pass", kind: "toggle", fallback: false },
    { key: "face_wildcard", label: "Face prompt", kind: "text" },
    { key: "face_denoise", label: "Denoise", kind: "slider", min: 0, max: 1, step: 0.01 },
    { key: "face_steps", label: "Steps", kind: "slider", min: 10, max: 60, step: 1, fold: "Sampler" },
    { key: "face_cfg", label: "CFG", kind: "slider", min: 1, max: 10, step: 0.1, fold: "Sampler" },
  ] }] },

  { id: "grain", title: "Sharpen + grain", sub: "the last touch inside the IL / krea2 / bgswap pass", panels: [{ title: "", ctls: [
    { key: "sharpen", label: "Sharpen", kind: "slider", min: 0, max: 2, step: 0.05 },
    { key: "grain_density", label: "Grain density", kind: "slider", min: 0, max: 1, step: 0.01, help: "0 = no grain" },
    { key: "grain_intensity", label: "Grain strength", kind: "slider", min: 0, max: 0.5, step: 0.01 },
  ] }] },

  { id: "background", title: "Background", sub: "bgswap: the Illustrious-painted room behind her", engines: ["bgswap"], panels: [{ title: "", ctls: [
    { key: "bg_denoise", label: "Room repaint", kind: "slider", min: 0.3, max: 1, step: 0.01, fallback: 1, help: "1 = paint from scratch; 0.5–0.75 restyles Krea2's own room" },
    { key: "depth_strength", label: "Depth ControlNet", kind: "slider", min: 0, max: 1, step: 0.01, fallback: 0.35 },
    { key: "unify_denoise", label: "Unify", kind: "slider", min: 0, max: 0.5, step: 0.01, fallback: 0.1, help: "0.1 ≈ photoreal with an anime sheen, 0.3 ≈ semi-anime, 0 skips it" },
    { key: "harmonize", label: "Harmonize lighting", kind: "slider", min: 0, max: 1, step: 0.01 },
    { key: "tint", label: "Tint", kind: "slider", min: 0, max: 1, step: 0.01 },
    { key: "il_checkpoint", label: "Checkpoint", kind: "select", models: true },
    { key: "edge_overlap", label: "Edge overlap", kind: "slider", min: 0, max: 24, step: 1, unit: "px", fallback: 6, fold: "Seams", help: "raise if a halo of the old background shows around her" },
    { key: "paste_feather", label: "Paste feather", kind: "slider", min: 0, max: 20, step: 0.5, unit: "px", fold: "Seams" },
    { key: "il_steps", label: "IL steps", kind: "slider", min: 10, max: 60, step: 1, fold: "Sampler" },
    { key: "il_cfg", label: "IL CFG", kind: "slider", min: 1, max: 10, step: 0.1, fold: "Sampler" },
    { key: "unify_steps", label: "Unify steps", kind: "slider", min: 10, max: 60, step: 1, fold: "Sampler" },
    ...sampler("il"),
  ] }] },

  { id: "finish", title: "Finish (SeedVR)", sub: "optional step between upscale and animate: SeedVR2 on the ✓'d upscale (hires/ or IL/) → finish/", panels: [{ title: "", ctls: [
    { key: "finish_resolution", label: "Size", kind: "slider", min: 1024, max: 2560, step: 8, unit: "px", fallback: 1440,
      help: "short edge; 1440 at 2:3 = 1440×2160. Its own pass, so it never shares VRAM with Krea2" },
    { key: "seedvr", label: "Also inside full cloud runs", kind: "toggle", fallback: false, fold: "Full cloud runs", help: "adds SeedVR after grain on a full cloud generate (project root)" },
    { key: "seedvr_resolution", label: "Size there", kind: "slider", min: 1024, max: 2560, step: 8, unit: "px", fallback: 1680, fold: "Full cloud runs" },
  ] }] },
];

export const SCHEMA_KEYS = new Set(SECTIONS.flatMap((s) => s.panels.flatMap((p) => p.ctls.map((c) => c.key))));
