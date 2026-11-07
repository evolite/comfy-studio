/** The studio's icon set: one inline-SVG theme (24px grid, 1.8px round strokes, currentColor).
 *  No emoji in the chrome — glyphs stay consistent across platforms and themes. */
const PATHS: Record<string, React.ReactNode> = {
  plus: <path d="M12 5v14M5 12h14" />,
  bell: <><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></>,
  "bell-off": <><path d="M8.7 3A6 6 0 0 1 18 8c0 4.5 1.2 7 2.2 8.2" /><path d="M6.3 6.3C6.1 6.9 6 7.4 6 8c0 7-3 9-3 9h14" /><path d="M13.7 21a2 2 0 0 1-3.4 0" /><path d="M2 2l20 20" /></>,
  sliders: <><path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3" /><path d="M14 2v4M8 10v4M16 18v4" /></>,
  trash: <><path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" /><path d="M10 11v6M14 11v6" /></>,
  shuffle: <><path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.8-1.1 2-1.7 3.3-1.7H22M18 2l4 4-4 4" /><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8" /><path d="m18 14 4 4-4 4" /></>,
  check: <path d="M20 6 9 17l-5-5" />,
  "check-circle": <><circle cx="12" cy="12" r="9" /><path d="m9 12 2 2 4-4" /></>,
  x: <path d="M18 6 6 18M6 6l12 12" />,
  cloud: <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" />,
  cpu: <><rect x="5" y="5" width="14" height="14" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" /></>,
  play: <path d="M6 4.5 19 12 6 19.5Z" />,
  list: <path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01" />,
  send: <path d="M22 2 11 13M22 2l-7 20-4-9-9-4Z" />,
  "arrow-down": <><path d="M12 3v12M7 10l5 5 5-5" /><path d="M4 21h16" /></>,
  restart: <><path d="M21 12a9 9 0 1 1-2.64-6.36" /><path d="M21 3v6h-6" /></>,
  stop: <rect x="6" y="6" width="12" height="12" rx="1.5" />,
  search: <><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></>,
  "chevron-down": <path d="m6 9 6 6 6-6" />,
  at: <><circle cx="12" cy="12" r="4" /><path d="M16 8v5a3 3 0 0 0 6 0v-1a10 10 0 1 0-4 8" /></>,
  zap: <path d="M13 2 3 14h7l-1 8 11-14h-7l1-6Z" />,
  layers: <><path d="m12 2 9 5-9 5-9-5Z" /><path d="m3 12 9 5 9-5M3 17l9 5 9-5" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 8h.01M12 11v5" /></>,
};

export function Icon({ n, size = 16, className }: { readonly n: keyof typeof PATHS | string; readonly size?: number; readonly className?: string }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {PATHS[n]}
    </svg>
  );
}
