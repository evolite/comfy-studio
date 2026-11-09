import { useEffect, useRef, useState } from "react";

/** The edit's opening as it will sound, mixed live in the browser (no render): the track from its start time under the first
 * voiced clip's own audio from its trim, at the voice × / music × levels, with video/make_edit.py's AMBIENT_MUSIC rebuilt in
 * Web Audio (keep the numbers in step). Duck is not simulated. */

const AMBIENT_GAIN = 0.4;
const MAX_SECONDS = 15;  // a clip without audio would otherwise play the music on forever

type Graph = { ctx: AudioContext; music: GainNode; voice: GainNode; dry: GainNode; wet: GainNode };

/** highpass 200 → lowpass 5000 → stereo narrowed (extrastereo m=0.4) → room echo (aecho 0.8:0.6:35|70:0.3|0.2) → ×0.4 */
function ambientChain(ctx: AudioContext, input: AudioNode, out: AudioNode) {
  const g = (v: number) => new GainNode(ctx, { gain: v });
  const hp = new BiquadFilterNode(ctx, { type: "highpass", frequency: 200 });
  const lp = new BiquadFilterNode(ctx, { type: "lowpass", frequency: 5000 });
  const split = new ChannelSplitterNode(ctx, { numberOfOutputs: 2 });
  const merge = new ChannelMergerNode(ctx, { numberOfInputs: 2 });
  input.connect(hp).connect(lp).connect(split);
  split.connect(g(0.7), 0).connect(merge, 0, 0);
  split.connect(g(0.3), 1).connect(merge, 0, 0);
  split.connect(g(0.3), 0).connect(merge, 0, 1);
  split.connect(g(0.7), 1).connect(merge, 0, 1);
  const base = 0.8 * 0.6 * AMBIENT_GAIN;
  merge.connect(g(base)).connect(out);
  for (const [ms, decay] of [[35, 0.3], [70, 0.2]]) merge.connect(new DelayNode(ctx, { delayTime: ms / 1000 })).connect(g(base * decay)).connect(out);
}

function build(musicEl: HTMLAudioElement, voiceEl: HTMLAudioElement): Graph {
  const ctx = new AudioContext();
  const limiter = new DynamicsCompressorNode(ctx, { threshold: -1, knee: 0, ratio: 20, attack: 0.003, release: 0.1 });  // ~alimiter
  limiter.connect(ctx.destination);
  const music = new GainNode(ctx), voice = new GainNode(ctx), dry = new GainNode(ctx), wet = new GainNode(ctx);
  const m = ctx.createMediaElementSource(musicEl);
  m.connect(dry).connect(music);
  ambientChain(ctx, m, wet);
  wet.connect(music);
  music.connect(limiter);
  ctx.createMediaElementSource(voiceEl).connect(voice).connect(limiter);
  return { ctx, music, voice, dry, wet };
}

export function MixPreview({ trackUrl, start, clipUrl, clipTrim, musicVolume, clipVolume, ambient }: {
  readonly trackUrl: string; readonly start: number; readonly clipUrl?: string; readonly clipTrim: number;
  readonly musicVolume: number; readonly clipVolume: number; readonly ambient: boolean;
}) {
  const musicEl = useRef<HTMLAudioElement>(null), voiceEl = useRef<HTMLAudioElement>(null);
  const graph = useRef<Graph | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {  // levels follow the fields live, also mid-play
    const gr = graph.current;
    if (!gr) return;
    const t = gr.ctx.currentTime;
    gr.music.gain.setTargetAtTime(Number.isFinite(musicVolume) ? musicVolume : 1, t, 0.02);
    gr.voice.gain.setTargetAtTime(Number.isFinite(clipVolume) ? clipVolume : 1, t, 0.02);
    gr.dry.gain.setTargetAtTime(ambient ? 0 : 1, t, 0.02);
    gr.wet.gain.setTargetAtTime(ambient ? 1 : 0, t, 0.02);
  }, [musicVolume, clipVolume, ambient, playing]);

  const stop = () => {
    clearTimeout(timer.current);
    musicEl.current?.pause(); voiceEl.current?.pause();
    setPlaying(false);
  };
  useEffect(() => () => { stop(); void graph.current?.ctx.close(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const play = async () => {
    const mu = musicEl.current, vo = voiceEl.current;
    if (!mu || !vo) return;
    graph.current ??= build(mu, vo);
    await graph.current.ctx.resume();
    mu.currentTime = start;
    vo.currentTime = clipUrl ? clipTrim : 0;
    setPlaying(true);
    await Promise.all([mu.play(), clipUrl ? vo.play() : Promise.resolve()]).catch(stop);
    timer.current = window.setTimeout(stop, MAX_SECONDS * 1000);
  };

  return <>
    <button type="button" className="btn sm" onClick={() => (playing ? stop() : void play())}
      title={`plays the edit's opening mixed live: the track from the start time under ${clipUrl ? "the first unmuted clip's own audio" : "(no clip to voice)"} at these levels; duck is not simulated`}>
      {playing ? "■ stop" : "▶ hear mix"}
    </button>
    <audio /* NOSONAR(S4084): the user's own audio, no captions exist */ ref={musicEl} preload="auto" src={trackUrl} />
    <audio /* NOSONAR(S4084): the clip's own generated sound */ ref={voiceEl} preload="auto" src={clipUrl} onEnded={() => setTimeout(stop, 1000)} />
  </>;
}
