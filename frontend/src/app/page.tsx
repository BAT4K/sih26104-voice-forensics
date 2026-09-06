"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import SmokeWarp from "@/components/SmokeWarp";
import SmoothScroll from "@/components/SmoothScroll";
import { FamilyBars, ScoreLine } from "@/components/charts";
import { SAMPLE, type Result } from "@/lib/sample";
import { analyzeAudio } from "@/lib/api";
import Link from "next/link";

type Phase = "idle" | "working" | "done" | "unavailable";

const VERDICT = {
  ai_clone:      { label: "AI-generated voice",  tone: "var(--state-clone)" },
  genuine:       { label: "Genuine human voice", tone: "var(--state-genuine)" },
  human_mimicry: { label: "Human mimicry",       tone: "var(--ink-1)" },
  unclear:       { label: "Unclear",             tone: "var(--state-unclear)" },
} as const;

const Rule = () => <div className="rule" />;

function Section({ title, aside, children }: {
  title: string; aside?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="mt-20">
      <div className="mb-5 flex items-baseline justify-between">
        <h2 className="eyebrow">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

const Gauge = ({ sigma }: { sigma: number }) => {
  const t = Math.max(-1, Math.min(1, sigma / 6));
  return (
    <svg width="64" height="10" aria-hidden className="shrink-0">
      <line x1="2" x2="62" y1="5" y2="5" stroke="var(--hair-strong)" strokeWidth="1" />
      <line x1="32" x2="32" y1="1.5" y2="8.5" stroke="var(--ink-5)" strokeWidth="1" />
      <circle cx={32 + t * 29} cy="5" r="2.5" fill="var(--ink-1)" />
    </svg>
  );
};

export default function Page() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [channel, setChannel] = useState("none");
  const [sample, setSample] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const slabRef = useRef<HTMLButtonElement>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [playbackProgress, setPlaybackProgress] = useState(0);

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      setIsPlaying(false);
      setPlaybackProgress(0);
      return () => URL.revokeObjectURL(url);
    }
  }, [file]);

  useEffect(() => {
    let frameId: number;
    function updateProgress() {
      if (audioRef.current && isPlaying) {
        setPlaybackProgress(audioRef.current.currentTime / (audioRef.current.duration || 1));
        frameId = requestAnimationFrame(updateProgress);
      }
    }
    if (isPlaying) {
      frameId = requestAnimationFrame(updateProgress);
    }
    return () => cancelAnimationFrame(frameId);
  }, [isPlaying]);

  function togglePlay(e: React.MouseEvent) {
    e.stopPropagation();
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        if (playbackProgress >= 1) setPlaybackProgress(0);
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  }

  function seekTo(index: number, e: React.MouseEvent) {
    e.stopPropagation();
    if (audioRef.current && audioRef.current.duration) {
      const newTime = (index / 68) * audioRef.current.duration;
      audioRef.current.currentTime = newTime;
      setPlaybackProgress(index / 68);
    }
  }

  function handleWaveformBoxClick(e: React.MouseEvent<HTMLSpanElement>) {
    e.stopPropagation();
    if (!audioRef.current || !audioRef.current.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    audioRef.current.currentTime = percentage * audioRef.current.duration;
    setPlaybackProgress(percentage);
  }

  /* Tilt toward the cursor, and move the cast shadow the OTHER way. When an
     object tilts toward you it pivots away on the far side, so a shadow that
     follows the tilt reads as a flat image being rotated. Note the negated ry. */
  function onSlabMove(e: React.PointerEvent<HTMLButtonElement>) {
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;
    const py = (e.clientY - r.top) / r.height;
    const rx = (0.5 - py) * 4;
    const ry = (px - 0.5) * 4;
    el.style.transition = "transform 120ms linear, box-shadow 120ms linear";
    el.style.transform = `perspective(900px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateY(-2px)`;
    el.style.boxShadow =
      `0 1px 0 rgba(0,0,0,.55), 0 2px 0 rgba(0,0,0,.68), 0 3px 0 rgba(0,0,0,.78), 0 4px 0 rgba(0,0,0,.86),` +
      ` ${(-ry * 3).toFixed(1)}px ${(rx * 3 + 26).toFixed(1)}px 50px -18px rgba(0,0,0,.92)`;
  }
  function onSlabLeave(e: React.PointerEvent<HTMLButtonElement>) {
    const el = e.currentTarget;
    el.style.transition = "transform 620ms cubic-bezier(.22,1.15,.36,1), box-shadow 620ms ease";
    el.style.transform = "perspective(900px) rotateX(0deg) rotateY(0deg)";
    el.style.boxShadow = "";
  }

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    document.documentElement.dataset.tone = q.get("tone") === "fog" ? "fog" : "vault";
    if (q.get("sample") === "1") {
      setSample(true); setResult(SAMPLE); setPhase("done");
      setFile(new File([], "sample_call_hi_in.wav"));
    }
  }, []);

  // once a result lands, carry the eye down to it
  useEffect(() => {
    if (phase === "done")
      requestAnimationFrame(() => {
        const el = resultsRef.current;
        if (!el) return;
        // go through Lenis when it is running, so the jump shares the same easing
        const lenis = (window as unknown as { __lenis?: { scrollTo: (t: Element, o?: object) => void } }).__lenis;
        if (lenis) lenis.scrollTo(el, { offset: -40 });
        else el.scrollIntoView({ behavior: "smooth", block: "start" });
      });
  }, [phase]);

  const [waveform, setWaveform] = useState<number[]>(
    Array.from({ length: 68 }, (_, i) =>
      +(18 + 82 * Math.abs(Math.sin(i * 0.55) * Math.cos(i * 0.17))).toFixed(2))
  );

  async function generateWaveform(file: File) {
    try {
      const buffer = await file.arrayBuffer();
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const audioBuffer = await ctx.decodeAudioData(buffer);
      const data = audioBuffer.getChannelData(0);
      const numBuckets = 68;
      const bucketSize = Math.floor(data.length / numBuckets);
      const newWaveform = [];
      let maxVal = 0;
      
      for (let i = 0; i < numBuckets; i++) {
        const start = i * bucketSize;
        let sum = 0;
        for (let j = 0; j < bucketSize; j++) {
          if (start + j < data.length) sum += Math.abs(data[start + j]);
        }
        const avg = sum / bucketSize;
        newWaveform.push(avg);
        if (avg > maxVal) maxVal = avg;
      }
      
      setWaveform(newWaveform.map(val => Math.max(2, Math.min(100, maxVal > 0 ? (val / maxVal) * 100 : 2))));
    } catch (e) {
      console.error("Waveform generation failed:", e);
    }
  }

  async function onPick(f: File, ch = channel) {
    setFile(f); setPhase("working");
    generateWaveform(f);
    try {
      setResult((await analyzeAudio(f, ch)) as Result);
      setPhase("done");
    } catch {
      // No checkpoint, no server, no verdict — never a fabricated one.
      setResult(null); setPhase("unavailable");
    }
  }

  const v = result ? (VERDICT[result.verdict as keyof typeof VERDICT] ?? VERDICT.unclear) : null;

  useEffect(() => {
    if (phase === "done" && resultsRef.current) {
      setTimeout(() => {
        const win = window as any;
        const lenis = win.__lenis;
        if (lenis) {
          // Wake the rAF loop — it sleeps after settling, and programmatic
          // scrollTo doesn't fire wheel/touch events to restart it.
          if (typeof win.__lenisWake === "function") win.__lenisWake();
          lenis.scrollTo(resultsRef.current, { offset: -50, duration: 1.2 });
        } else {
          resultsRef.current?.scrollIntoView({ block: "start" });
        }
      }, 100);
    }
  }, [phase]);

  return (
    <>
      <SmokeWarp />
      <SmoothScroll />

      {sample && (
        <div className="fixed left-1/2 top-0 z-50 -translate-x-1/2">
          <div className="eyebrow rounded-b-md px-3 py-1" style={{ background: "var(--lift)" }}>
            sample data · not a live result
          </div>
        </div>
      )}

      <main className="mx-auto w-full max-w-[1080px] px-14 pb-40">

        {/* ── hero ─────────────────────────────────────────────────────── */}
        <section className="flex min-h-[70vh] flex-col justify-center pt-16">
          <div className="scrim">
          <div className="flex items-center justify-between mb-8">
            <div className="eyebrow">Cipher · SIH 26104</div>
            <Link href="/settings" className="eyebrow hover:text-white transition-colors">
              Workflows & Alerts ↗
            </Link>
          </div>

          <h1 className="caps t1 text-[clamp(38px,6.4vw,78px)]">
            Detecting <span className="accent" style={{ color: "var(--state-clone)" }}>cloned</span> voices
            <br />
            across Indian <span className="accent">telephony</span>
          </h1>

          <p className="t3 mt-8 max-w-[52ch] text-[14px]">
            Upload a call recording. The detector reports a risk score, the decoder family
            behind it, and the measurements that produced the verdict — or reports that it
            cannot tell.
          </p>
          </div>

          {/* Perspective lives on the PARENT — put it on the element and every
              child gets its own vanishing point. 900px is the usable range. */}
          <div className="mt-12" style={{ perspective: "900px" }}>
            <button
              ref={slabRef}
              onClick={() => { if (!file) inputRef.current?.click(); }}
              onPointerMove={onSlabMove}
              onPointerLeave={onSlabLeave}
              className="slab lit-edge group relative block w-full rounded-2xl px-8 py-7 text-left"
              style={{
                background: "linear-gradient(180deg, var(--lift), rgba(255,255,255,.012))",
                transformStyle: "preserve-3d",
                willChange: "transform",
              }}
            >
              <span className="flex items-center gap-8">
                {/* extruded ridge — each bar has a lit top face */}
                <span className="flex h-16 flex-1 items-end gap-[3px] cursor-pointer group/waveform relative"
                      style={{ transform: "skewX(-10deg)" }}
                      onClick={handleWaveformBoxClick}>
                  {/* Invisible hover capture layer to ensure easy clicking even in the gaps */}
                  <span className="absolute inset-0 z-20" />
                  {waveform.map((h, i) => {
                    const isActive = (i / 68) <= playbackProgress;
                    return (
                      <span key={i} 
                            onClick={(e) => seekTo(i, e)}
                            className="relative flex-1 rounded-[1px] transition-colors duration-75 cursor-pointer hover:bg-[var(--state-genuine)] hover:opacity-100"
                            style={{ height: `${h}%`, background: isActive ? "var(--ink-1)" : "var(--ink-5)", zIndex: 10 }}>
                        <span className="absolute inset-x-0 top-0 h-[2px] rounded-[1px] pointer-events-none"
                              style={{ background: isActive ? "#fff" : "var(--ink-3)" }} />
                      </span>
                    );
                  })}
                </span>
                <span className="shrink-0 flex flex-col items-end text-right">
                  <span 
                    className="caps t1 block text-[19px] cursor-pointer hover:text-white transition-colors"
                    onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
                  >
                    {file ? "Replace file" : "Upload audio"}
                  </span>
                  <span className="mono t4 mt-1.5 block text-[11px]">
                    {phase === "working" ? "analysing…"
                      : file ? file.name : "wav or flac · 4 s window"}
                  </span>
                  {file && (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={togglePlay}
                      className="mt-3 px-3 py-1 bg-[var(--lift)] border border-[var(--ink-3)] rounded-md hover:border-[var(--ink-4)] hover:bg-[var(--ink-5)] transition-colors text-[11px] font-mono tracking-widest text-white/80 uppercase flex items-center justify-center min-w-[70px] cursor-pointer"
                    >
                      {isPlaying ? "Pause" : "Play"}
                    </span>
                  )}
                </span>
              </span>
            </button>
            <input ref={inputRef} type="file" accept="audio/*" className="hidden"
                   onChange={(e) => e.target.files?.[0] && onPick(e.target.files[0])} />
            {audioUrl && (
              <audio 
                ref={audioRef} 
                src={audioUrl} 
                onEnded={() => { setIsPlaying(false); setPlaybackProgress(0); }}
              />
            )}
          </div>

          <div className="mt-7 grid grid-cols-2 gap-8 md:grid-cols-4">
            <div>
              <div className="eyebrow">Channel</div>
              <div className="relative inline-block">
                <select 
                  value={channel}
                  onChange={(e) => {
                    const newCh = e.target.value;
                    setChannel(newCh);
                    if (file) onPick(file, newCh);
                  }}
                  className="mono t2 mt-1.5 text-[12.5px] bg-transparent outline-none cursor-pointer appearance-none border-b border-white/20 pb-0.5 hover:text-white transition-colors pr-4"
                >
                  <option className="bg-black text-white" value="none">none</option>
                  <option className="bg-black text-white" value="amr-nb">AMR-NB (4.75kbps)</option>
                  <option className="bg-black text-white" value="amr-wb">AMR-WB</option>
                  <option className="bg-black text-white" value="g711-ulaw">G.711 µ-law</option>
                  <option className="bg-black text-white" value="g711-alaw">G.711 A-law</option>
                  <option className="bg-black text-white" value="gsm-fr">GSM-FR</option>
                  <option className="bg-black text-white" value="g722">G.722</option>
                  <option className="bg-black text-white" value="g726">G.726</option>
                  <option className="bg-black text-white" value="opus">Opus</option>
                  <option className="bg-black text-white" value="speex">Speex</option>
                </select>
                <span className="pointer-events-none absolute right-0 top-1/2 -translate-y-1/2 text-[9px] opacity-50 mt-1.5">▼</span>
              </div>
            </div>
            {[["Window", "4 s / 1 s hop"],
              ["Families", "7 families"], ["Tier 3", "open set"]].map(([k, val]) => (
              <div key={k}>
                <div className="eyebrow">{k}</div>
                <div className="mono t2 mt-1.5 text-[12.5px]">{val}</div>
              </div>
            ))}
          </div>
        </section>

        <div ref={resultsRef} />

        {phase === "working" && (
          <p className="mono t4 py-32 text-center text-[11px]">analysing…</p>
        )}

        {phase === "unavailable" && (
          <div className="mt-8 py-1 pl-5" style={{ borderLeft: "2px solid var(--state-unclear)" }}>
            <div className="caps t1 text-[24px]">Detector unavailable</div>
            <p className="t3 mt-2 max-w-[62ch]">
              No checkpoint is loaded, so no verdict can be produced. Nothing is shown
              rather than an estimate.
            </p>
          </div>
        )}

        {phase === "done" && result && v && (
          <>
            <Section title="Verdict">
              <div className="scrim grid grid-cols-[1.5fr_0.9fr_1.1fr]">
                <div className="pr-10">
                  <div className="t4 mb-3 text-[11px]">Forensic verdict</div>
                  <div className="caps text-[30px]" style={{ color: v.tone }}>{v.label}</div>
                </div>

                <div className="px-10" style={{ borderLeft: "1px solid var(--hair)" }}>
                  <div className="t4 mb-3 text-[11px]">Risk</div>
                  <div className="flex items-end gap-3.5">
                    {/* the monolith — a standing slab with a lit right edge */}
                    {/* a standing solid: extruded top and right faces, lit from
                        the top left like everything else on the page */}
                    <span className="extrude relative block h-14 w-[9px] rounded-[2px]"
                          style={{ background: "var(--lift)" }}>
                      <span className="absolute inset-x-0 bottom-0 rounded-[2px]"
                            style={{ height: `${result.risk}%`, background: "var(--ink-1)",
                                     boxShadow: "inset 0 1px 0 rgba(255,255,255,.6)",
                                     /* one overshoot, then settle — mass, not bounce */
                                     transition: "height 820ms cubic-bezier(.22,1.2,.36,1)" }} />
                    </span>
                    <span className="caps tnum t1 text-[40px]">
                      {result.risk}<span className="t4 ml-1.5 text-[13px]">/100</span>
                    </span>
                  </div>
                </div>

                <div className="pl-10" style={{ borderLeft: "1px solid var(--hair)" }}>
                  <div className="t4 mb-3 text-[11px]">
                    Closest family <span className="mono t5 text-[9.5px]">TIER 2</span>
                  </div>
                  <div className="mono t1 text-[20px]">{result.family.replace(/_/g, ' ')}</div>
                  <div className="t4 mt-1.5 text-[11px]">
                    {(result.familyConfidence * 100).toFixed(1)}% of probability mass
                  </div>
                </div>
              </div>

              <div className="mt-9 py-1 pl-5" style={{ borderLeft: `2px solid ${v.tone}` }}>
                <span className="t1">Recommended action. </span>
                <span className="t3">
                  {result.recommendedAction}
                </span>
              </div>
            </Section>

            <Section title="Why">
              <div className="plate px-6">
                <div className="py-5">
                  <div className="t1">Risk model</div>
                  <p className="t3 mt-1">
                    Synthesis detector: {result.risk} percent likelihood this audio was
                    machine-generated.
                  </p>
                </div>
                {result.reasons.map((r) => (
                  <div key={r.name}>
                    <Rule />
                    <div className="flex items-start gap-6 py-5">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline gap-3">
                          <span className="mono t1 text-[12.5px]">{r.name}</span>
                          <span className="mono tnum t2 text-[12.5px]">
                            {r.value}{r.unit && <span className="t4"> {r.unit}</span>}
                          </span>
                        </div>
                        <p className="t3 mt-1">{r.note}</p>
                      </div>
                      <div className="pt-2"><Gauge sigma={r.sigma} /></div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Family breakdown"
                     aside={<span className="mono t5 text-[10px]">○ tell does not survive 8 kHz</span>}>
              <div className="plate px-6 py-7"><FamilyBars probs={result.probabilities} /></div>
              <p className="t2 mt-4 max-w-[78ch] text-[11.5px]">
                Four of seven synthetic families lose their primary tell at 8 kHz. Anything
                far from every known family is reported as unknown rather than forced into
                the nearest one.
              </p>
            </Section>

            <Section title="Score over time">
              <div className="plate px-6 py-7"><ScoreLine scores={result.scores} /></div>
            </Section>

            <footer className="mt-20">
              <Rule />
              <p className="mono tnum t3 mt-5 text-[10.5px]">
                {result.durationSec} s of speech analysed in {result.elapsedMs.toLocaleString()} ms
                · model {result.model} · 4 s window / 1 s hop
              </p>
            </footer>
          </>
        )}
      </main>
    </>
  );
}
