"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Hand-authored SVG — no charting library, per the stack constraint.
 *
 * Both charts stay monochrome: one series, so identity never needs a hue.
 * Emphasis comes from opacity and direct labels, which also keeps them legible
 * without colour vision.
 *
 * Reveal: bars grow and the score line draws itself the first time the section
 * scrolls into view. Bars animate with scaleX rather than the width attribute,
 * because transform is compositor-driven and width is not.
 */
function useReveal<T extends Element>() {
  const ref = useRef<T>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(true);
      return;
    }
    const io = new IntersectionObserver(
      ([e]) => e.isIntersecting && (setShown(true), io.disconnect()),
      { threshold: 0.25 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, shown] as const;
}

export const FAMILY_ORDER = [
  "real", "hifigan", "bigvgan", "vocos",
  "encodec", "modern_codec", "legacy", "diffusion"
] as const;

/** Families whose primary tell does not survive an 8 kHz channel. */
const NARROWBAND_LOSSY = new Set(["hifigan", "bigvgan", "encodec", "modern_codec"]);

export function FamilyBars({ probs }: { probs: Record<string, number> }) {
  const [ref, shown] = useReveal<HTMLDivElement>();
  const rows = FAMILY_ORDER.map((k) => ({ key: k, v: probs[k] ?? 0 }));
  const top = rows.reduce((a, b) => (b.v > a.v ? b : a), rows[0]);
  const ROW = 30, LABEL = 132, PAD_R = 56, W = 660;
  const track = W - LABEL - PAD_R;

  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${rows.length * ROW + 28}`} className="w-full"
           role="img" aria-label="Probability mass by decoder family">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line key={t} x1={LABEL + t * track} x2={LABEL + t * track}
                y1={2} y2={rows.length * ROW}
                stroke="var(--hair)" strokeWidth={1} />
        ))}

        {rows.map((r, i) => {
          const y = i * ROW + 9;
          const w = Math.max(r.v * track, r.v > 0 ? 2 : 0);
          const win = r.key === top.key;
          return (
            <g key={r.key}>
              <text x={LABEL - 14} y={y + 8} textAnchor="end" fontSize={12}
                    fontFamily="var(--font-mono)"
                    fill={win ? "var(--ink-1)" : "var(--ink-2)"}>
                {r.key}
              </text>
              {NARROWBAND_LOSSY.has(r.key) && (
                <circle cx={8} cy={y + 5} r={2.6} fill="none"
                        stroke="var(--ink-3)" strokeWidth={1} />
              )}
              <g style={{
                    transform: shown ? "scaleX(1)" : "scaleX(0)",
                    transformOrigin: `${LABEL}px 0px`,
                    transition: `transform 900ms cubic-bezier(.16,1,.3,1) ${i * 55}ms`,
                  }}>
                <rect x={LABEL} y={y} width={w} height={11} rx={2}
                      fill={win ? "var(--ink-1)" : "var(--ink-3)"}
                      opacity={win ? 1 : 0.20} />
                {/* lit top face — the bar reads as an extruded ridge */}
                {w > 3 && (
                  <rect x={LABEL} y={y} width={w} height={1.5} rx={1}
                        fill="var(--ink-1)" opacity={win ? 0.55 : 0.24} />
                )}
              </g>
              <text x={LABEL + w + 10} y={y + 8} fontSize={11}
                    fontFamily="var(--font-mono)"
                    fill={win ? "var(--ink-1)" : "var(--ink-2)"}
                    style={{ opacity: shown ? 1 : 0, transition: `opacity 500ms ${400 + i * 55}ms` }}>
                {(r.v * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}

        {[0, 0.5, 1].map((t) => (
          <text key={t} x={LABEL + t * track} y={rows.length * ROW + 18}
                fontSize={10} fontFamily="var(--font-mono)"
                fill="var(--ink-3)" textAnchor="middle">
            {t.toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  );
}

export function ScoreLine({
  scores, cloneAt = 0.72, genuineAt = 0.35,
}: { scores: number[]; cloneAt?: number; genuineAt?: number }) {
  const [ref, shown] = useReveal<HTMLDivElement>();
  const W = 660, H = 190, L = 38, R = 104, T = 12, B = 30;
  const px = (i: number) => L + (i / Math.max(scores.length - 1, 1)) * (W - L - R);
  const py = (v: number) => T + (1 - v) * (H - T - B);
  const d = scores.map((v, i) => `${i ? "L" : "M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
  const len = 1400;

  const Threshold = ({ v, label, colour, dash }: {
    v: number; label: string; colour: string; dash: string;
  }) => (
    <g>
      <line x1={L} x2={W - R} y1={py(v)} y2={py(v)} stroke={colour}
            strokeWidth={1} strokeDasharray={dash} opacity={0.8} />
      <text x={W - R + 10} y={py(v) + 3.5} fontSize={10}
            fontFamily="var(--font-mono)" fill={colour}>{label}</text>
    </g>
  );

  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
           aria-label="Synthesis score per analysis window">
        <rect x={L} y={T} width={W - L - R} height={py(cloneAt) - T}
              fill="var(--state-clone)" opacity={0.06} />
        {[0, 0.5, 1].map((t) => (
          <text key={t} x={L - 10} y={py(t) + 3.5} fontSize={10}
                fontFamily="var(--font-mono)" fill="var(--ink-3)" textAnchor="end">
            {t.toFixed(1)}
          </text>
        ))}
        {/* dash patterns differ as well as colour, so the thresholds stay
            distinguishable without colour vision */}
        <Threshold v={cloneAt}   label={`clone ${cloneAt}`}     colour="var(--state-clone)"   dash="5 3" />
        <Threshold v={genuineAt} label={`genuine ${genuineAt}`} colour="var(--state-genuine)" dash="1 4" />
        <path d={d} fill="none" stroke="var(--ink-1)" strokeWidth={1.8}
              strokeLinejoin="round" strokeLinecap="round"
              strokeDasharray={len}
              style={{
                strokeDashoffset: shown ? 0 : len,
                transition: "stroke-dashoffset 1100ms cubic-bezier(.16,1,.3,1) 120ms",
              }} />
        {scores.map((v, i) => (
          <circle key={i} cx={px(i)} cy={py(v)} r={2.4} fill="var(--ink-1)"
                  style={{ opacity: shown ? 1 : 0,
                           transition: `opacity 400ms ${500 + i * 60}ms` }} />
        ))}
        <text x={(L + W - R) / 2} y={H - 7} fontSize={10}
              fontFamily="var(--font-mono)" fill="var(--ink-3)" textAnchor="middle">
          window · 1 s apart
        </text>
      </svg>
    </div>
  );
}
