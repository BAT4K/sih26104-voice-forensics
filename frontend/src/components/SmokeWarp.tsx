"use client";

import { useEffect, useRef } from "react";

/**
 * Interactive smoke ground — per-pixel displacement on a 2D canvas.
 *
 * WHY NOT A MESH. The previous version translated ~140 rectangles as rigid
 * units. Neighbouring cells move by different amounts, so their shared edges
 * tear — that is the blockiness. No amount of blur fixes it, because the
 * discontinuity is structural: adjacent pixels either side of a cell boundary
 * genuinely come from far-apart parts of the source.
 *
 * WHY THIS IS AFFORDABLE. Displacement is computed per PIXEL, but on a small
 * buffer (~320px wide) that CSS scales to the viewport. The browser's own
 * bilinear filtering does the smoothing for free, and smoke has no hard edges
 * to lose. ~50k pixels a frame is cheap even with no GPU — where the old
 * version died was compositing a viewport-sized canvas, not the maths.
 *
 * The flow field is separable: the sine terms depend on x OR y alone, so they
 * are precomputed into row/column tables once per frame — two array lookups per
 * pixel instead of four trig calls.
 *
 * The field drifts continuously on its own; the pointer adds an impulse that
 * decays. No WebGL anywhere: it is unavailable in the target browser.
 */

const BUF_W = 512;        // buffer width; height follows the viewport aspect
const AMP = 3.6;          // resting drift, in buffer pixels (0.65x)
const PUSH_DECAY = 0.945; // how long a drag keeps pushing after you stop
const PUSH_GAIN = 0.36;   // 0.65x, matching the resting amplitude
const RADIUS = 78;        // pointer falloff, in buffer pixels

export default function SmokeWarp() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d", { alpha: false });
    if (!ctx) return;

    let W = BUF_W, H = 180;
    let src: Uint8ClampedArray | null = null;
    let out: ImageData | null = null;
    const rowA = new Float32Array(4096), rowB = new Float32Array(4096);
    const colA = new Float32Array(4096), colB = new Float32Array(4096);

    const img = new Image();
    let ready = false;

    const build = () => {
      const vw = window.innerWidth, vh = window.innerHeight;
      W = BUF_W;
      H = Math.max(2, Math.round(BUF_W * (vh / vw)));
      cv.width = W; cv.height = H;

      if (!img.width) return;
      // cover-fit the source: scale by the LARGER ratio, centre the overflow
      const off = document.createElement("canvas");
      off.width = W; off.height = H;
      const o = off.getContext("2d")!;
      const k = Math.max(W / img.width, H / img.height);
      const dw = img.width * k, dh = img.height * k;
      o.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
      src = o.getImageData(0, 0, W, H).data;
      out = ctx.createImageData(W, H);
      ready = true;
    };

    img.onload = build;
    img.onerror = () => { /* leave the ground flat rather than guess a texture */ };
    img.src = "/wallpaper.jpeg";

    window.addEventListener("resize", build);

    // pointer impulse, decaying
    let pxb = -999, pyb = -999, pushX = 0, pushY = 0, lx = -1, ly = -1;
    const onMove = (e: PointerEvent) => {
      const sx = (e.clientX / window.innerWidth) * W;
      const sy = (e.clientY / window.innerHeight) * H;
      if (lx >= 0) {
        pushX += (sx - lx) * PUSH_GAIN;
        pushY += (sy - ly) * PUSH_GAIN;
      }
      lx = sx; ly = sy; pxb = sx; pyb = sy;
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    let scrollY = 0;
    const onScroll = () => { scrollY = window.scrollY || 0; };
    window.addEventListener("scroll", onScroll, { passive: true });

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, t = 0, drift = 0;
    const R2 = RADIUS * RADIUS;

    const frame = () => {
      raf = requestAnimationFrame(frame);
      if (!ready || !src || !out) return;

      t += reduced ? 0 : 0.016;
      pushX *= PUSH_DECAY;
      pushY *= PUSH_DECAY;
      // small, and bounded — this is a subtle parallax, not a scroll offset
      drift += (Math.max(-14, Math.min(14, scrollY * 0.012)) - drift) * 0.07;

      // separable flow: computed once per row / per column, not per pixel
      for (let y = 0; y < H; y++) {
        const v = y / H;
        rowA[y] = Math.sin(v * 5.5 + t * 0.55);
        rowB[y] = Math.cos(v * 3.1 - t * 0.42);
      }
      for (let x = 0; x < W; x++) {
        const u = x / W;
        colA[x] = Math.sin(u * 4.2 - t * 0.48);
        colB[x] = Math.cos(u * 2.6 + t * 0.37);
      }

      const d = out.data;
      for (let y = 0; y < H; y++) {
        const ra = rowA[y], rb = rowB[y];
        const dy0 = y - pyb;
        for (let x = 0; x < W; x++) {
          let fx = AMP * (ra + colA[x]);
          let fy = AMP * (rb + colB[x]);

          // smooth inverse-square falloff — no sqrt, and no hard edge
          const dx0 = x - pxb;
          const f = R2 / (R2 + dx0 * dx0 + dy0 * dy0);
          fx += pushX * f;
          fy += pushY * f;

          // wrap, never clamp: a clamped sample repeats the edge row and
          // smears it down the page, which is exactly the streaking artifact
          let sx = (x + fx) | 0;
          let sy = (y + fy + drift) | 0;
          sx = sx < 0 ? sx + W : sx >= W ? sx - W : sx;
          sy = sy < 0 ? sy + H : sy >= H ? sy - H : sy;
          sx = sx < 0 ? 0 : sx >= W ? W - 1 : sx;
          sy = sy < 0 ? 0 : sy >= H ? H - 1 : sy;

          const si = (sy * W + sx) << 2;
          const di = (y * W + x) << 2;
          d[di] = src[si];
          d[di + 1] = src[si + 1];
          d[di + 2] = src[si + 2];
          d[di + 3] = 255;
        }
      }
      ctx.putImageData(out, 0, 0);
    };
    raf = requestAnimationFrame(frame);

    const onVis = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden) raf = requestAnimationFrame(frame);
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", build);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 bg-black">
      {/* the buffer is ~320px wide and stretched to the viewport; the browser's
          bilinear filter is what makes the displacement read as smooth */}
      <canvas ref={ref} className="absolute inset-0 h-full w-full"
              style={{ opacity: 0.55 }} />
      <div className="absolute inset-0"
           style={{ background:
             "radial-gradient(125% 92% at 50% 28%, rgba(0,0,0,.12) 0%, rgba(0,0,0,.42) 55%, rgba(0,0,0,.80) 100%)" }} />
    </div>
  );
}
