"use client";

import { useEffect } from "react";
import Lenis from "lenis";

/**
 * Interpolated scrolling — the "page moves with you" feel.
 *
 * Native scroll snaps to the wheel delta. Lenis keeps a target and eases the
 * real position toward it each frame, so the page glides and settles.
 *
 * The loop SLEEPS when the scroll has settled. Lenis's usual pattern is an
 * unconditional rAF loop running for the lifetime of the page; on a machine
 * without GPU acceleration that is a permanent CPU cost for a page that is
 * standing still. Here input wakes it and settling stops it, so an idle page
 * costs nothing at all.
 *
 * Disabled outright under prefers-reduced-motion, where hijacking the scroll
 * would be actively hostile.
 */
export default function SmoothScroll() {
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const lenis = new Lenis({
      lerp: 0.09,
      wheelMultiplier: 1,
      touchMultiplier: 1.6,
      smoothWheel: true,
    });

    let raf = 0;
    let running = false;
    let idleFrames = 0;

    const loop = (t: number) => {
      lenis.raf(t);
      // `velocity` decays as the eased position converges on the target
      if (Math.abs(lenis.velocity) < 0.05) idleFrames++;
      else idleFrames = 0;

      if (idleFrames > 12) { running = false; return; }   // settled — stop
      raf = requestAnimationFrame(loop);
    };

    const wake = () => {
      if (running) return;
      running = true;
      idleFrames = 0;
      raf = requestAnimationFrame(loop);
    };

    // Expose both lenis and wake so programmatic scrollTo actually animates
    (window as unknown as { __lenis?: Lenis; __lenisWake?: () => void }).__lenis = lenis;
    (window as unknown as { __lenis?: Lenis; __lenisWake?: () => void }).__lenisWake = wake;

    window.addEventListener("wheel", wake, { passive: true });
    window.addEventListener("touchstart", wake, { passive: true });
    window.addEventListener("keydown", wake, { passive: true });
    lenis.on("scroll", () => { idleFrames = 0; });

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("wheel", wake);
      window.removeEventListener("touchstart", wake);
      window.removeEventListener("keydown", wake);
      lenis.destroy();
      delete (window as unknown as { __lenis?: Lenis; __lenisWake?: () => void }).__lenis;
      delete (window as unknown as { __lenis?: Lenis; __lenisWake?: () => void }).__lenisWake;
    };
  }, []);

  return null;
}
