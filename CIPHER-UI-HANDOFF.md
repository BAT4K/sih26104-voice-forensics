# Cipher UI — handoff, and how to make it look identical

Send **one file**: `cipher-ui.zip`.

Everything the interface needs is inside it, including both images. He does not
need anything else for it to render exactly as it does here.

---

## What he does

```bash
unzip cipher-ui.zip
cd frontend
npm install
npm run dev
```

Then open **http://localhost:3000/?sample=1**

That flag loads the design fixture so the whole page is populated — verdict,
family bars, score chart. Without it he gets the empty state, which is correct
behaviour, not a bug.

---

## The five things that silently change how it looks

If it doesn't match, it is almost certainly one of these. In order of
likelihood.

### 1. No internet on first install → the type collapses

The three faces load through `next/font/google` **at build time**:
Archivo 700 (the caps headline), Instrument Serif italic (the accent words —
*CLONED*, *TELEPHONY*), IBM Plex Mono (every number and label).

They are **not bundled in the zip**. With no network on the first
`npm install` / `npm run dev`, Next silently falls back to Times and the system
sans, and the entire character of the page is gone. Nothing errors.

**Check:** the headline must be heavy condensed caps, and *CLONED* must be an
italic serif in red. If the headline looks like Helvetica and *CLONED* is not a
serif, the fonts didn't load. Reconnect and delete `.next`, then rerun.

### 2. "Reduce motion" turned on → everything is static

macOS **System Settings → Accessibility → Display → Reduce motion** disables,
by design:

- the drifting smoke background
- Lenis smooth scrolling
- the family bars growing on scroll
- the score line drawing itself

The page still looks right, but nothing moves and the reveals are already
finished. This is correct accessibility behaviour, not a fault — but if he is
comparing against your screen and his has it on, he'll think half the work is
missing.

### 3. He replaces or regenerates the wallpaper

`public/wallpaper.jpeg` is generated: an 8×2 grid of the original with alternate
columns mirrored and rows flipped so the tile edges match, then gamma 1.6 so
fewer filaments read as white.

`public/wallpaperoriginal.jpeg` is the original 536×1072 source. **Nothing may write
to it — it is the only copy.** (It was destroyed once already by generating in
place.)

Any substitute must be grayscale, soft-edged, landscape and ≥3000px wide. The
image is displaced per pixel at runtime, so a photo with a subject or a horizon
line will visibly smear.

### 4. He "upgrades" the background to WebGL

Do not. It was written as a shader first and rendered **nothing** — Krish's
Chrome has hardware acceleration off, so `getContext('webgl')` returns null and
fails silently to pure black. Hours went into finding that.

The current version is a 2D-canvas per-pixel displacement at a 512×256 buffer
that CSS stretches to the viewport. It costs **1.02 ms/frame** on that machine.
It works everywhere; a shader does not.

Related: the sampling **wraps** at the buffer edges and must never clamp. A
clamped sample repeats the edge row and smears it down the whole page.

### 5. Tailwind v4, not v3

`globals.css` uses `@import "tailwindcss"` and an `@theme` block. On v3 the
tokens do not resolve and the page renders unstyled. `package-lock.json` is in
the zip — installing from it avoids this entirely.

---

## What he should see, top to bottom

Use this to confirm it matches before wiring anything up.

1. **Background** — grey smoke drifting continuously over black. Moving the
   mouse across it pushes the smoke along the direction of travel; it carries
   momentum for about a second after stopping, then settles.
2. **Hero** — `DETECTING *CLONED* VOICES / ACROSS INDIAN *TELEPHONY*` in heavy
   caps, with the two accent words in italic serif, *CLONED* in red.
3. **Upload slab** — tilts toward the cursor, at most 4°, with its shadow
   shifting the opposite way. An extruded gold-lit waveform ridge runs across it.
4. **Verdict row** — three fields divided by hairlines, not boxes. The risk
   score is a standing slab with visible edge faces, filling to 98.
5. **Why** — one grouped panel with hairline dividers between rows, each with a
   deviation gauge on the right. Not four separate cards.
6. **Family breakdown** — eight rows. Bars grow left-to-right, staggered, *when
   that section scrolls into view*. One gold hue only; the winner is bright and
   the rest recessive.
7. **Score over time** — the line draws itself in on scroll. Two labelled dashed
   thresholds, red at 0.72 and green at 0.35.
8. **Scroll** — glides and settles rather than snapping. That's Lenis.

If 6 and 7 are already drawn when he arrives at them, he scrolled past too fast
or reduce-motion is on.

---

## What is NOT in the zip, and why

- **The backend.** He has it. This is the interface only.
- **`node_modules`.** `npm install` fetches it.
- **The checkpoint.** `./v0.3-telephony-xlsr-final` does not exist on this
  machine either. Nobody has it.

## The one piece of work waiting for him

The API contract does not match. `src/lib/api.ts` posts to
`:8000/v1/detect` and returns `{is_fake, confidence, vocoder.probabilities}`;
the page reads `{verdict, risk, family, reasons, scores}`. Until an adapter maps
between them, every real upload lands on "Detector unavailable".

Full field-by-field table is in `INTEGRATION.md` §1. `src/lib/sample.ts` is the
contract — code against that file.
