# Cipher UI — integration handoff

The frontend for **Indic Voice Forensic Inspector** (SIH 26104). Source only —
no `node_modules`, no `.next`, no git history.

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Open **http://localhost:3000/?sample=1** to see the whole page with fixture data.
Without that flag you get the empty state, which is correct: with no detector
reachable the UI shows nothing rather than inventing a result.

---

## 1. The only thing blocking integration

**The API contract does not match.** This is the work.

`src/lib/api.ts` posts the file to `POST http://localhost:8000/v1/detect` and
returns the JSON unchanged. The page then reads fields that response does not
contain.

| UI reads | Backend returns today |
|---|---|
| `verdict` — `ai_clone \| genuine \| human_mimicry \| unclear` | `is_fake` (boolean) |
| `risk` — 0–100 integer | `confidence` (0–1 float) |
| `family` + `familyConfidence` | `vocoder.probabilities` (map) |
| `probabilities` — 8 keys, see below | `vocoder.probabilities` — 5 keys |
| `reasons[]` — `{name, value, unit, note, sigma}` | *(absent)* |
| `scores[]` — per-window synthesis score | *(absent)* |
| `durationSec`, `elapsedMs`, `model` | *(absent)* |

Until an adapter maps one onto the other, **every real upload falls through to
the "Detector unavailable" state.** That is deliberate — see §4 — but it is
still a failure.

The exact shape the UI expects is `src/lib/sample.ts`. Treat that file as the
contract; it is the single source of truth for field names and types.

A separate session has written `types.ts` and `adapter.ts` against this shape.
If you have those, drop them in and point `api.ts` at the adapter. If not, the
mapping is mechanical except for two cases:

- **`risk`** is not `confidence * 100`. Confidence is the model's certainty in
  whichever class it picked; risk is the probability of *synthesis*. When
  `is_fake` is false, `risk = (1 - confidence) * 100`.
- **`human_mimicry` cannot be derived from `is_fake`.** It needs speaker
  verification to establish a human voice that fails to match the enrolled
  speaker. Return it only when `same_speaker === false`. Otherwise fall back to
  `unclear` — never guess.

## 2. Eight families, not five

`src/components/charts.tsx` exports `FAMILY_ORDER`, fixed:

```
real, hifigan, bigvgan, vocos, encodec, modern_codec, legacy, diffusion
```

`UI_INSTRUCTIONS.md` still documents five ending in `legacy_ar`. That predates
`TAXONOMY_PATCH.md`, which renames `legacy_ar` → `legacy` and adds `diffusion`.
The backend must emit all eight keys. Missing keys render as 0, they do not
crash.

`hifigan`, `bigvgan`, `encodec` and `modern_codec` are marked with a ring in the
chart — their primary tell does not survive an 8 kHz channel.

## 3. Tier 3 is a first-class result, not an error

When the embedding sits far from every known centroid, send `verdict: "unclear"`
with the family set to unknown. The UI styles this as information, calmly — not
as a failure. Do not force a novel generator into the nearest class.
**Update:** `tier2_family` can now return `"unknown_synthetic"` — please add a UI label for it.

## 3.5 Evidence API Changes

**Update:** The Evidence field `spectral_cutoff_hz` has been renamed to `spectral_cutoff_ratio`. Please update the UI adapter to map this correctly.

## 4. The rule that must not be broken

**No fabricated verdicts.** If there is no checkpoint, the endpoint should fail
— 503 is fine — and the UI will render "Detector unavailable — no checkpoint
loaded. No verdict can be produced." with every number blank.

Do not add a mock that returns `random.choice` with 85–99% confidence. That
already exists in `src/inference.py` as `mock_aasist_xlsr` and `app.py` is
hardwired to it. It is worse than no system, and it is the first thing a judge
will find.

## 5. Three suspect numbers — detector side, not UI

These come from the live Streamlit app, transcribed into the fixture. They are
displayed as evidence to an analyst, so they need checking before demo:

- **`hf energy ratio 0.00`** with the channel set to `none`. `ARCHITECTURE.md`'s
  own guidelines say not to rely on artifacts above 4 kHz because telephony
  destroys them. A ratio of exactly zero suggests the source is already
  band-limited or the extractor is reading an empty band.
- **`pause rate 134.1 / min`** is 2.2 per second. That is syllable rate, not
  pauses and breath gaps.
- **`20.0 standard deviations`**, exactly round, looks like a clamp being
  reported as a measurement. If it is clamped, flag it as such — a clamp shown
  as a measurement is a wrong number, not a large one.

## 6. Stack and constraints

Next.js 16.3.4 · React 19.2.8 · Tailwind v4 · lucide-react · lenis.
No charting library — both charts are hand-authored inline SVG in `charts.tsx`.

**No WebGL.** The target machine has hardware acceleration off;
`getContext('webgl')` returns null and a shader renders nothing at all. The
background is a 2D-canvas per-pixel displacement instead, at a 512×256 buffer
that CSS stretches to the viewport. Measured **1.02 ms/frame** on that machine.
Do not "upgrade" it to WebGL without checking that first.

Displacement **wraps** at the buffer edges. It must never clamp — a clamped
sample repeats the edge row and smears it down the page.

## 7. Assets

- `public/wallpaperoriginal.jpeg` — 536×1072, the original supplied by Krish.
  **Never write to this file.** It is the only copy.
- `public/wallpaper.jpeg` — 3840×1920, generated from it: an 8×2 grid with
  alternate columns mirrored and rows flipped so tile edges match, then gamma
  1.6 to reduce the number of bright filaments.

To regenerate, tile the original — do not edit the derivative. The ground must
stay grayscale, soft-edged and landscape ≥3000px: it is displaced per pixel at
runtime, so any subject or horizon line will visibly smear.

## 8. Files

| File | Purpose |
|---|---|
| `src/app/page.tsx` | The page: hero, verdict, why, charts, all six states |
| `src/app/globals.css` | Tokens, two tone ladders, the depth/lighting system |
| `src/components/SmokeWarp.tsx` | Interactive background |
| `src/components/SmoothScroll.tsx` | Lenis, sleeps when settled |
| `src/components/charts.tsx` | Family bars + score line, scroll-revealed |
| `src/lib/sample.ts` | **The data contract.** Design fixture only |
| `src/lib/api.ts` | The fetch call — needs the adapter |

`?tone=fog` switches to a lifted-charcoal variant. Default is `vault`.
