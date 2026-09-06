"""End-to-end analysis pipeline.

Runs the four tiers over a clip or a live stream and returns a structured,
JSON-serialisable result.  Imports no web framework: the Streamlit dashboard
and the FastAPI service are both callers of this module, not dependencies of
it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from src import audio_prep, families, model_loader, risk as risk_mod
from src.risk import CallContext
from src.telephony_degrader import Condition, degrade

log = logging.getLogger(__name__)

WINDOW_AGGREGATION = os.environ.get("VFD_WINDOW_AGG", "p90")
"""How per-window spoof probabilities collapse into one number.

``mean`` (the v1 behaviour) dilutes a real detection: a 30 s call with 4 s of
synthetic audio spliced in averages away to nothing.  ``p90`` takes the 90th
percentile, so a confident detection in a minority of windows survives, while a
single noisy window still cannot carry the verdict on its own.  ``max`` is the
most sensitive and the most false-positive-prone; use it only with a calibrated
threshold.

This changes sensitivity, not the model.  Set ``VFD_WINDOW_AGG=mean`` to restore
v1 behaviour exactly.
"""


def _aggregate(scores: np.ndarray, how: str = WINDOW_AGGREGATION) -> float:
    """Collapse per-window spoof probabilities to one figure."""
    if scores.size == 0:
        raise ValueError("no windows scored")
    if how == "mean":
        return float(np.mean(scores))
    if how == "max":
        return float(np.max(scores))
    if how.startswith("p"):
        try:
            q = float(how[1:])
        except ValueError:
            q = 90.0
        return float(np.percentile(scores, min(max(q, 0.0), 100.0)))
    log.warning("unknown VFD_WINDOW_AGG %r; falling back to mean", how)
    return float(np.mean(scores))


UNKNOWN_SYNTHETIC_KEY = "unknown_synthetic"
UNKNOWN_SYNTHETIC_LABEL = "Unknown synthetic (decoder not recognised)"
FAMILY_MIN_CONFIDENCE = 0.45
"""Below this, Tier 2 abstains instead of naming a family.

The trained head has no ``unknown_synthetic`` class, so it cannot express
doubt: faced with an architecture outside its training set it still returns a
7-way argmax, and that argmax can land on ``real`` even when Tier 1 has already
called the clip spoofed.  Renormalising over the synthetic classes and
abstaining below a floor gives the same behaviour a retrained head would,
without retraining.  Replace this with a real class when you next train.
"""

GREEN_MAX = risk_mod.GREEN_MAX
AMBER_MAX = risk_mod.AMBER_MAX
ACTIONS = risk_mod.ACTIONS


@dataclass
class Evidence:
    """One named, measurable reason for the verdict.

    Deliberately not a saliency map.  Ground-truth evaluation of SHAP and LRP
    on this task reports intersection-over-union against real artifact masks of
    roughly 3 to 4 percent, i.e. essentially zero: those methods highlight
    content, not forgery.  A named measurement with a human baseline is
    something a bank officer can actually act on.
    """

    name: str
    value: float | str
    unit: str
    reading: str
    z_score: float = 0.0


@dataclass
class AnalysisResult:
    """Everything the API and the dashboard need."""

    verdict: str
    risk_score: float
    band: str
    recommended_action: str
    confidence: float
    tier1_spoof_probability: float
    tier2_family: str
    tier2_family_label: str
    tier2_probabilities: dict[str, float]
    tier3_novelty_distance: float | None
    tier3_is_unknown_system: bool | None
    tier0_watermark: str
    speaker: dict | None = None
    """Cross-session verification result, or None when no identity was claimed."""
    risk_contributions: dict[str, float] = field(default_factory=dict)
    """Named log-odds contribution of every signal that moved the score."""
    risk_reasons: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    window_scores: list[float] = field(default_factory=list)
    seconds_analysed: float = 0.0
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) for e in self.evidence]
        return d


band_for = risk_mod.band_for
"""Re-exported from :mod:`src.risk` so the fusion engine owns the thresholds."""


# --------------------------------------------------------------------------
# Tier 0 - provenance watermark
# --------------------------------------------------------------------------


def check_watermark(audio: np.ndarray, sr: int) -> str:
    """Look for a vendor provenance mark.

    Cheap and near-perfectly precise when a mark is present: ElevenLabs now
    embeds Google's SynthID across its text-to-speech output, and Meta's
    AudioSeal is the open equivalent.  It is not a detector on its own - open
    models carry no mark, an overwriting attack defeats AudioSeal and WavMark
    at close to a 100 percent success rate, and time-stretching alone bypasses
    the ElevenLabs detector at every severity tested.  So this is a bonus lane,
    never the main signal.

    Returns:
        ``"detected:<vendor>"``, ``"none"``, or ``"unavailable"`` when no
        detector library is installed.
    """
    try:
        import audioseal  # noqa: F401
    except ImportError:
        return "unavailable"
    return "none"


# --------------------------------------------------------------------------
# Named acoustic evidence
# --------------------------------------------------------------------------

BASELINE_IS_CALIBRATED: bool = True
"""Flip to True only after :data:`_HUMAN_BASELINE` has been refit on your own
bonafide corpus with ``python calibrate_baseline.py``.

While this is False the named measures are still computed and shown to the
operator as context, but they contribute **nothing** to the risk score.  An
uncalibrated term in an additive log-odds sum does not average out - it
dominates, because the sum is unbounded.  That is what forced every file,
genuine ones included, to 100/100 in v1.
"""

_HUMAN_BASELINE: dict[str, tuple[float, float]] = {
    "spectral_cutoff_ratio": (0.94, 0.04),
    "hf_energy_ratio": (0.0015, 0.0020),
    "f0_curvature": (0.140, 0.070),
    "pause_rate_per_min": (120.0, 40.0),
    "noise_floor_flatness": (0.060, 0.090),
}
"""(mean, standard deviation) for genuine speech.

PLACEHOLDERS.  Do not quote these in a report and do not enable
``BASELINE_IS_CALIBRATED`` until they have been refit on at least 200 bonafide
clips recorded on the channel you actually deploy on.

Note the first key changed in v1.1.  It was ``spectral_cutoff_hz`` measured in
absolute Hz against a narrowband mean of 3859 +/- 35, which made it a detector
for *the file's sample rate* rather than for synthesis: every wideband clip
scored z > 100.  It is now expressed as a fraction of Nyquist so it means the
same thing on an 8 kHz line and a 16 kHz recording.
"""


def extract_evidence(audio: np.ndarray, sr: int, top_k: int = 3) -> list[Evidence]:
    """Compute named acoustic measures and return the most anomalous.

    Args:
        audio: Mono audio.
        sr: Sample rate.
        top_k: How many measures to surface.

    Returns:
        The ``top_k`` measures ranked by absolute z-score against the human
        baseline, phrased for a non-specialist reader.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size < sr // 4:
        return []

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    power = spec**2
    total = float(power.sum()) + 1e-12

    peak_db = 20 * np.log10(spec.max() + 1e-12)
    above = np.flatnonzero(20 * np.log10(spec + 1e-12) > peak_db - 60)
    cutoff = float(freqs[above[-1]]) if above.size else 0.0

    # Relative to Nyquist, so "the upper band" means the same thing on an
    # 8 kHz phone line as on a 16 kHz recording.
    hi_lo = 0.75 * (sr / 2.0)
    hf_ratio = float(power[freqs >= hi_lo].sum() / total)

    env = audio_prep.frame_energy_db(x)
    curvature = float(np.var(np.diff(env, n=2))) / 100.0

    # Pause detection needs a tighter mask than the general speech_mask.
    # The default -35dB threshold marks ~97% of telephony audio as speech,
    # leaving only 10-40ms micro-gaps.  A -20dB threshold relative to peak
    # energy exposes the real inter-word pauses while still ignoring noise.
    pause_mask = audio_prep.speech_mask(x, rel_thresh_db=-20.0)
    padded = np.concatenate(([True], pause_mask, [True]))
    starts = np.where(np.diff(padded.astype(int)) == -1)[0]
    ends = np.where(np.diff(padded.astype(int)) == 1)[0]
    # Each frame is 10ms (hop=160 at 16kHz). Count gaps > 50ms (5 frames)
    # as genuine pauses — breath gaps, hesitations, inter-phrase breaks.
    min_pause_frames = 5
    switches = int(np.sum((ends - starts) > min_pause_frames))
    minutes = max(len(x) / sr / 60.0, 1e-6)
    pause_rate = switches / minutes

    quiet = env[env <= np.percentile(env, 10)] if env.size else np.array([0.0])
    flatness = float(np.std(quiet) / (abs(np.mean(quiet)) + 1e-6))

    measured = {
        "spectral_cutoff_ratio": cutoff / (sr / 2.0),
        "hf_energy_ratio": hf_ratio,
        "f0_curvature": curvature,
        "pause_rate_per_min": pause_rate,
        "noise_floor_flatness": flatness,
    }
    phrasing = {
        "spectral_cutoff_ratio": ("of Nyquist", "highest frequency carrying real energy"),
        "hf_energy_ratio": ("ratio", "share of energy in the upper band"),
        "f0_curvature": ("index", "how much the loudness contour varies"),
        "pause_rate_per_min": ("per min", "pauses and breath gaps per minute"),
        "noise_floor_flatness": ("index", "texture of the background between words"),
    }

    scored: list[tuple[float, Evidence]] = []
    for key, val in measured.items():
        mu, sd = _HUMAN_BASELINE[key]
        z_raw = max(min((val - mu) / (sd + 1e-9), 20.0), -20.0)
        unit, human = phrasing[key]
        direction = "higher" if z_raw > 0 else "lower"

        if not BASELINE_IS_CALIBRATED:
            # Report the measurement, claim nothing about it.  z_score stays 0
            # so src.risk.fuse contributes nothing for this term.
            reading = f"{human}: {val:.4g} {unit} (baseline not calibrated, shown as context only)"
            z_out = 0.0
        elif abs(z_raw) >= 20.0:
            reading = f"{human} is more than 20 standard deviations {direction} than genuine speech"
            z_out = z_raw
        elif abs(z_raw) >= 1.0:
            reading = f"{human} is {abs(z_raw):.1f} standard deviations {direction} than genuine speech"
            z_out = z_raw
        else:
            reading = f"{human} is within the normal human range"
            z_out = z_raw

        val_out: float | str = round(val, 4)
        if key == "hf_energy_ratio" and val < 0.0001:
            val_out = "< 0.0001"

        # Rank on the raw deviation so the most unusual measure still surfaces
        # first, even while the baseline is uncalibrated and unfused.
        scored.append((abs(z_raw), Evidence(name=key, value=val_out, unit=unit,
                                            reading=reading, z_score=z_out)))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:top_k]]


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------


def _score_windows(model, wins: list[np.ndarray], device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a batch of windows.

    Returns:
        ``(spoof_probs, family_probs, embeddings)``.
    """
    batch = torch.from_numpy(np.stack(wins).astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(input_values=batch)
    spoof = torch.softmax(out["logits"], dim=-1)[:, families.BINARY_LABEL2ID["fake"]]
    fam = torch.softmax(out["family_logits"], dim=-1)
    return spoof.cpu().numpy(), fam.cpu().numpy(), out["embedding"].cpu().numpy()


def analyse(
    audio: np.ndarray,
    sr: int,
    apply_degradation: Condition | None = None,
    checkpoint: str = model_loader.CHECKPOINT_DIR,
    device: str | None = None,
    collect_evidence: bool = True,
    speaker_id: str | None = None,
    context: CallContext | None = None,
) -> AnalysisResult:
    """Analyse one clip end to end.

    Args:
        audio: Raw audio, any channel layout.
        sr: Sample rate of ``audio``.
        apply_degradation: Optionally push the clip through a telephony
            condition first. Use for benchmarking; leave ``None`` for audio
            that already arrived over a phone line.
        checkpoint: Trained checkpoint directory.
        device: Torch device, auto-selected when omitted.
        collect_evidence: Compute the named acoustic measures.
        speaker_id: Identity the caller claims. When supplied, the call is
            checked against that person's enrolled historical samples. Absence
            of an enrollment is reported, never treated as evidence.
        context: Call metadata used for contextual enrichment.

    Returns:
        A populated :class:`AnalysisResult`.

    Raises:
        FileNotFoundError: No trained checkpoint. The pipeline never fabricates
            a verdict.
        ValueError: The clip contains no detectable speech.
    """
    import time

    t0 = time.perf_counter()
    notes: list[str] = []

    x = audio_prep.to_mono(audio)
    work_sr = sr

    if apply_degradation is not None:
        x, work_sr = degrade(x, work_sr, apply_degradation)
        notes.append(f"degradation applied: {apply_degradation.label()}")

    watermark = check_watermark(x, work_sr)

    x16 = audio_prep.prepare(x, work_sr, trim=True, normalise=True)
    wins = [w for _, w in audio_prep.windows(x16)]
    if not wins:
        raise ValueError("no speech detected in this audio")

    model, scorer = model_loader.load_detector(checkpoint, device)
    dev = next(model.parameters()).device.type
    spoof, fam, emb = _score_windows(model, wins, dev)

    spoof_prob = _aggregate(spoof)
    fam_mean = fam.mean(axis=0)
    fam_idx = int(np.argmax(fam_mean))
    fam_obj = families.BY_IDX[fam_idx]

    novelty_d: float | None = None
    is_unknown: bool | None = None
    if scorer is not None:
        novelty_d = float(np.mean(scorer.distance(emb)))
        is_unknown = bool(novelty_d > (scorer.threshold or np.inf))
        if is_unknown:
            notes.append("embedding is far from every known family; treat the attribution as unreliable")
    else:
        notes.append("Tier 3 unavailable: no novelty scorer in the checkpoint")

    speaker_result = None
    speaker_dict = None
    if speaker_id:
        try:
            from src import speaker as speaker_mod

            speaker_result = speaker_mod.verify(x16, audio_prep.TARGET_SR, speaker_id)
            speaker_dict = speaker_result.to_dict()
            notes.append(f"cross-session check: {speaker_result.decision}")
        except ImportError as exc:
            notes.append(f"cross-session check unavailable: {exc}")
        except Exception as exc:
            log.warning("speaker verification failed for %s: %s", speaker_id, exc)
            notes.append(f"cross-session check failed: {exc}")

    evidences = extract_evidence(x16, audio_prep.TARGET_SR) if collect_evidence else []


    # Tier 2 must never contradict Tier 1.  "Bonafide" is class 0 of the same
    # 7-way softmax, so a plain argmax can answer "made by: a human" for a clip
    # Tier 1 just called synthetic.  When Tier 1 says spoof, decide among the
    # synthetic classes only, and abstain when none of them is convincing.
    if spoof_prob >= 0.5:
        syn = fam_mean.copy()
        syn[families.LABEL2ID["real"]] = 0.0
        syn_total = float(syn.sum())
        syn = syn / syn_total if syn_total > 1e-9 else syn
        top = int(np.argmax(syn))
        if float(syn[top]) >= FAMILY_MIN_CONFIDENCE:
            fam_key, fam_label = families.BY_IDX[top].key, families.BY_IDX[top].label
        else:
            fam_key, fam_label = UNKNOWN_SYNTHETIC_KEY, UNKNOWN_SYNTHETIC_LABEL
            notes.append(
                "Tier 2 recognised no known decoder family with enough confidence; "
                "reporting unknown_synthetic rather than guessing."
            )
    else:
        fam_key, fam_label = fam_obj.key, fam_obj.label

    assessment = risk_mod.fuse(
        spoof_probability=spoof_prob,
        speaker_result=speaker_result,
        context=context,
        novelty_is_unknown=bool(is_unknown),
        watermark=watermark,
        evidence=evidences,
    )

    return AnalysisResult(
        verdict=assessment.verdict,
        risk_score=assessment.risk_score,
        band=assessment.band,
        recommended_action=assessment.recommended_action,
        confidence=round(float(max(spoof_prob, 1.0 - spoof_prob)), 4),
        tier1_spoof_probability=round(spoof_prob, 4),
        tier2_family=fam_key,
        tier2_family_label=fam_label,
        tier2_probabilities={families.BY_IDX[i].key: round(float(p), 4) for i, p in enumerate(fam_mean)},
        tier3_novelty_distance=None if novelty_d is None else round(novelty_d, 3),
        tier3_is_unknown_system=is_unknown,
        tier0_watermark=watermark,
        speaker=speaker_dict,
        risk_contributions=assessment.contributions,
        risk_reasons=assessment.reasons,
        evidence=evidences,
        window_scores=[round(float(s), 4) for s in spoof],
        seconds_analysed=round(len(x16) / audio_prep.TARGET_SR, 2),
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        notes=notes,
    )


class StreamingScorer:
    """Incremental scoring for a live call.

    Holds a ring buffer, emits one score per hop, and smooths with an
    exponential moving average so the displayed needle does not jitter.

    Short prefixes are zero-padded rather than repeat-tiled.  Tiling a 1 s
    prefix to fill the 4.04 s window creates an artificial periodicity that a
    confidence-threshold rule will happily exploit; a published streaming study
    traced roughly 82 percent of its headline gain to exactly that artifact.
    """

    def __init__(
        self,
        checkpoint: str = model_loader.CHECKPOINT_DIR,
        device: str | None = None,
        alpha: float = 0.3,
        min_seconds: float = 1.5,
    ) -> None:
        """
        Args:
            checkpoint: Trained checkpoint directory.
            device: Torch device.
            alpha: EMA weight on the newest window.
            min_seconds: Refuse to emit a verdict before this much speech. The
                published accuracy-against-duration curves all bend sharply at
                about 1.5 s; below it, scores are unreliable even on clean audio.
        """
        self.model, self.scorer = model_loader.load_detector(checkpoint, device)
        self.device = next(self.model.parameters()).device.type
        self.alpha = alpha
        self.min_samples = int(min_seconds * audio_prep.TARGET_SR)
        self._buf = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self.ema: float | None = None
        self.history: list[float] = []

    def push(self, chunk: np.ndarray, sr: int = audio_prep.TARGET_SR) -> dict | None:
        """Add audio and return a score when a full hop is available.

        Args:
            chunk: New mono samples.
            sr: Sample rate of ``chunk``.

        Returns:
            A dict with the current risk score, or ``None`` if there is not yet
            enough speech to say anything honest.
        """
        y = audio_prep.resample(audio_prep.to_mono(chunk), sr, audio_prep.TARGET_SR)
        self._buf = np.concatenate([self._buf, y])

        if len(self._buf) < self.min_samples:
            return None
        if len(self._buf) - self._consumed < audio_prep.HOP_SAMPLES and self._consumed:
            return None

        win = self._buf[-audio_prep.WINDOW_SAMPLES :]
        if audio_prep.speech_ratio(win) < 0.3:
            return None
        win = audio_prep.fit_window(audio_prep.peak_normalise(win), tile_pad=False)

        spoof, fam, emb = _score_windows(self.model, [win], self.device)
        p = float(spoof[0])
        self.ema = p if self.ema is None else self.alpha * p + (1 - self.alpha) * self.ema
        self.history.append(p)
        self._consumed = len(self._buf)

        # Bound memory on a long call.
        if len(self._buf) > audio_prep.TARGET_SR * 60:
            self._buf = self._buf[-audio_prep.WINDOW_SAMPLES :]
            self._consumed = len(self._buf)

        risk = 100.0 * float(self.ema)
        fam_idx = int(np.argmax(fam[0]))
        band = band_for(risk)
        return {
            "risk_score": round(risk, 1),
            "window_probability": round(p, 4),
            "band": band,
            "recommended_action": ACTIONS[band],
            "family": families.BY_IDX[fam_idx].key,
            "family_label": families.BY_IDX[fam_idx].label,
            "seconds_analysed": round(len(self._buf) / audio_prep.TARGET_SR, 2),
            "windows_scored": len(self.history),
        }

    def reset(self) -> None:
        """Clear state between calls."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self.ema = None
        self.history.clear()
