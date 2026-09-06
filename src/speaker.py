"""Cross-session speaker verification.

The problem statement asks for "cross-session consistency checks comparing
ongoing call features against historical genuine samples (where available) to
detect anomalies in speaker identity".  That is a different question from the
one Tiers 0 to 3 answer.

    Tiers 0-3 ask:  was this audio made by a machine?
    This module asks: is this the person it claims to be?

They fail differently, which is why both are needed:

* A clone good enough to leave no artifacts still has to match an enrolled
  voiceprint, and a poor clone will not.
* A **human impersonator** leaves no synthesis artifacts at all. No artifact
  detector will ever catch one. Only speaker verification can.

Nothing here is trained
-----------------------
ECAPA-TDNN ships pretrained on VoxCeleb.  "Enrollment" is a forward pass over a
few genuine clips, not a training run: no extra dataset, no GPU hours.  The
only fitted quantity is a similarity threshold, which needs a few dozen pairs
and no gradient descent.

Privacy
-------
The store holds 192-dimension embeddings and metadata.  It never holds audio,
and an embedding cannot be inverted back to a waveform.  That satisfies the
"feature-only logging" requirement directly rather than by policy.

A published result worth knowing before you read the consistency code
---------------------------------------------------------------------
Zhang et al. (Institute of Acoustics, CAS, 2023) measured embedding stability
across a clip and found the sign runs **opposite to intuition**: synthetic
speech is *more* stable than genuine speech, because a clone is generated from
one fixed speaker vector while a real speaker's voice drifts with effort,
emotion and posture.  If you assume clones sound jittery you will invert your
decision rule.  Their within-clip result also generalises poorly (1.79 percent
EER in-domain against 29.66 percent in the wild), so it is used here as a
supporting signal, never as a standalone verdict.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from src import audio_prep

log = logging.getLogger(__name__)

ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
EMBED_DIM = 192

CALIBRATION_FILE = os.environ.get("VFD_SPEAKER_CALIBRATION", "calibration/speaker_thresholds.json")
OPERATING_POINT = os.environ.get("VFD_OPERATING_POINT", "narrowband")
"""``narrowband`` or ``clean``. Narrowband is the default because the deployment
target is a phone line, and thresholds fitted on clean audio are wrong there."""


def _load_calibration() -> tuple[float, float, str]:
    """Read measured thresholds, falling back to library defaults with a warning.

    The fallback values are ECAPA defaults and were never fitted to anything.
    Running on them is not an error, but it must be visible - so it logs.
    """
    fallback = (0.55, 0.35, "UNCALIBRATED (ECAPA library defaults)")
    try:
        with open(CALIBRATION_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        point = data["operating_points"][OPERATING_POINT]
        return (float(point["accept"]), float(point["reject"]),
                f"{OPERATING_POINT} (measured, EER {100 * point['eer']:.2f}%)")
    except (OSError, KeyError, ValueError, TypeError):
        log.warning(
            "no calibration at %s - falling back to ECAPA library defaults, which were "
            "never fitted. Run tools/eval_crosssession.py and do not quote these numbers.",
            CALIBRATION_FILE,
        )
        return fallback


DEFAULT_ACCEPT, DEFAULT_REJECT, CALIBRATION_SOURCE = _load_calibration()
"""Similarity above DEFAULT_ACCEPT accepts the claimed identity; below
DEFAULT_REJECT rejects it. Between the two is 'inconclusive', which is an honest
third answer and better than forcing a binary call on a noisy line."""

MIN_ENROLL_CLIPS = 3
_STORE_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Embedding extractor
# --------------------------------------------------------------------------


class SpeakerEncoder:
    """ECAPA-TDNN wrapper. Pretrained, never fine-tuned here.

    Loaded lazily so that importing this module costs nothing and the rest of
    the system still runs if SpeechBrain is not installed.
    """

    _instance: "SpeakerEncoder | None" = None

    def __init__(self, model_id: str = ECAPA_MODEL_ID, device: str | None = None) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Speaker verification needs SpeechBrain. Install it with:\n"
                "    pip install speechbrain\n"
                "Nothing is trained - the model is downloaded pretrained from VoxCeleb."
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=os.environ.get("VFD_ECAPA_CACHE", ".cache/ecapa"),
            run_opts={"device": self.device},
        )

    @classmethod
    def shared(cls, device: str | None = None) -> "SpeakerEncoder":
        """Process-wide singleton. Plain module state, no framework involved."""
        with _STORE_LOCK:
            if cls._instance is None:
                cls._instance = cls(device=device)
            return cls._instance

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """One L2-normalised 192-d voiceprint for a clip.

        Args:
            audio: Mono audio.
            sr: Sample rate.

        Returns:
            Unit-norm embedding of shape ``(192,)``.
        """
        import torch

        x = audio_prep.prepare(audio, sr, trim=True, normalise=True)
        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(x).unsqueeze(0).to(self.device))
        v = emb.squeeze().detach().cpu().numpy().astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    def embed_windows(self, audio: np.ndarray, sr: int, hop_seconds: float = 1.5) -> np.ndarray:
        """One embedding per window, for within-call analysis.

        Args:
            audio: Mono audio.
            sr: Sample rate.
            hop_seconds: Spacing between windows.

        Returns:
            ``(n_windows, 192)``. Empty if there is not enough speech.
        """
        x = audio_prep.prepare(audio, sr, trim=True, normalise=True)
        win = audio_prep.WINDOW_SAMPLES
        hop = int(hop_seconds * audio_prep.TARGET_SR)
        if len(x) < win:
            return self.embed(x, audio_prep.TARGET_SR)[None, :]
        out = [
            self.embed(x[s : s + win], audio_prep.TARGET_SR)
            for s in range(0, len(x) - win + 1, hop)
            if audio_prep.speech_ratio(x[s : s + win]) >= 0.4
        ]
        return np.stack(out) if out else np.empty((0, EMBED_DIM))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, in [-1, 1]."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


# --------------------------------------------------------------------------
# Enrollment store - embeddings only, never audio
# --------------------------------------------------------------------------


@dataclass
class Session:
    """One historical genuine interaction with an enrolled speaker."""

    session_id: str
    centroid: list[float]
    n_windows: int
    recorded_at: str
    channel: str = "unknown"
    note: str = ""


@dataclass
class Enrollment:
    """Everything held about one enrolled person. No audio, ever."""

    speaker_id: str
    display_name: str
    sessions: list[Session] = field(default_factory=list)
    accept_threshold: float = DEFAULT_ACCEPT
    reject_threshold: float = DEFAULT_REJECT

    def centroid(self) -> np.ndarray:
        """Mean voiceprint across every enrolled session."""
        if not self.sessions:
            raise ValueError(f"speaker {self.speaker_id!r} has no enrolled sessions")
        c = np.mean([s.centroid for s in self.sessions], axis=0)
        return c / (np.linalg.norm(c) + 1e-12)

    def session_matrix(self) -> np.ndarray:
        """``(n_sessions, 192)`` of per-session centroids."""
        return np.asarray([s.centroid for s in self.sessions], dtype=np.float64)

    def self_similarity(self) -> tuple[float, float]:
        """How consistent this speaker is with themselves across sessions.

        Returns:
            ``(mean, std)`` of pairwise cosine similarity between sessions.
            This is the personal baseline a live call is judged against - a
            speaker who is naturally variable should not be penalised for it.
        """
        m = self.session_matrix()
        if len(m) < 2:
            return 1.0, 0.0
        sims = [cosine(m[i], m[j]) for i in range(len(m)) for j in range(i + 1, len(m))]
        return float(np.mean(sims)), float(np.std(sims))


class EnrollmentStore:
    """On-disk store of voiceprints.

    Layout: one JSON index plus one ``.npz`` of embeddings. Deliberately
    file-backed rather than a database so it is auditable - you can open it and
    confirm there is no audio in it.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("VFD_SPEAKER_STORE", "./data/speakers")
        self.index_file = os.path.join(self.path, "index.json")
        self._enrollments: dict[str, Enrollment] = {}
        self.load()

    def load(self) -> None:
        """Read the store from disk. Missing store is not an error."""
        if not os.path.isfile(self.index_file):
            return
        with open(self.index_file, encoding="utf-8") as fh:
            raw = json.load(fh)
        self._enrollments = {
            sid: Enrollment(
                speaker_id=rec["speaker_id"],
                display_name=rec["display_name"],
                sessions=[Session(**s) for s in rec["sessions"]],
                accept_threshold=rec.get("accept_threshold", DEFAULT_ACCEPT),
                reject_threshold=rec.get("reject_threshold", DEFAULT_REJECT),
            )
            for sid, rec in raw.items()
        }

    def save(self) -> None:
        """Persist. Writes embeddings and metadata; never audio."""
        os.makedirs(self.path, exist_ok=True)
        with _STORE_LOCK, open(self.index_file, "w", encoding="utf-8") as fh:
            json.dump({sid: asdict(e) for sid, e in self._enrollments.items()}, fh, indent=2)

    def enroll(
        self,
        speaker_id: str,
        embeddings: np.ndarray,
        display_name: str = "",
        session_id: str | None = None,
        channel: str = "unknown",
        note: str = "",
    ) -> Enrollment:
        """Add one genuine session for a speaker.

        Args:
            speaker_id: Stable identifier, e.g. an employee number.
            embeddings: ``(n_windows, 192)`` from :meth:`SpeakerEncoder.embed_windows`.
            display_name: Human-readable name, used in alerts.
            session_id: Defaults to a UTC timestamp.
            channel: "pstn", "voip", "in-person" - lets you spot a voiceprint
                that only ever came from one channel, which would confound.
            note: Free text.

        Returns:
            The updated enrollment.

        Raises:
            ValueError: Wrong embedding shape.
        """
        emb = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        if emb.shape[1] != EMBED_DIM:
            raise ValueError(f"expected {EMBED_DIM}-d embeddings, got {emb.shape[1]}")

        c = emb.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        rec = self._enrollments.setdefault(
            speaker_id, Enrollment(speaker_id=speaker_id, display_name=display_name or speaker_id)
        )
        rec.sessions.append(
            Session(
                session_id=session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                centroid=c.tolist(),
                n_windows=int(emb.shape[0]),
                recorded_at=datetime.now(timezone.utc).isoformat(),
                channel=channel,
                note=note,
            )
        )
        self.save()
        return rec

    def get(self, speaker_id: str) -> Enrollment | None:
        return self._enrollments.get(speaker_id)

    def list_speakers(self) -> list[dict]:
        """Summary of everyone enrolled, for the API."""
        out = []
        for e in self._enrollments.values():
            mean, std = e.self_similarity()
            out.append({
                "speaker_id": e.speaker_id,
                "display_name": e.display_name,
                "sessions": len(e.sessions),
                "ready": len(e.sessions) >= MIN_ENROLL_CLIPS,
                "self_similarity_mean": round(mean, 4),
                "self_similarity_std": round(std, 4),
                "accept_threshold": e.accept_threshold,
            })
        return out

    def delete(self, speaker_id: str) -> bool:
        """Remove a speaker. Right-to-erasure, one call."""
        if speaker_id in self._enrollments:
            del self._enrollments[speaker_id]
            self.save()
            return True
        return False


# --------------------------------------------------------------------------
# Verification and consistency
# --------------------------------------------------------------------------


@dataclass
class SpeakerResult:
    """Outcome of a cross-session check."""

    speaker_id: str
    display_name: str
    decision: str
    """accept, reject, inconclusive, not_enrolled, or insufficient_enrollment."""
    similarity: float
    similarity_to_each_session: list[float]
    enrolled_sessions: int
    self_similarity_mean: float
    consistency_index: float
    """Within-call embedding stability. Higher means more stable, which -
    counter-intuitively - leans synthetic."""
    change_points: list[float]
    """Seconds into the call where the voice appears to change."""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def temporal_consistency(window_embeddings: np.ndarray) -> float:
    """How stable the voiceprint is across a call, in [0, 1].

    Read the direction carefully. A clone is rendered from one fixed speaker
    vector, so its embedding barely moves. A real person's does, because their
    vocal effort, emotion and distance from the handset change. **High
    stability leans synthetic.** Zhang et al. (2023) measured this and it is
    the opposite of what most people assume.

    Args:
        window_embeddings: ``(n, 192)``.

    Returns:
        0 for a wildly varying voice, 1 for a perfectly constant one. Returns
        0.5 when there are too few windows to say anything.
    """
    e = np.atleast_2d(np.asarray(window_embeddings, dtype=np.float64))
    if len(e) < 3:
        return 0.5
    sims = [cosine(e[i], e[i + 1]) for i in range(len(e) - 1)]
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def detect_change_points(
    window_embeddings: np.ndarray, hop_seconds: float = 1.5, drop: float = 0.25, sustain: int = 2
) -> list[float]:
    """Find moments where the speaker appears to change mid-call.

    This is the fraud pattern nobody in the published literature covers: a live
    social engineer builds rapport, then hands the call to a clone for the
    sentence that authorises the transfer - or the reverse. A whole-call
    verdict averages that away entirely.

    Args:
        window_embeddings: ``(n, 192)``, evenly spaced.
        hop_seconds: Spacing between windows, used to report a timestamp.
        drop: Similarity fall against the running mean that counts as a change.
        sustain: Consecutive windows the drop must persist for, so a cough or a
            single noisy window does not fire it.

    Returns:
        Seconds into the call at which a sustained change begins.
    """
    e = np.atleast_2d(np.asarray(window_embeddings, dtype=np.float64))
    if len(e) < sustain + 3:
        return []

    points: list[float] = []
    run = 0
    for i in range(2, len(e)):
        ref = e[:i].mean(axis=0)
        if cosine(e[i], ref) < cosine(e[i - 1], ref) - drop or cosine(e[i], ref) < (
            np.mean([cosine(e[j], ref) for j in range(i)]) - drop
        ):
            run += 1
            if run == sustain:
                points.append(round((i - sustain + 1) * hop_seconds, 2))
        else:
            run = 0
    return points


def verify(
    audio: np.ndarray,
    sr: int,
    speaker_id: str,
    store: EnrollmentStore | None = None,
    encoder: SpeakerEncoder | None = None,
    hop_seconds: float = 1.5,
) -> SpeakerResult:
    """Cross-session check of a live call against an enrolled voiceprint.

    Args:
        audio: The call audio.
        sr: Sample rate.
        speaker_id: The identity being claimed.
        store: Enrollment store. A default-located one is used if omitted.
        encoder: Speaker encoder. The shared singleton is used if omitted.
        hop_seconds: Window spacing for the within-call analysis.

    Returns:
        A :class:`SpeakerResult`. Never raises for an unknown speaker - it
        returns ``decision="not_enrolled"``, because "we have no baseline for
        this person" is a real operational answer and must not be confused with
        "this person failed verification".
    """
    store = store or EnrollmentStore()
    rec = store.get(speaker_id)

    if rec is None:
        return SpeakerResult(
            speaker_id, speaker_id, "not_enrolled", 0.0, [], 0, 0.0, 0.5, [],
            ["No historical genuine samples on file for this identity. "
             "Verification cannot run; rely on the synthesis tiers and call context."],
        )
    if len(rec.sessions) < MIN_ENROLL_CLIPS:
        return SpeakerResult(
            speaker_id, rec.display_name, "insufficient_enrollment", 0.0, [], len(rec.sessions),
            0.0, 0.5, [],
            [f"Only {len(rec.sessions)} enrolled session(s); {MIN_ENROLL_CLIPS} are needed "
             f"before a similarity threshold means anything."],
        )

    enc = encoder or SpeakerEncoder.shared()
    wins = enc.embed_windows(audio, sr, hop_seconds)
    if len(wins) == 0:
        return SpeakerResult(
            speaker_id, rec.display_name, "inconclusive", 0.0, [], len(rec.sessions),
            0.0, 0.5, [], ["Not enough speech in this call to extract a voiceprint."],
        )

    live = wins.mean(axis=0)
    live = live / (np.linalg.norm(live) + 1e-12)

    per_session = [round(cosine(live, s.centroid), 4) for s in rec.sessions]
    sim = cosine(live, rec.centroid())
    self_mean, self_std = rec.self_similarity()
    consistency = temporal_consistency(wins)
    changes = detect_change_points(wins, hop_seconds)

    reasons: list[str] = []
    if sim >= rec.accept_threshold:
        decision = "accept"
        reasons.append(f"Voice matches the enrolled profile for {rec.display_name} (similarity {sim:.2f}).")
    elif sim <= rec.reject_threshold:
        decision = "reject"
        reasons.append(
            f"Voice does NOT match the enrolled profile for {rec.display_name} "
            f"(similarity {sim:.2f}, needs {rec.accept_threshold:.2f})."
        )
    else:
        decision = "inconclusive"
        reasons.append(
            f"Similarity {sim:.2f} sits between the reject and accept thresholds. "
            f"Verify by a second channel."
        )

    # Judge against the speaker's own variability, not a global constant.
    if self_std > 0 and sim < (self_mean - 2.5 * self_std):
        reasons.append(
            f"This call is further from the profile than {rec.display_name}'s own sessions "
            f"ever are from each other (baseline {self_mean:.2f} ± {self_std:.2f})."
        )
    if consistency > 0.97:
        reasons.append(
            "Voiceprint is unusually constant across the call. Genuine speech drifts with "
            "effort and emotion; a clone rendered from one fixed speaker vector does not."
        )
    if changes:
        reasons.append(
            f"Voice appears to change at {', '.join(f'{t:.1f}s' for t in changes)}. "
            f"Possible hand-off between a live caller and a synthetic one."
        )

    return SpeakerResult(
        speaker_id, rec.display_name, decision, round(sim, 4), per_session,
        len(rec.sessions), round(self_mean, 4), round(consistency, 4), changes, reasons,
    )


def enroll_from_audio(
    speaker_id: str,
    clips: list[tuple[np.ndarray, int]],
    display_name: str = "",
    store: EnrollmentStore | None = None,
    encoder: SpeakerEncoder | None = None,
    channel: str = "unknown",
) -> Enrollment:
    """Enrol a speaker from genuine clips. One session per clip.

    This is not training. Each clip is a single forward pass through a
    pretrained network, and the audio is discarded immediately afterwards.

    Args:
        speaker_id: Stable identifier.
        clips: ``[(audio, sample_rate), ...]``. Three or more, from separate
            genuine occasions - three clips from one recording teach the store
            nothing about how this person varies.
        display_name: Name used in alerts.
        store: Enrollment store.
        encoder: Speaker encoder.
        channel: Channel these clips came from.

    Returns:
        The enrollment record.

    Raises:
        ValueError: Fewer than :data:`MIN_ENROLL_CLIPS` clips.
    """
    if len(clips) < MIN_ENROLL_CLIPS:
        raise ValueError(
            f"need at least {MIN_ENROLL_CLIPS} genuine clips, got {len(clips)}. "
            f"Use clips from separate occasions, not slices of one recording."
        )
    store = store or EnrollmentStore()
    enc = encoder or SpeakerEncoder.shared()
    rec = None
    for i, (audio, sr) in enumerate(clips):
        emb = enc.embed_windows(audio, sr)
        if len(emb) == 0:
            log.warning("clip %d for %s has no usable speech; skipped", i, speaker_id)
            continue
        rec = store.enroll(speaker_id, emb, display_name, channel=channel, note=f"clip {i}")
    if rec is None:
        raise ValueError("none of the supplied clips contained usable speech")
    return rec
