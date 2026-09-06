"""Audio preparation: loading, trimming, windowing and voice activity.

Two things here are not cosmetic.

1. Silence trimming.  ASVspoof 2019 and 2021 contain an uneven distribution of
   leading and trailing silence that correlates with the label.  Detectors
   learn it.  Masking silence has been shown to drive AASIST to a 99.99 percent
   false-accept rate, which means a model that scores well in the lab can be
   reading nothing but the silence.  We trim both classes identically.

2. Windowing at 64,600 samples.  The AASIST lineage crops to exactly that
   (4.0375 s at 16 kHz) and repeat-tiles anything shorter.  Feeding it a
   tile-padded 1 s clip creates an artificial periodicity that a confidence
   selector can exploit - a published streaming study found ~82 percent of its
   headline gain was this artifact.  We window at 4.04 s with a 1 s hop and
   never tile-pad a short prefix in the streaming path.
"""

from __future__ import annotations

import numpy as np

TARGET_SR: int = 16_000
WINDOW_SAMPLES: int = 64_600
"""4.0375 seconds at 16 kHz - the AASIST architectural crop."""
HOP_SAMPLES: int = 16_000
"""1 second hop, so the risk score updates once per second."""


def to_mono(x: np.ndarray) -> np.ndarray:
    """Mix down to mono, accepting (n,), (n, ch) or (ch, n)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    # Heuristic: the channel axis is the short one.
    return x.mean(axis=1 if x.shape[0] > x.shape[1] else 0).astype(np.float32)


def resample(x: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample with librosa when available, otherwise linear interpolation.

    The fallback keeps this module importable in a minimal container that has
    no librosa; it is adequate for integer-ratio telephony rates.
    """
    if orig_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    try:
        import librosa

        return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        n_out = int(round(len(x) * target_sr / orig_sr))
        src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(dst, src, np.asarray(x, dtype=np.float64)).astype(np.float32)


def frame_energy_db(x: np.ndarray, frame: int = 320, hop: int = 160) -> np.ndarray:
    """Per-frame RMS energy in dBFS. 20 ms frames, 10 ms hop at 16 kHz."""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(np.square(x[idx].astype(np.float64)), axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def speech_mask(x: np.ndarray, rel_thresh_db: float = -35.0, frame: int = 320, hop: int = 160) -> np.ndarray:
    """Boolean per-frame speech mask, thresholded relative to the loudest frame.

    Deliberately energy-based rather than a neural VAD: it is deterministic,
    has no download, and adds no latency. Swap in silero-vad if you need
    robustness to babble noise.
    """
    e = frame_energy_db(x, frame, hop)
    return e > (float(e.max()) + rel_thresh_db)


def speech_ratio(x: np.ndarray) -> float:
    """Fraction of frames that look like speech. Used to drop empty windows."""
    m = speech_mask(x)
    return float(m.mean()) if m.size else 0.0


def trim_silence(x: np.ndarray, rel_thresh_db: float = -35.0, keep_pad_ms: int = 50) -> np.ndarray:
    """Remove leading and trailing silence, keeping a small pad.

    Apply this identically to bonafide and spoof. Applying it to one class only
    reintroduces exactly the artifact it is meant to remove.
    """
    m = speech_mask(x, rel_thresh_db)
    if not m.any():
        return np.asarray(x, dtype=np.float32)
    hop = 160
    pad = int(keep_pad_ms * TARGET_SR / 1000)
    first, last = int(np.argmax(m)), int(len(m) - 1 - np.argmax(m[::-1]))
    a = max(0, first * hop - pad)
    b = min(len(x), (last + 1) * hop + pad)
    return np.asarray(x[a:b], dtype=np.float32)


def peak_normalise(x: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale to a fixed peak. Prevents level from leaking as a class cue."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return np.asarray(x, dtype=np.float32) if peak < 1e-9 else (x * (target_peak / peak)).astype(np.float32)


def fit_window(x: np.ndarray, n: int = WINDOW_SAMPLES, tile_pad: bool = True) -> np.ndarray:
    """Crop or pad a clip to exactly ``n`` samples.

    Args:
        x: Mono audio.
        n: Target length.
        tile_pad: If True, repeat-tile short clips (the AASIST training
            convention). If False, zero-pad. Use ``False`` in the streaming
            path so short prefixes do not gain an artificial periodicity.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= n:
        return x[:n]
    if not tile_pad:
        return np.pad(x, (0, n - len(x)))
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n].astype(np.float32)


def windows(
    x: np.ndarray,
    win: int = WINDOW_SAMPLES,
    hop: int = HOP_SAMPLES,
    min_speech_ratio: float = 0.5,
) -> list[tuple[int, np.ndarray]]:
    """Split a clip into overlapping analysis windows.

    Args:
        x: Mono 16 kHz audio.
        win: Window length in samples.
        hop: Hop in samples.
        min_speech_ratio: Windows below this speech fraction are dropped.

    Returns:
        List of ``(start_sample, window)``. Empty if the clip is all silence.
    """
    x = np.asarray(x, dtype=np.float32)
    out: list[tuple[int, np.ndarray]] = []
    if len(x) < win:
        w = fit_window(x, win, tile_pad=False)
        return [(0, w)] if speech_ratio(w) >= min_speech_ratio else []
    for start in range(0, len(x) - win + 1, hop):
        w = x[start : start + win]
        if speech_ratio(w) >= min_speech_ratio:
            out.append((start, w))
    return out


def prepare(x: np.ndarray, sr: int, trim: bool = True, normalise: bool = True) -> np.ndarray:
    """Full preparation: mono, resample to 16 kHz, trim, normalise."""
    y = to_mono(x)
    y = resample(y, sr, TARGET_SR)
    if trim:
        y = trim_silence(y)
    if normalise:
        y = peak_normalise(y)
    return y
