"""Telephony channel simulation.

Why this file is the centre of the project
------------------------------------------
Every published voice-deepfake detector is evaluated on clean wideband studio
audio.  Real Indian fraud calls arrive at 8 kHz over AMR-NB with packet loss
and discontinuous transmission.  Published benchmarks show the *same* detector
spanning roughly 0.8 percent to 29 percent EER purely as a function of the
codec.  That gap is the project's research contribution, and it is measured
here.

Two facts drive the design:

* AMR-NB per-mode results have never been published, in any language.  Every
  codec benchmark to date uses AMR-*WB*.  AMR-NB is what carries a large share
  of Indian mobile-originated calls, so all eight modes are first-class here.
* Most vocoder fingerprints live above 4 kHz, which narrowband discards.  A
  detector that depends on them will collapse.  Measuring that collapse is the
  point, not a failure.

Design notes
------------
* Array in, array out.  No disk in the request path.
* The random generator is passed in explicitly.  The original implementation
  called ``np.random.seed(42)`` inside this function, which silently reset the
  global stream on every call and would destroy shuffling inside a dataloader.
* Codecs that need ffmpeg degrade gracefully: :func:`available_codecs` reports
  what this machine can actually do, so the same code runs on a laptop without
  ffmpeg and on a training box with the full 3GPP set.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src import audio_prep

NARROWBAND_SR: int = 8_000
WIDEBAND_SR: int = 16_000

AMR_NB_MODES: tuple[float, ...] = (4.75, 5.15, 5.90, 6.70, 7.40, 7.95, 10.2, 12.2)
"""All eight AMR-NB bitrates in kbps.  No paper reports these separately."""

AMR_WB_MODES: tuple[float, ...] = (6.60, 8.85, 12.65, 15.85, 18.25, 23.05, 23.85)

G726_RATES: tuple[float, ...] = (16.0, 24.0, 32.0, 40.0)
"""G.726 ADPCM bitrates in kbps. A real narrowband telephony codec (DECT, some
VoIP trunks) and, unlike AMR-NB, available in a stock ffmpeg build - which
makes it the practical way to get a bitrate sweep without compiling ffmpeg
against libopencore-amrnb."""


@dataclass(frozen=True)
class Condition:
    """One transmission condition.

    Attributes:
        codec: One of the keys in :data:`CODECS`.
        bitrate_kbps: Encoder bitrate. Ignored by fixed-rate codecs.
        packet_loss: Fraction of 20 ms frames dropped, in [0, 1].
        plc: If True, conceal a lost frame by repeating the previous one.
            If False, the frame is silenced. Toggling this isolates whether
            concealment is what destroys the forgery artifacts - published work
            observed the effect but never made it a variable.
        name: Optional label used in result tables.
    """

    codec: str = "g711_ulaw"
    bitrate_kbps: float | None = None
    packet_loss: float = 0.0
    plc: bool = True
    name: str | None = None

    def label(self) -> str:
        """Human-readable identifier for report tables."""
        if self.name:
            return self.name
        parts = [self.codec]
        if self.bitrate_kbps is not None:
            parts.append(f"{self.bitrate_kbps:g}k")
        if self.packet_loss > 0:
            parts.append(f"pl{int(self.packet_loss * 100)}")
            parts.append("plc" if self.plc else "noplc")
        return "-".join(parts)


# codec key -> (ffmpeg encoder, container, native sample rate, needs_ffmpeg)
CODECS: dict[str, tuple[str | None, str | None, int, bool]] = {
    "none":       (None, None, WIDEBAND_SR, False),
    "g711_ulaw":  (None, None, NARROWBAND_SR, False),
    "g711_alaw":  (None, None, NARROWBAND_SR, False),
    "amr_nb":     ("libopencore_amrnb", "amr", NARROWBAND_SR, True),
    "gsm_fr":     ("libgsm", "gsm", NARROWBAND_SR, True),
    "g726":       ("g726", "wav", NARROWBAND_SR, True),
    "g722":       ("g722", "g722", WIDEBAND_SR, True),
    "amr_wb":     ("libvo_amrwbenc", "amr", WIDEBAND_SR, True),
    "opus_nb":    ("libopus", "ogg", WIDEBAND_SR, True),
    "speex_nb":   ("libspeex", "ogg", NARROWBAND_SR, True),
}

NARROWBAND_CODECS: tuple[str, ...] = ("g711_ulaw", "g711_alaw", "g726", "amr_nb", "gsm_fr", "speex_nb")
WIDEBAND_CONTROLS: tuple[str, ...] = ("g722", "amr_wb", "opus_nb")


def _ffmpeg() -> str | None:
    """Path to ffmpeg, or None."""
    return shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def _ffmpeg_encoders() -> frozenset[str]:
    """Encoder names this ffmpeg build supports."""
    exe = _ffmpeg()
    if not exe:
        return frozenset()
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


def available_codecs() -> dict[str, bool]:
    """Which codecs this machine can actually run.

    Call this at startup and log it. A silently-skipped codec is how a
    benchmark ends up claiming coverage it does not have.
    """
    enc = _ffmpeg_encoders()
    return {k: (not needs_ff) or (name in enc) for k, (name, _, _, needs_ff) in CODECS.items()}


def missing_codecs() -> list[str]:
    """Codecs unavailable here. Report these rather than dropping them quietly."""
    return sorted(k for k, ok in available_codecs().items() if not ok)


def _to_pcm16(x: np.ndarray) -> bytes:
    return np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0).__mul__(32767.0).astype("<i2").tobytes()


def _from_pcm16(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()


def _run(cmd: list[str], payload: bytes, timeout: int = 60) -> bytes:
    proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd[:6])}...): {' | '.join(tail)}")
    return proc.stdout


def _ffmpeg_roundtrip(x: np.ndarray, sr: int, codec: str, bitrate_kbps: float | None) -> np.ndarray:
    """Encode through a real codec and decode back, entirely in memory."""
    exe = _ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found; install it or pick a codec from available_codecs()")
    encoder, container, _, _ = CODECS[codec]
    if encoder is None or container is None:
        raise ValueError(f"{codec} is not an ffmpeg codec")

    enc_cmd = [exe, "-hide_banner", "-loglevel", "error",
               "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
               "-c:a", encoder]
    if bitrate_kbps is not None:
        enc_cmd += ["-b:a", f"{int(round(bitrate_kbps * 1000))}"]
    if codec == "opus_nb":
        enc_cmd += ["-application", "voip", "-frame_duration", "20"]
    enc_cmd += ["-f", container, "pipe:1"]

    encoded = _run(enc_cmd, _to_pcm16(x))

    dec_cmd = [exe, "-hide_banner", "-loglevel", "error",
               "-f", container, "-i", "pipe:0",
               "-f", "s16le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    return _from_pcm16(_run(dec_cmd, encoded))


def _g711(x: np.ndarray, law: str) -> np.ndarray:
    """G.711 companding round-trip in pure NumPy. No ffmpeg required."""
    s = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    if law == "ulaw":
        mu = 255.0
        code = np.round(np.sign(s) * np.log1p(mu * np.abs(s)) / np.log1p(mu) * 127.0) / 127.0
        return (np.sign(code) * (np.expm1(np.abs(code) * np.log1p(mu)) / mu)).astype(np.float32)
    a = 87.6
    absx = np.abs(s)
    lin = absx < (1.0 / a)
    y = np.where(lin, a * absx, 1.0 + np.log(np.maximum(a * absx, 1e-12))) / (1.0 + np.log(a))
    code = np.round(np.sign(s) * y * 127.0) / 127.0
    ay = np.abs(code) * (1.0 + np.log(a))
    inv = np.where(ay < 1.0, ay / a, np.exp(ay - 1.0) / a)
    return (np.sign(code) * inv).astype(np.float32)


def apply_packet_loss(
    x: np.ndarray, sr: int, loss_rate: float, plc: bool = True, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Drop 20 ms frames at random, with or without concealment.

    Args:
        x: Mono audio.
        sr: Sample rate.
        loss_rate: Fraction of frames to drop, in [0, 1].
        plc: Repeat the previous frame instead of silencing it.
        rng: Explicit generator. A fresh one is used if omitted; the global
            NumPy stream is never touched.
    """
    if loss_rate <= 0.0:
        return np.asarray(x, dtype=np.float32)
    if not 0.0 <= loss_rate <= 1.0:
        raise ValueError("loss_rate must be in [0, 1]")
    rng = rng or np.random.default_rng()
    frame = max(int(0.020 * sr), 1)
    y = np.array(x, dtype=np.float32, copy=True)
    n_frames = len(y) // frame
    drop = rng.random(n_frames) < loss_rate
    for i in np.flatnonzero(drop):
        a, b = i * frame, (i + 1) * frame
        if plc and i > 0:
            prev = y[a - frame : a]
            y[a:b] = prev * np.linspace(1.0, 0.3, num=len(prev), dtype=np.float32)
        else:
            y[a:b] = 0.0
    return y


def degrade(
    x: np.ndarray,
    sr: int,
    condition: Condition | None = None,
    rng: np.random.Generator | None = None,
    return_sr: int | None = NARROWBAND_SR,
) -> tuple[np.ndarray, int]:
    """Push audio through one transmission condition.

    Args:
        x: Audio, any shape accepted by :func:`audio_prep.to_mono`.
        sr: Input sample rate.
        condition: Transmission condition. Defaults to plain G.711 mu-law.
        rng: Generator for packet loss. The global stream is never seeded.
        return_sr: Resample the result to this rate, or None to keep the
            codec's native rate.

    Returns:
        ``(audio, sample_rate)``.

    Raises:
        ValueError: Unknown codec.
        RuntimeError: A codec was requested that this machine cannot run.
    """
    cond = condition or Condition()
    if cond.codec not in CODECS:
        raise ValueError(f"unknown codec {cond.codec!r}; known: {sorted(CODECS)}")

    _, _, native_sr, needs_ff = CODECS[cond.codec]
    y = audio_prep.to_mono(x)

    if needs_ff and not available_codecs()[cond.codec]:
        raise RuntimeError(
            f"codec {cond.codec!r} needs an ffmpeg build with its encoder. "
            f"Missing here: {missing_codecs()}"
        )

    y = audio_prep.resample(y, sr, native_sr)
    work_sr = native_sr

    if cond.codec == "g711_ulaw":
        y = _g711(y, "ulaw")
    elif cond.codec == "g711_alaw":
        y = _g711(y, "alaw")
    elif cond.codec != "none":
        y = _ffmpeg_roundtrip(y, work_sr, cond.codec, cond.bitrate_kbps)

    if cond.packet_loss > 0:
        y = apply_packet_loss(y, work_sr, cond.packet_loss, cond.plc, rng)

    if return_sr is not None and return_sr != work_sr:
        y = audio_prep.resample(y, work_sr, return_sr)
        work_sr = return_sr
    return np.asarray(y, dtype=np.float32), work_sr


def condition_grid(
    codecs: tuple[str, ...] | None = None,
    packet_loss_rates: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10, 0.20),
    include_plc_ablation: bool = True,
    only_available: bool = True,
) -> list[Condition]:
    """The full evaluation grid: every codec crossed with every loss rate.

    This is the benchmark axis.  With AMR-NB expanded to all eight modes it
    produces the language-by-codec-by-loss matrix that no published paper
    contains.

    Args:
        codecs: Restrict to these codec keys. Defaults to narrowband plus
            wideband controls.
        packet_loss_rates: Loss fractions to sweep.
        include_plc_ablation: Also emit a no-concealment variant at each
            non-zero loss rate.
        only_available: Skip codecs this machine cannot run.

    Returns:
        Conditions in a stable order, suitable as table rows.
    """
    keys = codecs or (NARROWBAND_CODECS + WIDEBAND_CONTROLS)
    have = available_codecs()
    out: list[Condition] = [Condition(codec="none", name="clean-16k")]

    for key in keys:
        if only_available and not have.get(key, False):
            continue
        if key == "amr_nb":
            rates: tuple[float | None, ...] = AMR_NB_MODES
        elif key == "amr_wb":
            rates = AMR_WB_MODES
        elif key == "opus_nb":
            rates = (6.0, 12.0, 24.0)
        elif key == "speex_nb":
            rates = (8.0, 15.0)
        elif key == "g726":
            rates = G726_RATES
        else:
            rates = (None,)
        for br in rates:
            for pl in packet_loss_rates:
                out.append(Condition(codec=key, bitrate_kbps=br, packet_loss=pl, plc=True))
                if include_plc_ablation and pl > 0:
                    out.append(Condition(codec=key, bitrate_kbps=br, packet_loss=pl, plc=False))
    return out


def simulate_telephony_degradation(input_path: str, output_path: str) -> str:
    """File-in, file-out wrapper kept for the dataset builder and old callers.

    The original signature is preserved so nothing downstream breaks. New code
    should call :func:`degrade` and stay in memory.
    """
    import os

    import soundfile as sf

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    data, sr = sf.read(input_path, dtype="float32")
    y, out_sr = degrade(data, sr, Condition(codec="g711_ulaw"))
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(output_path, y, out_sr, format="WAV", subtype="PCM_16")
    return os.path.abspath(output_path)
