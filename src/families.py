"""Vocoder-family taxonomy for Tier 2 attribution.

Rationale
---------
Roughly 35 public voice-cloning systems exist, but almost none ship their own
waveform generator: they reuse an existing vocoder or neural codec.  Published
work (Zhang et al., CCL 2024) measured this directly - when the vocoder varies
at test time, acoustic-model identification collapses to ~10% F1 while vocoder
identification holds at ~98% F1.  The vocoder fingerprint dominates.

So we classify the *decoder family*, not the system.  Six families cover the
field, and a system released next month will almost certainly reuse one of
them.  Anything that matches none of them is handled by Tier 3 (open-set).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    """One decoder family."""

    idx: int
    key: str
    label: str
    mechanism: str
    systems: tuple[str, ...]
    tell: str
    narrowband_safe: bool = field(default=False)
    """True if the family's primary artifact survives an 8 kHz phone channel."""


REAL = Family(
    idx=0,
    key="real",
    label="Bonafide (human)",
    mechanism="Physical vocal tract",
    systems=("human speaker",),
    tell="Cycle-to-cycle jitter and shimmer, audible breathing, drifting speaker embedding",
    narrowband_safe=True,
)

FAMILIES: tuple[Family, ...] = (
    REAL,
    Family(
        idx=1,
        key="hifigan",
        label="HiFi-GAN family",
        mechanism="Transposed-convolution upsampling, mel to waveform",
        systems=("XTTS-v2", "VITS", "Piper", "StyleTTS 2", "CosyVoice 2", "RVC", "so-vits-svc"),
        tell="Checkerboard periodicity at the upsampling factors (8/8/2/2); band-limited top end",
    ),
    Family(
        idx=2,
        key="bigvgan",
        label="BigVGAN family",
        mechanism="Anti-aliased upsampling with snake activation",
        systems=("BigVGAN", "BigVSAN", "UnivNet", "Avocodo"),
        tell="Weak checkerboard, but harmonics above 4 kHz are unnaturally regular",
    ),
    Family(
        idx=3,
        key="vocos",
        label="Vocos / iSTFT head",
        mechanism="Predicts STFT coefficients, inverse STFT to waveform",
        systems=("F5-TTS", "IndicF5", "E2-TTS", "Vocos"),
        tell="No transposed-conv artifact at all; phase discontinuity at the STFT hop boundary",
        narrowband_safe=True,
    ),
    Family(
        idx=4,
        key="encodec",
        label="EnCodec RVQ",
        mechanism="Residual vector quantisation, discrete tokens to decoder",
        systems=("Bark", "VALL-E", "MusicGen", "AudioGen"),
        tell="Quantisation noise shaped by the codebook; 75 Hz frame periodicity at 24 kHz (13.3 ms)",
    ),
    Family(
        idx=5,
        key="modern_codec",
        label="DAC / SNAC / Mimi / Firefly",
        mechanism="Newer neural audio codecs, multi-scale or low frame rate",
        systems=("Parler-TTS", "Orpheus", "Fish-Speech", "Moshi", "CosyVoice 2 tokens"),
        tell="Same class as EnCodec but different frame rates; SNAC is multi-scale so several periodicities",
    ),
    Family(
        idx=6,
        key="legacy_ar",
        label="Legacy autoregressive",
        mechanism="Sample-level autoregression or classical signal reconstruction",
        systems=("WaveNet", "WaveRNN", "LPCNet", "Griffin-Lim", "STRAIGHT"),
        tell="No upsampling artifact; buzzy excitation and over-smoothed spectral envelope",
        narrowband_safe=True,
    ),
)

NUM_FAMILIES: int = len(FAMILIES)
"""7 classes: bonafide plus six synthetic decoder families."""

BY_KEY: dict[str, Family] = {f.key: f for f in FAMILIES}
BY_IDX: dict[int, Family] = {f.idx: f for f in FAMILIES}

ID2LABEL: dict[int, str] = {f.idx: f.key for f in FAMILIES}
LABEL2ID: dict[str, int] = {f.key: f.idx for f in FAMILIES}

BINARY_ID2LABEL: dict[int, str] = {0: "real", 1: "fake"}
BINARY_LABEL2ID: dict[str, int] = {"real": 0, "fake": 1}
"""Matches the convention of Gustking/wav2vec2-large-xlsr-deepfake-audio-classification,
which publishes id2label = {0: real, 1: fake}.  Do not silently invert this."""


def family_to_binary(family_idx: int) -> int:
    """Map a 7-way family index to the binary real/fake label.

    Args:
        family_idx: Index into ``FAMILIES``.

    Returns:
        0 for bonafide, 1 for any synthetic family.

    Raises:
        KeyError: If ``family_idx`` is not a known family.
    """
    if family_idx not in BY_IDX:
        raise KeyError(f"Unknown family index {family_idx}. Valid: {sorted(BY_IDX)}")
    return 0 if family_idx == REAL.idx else 1


def describe(family_idx: int) -> dict[str, object]:
    """Return a JSON-serialisable description of a family, for the API and UI."""
    f = BY_IDX[family_idx]
    return {
        "key": f.key,
        "label": f.label,
        "mechanism": f.mechanism,
        "known_systems": list(f.systems),
        "acoustic_tell": f.tell,
        "survives_narrowband": f.narrowband_safe,
    }
