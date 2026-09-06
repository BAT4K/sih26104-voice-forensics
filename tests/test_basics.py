"""Tests that catch the bugs v0.3 actually had.

Run with: pytest tests/ -v
None of these need a GPU, a checkpoint, or the data folder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import audio_prep, families, metrics
from src.telephony_degrader import Condition, apply_packet_loss, available_codecs, degrade


# --------------------------------------------------------------------- labels

def test_label_convention_matches_pretrained_checkpoint():
    """real=0, fake=1.

    v0.3 forced {0: fake, 1: real} onto a checkpoint that publishes
    {0: real, 1: fake}. Because num_labels matched, the head loaded instead of
    reinitialising, so every training example contradicted the pretrained
    weights. This test is the guard.
    """
    assert families.BINARY_LABEL2ID["real"] == 0
    assert families.BINARY_LABEL2ID["fake"] == 1
    assert families.BINARY_ID2LABEL[0] == "real"


def test_family_taxonomy_is_complete_and_consistent():
    assert families.NUM_FAMILIES == 7, "bonafide plus six synthetic families"
    assert families.REAL.idx == 0
    assert [f.idx for f in families.FAMILIES] == list(range(7)), "indices must be contiguous from 0"
    assert len({f.key for f in families.FAMILIES}) == 7, "family keys must be unique"


def test_family_to_binary_mapping():
    assert families.family_to_binary(0) == 0
    for idx in range(1, families.NUM_FAMILIES):
        assert families.family_to_binary(idx) == 1
    with pytest.raises(KeyError):
        families.family_to_binary(99)


# ---------------------------------------------------------------- annotations

def test_every_module_imports_on_this_python():
    """v0.3 crashed at import on Python 3.9-3.13 with two NameErrors.

    `from __future__ import annotations` at the top of every module makes that
    class of bug impossible.
    """
    import importlib

    for mod in ("src.families", "src.metrics", "src.audio_prep",
                "src.telephony_degrader", "src.model_loader", "src.inference"):
        importlib.import_module(mod)


def test_model_layer_does_not_import_streamlit():
    """The model layer must be usable from an API worker and a unit test.

    v0.3 decorated the loader with @st.cache_resource, so everything had to
    drag in the Streamlit runtime to load a model.
    """
    import ast
    import pathlib

    for name in ("model_loader.py", "telephony_degrader.py", "audio_prep.py",
                 "metrics.py", "families.py", "inference.py"):
        src = (pathlib.Path(__file__).parent.parent / "src" / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert "streamlit" not in names, f"{name} imports streamlit"
            assert "fastapi" not in names, f"{name} imports fastapi"


# ---------------------------------------------------------------- degradation

def _tone(sr=16000, secs=3.0):
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 1500 * t)
            + 0.25 * np.sin(2 * np.pi * 5200 * t)).astype(np.float32)


def test_narrowband_removes_energy_above_4khz():
    x = _tone()
    y, sr = degrade(x, 16000, Condition(codec="g711_ulaw"))
    assert sr == 8000

    def hf_share(sig, fs):
        p = np.abs(np.fft.rfft(sig)) ** 2
        f = np.fft.rfftfreq(len(sig), 1 / fs)
        return p[f >= 4000].sum() / (p.sum() + 1e-12)

    assert hf_share(x, 16000) > 0.1, "test signal must have high-frequency energy to begin with"
    assert hf_share(y, sr) < 0.01, "narrowband must discard content above 4 kHz"


def test_degradation_does_not_touch_the_global_random_stream():
    """v0.3 called np.random.seed(42) inside the degrader.

    Every call silently reset the global stream, which would destroy dataloader
    shuffling and any stochastic augmentation.
    """
    np.random.seed(1)
    _ = np.random.rand()
    degrade(_tone(), 16000, Condition(codec="g711_ulaw"))
    after = np.random.rand()

    np.random.seed(1)
    _ = np.random.rand()
    expected = np.random.rand()
    assert after == pytest.approx(expected), "the degrader must not seed the global RNG"


def test_packet_loss_is_reproducible_from_an_explicit_generator():
    x = _tone()
    a = apply_packet_loss(x, 16000, 0.2, plc=True, rng=np.random.default_rng(7))
    b = apply_packet_loss(x, 16000, 0.2, plc=True, rng=np.random.default_rng(7))
    assert np.allclose(a, b)
    assert not np.allclose(a, x), "20 percent loss must change the signal"


def test_available_codecs_reports_rather_than_hides():
    av = available_codecs()
    assert av["g711_ulaw"] and av["g711_alaw"] and av["none"], "pure-NumPy codecs always work"
    assert set(av) >= {"amr_nb", "gsm_fr", "opus_nb"}


# ------------------------------------------------------------------- windowing

def test_window_is_the_aasist_crop():
    assert audio_prep.WINDOW_SAMPLES == 64_600
    assert abs(audio_prep.WINDOW_SAMPLES / audio_prep.TARGET_SR - 4.0375) < 1e-3


def test_short_clips_are_not_tile_padded_when_asked():
    """Tiling a short prefix creates an artificial periodicity a confidence
    rule can exploit; a published streaming study traced ~82 percent of its
    headline gain to exactly that artifact."""
    x = np.ones(1000, dtype=np.float32)
    tiled = audio_prep.fit_window(x, 5000, tile_pad=True)
    zeroed = audio_prep.fit_window(x, 5000, tile_pad=False)
    assert np.count_nonzero(tiled) == 5000
    assert np.count_nonzero(zeroed) == 1000


def test_silence_trimming_removes_leading_and_trailing_quiet():
    speech = _tone(secs=1.0)
    padded = np.concatenate([np.zeros(16000, np.float32), speech, np.zeros(16000, np.float32)])
    trimmed = audio_prep.trim_silence(padded)
    assert len(trimmed) < len(padded)
    assert len(trimmed) >= len(speech) * 0.8


# --------------------------------------------------------------------- metrics

def test_eer_is_zero_for_a_perfect_separator():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    e, _ = metrics.eer(scores, labels)
    assert e == pytest.approx(0.0, abs=1e-9)


def test_eer_is_half_for_a_coin_flip():
    rng = np.random.default_rng(0)
    scores = rng.random(4000)
    labels = rng.integers(0, 2, 4000)
    e, _ = metrics.eer(scores, labels)
    assert 0.42 < e < 0.58


def test_alert_precision_at_a_realistic_base_rate():
    """At the reported 0.17 percent banking base rate, 1 percent FPR and
    90 percent TPR means most alerts are innocent customers. Nobody publishes
    this number; it decides deployability."""
    p = metrics.alert_precision(fpr=0.01, tpr=0.90, base_rate=0.0017)
    assert p < 0.20, "most alerts should be false at this base rate"


def test_summarise_reports_operating_points_not_just_eer():
    rng = np.random.default_rng(3)
    labels = np.r_[np.zeros(500), np.ones(500)].astype(int)
    scores = np.r_[rng.normal(0.3, 0.15, 500), rng.normal(0.7, 0.15, 500)]
    out = metrics.summarise(scores, labels)
    for key in ("eer", "auc", "fpr_at_tpr90", "fpr_at_tpr95",
                "fpr_at_tpr99", "alert_precision_at_tpr90"):
        assert key in out
    assert 0.0 <= out["eer"] <= 0.5


def test_macro_f1_perfect_and_chance():
    assert metrics.macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2]), 3) == pytest.approx(1.0)
    assert metrics.macro_f1(np.array([0, 0, 0]), np.array([0, 1, 2]), 3) < 0.5
