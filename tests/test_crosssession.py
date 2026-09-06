"""Tests for the cross-session verification protocol.

This is the part of the system the problem statement names directly:

    "Cross-session consistency checks comparing ongoing call features against
     historical genuine samples (where available) to detect anomalies in
     speaker identity."

The protocol code is what turns audio into a number. If it has a bug the number
is wrong and nothing downstream will notice, so it is tested against mock
embeddings whose ground truth is known by construction.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import metrics
from tools.eval_crosssession import load_sessions, run_trials


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def speaker_embeddings(seed: int, n_sessions: int, spread: float) -> list[np.ndarray]:
    """Sessions from one speaker: an identity vector plus session variation."""
    rng = np.random.default_rng(seed)
    identity = unit(rng.normal(size=64))
    return [unit(identity + spread * rng.normal(size=64)) for _ in range(n_sessions)]


# ------------------------------------------------------------ session grouping

def _manifest(tmp_path, records):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def test_two_clips_from_one_recording_are_one_session(tmp_path):
    """The mistake that inflates every cross-session number.

    Two clips cut from the same recording share a channel, a room and a
    microphone. Counting them as two sessions measures how well the system
    recognises a recording, not a person.
    """
    recs = [
        {"path": "a.wav", "label": "real", "speaker": "s1", "source_url": "vid1"},
        {"path": "b.wav", "label": "real", "speaker": "s1", "source_url": "vid1"},
        {"path": "c.wav", "label": "real", "speaker": "s1", "source_url": "vid2"},
    ]
    sessions = load_sessions(_manifest(tmp_path, recs))
    assert "s1" in sessions
    assert len(sessions["s1"]) == 2, "two clips from vid1 must collapse to one session"


def test_single_session_speakers_are_excluded(tmp_path):
    """No history means no cross-session check. That is the 'where available'
    clause in the statement, and it must not silently become a trial."""
    recs = [
        {"path": "a.wav", "label": "real", "speaker": "once", "source_url": "v1"},
        {"path": "b.wav", "label": "real", "speaker": "twice", "source_url": "v1"},
        {"path": "c.wav", "label": "real", "speaker": "twice", "source_url": "v2"},
    ]
    sessions = load_sessions(_manifest(tmp_path, recs))
    assert "once" not in sessions
    assert "twice" in sessions


def test_spoofed_audio_is_excluded_from_enrollment(tmp_path):
    """Historical *genuine* samples. Enrolling a fake would poison the profile."""
    recs = [
        {"path": "a.wav", "label": "real", "speaker": "s", "source_url": "v1"},
        {"path": "b.wav", "label": "real", "speaker": "s", "source_url": "v2"},
        {"path": "c.wav", "label": "fake", "speaker": "s", "source_url": "v3"},
    ]
    assert len(load_sessions(_manifest(tmp_path, recs))["s"]) == 2


# ------------------------------------------------------------------- trials

def test_trial_counts_are_leave_one_session_out():
    embs = {f"s{i}": speaker_embeddings(i, 3, 0.15) for i in range(4)}
    scores, labels = run_trials(embs)
    # 4 speakers x 3 held-out sessions = 12 target trials
    assert int(labels.sum()) == 12
    # each held-out session also scores against the other 3 speakers
    assert int((1 - labels).sum()) == 12 * 3
    assert len(scores) == len(labels)


def test_held_out_session_is_never_in_its_own_profile():
    """If the live call is inside the profile it is compared against, target
    scores are trivially perfect and the EER is meaningless."""
    embs = {"a": speaker_embeddings(1, 4, 0.4), "b": speaker_embeddings(2, 4, 0.4)}
    scores, labels = run_trials(embs)
    assert np.max(scores[labels == 1]) < 0.9999, "a self-match leaked into the profile"


def test_tight_speakers_separate_and_loose_ones_do_not():
    """Sanity of the whole protocol: the EER must track how distinguishable
    the speakers actually are."""
    tight = {f"s{i}": speaker_embeddings(i, 3, 0.05) for i in range(6)}
    loose = {f"s{i}": speaker_embeddings(i, 3, 1.60) for i in range(6)}
    s_t, l_t = run_trials(tight)
    s_l, l_l = run_trials(loose)
    eer_tight, _ = metrics.eer(-s_t, 1 - l_t)   # score high = same speaker
    eer_loose, _ = metrics.eer(-s_l, 1 - l_l)
    assert eer_tight < 0.05, f"consistent speakers should separate cleanly, got {eer_tight:.3f}"
    assert eer_loose > eer_tight, "highly variable speakers must be harder"


def test_protocol_needs_at_least_two_speakers():
    """One speaker gives no non-target trials, so there is no EER to compute."""
    scores, labels = run_trials({"only": speaker_embeddings(1, 3, 0.1)})
    assert int((1 - labels).sum()) == 0


def test_degradation_can_only_make_verification_harder():
    """Adding channel noise to embeddings must not improve separability.

    Guards against a sign error in the trial or scoring code, which would show
    up as codecs apparently *helping*.
    """
    rng = np.random.default_rng(0)
    clean = {f"s{i}": speaker_embeddings(i, 3, 0.10) for i in range(6)}
    degraded = {k: [unit(v + 0.5 * rng.normal(size=64)) for v in vs] for k, vs in clean.items()}
    s_c, l_c = run_trials(clean)
    s_d, l_d = run_trials(degraded)
    eer_c, _ = metrics.eer(-s_c, 1 - l_c)
    eer_d, _ = metrics.eer(-s_d, 1 - l_d)
    assert eer_d >= eer_c, "degradation must not improve the EER"


# --------------------------------------------------------------- calibration

def test_shipped_thresholds_are_measured_not_defaults():
    """The shipped values must come from the calibration file, not the
    ECAPA library defaults that were never fitted to anything."""
    from src import speaker

    assert "UNCALIBRATED" not in speaker.CALIBRATION_SOURCE
    assert "measured" in speaker.CALIBRATION_SOURCE
    assert speaker.DEFAULT_ACCEPT != 0.55, "0.55 is the unfitted library default"
    assert 0.2 < speaker.DEFAULT_REJECT < speaker.DEFAULT_ACCEPT < 0.9


def test_calibration_file_records_its_provenance():
    """A threshold with no protocol behind it is not a threshold."""
    import json
    import pathlib

    p = pathlib.Path(__file__).parent.parent / "calibration" / "speaker_thresholds.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("corpus", "speakers", "sessions", "encoder", "protocol",
                "target_trials", "nontarget_trials"):
        assert key in data, f"calibration must record {key}"
    assert data["speakers"] >= 10
    assert "leave-one-session-out" in data["protocol"]
    for point in ("clean", "narrowband"):
        assert data["operating_points"][point]["accept"] > data["operating_points"][point]["reject"]


def test_narrowband_is_harder_than_clean():
    """Measured, not assumed: the phone channel must cost EER."""
    import json
    import pathlib

    p = pathlib.Path(__file__).parent.parent / "calibration" / "speaker_thresholds.json"
    ops = json.loads(p.read_text(encoding="utf-8"))["operating_points"]
    assert ops["narrowband"]["eer"] > ops["clean"]["eer"]
