"""Tests for cross-session speaker verification and risk fusion.

None of these download ECAPA. The embedding extractor is the only part that
needs SpeechBrain, and it is a pretrained forward pass with nothing to test
that SpeechBrain does not already test. Everything else - the store, the
similarity maths, change-point detection and the fusion rule - is ours, and is
covered here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import risk, speaker


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def fake_voiceprints(seed: int, n: int = 8, drift: float = 0.05) -> np.ndarray:
    """Windows from one speaker: a base identity plus small natural drift."""
    rng = np.random.default_rng(seed)
    base = unit(rng.normal(size=speaker.EMBED_DIM))
    return np.stack([unit(base + drift * rng.normal(size=speaker.EMBED_DIM)) for _ in range(n)])


# ------------------------------------------------------------------- store

def test_store_holds_no_audio(tmp_path):
    """The privacy guarantee, asserted rather than promised."""
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    st.enroll("cfo_001", fake_voiceprints(1), display_name="A. Rao")

    text = (tmp_path / "spk" / "index.json").read_text()
    assert "cfo_001" in text and "A. Rao" in text
    for banned in ("audio", ".wav", "waveform", "pcm"):
        assert banned not in text.lower(), f"store must not reference {banned}"
    assert not list((tmp_path / "spk").glob("*.wav"))


def test_enrollment_roundtrips_through_disk(tmp_path):
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    for i in range(3):
        st.enroll("cfo_001", fake_voiceprints(1 + i), display_name="A. Rao")

    reopened = speaker.EnrollmentStore(str(tmp_path / "spk"))
    rec = reopened.get("cfo_001")
    assert rec is not None
    assert len(rec.sessions) == 3
    assert len(rec.centroid()) == speaker.EMBED_DIM
    assert reopened.list_speakers()[0]["ready"] is True


def test_delete_is_immediate(tmp_path):
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    st.enroll("x", fake_voiceprints(1))
    assert st.delete("x") is True
    assert st.get("x") is None
    assert st.delete("x") is False


def test_wrong_embedding_dimension_is_rejected(tmp_path):
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    with pytest.raises(ValueError, match="192"):
        st.enroll("x", np.zeros((4, 64)))


# -------------------------------------------------------------- similarity

def test_same_speaker_scores_higher_than_a_different_one():
    a1, a2 = fake_voiceprints(11).mean(axis=0), fake_voiceprints(11, seed_shift := 11).mean(axis=0)
    b = fake_voiceprints(99).mean(axis=0)
    assert speaker.cosine(a1, a2) > speaker.cosine(a1, b)
    assert speaker.cosine(a1, b) < 0.5


def test_self_similarity_gives_a_personal_baseline(tmp_path):
    """A naturally variable speaker must not be penalised for being variable."""
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    rng = np.random.default_rng(5)
    base = unit(rng.normal(size=speaker.EMBED_DIM))
    for i in range(4):
        wins = np.stack([unit(base + 0.15 * rng.normal(size=speaker.EMBED_DIM)) for _ in range(6)])
        st.enroll("variable_person", wins)
    mean, std = st.get("variable_person").self_similarity()
    assert 0.0 < mean <= 1.0
    assert std >= 0.0


# ------------------------------------------------------- temporal behaviour

def test_synthetic_speech_is_MORE_stable_than_real():
    """The sign here is counter-intuitive and published.

    A clone is rendered from one fixed speaker vector so its embedding barely
    moves; a real speaker's drifts with effort and emotion. Assuming the
    opposite inverts the decision rule.
    """
    clone = fake_voiceprints(3, n=10, drift=0.01)
    human = fake_voiceprints(3, n=10, drift=0.25)
    assert speaker.temporal_consistency(clone) > speaker.temporal_consistency(human)


def test_temporal_consistency_abstains_on_short_calls():
    assert speaker.temporal_consistency(fake_voiceprints(1, n=2)) == 0.5


def test_change_point_detects_a_mid_call_voice_swap():
    """The fraud pattern no published work covers: a live social engineer
    hands the call to a clone for the sentence that authorises the transfer."""
    person_a = fake_voiceprints(21, n=8, drift=0.03)
    person_b = fake_voiceprints(77, n=8, drift=0.03)
    swapped = np.concatenate([person_a, person_b])
    assert speaker.detect_change_points(swapped, hop_seconds=1.5), "a swap must be detected"


def test_change_point_does_not_fire_on_one_steady_speaker():
    assert speaker.detect_change_points(fake_voiceprints(21, n=16, drift=0.04)) == []


# ------------------------------------------------------------- verification

def test_unknown_speaker_is_reported_not_rejected(tmp_path):
    """"We have no baseline" and "this person failed" are different answers.

    Conflating them would flag every caller who was never enrolled.
    """
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    res = speaker.verify(np.zeros(16000, dtype=np.float32), 16000, "nobody", store=st)
    assert res.decision == "not_enrolled"
    assert res.decision != "reject"
    assert "no historical genuine samples" in res.reasons[0].lower()


def test_thin_enrollment_is_reported(tmp_path):
    st = speaker.EnrollmentStore(str(tmp_path / "spk"))
    st.enroll("thin", fake_voiceprints(2))
    res = speaker.verify(np.zeros(16000, dtype=np.float32), 16000, "thin", store=st)
    assert res.decision == "insufficient_enrollment"


# ------------------------------------------------------------------ fusion

def test_speaker_rejection_raises_risk_above_the_detector_alone():
    class Rejected:
        decision, similarity, display_name = "reject", 0.21, "A. Rao"
        change_points, consistency_index = [], 0.8

    alone = risk.fuse(0.5)
    fused = risk.fuse(0.5, speaker_result=Rejected())
    assert fused.risk_score > alone.risk_score
    assert fused.speaker_decision == "reject"


def test_speaker_acceptance_lowers_risk():
    class Accepted:
        decision, similarity, display_name = "accept", 0.81, "A. Rao"
        change_points, consistency_index = [], 0.8

    assert risk.fuse(0.5, speaker_result=Accepted()).risk_score < risk.fuse(0.5).risk_score


def test_missing_enrollment_is_neutral_not_incriminating():
    """Not being enrolled must not push a caller toward 'fake'."""
    class NotEnrolled:
        decision, similarity, display_name = "not_enrolled", 0.0, "unknown"
        change_points, consistency_index = [], 0.5

    a = risk.fuse(0.4)
    b = risk.fuse(0.4, speaker_result=NotEnrolled())
    assert b.risk_score == pytest.approx(a.risk_score, abs=0.2)


def test_mid_call_voice_change_is_escalated():
    class Swapped:
        decision, similarity, display_name = "accept", 0.7, "A. Rao"
        change_points, consistency_index = [12.0], 0.8

    class Steady:
        decision, similarity, display_name = "accept", 0.7, "A. Rao"
        change_points, consistency_index = [], 0.8

    assert risk.fuse(0.3, speaker_result=Swapped()).risk_score > risk.fuse(0.3, speaker_result=Steady()).risk_score


def test_context_tips_but_does_not_manufacture_a_verdict():
    """A real executive on a new phone at 9pm is common. Metadata alone must
    not produce a red verdict."""
    ctx = risk.CallContext(
        number_is_known=False, first_contact=True, outside_business_hours=True,
        is_privileged_action=True, transaction_value_inr=5_000_000,
    )
    clean_voice = risk.fuse(0.02, context=ctx)
    assert clean_voice.band != "red", "context alone must not force a red verdict"
    assert clean_voice.risk_score > risk.fuse(0.02).risk_score


def test_watermark_short_circuits_to_certainty():
    a = risk.fuse(0.05, watermark="detected:SynthID")
    assert a.risk_score == 100.0 and a.band == "red"


def test_every_contribution_is_named_and_auditable():
    ctx = risk.CallContext(number_is_known=False, prior_fraud_flag=True)
    out = risk.fuse(0.8, context=ctx, novelty_is_unknown=True)
    assert "synthesis_detector" in out.contributions
    assert "context.unknown_number" in out.contributions
    assert "context.prior_fraud" in out.contributions
    assert "unknown_synthesis_family" in out.contributions
    assert "asymmetric_cost_bias" in out.contributions
    assert len(out.reasons) >= 4
    assert out.cost_ratio == 10.0, "the asymmetry must be stated, not hidden in a threshold"


def test_bands_and_actions_line_up():
    assert risk.band_for(10) == "green"
    assert risk.band_for(50) == "amber"
    assert risk.band_for(90) == "red"
    assert "call back" in risk.ACTIONS["red"].lower()
