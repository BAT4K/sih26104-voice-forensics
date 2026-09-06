"""Risk fusion: turn several signals into one number and a reason.

The problem statement asks for a risk score with "contextual enrichment using
metadata such as call origin, known contact information, transaction context,
and historical fraud indicators".  This is that engine.

Why fusion in log-odds rather than a bigger network
---------------------------------------------------
The Spoofing-Aware Speaker Verification challenge settled this.  Speaker
verification alone scored 23.83 percent SASV-EER; fused with a countermeasure
the best entry reached 0.13 percent.  More usefully for a team with one GPU, a
*probabilistic scoring rule* - not a larger model - took the baseline from
19.31 percent to 1.53 percent (Zhang, Zhu and Duan, Odyssey 2022), essentially
matching twelve-model ensembles.

So the fusion here is deliberately a calibrated weighted sum in log-odds space.
It is interpretable, every term can be traced to a reason string, and the
weights can be refitted on a few hundred labelled calls without touching the
detector.

Asymmetric cost
---------------
A missed clone on a bank call costs far more than a deferral.  Every published
stopping rule is symmetric, which is wrong for this deployment.  ``cost_ratio``
below makes the asymmetry explicit and auditable rather than hidden in a
threshold.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

GREEN_MAX = 30.0
AMBER_MAX = 70.0

DSP_MIN_Z = 2.0
"""Deviation below which an acoustic measure contributes nothing."""
DSP_PER_FEATURE_MAX = 0.45
"""Ceiling on one acoustic measure, in log-odds."""
DSP_TOTAL_MAX = 1.2
"""Ceiling on all acoustic measures combined, in log-odds.

Deliberately in the same range as the context multipliers below: an acoustic
measure should tip a borderline call, never manufacture a verdict.  See the
note in :func:`fuse` for why this is capped rather than merely rescaled.
"""

ACTIONS: dict[str, str] = {
    "green": "No action. Proceed normally.",
    "amber": "Verify by a second channel before acting on any instruction from this call.",
    "red": "Do not act on this call. Call back on a number already on file before approving "
           "any transfer or disclosing anything.",
}


@dataclass
class CallContext:
    """Metadata about the call. Every field is optional and defaults to neutral."""

    caller_number: str | None = None
    number_is_known: bool | None = None
    """True if the number is already associated with the claimed identity."""
    first_contact: bool = False
    """No prior history with this number."""
    claimed_identity: str | None = None
    transaction_value_inr: float | None = None
    is_privileged_action: bool = False
    """Fund transfer, credential reset, access grant."""
    outside_business_hours: bool = False
    prior_fraud_flag: bool = False
    """This number or identity has been flagged before."""
    channel: str = "unknown"

    def multipliers(self) -> list[tuple[str, float, str]]:
        """Context terms as (name, log-odds delta, human reason).

        Values are deliberately modest. Context should tip a borderline call,
        never manufacture a verdict on its own - a real executive calling from
        a new phone at 9pm is common, and flagging them on metadata alone is
        how a fraud system loses its users.
        """
        out: list[tuple[str, float, str]] = []
        if self.number_is_known is False:
            out.append(("unknown_number", 0.35,
                        "Calling number is not on file for the claimed identity."))
        if self.first_contact:
            out.append(("first_contact", 0.20, "No prior contact history with this number."))
        if self.prior_fraud_flag:
            out.append(("prior_fraud", 0.90, "This number or identity was previously flagged."))
        if self.is_privileged_action:
            out.append(("privileged_action", 0.30,
                        "Call concerns a privileged action, so the cost of a miss is high."))
        if self.transaction_value_inr and self.transaction_value_inr >= 1_000_000:
            out.append(("high_value", 0.40,
                        f"Transaction value is INR {self.transaction_value_inr:,.0f}."))
        if self.outside_business_hours:
            out.append(("off_hours", 0.15, "Call is outside normal business hours."))
        if self.number_is_known is True and not self.prior_fraud_flag:
            out.append(("known_number", -0.30,
                        "Calling number is already on file for this identity."))
        return out


@dataclass
class RiskAssessment:
    """The final, explainable verdict."""

    risk_score: float
    band: str
    verdict: str
    recommended_action: str
    contributions: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    speaker_decision: str | None = None
    cost_ratio: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def band_for(risk: float) -> str:
    """Traffic-light band for a 0-100 risk score."""
    if risk < GREEN_MAX:
        return "green"
    return "amber" if risk < AMBER_MAX else "red"


def fuse(
    spoof_probability: float,
    speaker_result: "object | None" = None,
    context: CallContext | None = None,
    novelty_is_unknown: bool = False,
    watermark: str = "unavailable",
    cost_ratio: float = 10.0,
    speaker_weight: float = 1.0,
    evidence: list | None = None,
) -> RiskAssessment:
    """Combine every available signal into one score and a set of reasons.

    Args:
        spoof_probability: Tier 1 output in [0, 1].
        speaker_result: A :class:`src.speaker.SpeakerResult`, or None when the
            caller is not enrolled.
        context: Call metadata.
        novelty_is_unknown: Tier 3 says the audio is far from every known family.
        watermark: Tier 0 output.
        cost_ratio: How much worse a missed clone is than a false alarm. 10
            means one miss costs as much as ten unnecessary call-backs; it
            shifts the decision boundary, and it is stated in the output so an
            auditor can see what was assumed.
        speaker_weight: Weight on the speaker term. Drop it below 1 while the
            enrollment store is small and thresholds are uncalibrated.

    Returns:
        A :class:`RiskAssessment` in which every contribution is named.
    """
    ctx = context or CallContext()
    contributions: dict[str, float] = {}
    reasons: list[str] = []

    z = _logit(spoof_probability)
    contributions["synthesis_detector"] = round(z, 4)
    if spoof_probability >= 0.7:
        reasons.append(f"Synthesis detector: {spoof_probability * 100:.0f} percent likelihood this audio was machine-generated.")
    elif spoof_probability <= 0.3:
        reasons.append(f"Synthesis detector: no strong evidence of machine generation ({spoof_probability * 100:.0f} percent).")

    speaker_decision = None
    if speaker_result is not None:
        speaker_decision = getattr(speaker_result, "decision", None)
        sim = float(getattr(speaker_result, "similarity", 0.0))
        delta = 0.0
        if speaker_decision == "reject":
            delta = 1.6
            reasons.append(
                f"Cross-session check: this voice does not match the enrolled profile for "
                f"{getattr(speaker_result, 'display_name', 'the claimed identity')} "
                f"(similarity {sim:.2f})."
            )
        elif speaker_decision == "accept":
            delta = -1.2
            reasons.append(
                f"Cross-session check: voice matches the enrolled profile (similarity {sim:.2f})."
            )
        elif speaker_decision == "inconclusive":
            delta = 0.35
            reasons.append("Cross-session check: inconclusive. Similarity sits between thresholds.")
        elif speaker_decision in ("not_enrolled", "insufficient_enrollment"):
            delta = 0.0
            reasons.append(
                "Cross-session check unavailable: no historical genuine samples for this identity. "
                "This is not evidence either way."
            )
        contributions["speaker_verification"] = round(delta * speaker_weight, 4)
        z += delta * speaker_weight

        if getattr(speaker_result, "change_points", None):
            pts = speaker_result.change_points
            contributions["mid_call_voice_change"] = 1.1
            z += 1.1
            reasons.append(
                f"The voice changes {len(pts)} time(s) during the call "
                f"(first at {pts[0]:.1f}s). Possible hand-off between a live caller and a clone."
            )
        if float(getattr(speaker_result, "consistency_index", 0.5)) > 0.97:
            contributions["unnatural_stability"] = 0.5
            z += 0.5
            reasons.append(
                "The voiceprint barely moves across the call. Genuine speech drifts; a clone "
                "rendered from one fixed speaker vector does not."
            )

    if novelty_is_unknown:
        contributions["unknown_synthesis_family"] = 0.6
        z += 0.6
        reasons.append(
            "This audio does not resemble any synthesis family the model knows. Treat the "
            "family attribution as unreliable and the call as suspicious."
        )

    for name, delta, reason in ctx.multipliers():
        contributions[f"context.{name}"] = delta
        z += delta
        reasons.append(reason)

    if evidence:
        # Bounded, and capped in total.
        #
        # v1 used `delta = 2.0 + abs(z) * 0.5` with z clamped at 20, so a single
        # measure could contribute 12 log-odds and three could contribute 36.
        # The sigmoid saturates past about 16, which made every other signal -
        # Tier 1 included - arithmetically unable to change the verdict.  Every
        # file in the demo set scored 100/100, both genuine ones included.
        #
        # Two things are wrong with an unbounded term here.  The magnitude, and
        # the independence assumption: these measures are computed from the same
        # spectrum the encoder already consumes, so at full weight the fusion
        # counts one piece of evidence twice.  Hence a hard ceiling well below
        # the Tier 1 term's range.
        dsp_total = 0.0
        for ev in evidence:
            z_score = getattr(ev, "z_score", 0.0)
            if abs(z_score) < DSP_MIN_Z:
                continue
            delta = math.copysign(
                DSP_PER_FEATURE_MAX * math.tanh((abs(z_score) - DSP_MIN_Z) / 3.0), z_score
            )
            contributions[f"dsp_anomaly.{ev.name}"] = round(delta, 4)
            dsp_total += delta
            reasons.append(
                f"Acoustic measure out of range: {ev.name} sits {abs(z_score):.1f} "
                f"standard deviations from the human baseline. {ev.reading}"
            )
        dsp_total = max(min(dsp_total, DSP_TOTAL_MAX), -DSP_TOTAL_MAX)
        if dsp_total:
            contributions["dsp_total_after_cap"] = round(dsp_total, 4)
        z += dsp_total

    # Asymmetric cost: shift the boundary rather than the score, so the score
    # stays a probability and the assumption stays visible.
    bias = math.log(max(cost_ratio, 1e-6)) * 0.15
    contributions["asymmetric_cost_bias"] = round(bias, 4)
    z += bias

    risk = 100.0 * _sigmoid(z)

    if watermark.startswith("detected:"):
        risk = 100.0
        contributions["provenance_watermark"] = float("inf")
        reasons.insert(0, f"A provenance watermark was found ({watermark.split(':', 1)[1]}). "
                          f"This audio is machine-generated; no further inference is needed.")

    b = band_for(risk)
    verdict = ("ai_clone" if risk >= AMBER_MAX
               else "unclear" if risk >= GREEN_MAX else "genuine")

    if speaker_decision == "reject":
        risk = 100.0
        b = "red"
        verdict = "human_mimicry"
        contributions["speaker_mismatch_fatal"] = float("inf")
        # Ensure the reason is at the top
        reasons.insert(0, "CRITICAL: Caller does not match the enrolled voiceprint. Impersonator detected.")
    return RiskAssessment(
        risk_score=round(risk, 1), band=b, verdict=verdict, recommended_action=ACTIONS[b],
        contributions=contributions, reasons=reasons,
        speaker_decision=speaker_decision, cost_ratio=cost_ratio,
    )
