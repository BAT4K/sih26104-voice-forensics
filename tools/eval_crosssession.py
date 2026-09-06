"""Cross-session consistency: the full evaluation protocol.

What the problem statement asks for
-----------------------------------
    "Cross-session consistency checks comparing ongoing call features against
     historical genuine samples (where available) to detect anomalies in
     speaker identity."

In a bank or a telecom that means: a customer has called before, those calls
are the historical genuine samples, and the live call is checked against them.
So this is speaker verification with session structure - and, crucially, over a
telephone channel.

Nothing here is trained.  ECAPA-TDNN ships pretrained; the only fitted quantity
is a decision threshold.  What this script does is *measure*: it runs the
verification protocol under every telephony condition and reports how far the
channel moves the numbers.

Why that measurement is the interesting part
--------------------------------------------
ECAPA was trained on wideband VoxCeleb.  Real Indian calls arrive at 8 kHz over
AMR-NB.  Nobody has published how much speaker verification degrades across
that gap, and the answer decides whether cross-session checking is deployable
on a phone line at all.

Protocol
--------
For every speaker with at least two sessions, one session is held out as the
"live call" and the rest form the enrolled profile.  Target trials compare the
held-out session to its own profile; non-target trials compare it to every
other speaker's profile.  Both sides of every trial go through the same
degradation, because degrading only one side would leak the answer.

    python tools/eval_crosssession.py --manifest data/train.jsonl --out results/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tools/<script>.py` from the project root as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import metrics
from src.telephony_degrader import Condition, available_codecs, missing_codecs

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("crosssession")

DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition(codec="none", name="clean-16k"),
    Condition(codec="g711_ulaw", name="g711-ulaw-8k"),
    Condition(codec="g726", bitrate_kbps=32.0, name="g726-32k"),
    Condition(codec="g726", bitrate_kbps=24.0, name="g726-24k"),
    Condition(codec="g726", bitrate_kbps=16.0, name="g726-16k"),
    Condition(codec="opus_nb", bitrate_kbps=12.0, name="opus-12k"),
    Condition(codec="g726", bitrate_kbps=16.0, packet_loss=0.05, name="g726-16k+5%loss"),
    Condition(codec="g726", bitrate_kbps=16.0, packet_loss=0.10, name="g726-16k+10%loss"),
    # Requires an ffmpeg built with libopencore-amrnb / libgsm. Skipped with a
    # warning when absent rather than silently dropped.
    Condition(codec="amr_nb", bitrate_kbps=12.2, name="amr-nb-12.2k"),
    Condition(codec="amr_nb", bitrate_kbps=4.75, name="amr-nb-4.75k"),
    Condition(codec="gsm_fr", name="gsm-fr"),
)


def load_sessions(manifest: Path, min_sessions: int = 2) -> dict[str, list[dict]]:
    """Group clips by speaker, keeping only speakers with enough sessions.

    A "session" is one source recording. Two clips cut from the same recording
    are the same session and must not be treated as independent - that is the
    mistake that makes cross-session numbers look far better than they are.
    """
    by_speaker: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        spk = rec.get("speaker") or rec.get("identity")
        if not spk or str(rec.get("label", "real")).lower() != "real":
            continue
        session = rec.get("source_url") or rec.get("context") or rec["path"]
        by_speaker[spk][session].append(rec)

    out: dict[str, list[dict]] = {}
    for spk, sessions in by_speaker.items():
        if len(sessions) >= min_sessions:
            out[spk] = [clips[0] for clips in sessions.values()]
    return out


def embed_all(sessions: dict[str, list[dict]], condition: Condition, encoder) -> dict[str, list[np.ndarray]]:
    """One embedding per session, under one transmission condition."""
    import soundfile as sf

    from src.telephony_degrader import degrade

    rng = np.random.default_rng(0)
    out: dict[str, list[np.ndarray]] = {}
    for spk, recs in sessions.items():
        vecs = []
        for rec in recs:
            try:
                x, sr = sf.read(rec["path"], dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if condition.codec != "none":
                    x, sr = degrade(x, sr, condition, rng=rng)
                vecs.append(encoder.embed(x, sr))
            except Exception as exc:
                log.debug("skipped %s: %s", rec.get("path"), exc)
        if len(vecs) >= 2:
            out[spk] = vecs
    return out


def run_trials(embeddings: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-session-out target and non-target trials.

    Returns:
        ``(scores, labels)`` with label 1 for a target trial.
    """
    from src.speaker import cosine

    scores, labels = [], []
    speakers = sorted(embeddings)
    for spk in speakers:
        vecs = embeddings[spk]
        for held in range(len(vecs)):
            profile = np.mean([v for i, v in enumerate(vecs) if i != held], axis=0)
            profile = profile / (np.linalg.norm(profile) + 1e-12)
            scores.append(cosine(vecs[held], profile))
            labels.append(1)
            for other in speakers:
                if other == spk:
                    continue
                other_profile = np.mean(embeddings[other], axis=0)
                other_profile = other_profile / (np.linalg.norm(other_profile) + 1e-12)
                scores.append(cosine(vecs[held], other_profile))
                labels.append(0)
    return np.array(scores), np.array(labels)


def evaluate(
    manifest: Path, out_dir: Path, conditions=DEFAULT_CONDITIONS,
    encoder=None, base_rate: float = 0.0017,
) -> dict:
    """Run the protocol under every condition and write the results table."""
    sessions = load_sessions(manifest)
    if len(sessions) < 2:
        raise SystemExit(
            f"need at least 2 speakers with 2+ distinct sessions; found {len(sessions)}. "
            f"Cross-session verification is not measurable without repeat callers."
        )
    n_sessions = sum(len(v) for v in sessions.values())
    log.info("%d speakers, %d sessions (%.1f per speaker)",
             len(sessions), n_sessions, n_sessions / len(sessions))

    if encoder is None:
        from src.speaker import SpeakerEncoder

        encoder = SpeakerEncoder.shared()

    if missing := missing_codecs():
        log.warning("NOT COVERED on this machine: %s", ", ".join(missing))

    have = available_codecs()
    rows, results = [], {}
    for cond in conditions:
        if not have.get(cond.codec, False):
            log.warning("skipping %s - codec unavailable here", cond.label())
            continue
        embs = embed_all(sessions, cond, encoder)
        if len(embs) < 2:
            log.warning("skipping %s - too few usable speakers", cond.label())
            continue
        scores, labels = run_trials(embs)
        eer, thr = metrics.eer(scores, labels)
        fpr90, t90 = metrics.fpr_at_tpr(scores, labels, 0.90)
        row = {
            "condition": cond.label(),
            "speakers": len(embs),
            "target_trials": int(labels.sum()),
            "nontarget_trials": int((1 - labels).sum()),
            "eer": round(float(eer), 5),
            "threshold": round(float(thr), 4),
            "auc": round(metrics.auc(scores, labels), 5),
            "fpr_at_tpr90": round(float(fpr90), 5),
            "threshold_at_tpr90": round(float(t90), 4),
            "alert_precision_at_tpr90": round(
                metrics.alert_precision(float(fpr90), 0.90, base_rate), 5),
        }
        rows.append(row)
        results[cond.label()] = row
        log.info("%-24s EER %6.2f%%   thr %.3f   FPR@TPR90 %6.2f%%",
                 cond.label(), 100 * eer, thr, 100 * fpr90)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crosssession_results.json").write_text(
        json.dumps({"speakers": len(sessions), "sessions": n_sessions,
                    "missing_codecs": missing_codecs(), "conditions": results},
                   indent=2), encoding="utf-8")
    if rows:
        import csv

        with open(out_dir / "crosssession_matrix.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        clean = results.get("clean-16k", {}).get("eer")
        worst = max(rows, key=lambda r: r["eer"])
        if clean is not None and clean > 0:
            log.info("")
            log.info("HEADLINE: clean %.2f%% EER -> %.2f%% under %s  (%.1fx worse)",
                     100 * clean, 100 * worst["eer"], worst["condition"],
                     worst["eer"] / max(clean, 1e-9))
            log.info("Speaker verification degrades this much on a phone line. "
                     "No published paper reports this for Indian telephony.")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-session speaker verification benchmark.")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", default=Path("results"), type=Path)
    ap.add_argument("--base-rate", type=float, default=0.0017)
    args = ap.parse_args()
    evaluate(args.manifest, args.out, base_rate=args.base_rate)


if __name__ == "__main__":
    main()
