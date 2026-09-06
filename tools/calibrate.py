"""Fit the speaker-verification thresholds on real data.

The accept and reject values shipped in ``src/speaker.py`` are ECAPA defaults.
They are **not measured**, and nobody should quote a similarity threshold they
have not fitted on their own audio through their own channel.

This builds every same-speaker and different-speaker pair from an enrollment
corpus, scores them, finds the equal-error threshold, and writes the result
back. Optionally it does the whole thing **through the telephony codec chain**,
which is the number that actually matters: ECAPA was trained on wideband
VoxCeleb, and nobody has published how much it degrades at 8 kHz over AMR-NB.

    python tools/calibrate.py --data data/enrollment
    python tools/calibrate.py --data data/enrollment --codec amr_nb --bitrate 4.75
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tools/<script>.py` from the project root as well as `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np

from src import metrics
from src.telephony_degrader import Condition, degrade

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("calibrate")


def collect_embeddings(root: Path, condition: Condition | None) -> dict[str, list[np.ndarray]]:
    """One embedding per clip, grouped by speaker."""
    import soundfile as sf

    from src.speaker import SpeakerEncoder

    enc = SpeakerEncoder.shared()
    rng = np.random.default_rng(0)
    out: dict[str, list[np.ndarray]] = {}
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise SystemExit(f"no .wav files under {root}")

    for i, f in enumerate(files, 1):
        speaker = f.parent.name
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if condition is not None:
            x, sr = degrade(x, sr, condition, rng=rng)
        try:
            out.setdefault(speaker, []).append(enc.embed(x, sr))
        except Exception as exc:
            log.warning("skipped %s: %s", f.name, exc)
        if i % 25 == 0:
            log.info("  %d/%d clips", i, len(files))
    return out


def score_pairs(by_speaker: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """All within-speaker and between-speaker cosine similarities.

    Returns:
        ``(scores, labels)`` with label 1 for a same-speaker pair.
    """
    from src.speaker import cosine

    scores, labels = [], []
    for embs in by_speaker.values():
        for a, b in itertools.combinations(embs, 2):
            scores.append(cosine(a, b))
            labels.append(1)
    names = sorted(by_speaker)
    for s1, s2 in itertools.combinations(names, 2):
        for a in by_speaker[s1]:
            for b in by_speaker[s2]:
                scores.append(cosine(a, b))
                labels.append(0)
    return np.array(scores), np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate speaker-verification thresholds.")
    ap.add_argument("--data", default=Path("data/enrollment"), type=Path)
    ap.add_argument("--codec", default=None, help="e.g. amr_nb, g711_ulaw. Omit for clean audio.")
    ap.add_argument("--bitrate", type=float, default=None)
    ap.add_argument("--packet-loss", type=float, default=0.0)
    ap.add_argument("--out", default=Path("checkpoints/speaker_thresholds.json"), type=Path)
    args = ap.parse_args()

    cond = (Condition(codec=args.codec, bitrate_kbps=args.bitrate, packet_loss=args.packet_loss)
            if args.codec else None)
    log.info("condition: %s", cond.label() if cond else "clean")

    by_speaker = collect_embeddings(args.data, cond)
    n_clips = sum(len(v) for v in by_speaker.values())
    usable = {k: v for k, v in by_speaker.items() if len(v) >= 2}
    log.info("%d speakers, %d clips (%d speakers have 2+ clips)",
             len(by_speaker), n_clips, len(usable))
    if len(usable) < 2:
        raise SystemExit("need at least 2 speakers with 2+ clips each")

    scores, labels = score_pairs(usable)
    same, diff = scores[labels == 1], scores[labels == 0]
    log.info("%d same-speaker pairs, %d different-speaker pairs", len(same), len(diff))
    log.info("  same-speaker      mean %.3f  sd %.3f", same.mean(), same.std())
    log.info("  different-speaker mean %.3f  sd %.3f", diff.mean(), diff.std())

    # Label 1 = same speaker = higher score, so EER is computed on the score directly.
    eer, thr = metrics.eer(scores, labels)
    fpr90, t90 = metrics.fpr_at_tpr(scores, labels, 0.90)
    fpr95, t95 = metrics.fpr_at_tpr(scores, labels, 0.95)

    accept = float(np.quantile(diff, 0.99))
    reject = float(np.quantile(same, 0.01))
    if reject >= accept:  # distributions overlap badly - fall back to the EER point
        accept = reject = float(thr)
        log.warning("same and different distributions overlap heavily; falling back to a "
                    "single EER threshold. More enrolled speakers would help.")

    result = {
        "condition": cond.label() if cond else "clean",
        "n_speakers": len(usable), "n_clips": n_clips,
        "n_same_pairs": int(len(same)), "n_diff_pairs": int(len(diff)),
        "eer": round(float(eer), 5), "eer_threshold": round(float(thr), 5),
        "auc": round(metrics.auc(scores, labels), 5),
        "fpr_at_tpr90": round(float(fpr90), 5), "threshold_at_tpr90": round(float(t90), 5),
        "fpr_at_tpr95": round(float(fpr95), 5), "threshold_at_tpr95": round(float(t95), 5),
        "accept_threshold": round(accept, 5), "reject_threshold": round(reject, 5),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    log.info("")
    log.info("EER %.2f%%  at similarity %.3f", 100 * eer, thr)
    log.info("suggested accept %.3f / reject %.3f", accept, reject)
    log.info("written to %s", args.out)
    if len(usable) < 10:
        log.warning("only %d speakers. Thresholds from this few are unstable - collect 20+ "
                    "before quoting them anywhere.", len(usable))


if __name__ == "__main__":
    main()
