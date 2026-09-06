"""The benchmark: EER as a function of language, codec and packet loss.

This is the research contribution, and it needs no training - it is pure
evaluation over public checkpoints, which is why it can be finished in the
first fortnight and stand on its own even if every other track slips.

What it produces that no published paper contains:

* AMR-NB scored at all eight modes. Every existing codec benchmark uses
  AMR-*WB*; AMR-NB carries a large share of Indian mobile calls.
* Narrowband crossed with packet loss. Existing work has one or the other,
  never both.
* Packet-loss concealment as an ablation rather than an explanation.
* All of it per Indian language.

Run::

    python -m src.benchmark --manifest data/eval.jsonl --out results/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src import audio_prep, families, metrics
from src.telephony_degrader import Condition, condition_grid, missing_codecs

log = logging.getLogger("vfd.bench")


def load_manifest(path: str) -> list[dict]:
    """Read a JSON Lines manifest written by :mod:`src.build_dataset`."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def score_condition(
    records: list[dict], condition: Condition, checkpoint: str, device: str | None, seed: int = 0
) -> dict[str, list]:
    """Score every clip under one transmission condition.

    Returns:
        Dict with parallel lists: ``scores``, ``labels``, ``languages``,
        ``families``.
    """
    import soundfile as sf
    import torch

    from src.inference import _score_windows
    from src import model_loader

    model, _ = model_loader.load_detector(checkpoint, device)
    dev = next(model.parameters()).device.type
    rng = np.random.default_rng(seed)

    scores, labels, langs, fams = [], [], [], []
    for rec in records:
        try:
            data, sr = sf.read(rec["path"], dtype="float32")
            x = audio_prep.to_mono(data)
            if condition.codec != "none":
                from src.telephony_degrader import degrade

                x, sr = degrade(x, sr, condition, rng=rng)
            x = audio_prep.prepare(x, sr)
            wins = [w for _, w in audio_prep.windows(x)]
            if not wins:
                continue
            spoof, _, _ = _score_windows(model, wins, dev)
        except Exception as exc:
            log.warning("skipped %s under %s: %s", rec.get("path"), condition.label(), exc)
            continue
        scores.append(float(np.mean(spoof)))
        labels.append(families.BINARY_LABEL2ID[str(rec["label"]).lower()])
        langs.append(rec.get("language", "unknown"))
        fams.append(rec.get("family", "unknown"))
    return {"scores": scores, "labels": labels, "languages": langs, "families": fams}


def run(
    manifest: str,
    out_dir: str,
    checkpoint: str,
    device: str | None = None,
    codecs: tuple[str, ...] | None = None,
    per_language: bool = True,
    base_rate: float = 0.0017,
) -> dict:
    """Run the full grid and write results.

    Args:
        manifest: Evaluation manifest.
        out_dir: Directory for ``results.json`` and ``matrix.csv``.
        checkpoint: Trained checkpoint directory.
        device: Torch device.
        codecs: Restrict the codec axis.
        per_language: Also break every condition down by language.
        base_rate: Fraud prior used for alert precision.

    Returns:
        The results dictionary that was written.
    """
    records = load_manifest(manifest)
    grid = condition_grid(codecs=codecs)
    if missing := missing_codecs():
        log.warning("NOT COVERED on this machine (install ffmpeg codecs): %s", ", ".join(missing))

    log.info("%d clips x %d conditions = %d evaluations", len(records), len(grid), len(records) * len(grid))
    rows, out = [], {"n_clips": len(records), "conditions": {}, "missing_codecs": missing_codecs()}

    for i, cond in enumerate(grid, 1):
        res = score_condition(records, cond, checkpoint, device)
        if len(set(res["labels"])) < 2:
            log.warning("condition %s has only one class; skipping", cond.label())
            continue
        s, y = np.array(res["scores"]), np.array(res["labels"])
        summary = metrics.summarise(s, y, base_rate)
        summary["n"] = int(len(y))

        if per_language:
            by_lang = {}
            for lang in sorted(set(res["languages"])):
                m = np.array([l == lang for l in res["languages"]])
                if m.sum() >= 20 and len(set(y[m])) == 2:
                    by_lang[lang] = {"eer": metrics.eer(s[m], y[m])[0], "n": int(m.sum())}
            summary["by_language"] = by_lang

        out["conditions"][cond.label()] = summary
        rows.append({
            "condition": cond.label(), "codec": cond.codec,
            "bitrate_kbps": cond.bitrate_kbps or "", "packet_loss": cond.packet_loss,
            "plc": cond.plc, "eer": round(summary["eer"], 5),
            "fpr_at_tpr90": round(summary["fpr_at_tpr90"], 5),
            "alert_precision_at_tpr90": round(summary["alert_precision_at_tpr90"], 5),
            "n": summary["n"],
        })
        log.info("[%3d/%3d] %-28s EER %6.2f%%  FPR@TPR90 %6.2f%%",
                 i, len(grid), cond.label(), 100 * summary["eer"], 100 * summary["fpr_at_tpr90"])

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if rows:
        import csv

        with open(dest / "matrix.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    clean = out["conditions"].get("clean-16k", {}).get("eer")
    if clean is not None and out["conditions"]:
        worst = max(out["conditions"].items(), key=lambda kv: kv[1]["eer"])
        log.info("HEADLINE: clean %.2f%% EER -> worst %.2f%% EER under %s (%.1fx)",
                 100 * clean, 100 * worst[1]["eer"], worst[0], worst[1]["eer"] / max(clean, 1e-6))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Telephony-condition benchmark.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--checkpoint", default="./checkpoints/vfd-v1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--codecs", nargs="*", default=None)
    args = ap.parse_args()
    run(args.manifest, args.out, args.checkpoint, args.device,
        tuple(args.codecs) if args.codecs else None)


if __name__ == "__main__":
    main()
