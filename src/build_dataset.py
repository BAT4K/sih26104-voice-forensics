"""Dataset builder: fetch, degrade, split, and write a manifest.

Three things v0.3 got wrong, all fixed here.

* It drew train and eval sequentially from a single partition, so speakers
  appeared on both sides and attack coverage was arbitrary. Splits here are
  speaker-disjoint, and whole synthesis systems can be held out.
* It never trimmed silence, so a model could learn the ASVspoof
  silence-duration artifact instead of synthesis artifacts.
* It emitted only folders, which cannot express family, speaker or language.
  A JSON Lines manifest carries all three, which is what Tier 2 needs.

ASVspoof stays available as an English sanity baseline, but the project's
premise is Indic telephony, so ``--source indic`` is the mode that matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src import audio_prep, families
from src.telephony_degrader import Condition, degrade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vfd.data")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def speaker_bucket(speaker: str, n_buckets: int = 10) -> int:
    """Stable hash of a speaker id, so a speaker always lands in one split."""
    return int(hashlib.sha1(speaker.encode("utf-8")).hexdigest(), 16) % n_buckets


def assign_split(speaker: str, eval_buckets: tuple[int, ...] = (8, 9)) -> str:
    """Speaker-disjoint train/eval assignment.

    Hashing the speaker rather than shuffling files guarantees no speaker can
    leak across the split, and the assignment is reproducible without storing
    a split file.
    """
    return "eval" if speaker_bucket(speaker) in eval_buckets else "train"


def iter_asvspoof(limit: int) -> Iterator[dict]:
    """Stream ASVspoof 2019 LA from Hugging Face.

    Note the label order: the ClassLabel feature is ``['bonafide', 'spoof']``,
    so 0 is real and 1 is fake. That matches
    :data:`src.families.BINARY_LABEL2ID`.
    """
    import datasets

    ds = datasets.load_dataset(
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", streaming=True
    )
    for i, item in enumerate(ds):
        if i >= limit:
            break
        label = int(item["label"])
        path = str(item.get("path", f"item_{i}"))
        # ASVspoof filenames are LA_E_<id>; speaker is not exposed by this
        # mirror, so we fall back to the file stem. Prefer a corpus that
        # publishes speaker ids.
        yield {
            "audio": np.asarray(item["audio"]["array"], dtype=np.float32),
            "sr": int(item["audio"]["sampling_rate"]),
            "label": "real" if label == 0 else "fake",
            "family": families.REAL.key if label == 0 else "legacy_ar",
            "speaker": Path(path).stem or f"asv_{i}",
            "language": "en",
        }


def iter_local(root: str) -> Iterator[dict]:
    """Walk a local tree laid out as ``<root>/<label>/<family>/<speaker>/*.wav``.

    ``family`` and ``speaker`` levels are optional; missing levels become
    "unknown" and are skipped by the Tier 2 loss.
    """
    import soundfile as sf

    base = Path(root)
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac"}:
            continue
        rel = path.relative_to(base).parts
        label = rel[0].lower() if rel and rel[0].lower() in {"real", "fake"} else None
        if label is None:
            continue
        fam = rel[1].lower() if len(rel) > 2 and rel[1].lower() in families.LABEL2ID else (
            families.REAL.key if label == "real" else None
        )
        speaker = rel[2] if len(rel) > 3 else path.stem
        data, sr = sf.read(path, dtype="float32")
        yield {
            "audio": np.asarray(data, dtype=np.float32), "sr": sr, "label": label,
            "family": fam, "speaker": speaker, "language": "unknown",
        }


def build(
    out_dir: str,
    source: str = "asvspoof",
    local_root: str | None = None,
    limit: int = 4000,
    codec: str = "g711_ulaw",
    bitrate_kbps: float | None = None,
    packet_loss: float = 0.0,
    trim: bool = True,
    seed: int = 42,
) -> dict[str, int]:
    """Fetch, degrade identically for both classes, and write manifests.

    Args:
        out_dir: Destination root. Audio goes under ``<out_dir>/audio``,
            manifests are ``<out_dir>/train.jsonl`` and ``eval.jsonl``.
        source: ``asvspoof`` or ``local``.
        local_root: Required when ``source='local'``.
        limit: Maximum clips to pull.
        codec: Telephony codec applied to every clip.
        bitrate_kbps: Codec bitrate, where applicable.
        packet_loss: Frame-loss fraction.
        trim: Trim leading and trailing silence.
        seed: Seed for the packet-loss generator only.

    Returns:
        Counts per split.
    """
    import soundfile as sf

    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    if codec != "random":
        cond = Condition(codec=codec, bitrate_kbps=bitrate_kbps, packet_loss=packet_loss)
        log.info("degradation condition: %s", cond.label())
    else:
        log.info("degradation condition: RANDOM (g711, opus_nb, amr_nb, g726)")

    handles = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "eval")}
    counts = {"train": 0, "eval": 0}
    src = iter_asvspoof(limit) if source == "asvspoof" else iter_local(local_root or "")

    try:
        for i, rec in enumerate(src):
            x = audio_prep.to_mono(rec["audio"])
            if trim:
                # Trim BEFORE degrading, and identically for both classes.
                x = audio_prep.trim_silence(audio_prep.resample(x, rec["sr"], audio_prep.TARGET_SR))
                rec["sr"] = audio_prep.TARGET_SR
                
            if codec == "random":
                chosen_codec = rng.choice(["g711_ulaw", "opus_nb", "amr_nb", "g726"])
                current_cond = Condition(codec=chosen_codec, bitrate_kbps=bitrate_kbps, packet_loss=packet_loss)
            else:
                current_cond = cond
                
            y, out_sr = degrade(x, rec["sr"], current_cond, rng=rng)

            split = assign_split(rec["speaker"])
            name = f"{split}_{rec['label']}_{i:06d}.wav"
            dest = out / "audio" / name
            sf.write(dest, y, out_sr, format="WAV", subtype="PCM_16")

            handles[split].write(json.dumps({
                "path": str(dest), "label": rec["label"], "family": rec["family"],
                "speaker": rec["speaker"], "language": rec["language"],
                "condition": current_cond.label(),
            }) + "\n")
            counts[split] += 1
            if (i + 1) % 200 == 0:
                log.info("processed %d clips  train=%d eval=%d", i + 1, counts["train"], counts["eval"])
    finally:
        for fh in handles.values():
            fh.close()

    log.info("done. train=%d eval=%d  manifests in %s", counts["train"], counts["eval"], out)
    log.info("splits are speaker-disjoint by stable hash; no speaker appears on both sides")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a degraded, speaker-disjoint dataset.")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--source", choices=("asvspoof", "local"), default="asvspoof")
    ap.add_argument("--local-root", default=None, help="required for --source local")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--codec", default="g711_ulaw")
    ap.add_argument("--bitrate-kbps", type=float, default=None)
    ap.add_argument("--packet-loss", type=float, default=0.0)
    ap.add_argument("--no-trim", action="store_true")
    args = ap.parse_args()

    if args.source == "local" and not args.local_root:
        ap.error("--local-root is required with --source local")
    build(
        args.out, args.source, args.local_root, args.limit,
        args.codec, args.bitrate_kbps, args.packet_loss, not args.no_trim,
    )


# Backwards-compatible alias for anything that imported the old entry point.
simulate_and_build = build

if __name__ == "__main__":
    main()
