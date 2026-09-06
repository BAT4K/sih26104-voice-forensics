"""Multi-task fine-tuning: binary detection plus vocoder-family attribution.

What changed from v0.3, and why it matters
------------------------------------------
1. Label convention.  v0.3 forced ``id2label = {0: fake, 1: real}`` onto a
   checkpoint that publishes ``{0: real, 1: fake}``.  Because ``num_labels``
   matched, the head was loaded rather than reinitialised, so every training
   example contradicted the pretrained weights.  The convention here matches
   :mod:`src.families` and is asserted in the tests.
2. Learning rate.  1e-3 against a wav2vec2-large backbone is roughly thirty
   times too high and causes catastrophic forgetting.  3e-5 with warmup.
3. Metric.  Accuracy is threshold-dependent and hides the operating point.
   EER is primary here, with false-positive rate at fixed true-positive rate.
4. Silence.  ASVspoof carries a silence-duration artifact that correlates with
   the label.  Both classes are trimmed identically.
5. Splits.  A manifest with speaker and family columns lets you hold out whole
   speakers and whole synthesis systems, which folder scanning cannot express.

Sanity check first, always::

    python train.py --overfit-batch

If the loss does not fall to near zero on sixteen files in two hundred steps,
something is wrong with the head or the learning rate and a full run will only
waste a night finding that out.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src import audio_prep, families, metrics
from src.model_loader import DetectorConfig, VoiceForensicDetector, NoveltyScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vfd.train")


def set_seed(seed: int = 42) -> None:
    """Seed every generator once, at process entry.

    Never seed inside a per-item helper: v0.3 called ``np.random.seed(42)``
    inside the degradation function, which silently reset the global stream on
    every file and would destroy dataloader shuffling.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Sample:
    """One training item."""

    path: str
    label: int
    family: int
    speaker: str = "unknown"
    language: str = "unknown"


class VoiceForensicsDataset(Dataset):
    """Audio dataset backed by a manifest, with folder scanning as a fallback.

    Manifest format is JSON Lines, one object per clip::

        {"path": "...", "label": "fake", "family": "vocos",
         "speaker": "spk_012", "language": "hi"}

    ``family`` may be omitted for bonafide clips or when the generator is
    unknown; those samples are given ``-100`` and skipped by the Tier 2 loss
    while still contributing to Tier 1.
    """

    def __init__(self, root: str, split: str = "train", manifest: str | None = None, augment: bool = False) -> None:
        self.augment = augment
        self.samples: list[Sample] = []
        manifest_path = manifest or os.path.join(root, f"{split}.jsonl")

        if os.path.isfile(manifest_path):
            self._from_manifest(manifest_path)
            log.info("loaded %d samples from manifest %s", len(self.samples), manifest_path)
        else:
            self._from_folders(os.path.join(root, split))
            log.info(
                "no manifest at %s; scanned folders and found %d samples. "
                "Family labels are unknown, so Tier 2 will not train.",
                manifest_path, len(self.samples),
            )
        if not self.samples:
            raise FileNotFoundError(f"no audio found for split {split!r} under {root!r}")

    def _from_manifest(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                label = families.BINARY_LABEL2ID[str(rec["label"]).lower()]
                fam_key = rec.get("family")
                if fam_key is None:
                    fam = families.REAL.idx if label == 0 else -100
                elif isinstance(fam_key, int) and fam_key in families.BY_IDX:
                    fam = fam_key
                else:
                    fam = families.LABEL2ID.get(str(fam_key).lower(), -100)
                self.samples.append(
                    Sample(rec["path"], label, fam, rec.get("speaker", "unknown"), rec.get("language", "unknown"))
                )

    def _from_folders(self, split_dir: str) -> None:
        for name, label in (("real", 0), ("fake", 1)):
            d = os.path.join(split_dir, name)
            if not os.path.isdir(d):
                continue
            fam = families.REAL.idx if label == 0 else -100
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".wav", ".flac")):
                    self.samples.append(Sample(os.path.join(d, fn), label, fam))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import soundfile as sf

        s = self.samples[idx]
        data, sr = sf.read(s.path, dtype="float32")
        x = audio_prep.prepare(data, sr, trim=True, normalise=True)

        if self.augment and len(x) > audio_prep.WINDOW_SAMPLES:
            start = random.randint(0, len(x) - audio_prep.WINDOW_SAMPLES)
            x = x[start : start + audio_prep.WINDOW_SAMPLES]
        x = audio_prep.fit_window(x, audio_prep.WINDOW_SAMPLES, tile_pad=True)

        return {
            "input_values": torch.from_numpy(x),
            "labels": torch.tensor(s.label, dtype=torch.long),
            "family_labels": torch.tensor(s.family, dtype=torch.long),
        }

    def speakers(self) -> set[str]:
        return {s.speaker for s in self.samples}


def compute_metrics(eval_pred) -> dict[str, float]:
    """EER first, family F1 second, accuracy last."""
    preds, labels = eval_pred
    binary_logits, family_logits = (preds if isinstance(preds, (tuple, list)) else (preds, None))[:2]
    y = labels[0] if isinstance(labels, (tuple, list)) else labels

    p = torch.softmax(torch.as_tensor(binary_logits), dim=-1).numpy()
    spoof_score = p[:, families.BINARY_LABEL2ID["fake"]]
    out = metrics.summarise(spoof_score, np.asarray(y))
    out["accuracy"] = float(np.mean(np.argmax(p, axis=1) == np.asarray(y)))

    if family_logits is not None and isinstance(labels, (tuple, list)) and len(labels) > 1:
        fam_true = np.asarray(labels[1])
        keep = fam_true != -100
        if keep.any():
            fam_pred = np.argmax(np.asarray(family_logits)[keep], axis=1)
            out["family_macro_f1"] = metrics.macro_f1(fam_pred, fam_true[keep], families.NUM_FAMILIES)
    return out


def overfit_batch(model: VoiceForensicDetector, ds: Dataset, steps: int = 200, n: int = 16, device: str = "cpu") -> float:
    """Sanity check: can the model memorise sixteen files?

    If this does not drive the loss to near zero, the head, the label mapping
    or the learning rate is wrong. Five minutes here saves a wasted night.
    """
    model.train().to(device)
    batch = [ds[i] for i in range(min(n, len(ds)))]
    x = torch.stack([b["input_values"] for b in batch]).to(device)
    y = torch.stack([b["labels"] for b in batch]).to(device)
    f = torch.stack([b["family_labels"] for b in batch]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss_val = float("inf")
    for step in range(steps):
        opt.zero_grad()
        out = model(input_values=x, labels=y, family_labels=f)
        out["loss"].backward()
        opt.step()
        loss_val = float(out["loss"].item())
        if step % 25 == 0:
            log.info("overfit step %3d  loss %.4f", step, loss_val)
    log.info("final overfit loss %.4f (want < 0.05)", loss_val)
    return loss_val


def fit_novelty(model: VoiceForensicDetector, ds: Dataset, out_dir: str, device: str, max_items: int = 2000) -> None:
    """Fit the Tier 3 scorer on training embeddings and save it."""
    model.eval().to(device)
    embs, fams = [], []
    with torch.no_grad():
        for i in range(0, min(len(ds), max_items), 16):
            batch = [ds[j] for j in range(i, min(i + 16, len(ds), max_items))]
            if not batch:
                break
            x = torch.stack([b["input_values"] for b in batch]).to(device)
            embs.append(model.embed(x).cpu().numpy())
            fams.append(np.array([int(b["family_labels"]) for b in batch]))
    emb = np.concatenate(embs)
    fam = np.concatenate(fams)
    keep = fam != -100
    if keep.sum() < 20:
        log.warning("too few family-labelled samples (%d); skipping Tier 3", int(keep.sum()))
        return
    NoveltyScorer().fit(emb[keep], fam[keep]).save(os.path.join(out_dir, "novelty.npz"))
    log.info("Tier 3 novelty scorer saved with %d classes", len(set(fam[keep].tolist())))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the Indic voice forensic detector.")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./checkpoints/vfd-v1")
    ap.add_argument("--encoder", default=DetectorConfig().encoder_id)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-5, help="30x lower than v0.3; do not raise this")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze the lowest N transformer layers to fit a smaller GPU")
    ap.add_argument("--family-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overfit-batch", action="store_true", help="run the sanity check and exit")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("device=%s", device)

    cfg = DetectorConfig(encoder_id=args.encoder, freeze_encoder_layers=args.freeze_layers)
    model = VoiceForensicDetector(cfg)

    train_ds = VoiceForensicsDataset(args.data, "train", augment=True)
    if args.overfit_batch:
        loss = overfit_batch(model, train_ds, device=device)
        raise SystemExit(0 if loss < 0.05 else 1)

    eval_ds = VoiceForensicsDataset(args.data, "eval")
    overlap = train_ds.speakers() & eval_ds.speakers()
    if overlap - {"unknown"}:
        log.warning(
            "%d speakers appear in BOTH train and eval (%s...). Every number from this run is "
            "inflated. Rebuild the split with tools in src/build_dataset.py.",
            len(overlap), sorted(overlap)[:5],
        )

    from transformers import Trainer, TrainingArguments

    targs = TrainingArguments(
        output_dir=args.out,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eer",
        greater_is_better=False,
        fp16=(device == "cuda"),
        seed=args.seed,
        report_to=[],
        label_names=["labels", "family_labels"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    fit_novelty(model, train_ds, args.out, device)
    log.info("saved to %s", args.out)
    log.info("final metrics: %s", json.dumps(trainer.evaluate(), indent=2, default=float))


if __name__ == "__main__":
    main()
