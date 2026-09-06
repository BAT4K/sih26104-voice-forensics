"""The detector: a shared encoder with three tiers on top.

Tier 1  binary head        real against fake
Tier 2  family head        which of six decoder families produced it
Tier 3  novelty scorer     Mahalanobis distance from every known family

Why one encoder and two heads rather than two models: the vocoder fingerprint
dominates the acoustic-model fingerprint, so the representation that separates
families is the same one that separates real from fake.  Training them jointly
gives the binary head a harder, more informative objective and costs one
forward pass at inference instead of two.

Nothing in this module imports Streamlit or FastAPI.  That is deliberate and
load-bearing: the original code decorated the loader with
``@st.cache_resource``, which meant a training script, a unit test and an API
worker all had to drag in the Streamlit runtime to load a model, and none of
them got correct caching when they did.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src import families

log = logging.getLogger(__name__)

DEFAULT_ENCODER = "facebook/wav2vec2-xls-r-300m"
"""Multilingual SSL front end.  Pretrained on 128 languages, which is most of
why cross-lingual transfer to Indic works at all once fine-tuned."""

CHECKPOINT_DIR = os.environ.get("VFD_CHECKPOINT", "./checkpoints/vfd-v1")


@dataclass
class DetectorConfig:
    """Architecture and training-shape configuration."""

    encoder_id: str = DEFAULT_ENCODER
    num_families: int = families.NUM_FAMILIES
    hidden_dropout: float = 0.1
    proj_dim: int = 256
    freeze_feature_encoder: bool = True
    freeze_encoder_layers: int = 0
    """Freeze the lowest N transformer layers. Use this to fit a smaller GPU
    instead of raising the learning rate, which destroys the pretrained weights."""

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DetectorConfig":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


class AttentiveStatsPooling(nn.Module):
    """Attention-weighted mean and standard deviation over time.

    Preferred over plain mean pooling because synthesis artifacts are not
    uniformly distributed across an utterance; the attention lets the model
    weight the frames that carry them.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, 2D)."""
        w = torch.softmax(self.attn(x), dim=1)
        mean = torch.sum(w * x, dim=1)
        var = torch.sum(w * x.pow(2), dim=1) - mean.pow(2)
        return torch.cat([mean, torch.sqrt(var.clamp(min=1e-8))], dim=-1)


class VoiceForensicDetector(nn.Module):
    """Shared SSL encoder, attentive pooling, two classification heads."""

    def __init__(self, cfg: DetectorConfig | None = None) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.cfg = cfg or DetectorConfig()
        enc_cfg = AutoConfig.from_pretrained(self.cfg.encoder_id)
        self.encoder = AutoModel.from_pretrained(self.cfg.encoder_id, config=enc_cfg)

        if self.cfg.freeze_feature_encoder and hasattr(self.encoder, "freeze_feature_encoder"):
            self.encoder.freeze_feature_encoder()
        if self.cfg.freeze_encoder_layers > 0:
            layers = getattr(self.encoder.encoder, "layers", [])
            for layer in list(layers)[: self.cfg.freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        dim = enc_cfg.hidden_size
        self.pool = AttentiveStatsPooling(dim)
        self.project = nn.Sequential(
            nn.Linear(2 * dim, self.cfg.proj_dim),
            nn.LayerNorm(self.cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.hidden_dropout),
        )
        self.binary_head = nn.Linear(self.cfg.proj_dim, 2)
        self.family_head = nn.Linear(self.cfg.proj_dim, self.cfg.num_families)

    def embed(self, input_values: torch.Tensor) -> torch.Tensor:
        """Pooled 256-d embedding. Tier 3 operates on this."""
        h = self.encoder(input_values=input_values).last_hidden_state
        return self.project(self.pool(h))

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        family_labels: torch.Tensor | None = None,
        family_loss_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Multi-task forward pass.

        Args:
            input_values: (B, 64600) normalised waveform.
            labels: Optional (B,) binary targets, 0 real / 1 fake.
            family_labels: Optional (B,) family targets, 0..6. Use -100 to
                ignore a sample whose family is unknown.
            family_loss_weight: Weight on the Tier 2 term.

        Returns:
            Dict with ``embedding``, ``logits``, ``family_logits`` and, when
            targets are supplied, ``loss``.
        """
        emb = self.embed(input_values)
        out: dict[str, torch.Tensor] = {
            "embedding": emb,
            "logits": self.binary_head(emb),
            "family_logits": self.family_head(emb),
        }
        if labels is not None:
            loss = nn.functional.cross_entropy(out["logits"], labels)
            if family_labels is not None:
                loss = loss + family_loss_weight * nn.functional.cross_entropy(
                    out["family_logits"], family_labels, ignore_index=-100
                )
            out["loss"] = loss
        return out

    def save(self, directory: str) -> None:
        """Persist weights and config."""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(directory, "pytorch_model.bin"))
        self.cfg.to_json(os.path.join(directory, "vfd_config.json"))

    @classmethod
    def load(cls, directory: str, map_location: str | torch.device = "cpu") -> "VoiceForensicDetector":
        """Restore from a directory written by :meth:`save`."""
        config_path = os.path.join(directory, "vfd_config.json")
        if os.path.exists(config_path):
            cfg = DetectorConfig.from_json(config_path)
        else:
            cfg = DetectorConfig()
            
        model = cls(cfg)
        
        bin_path = os.path.join(directory, "pytorch_model.bin")
        safe_path = os.path.join(directory, "model.safetensors")
        
        if os.path.exists(safe_path):
            from safetensors.torch import load_file
            state = load_file(safe_path)
        else:
            state = torch.load(bin_path, map_location=map_location)
            
        model.load_state_dict(state)
        return model


class NoveltyScorer:
    """Tier 3: is this a synthesis method we have never seen?

    A class-conditional Mahalanobis distance with a shared covariance.  The
    open-set literature is clear that a detector confident on its training
    families will still be confident on an unseen one, so the honest output is
    a distance, not a class.  Fit on training embeddings, then anything far
    from every family centroid is reported as unknown rather than guessed.
    """

    def __init__(self) -> None:
        self.centroids: np.ndarray | None = None
        self.precision: np.ndarray | None = None
        self.threshold: float | None = None
        self.class_ids: list[int] = []

    def fit(self, embeddings: np.ndarray, family_labels: np.ndarray, quantile: float = 0.95,
            shrinkage: float = 0.1) -> "NoveltyScorer":
        """Estimate centroids, shared precision, and a rejection threshold.

        Args:
            embeddings: (N, D) training embeddings.
            family_labels: (N,) family indices.
            quantile: In-distribution quantile used as the reject threshold, so
                roughly ``1 - quantile`` of genuine training data is flagged.
            shrinkage: Weight on the scaled-identity target, in [0, 1]. Raise it
                if the covariance is near-singular; 0 reproduces the unstable
                v1 behaviour and is not recommended.
        """
        emb = np.asarray(embeddings, dtype=np.float64)
        lab = np.asarray(family_labels).ravel().astype(int)
        self.class_ids = sorted(set(lab.tolist()))
        self.centroids = np.stack([emb[lab == c].mean(axis=0) for c in self.class_ids])

        centred = np.concatenate([emb[lab == c] - self.centroids[i] for i, c in enumerate(self.class_ids)])
        cov = np.cov(centred, rowvar=False)

        # Shrink toward a scaled identity before inverting.
        #
        # The wav2vec2 embedding covariance is badly conditioned - measured at
        # ~3e6 on this checkpoint, smallest eigenvalue ~2e-5 against a mean
        # variance of 0.33.  A fixed 1e-6 ridge is six orders of magnitude too
        # small to help, so pinv amplifies the noise directions and the
        # quadratic form below overflows to inf, which then silently poisons
        # anything consuming the distance.  Ledoit-Wolf style shrinkage scaled
        # to the trace keeps the inverse finite and the ranking intact.
        d_dim = emb.shape[1]
        mean_var = float(np.trace(cov) / d_dim)
        cov = (1.0 - shrinkage) * cov + shrinkage * mean_var * np.eye(d_dim)
        self.precision = np.linalg.inv(cov)

        dist = self.distance(emb)
        if not np.isfinite(dist).all():
            raise RuntimeError("novelty distances are not finite; raise `shrinkage`")
        self.threshold = float(np.quantile(dist, quantile))
        return self

    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Minimum Mahalanobis distance to any known family centroid."""
        if self.centroids is None or self.precision is None:
            raise RuntimeError("NoveltyScorer.fit must be called first")
        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim == 1:
            emb = emb[None, :]
        d = np.empty((emb.shape[0], self.centroids.shape[0]))
        for i, c in enumerate(self.centroids):
            diff = emb - c
            d[:, i] = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return np.sqrt(np.maximum(d.min(axis=1), 0.0))

    def is_novel(self, embeddings: np.ndarray) -> np.ndarray:
        """True where the embedding is further than the fitted threshold."""
        if self.threshold is None:
            raise RuntimeError("NoveltyScorer.fit must be called first")
        return self.distance(embeddings) > self.threshold

    def save(self, path: str) -> None:
        if self.centroids is None or self.precision is None:
            raise RuntimeError("nothing to save; call fit first")
        np.savez(
            path,
            centroids=self.centroids,
            precision=self.precision,
            threshold=np.array([self.threshold]),
            class_ids=np.array(self.class_ids),
        )

    @classmethod
    def load(cls, path: str) -> "NoveltyScorer":
        z = np.load(path)
        s = cls()
        s.centroids, s.precision = z["centroids"], z["precision"]
        s.threshold, s.class_ids = float(z["threshold"][0]), z["class_ids"].tolist()
        return s


# --------------------------------------------------------------------------
# Process-wide cache. Plain module state and a lock - no framework involved.
# --------------------------------------------------------------------------

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def checkpoint_available(directory: str = CHECKPOINT_DIR) -> bool:
    """True if a trained checkpoint exists at ``directory``."""
    has_bin = os.path.isfile(os.path.join(directory, "pytorch_model.bin"))
    has_safe = os.path.isfile(os.path.join(directory, "model.safetensors"))
    return has_bin or has_safe


def load_detector(
    directory: str = CHECKPOINT_DIR, device: str | None = None, novelty: bool = True
) -> tuple[VoiceForensicDetector, NoveltyScorer | None]:
    """Load and cache the detector for this process.

    Args:
        directory: Checkpoint directory.
        device: Torch device string. Auto-selected when omitted.
        novelty: Also load the Tier 3 scorer if present.

    Returns:
        ``(model, novelty_scorer_or_None)``.

    Raises:
        FileNotFoundError: No checkpoint. This is deliberate: the system must
            never silently fall back to fabricated verdicts.
    """
    key = f"{directory}|{device}|{novelty}"
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if not checkpoint_available(directory):
            raise FileNotFoundError(
                f"No trained checkpoint at {directory!r}. Train one with `python train.py`, or point "
                f"VFD_CHECKPOINT at an existing directory. The service will not invent a verdict."
            )
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = VoiceForensicDetector.load(directory, map_location=dev).to(dev).eval()

        scorer = None
        npz = os.path.join(directory, "novelty.npz")
        if novelty and os.path.isfile(npz):
            scorer = NoveltyScorer.load(npz)
        elif novelty:
            log.warning("Tier 3 novelty scorer missing at %s; unknown-system detection disabled", npz)

        _CACHE[key] = (model, scorer)
        return model, scorer


def clear_cache() -> None:
    """Drop cached models. Used by tests."""
    with _LOCK:
        _CACHE.clear()
