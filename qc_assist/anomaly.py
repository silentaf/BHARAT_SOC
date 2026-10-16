"""PatchCore-style anomaly detector.

Trained only on *good* parts: a pretrained CNN turns every image patch into a
feature vector, a compact "memory bank" of normal patches is kept, and at test
time each patch is scored by its distance to the nearest normal patch.
Real factories rarely have many labelled defect images, so needing only good
samples is the main practical advantage.

On the Pi + AI HAT+, the plan is to run the backbone on the Hailo NPU (HEF) and
keep the nearest-neighbour search on the Arm CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as TF

import qc_assist  # noqa: F401  (sets TORCH_HOME)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_input(img: np.ndarray, size: int) -> np.ndarray:
    """Area-downscale to the network's square input. Cheap on uint8, so do it as early as possible."""
    if img.shape[:2] == (size, size):
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class DetectorConfig:
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 224
    coreset_ratio: float = 0.01
    threshold_percentile: float = 99.0
    blur_sigma: float = 4.0


@dataclass
class Prediction:
    score: float          # image anomaly score normalised so 1.0 == decision threshold
    is_defect: bool
    anomaly_map: np.ndarray  # HxW float32, same normalisation, at image_size resolution


class PatchEmbedder(torch.nn.Module):
    """Backbone + local pooling + multi-scale concat: image batch -> (B, C, H, W) patch features.

    This is the part that runs on the Hailo NPU; scripts/export_onnx.py exports exactly this module.
    """

    def __init__(self, backbone: str, layers: tuple[str, ...], pretrained: bool = True):
        super().__init__()
        net = get_model(backbone, weights="DEFAULT" if pretrained else None)
        self.extractor = create_feature_extractor(net, {l: l for l in layers})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = list(self.extractor(x).values())
        grid = feats[0].shape[-2:]
        maps = []
        for f in feats:
            f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)  # local neighbourhood context
            if f.shape[-2:] != grid:
                f = F.interpolate(f, size=grid, mode="bilinear", align_corners=False)
            maps.append(f)
        return torch.cat(maps, dim=1)


class OnnxEmbedder:
    """Drop-in replacement for PatchEmbedder running the exported graph (scripts/export_onnx.py)
    with ONNX Runtime. The graph expects raw 0..255 pixels, so normalisation is undone first.
    A HailoRT embedder for the AI HAT+ will follow the same interface."""

    def __init__(self, onnx_path: str | Path, threads: int | None = None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        # ORT worker threads busy-wait by default and starve the PyTorch kNN/blur that runs
        # right after (measured 166 ms vs 11 ms per frame on the laptop). Let them sleep.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raw = ((x.cpu() * self.std + self.mean) * 255).numpy()
        outs = [self.sess.run(None, {"image": raw[i:i + 1]})[0] for i in range(len(raw))]
        return torch.from_numpy(np.concatenate(outs))


class PatchCore:
    def __init__(self, config: DetectorConfig | None = None, device: str = "cpu", pretrained: bool = True):
        self.cfg = config or DetectorConfig()
        self.device = torch.device(device)
        self.embedder = PatchEmbedder(self.cfg.backbone, self.cfg.layers, pretrained).eval().to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None

    # ---------- features ----------

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """RGB uint8 HxWx3 arrays -> normalised batch tensor."""
        batch = []
        for img in images:
            img = resize_input(img, self.cfg.image_size)
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch.append(TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD))
        return torch.stack(batch).to(self.device)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Returns (B, H*W, C) patch features and the patch grid size."""
        emb = self.embedder(x)
        b, c, h, w = emb.shape
        return emb.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)

    # ---------- training ----------

    @torch.no_grad()
    def fit(self, good_images: list[np.ndarray], calib_images: list[np.ndarray] | None = None,
            val_fraction: float = 0.1, batch_size: int = 16, seed: int = 0) -> None:
        """Build the memory bank from good_images. The decision threshold is calibrated on
        calib_images (separate good images) if given, else on a held-out val_fraction."""
        if calib_images:
            train, val = good_images, calib_images
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(good_images))
            n_val = max(1, int(len(good_images) * val_fraction)) if len(good_images) > 5 else 0
            val = [good_images[i] for i in idx[:n_val]]
            train = [good_images[i] for i in idx[n_val:]]

        patches = []
        for i in range(0, len(train), batch_size):
            emb, _ = self.embed(self.preprocess(train[i:i + batch_size]))
            patches.append(emb.reshape(-1, emb.shape[-1]))
        bank = torch.cat(patches)
        n_keep = max(1, int(len(bank) * self.cfg.coreset_ratio))
        self.memory_bank = greedy_coreset(bank, n_keep)

        # Threshold = high percentile of scores on held-out good parts.
        calib = val if val else train
        raw = [self._raw_map(self.preprocess(calib[i:i + batch_size])) for i in range(0, len(calib), batch_size)]
        scores = torch.cat([m.flatten(1).max(dim=1).values for m in raw]).cpu().numpy()
        self.threshold = float(np.percentile(scores, self.cfg.threshold_percentile))

    # ---------- inference ----------

    @torch.no_grad()
    def _raw_map(self, x: torch.Tensor) -> torch.Tensor:
        assert self.memory_bank is not None, "call fit() or load() first"
        emb, (h, w) = self.embed(x)
        b = emb.shape[0]
        dists = torch.cdist(emb.reshape(-1, emb.shape[-1]), self.memory_bank)
        patch_scores = dists.min(dim=1).values.reshape(b, 1, h, w)
        amap = F.interpolate(patch_scores, size=(self.cfg.image_size,) * 2, mode="bilinear", align_corners=False)
        k = int(4 * self.cfg.blur_sigma) * 2 + 1
        return TF.gaussian_blur(amap, kernel_size=k, sigma=self.cfg.blur_sigma).squeeze(1)

    def predict(self, image: np.ndarray) -> Prediction:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        maps = (self._raw_map(self.preprocess(images)) / self.threshold).cpu().numpy().astype(np.float32)
        out = []
        for m in maps:
            score = float(m.max())
            out.append(Prediction(score=score, is_defect=score > 1.0, anomaly_map=m))
        return out

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg), "memory_bank": self.memory_bank.cpu(), "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", backend: str = "torch") -> "PatchCore":
        """backend: "torch" (default) or "onnx" (uses <model>_embedder.onnx next to the model file)."""
        state = torch.load(path, map_location=device)
        cfg = state["config"]
        cfg["layers"] = tuple(cfg["layers"])
        model = cls(DetectorConfig(**cfg), device=device)
        model.memory_bank = state["memory_bank"].to(device)
        model.threshold = state["threshold"]
        if backend == "onnx":
            model.embedder = OnnxEmbedder(Path(path).with_suffix("").as_posix() + "_embedder.onnx",
                                          threads=torch.get_num_threads())
        elif backend != "torch":
            raise ValueError(f"unknown backend: {backend}")
        return model


@torch.no_grad()
def greedy_coreset(features: torch.Tensor, n: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """k-center greedy subsampling (as in PatchCore) on a random projection for speed."""
    if n >= len(features):
        return features
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, generator=g).to(features) / proj_dim ** 0.5
    z = features @ proj
    sq = (z * z).sum(1)

    def sq_dist(i: int) -> torch.Tensor:
        # |z - c|^2 = |z|^2 - 2 z.c + |c|^2 — one mat-vec; ~17x faster than cdist against a single point
        return sq - 2 * (z @ z[i]) + sq[i]

    selected = [int(torch.randint(len(z), (1,), generator=g))]
    min_d = sq_dist(selected[0])
    for _ in range(n - 1):
        nxt = int(min_d.argmax())
        selected.append(nxt)
        min_d = torch.minimum(min_d, sq_dist(nxt))
    return features[selected].clone()
