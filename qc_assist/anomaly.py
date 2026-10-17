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

import threading

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
from .defect_types import TOP_PATCHES, DefectTypeClassifier

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
    descriptor: np.ndarray | None = None  # what changed at the worst patches (defect_types.py)
    defect_type: str | None = None        # set when the product has a defect-type model and is_defect
    type_confidence: float | None = None


class PatchEmbedder(torch.nn.Module):
    """Backbone + local pooling + multi-scale concat: image batch -> (B, C, H, W) patch features.

    Split where the NPU hands over: `maps` (backbone + 3x3 local averaging, one output per layer)
    is what scripts/export_onnx.py exports and the Hailo NPU runs; `merge` (upsample + concat)
    runs on the CPU. One output per layer also lets the Hailo compiler give each its own
    quantisation scale; a single concatenated output failed to quantise.
    """

    def __init__(self, backbone: str, layers: tuple[str, ...], pretrained: bool = True):
        super().__init__()
        net = get_model(backbone, weights="DEFAULT" if pretrained else None)
        self.extractor = create_feature_extractor(net, {l: l for l in layers})

    def maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        # 3x3 average = local neighbourhood context for every patch
        return [F.avg_pool2d(f, kernel_size=3, stride=1, padding=1) for f in self.extractor(x).values()]

    @staticmethod
    def merge(maps: list[torch.Tensor]) -> torch.Tensor:
        grid = maps[0].shape[-2:]
        return torch.cat([m if m.shape[-2:] == grid else
                          F.interpolate(m, size=grid, mode="bilinear", align_corners=False) for m in maps], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(self.maps(x))


class OnnxEmbedder:
    """Drop-in replacement for PatchEmbedder running the exported graph (scripts/export_onnx.py)
    with ONNX Runtime. The graph expects raw 0..255 pixels, so normalisation is undone first;
    it returns one map per layer, merged on the CPU as in PatchEmbedder.
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
        return torch.cat([PatchEmbedder.merge([torch.from_numpy(m) for m in self.sess.run(None, {"image": raw[i:i + 1]})])
                          for i in range(len(raw))])


class HailoEmbedder:
    """The NPU part on a Hailo chip, same interface as OnnxEmbedder (see docs/hailo.md).

    mode "hailort":  HailoRT on the device (Raspberry Pi 5 + AI HAT+), runs the compiled .hef.
    mode "emulator": the Dataflow Compiler's INT8 emulator on an x86 PC, runs the quantised .har,
                     so the accuracy cost of quantisation is measured before the hardware exists.
    """

    def __init__(self, path: str | Path, mode: str = "hailort"):
        self._lock = threading.Lock()  # one set of device buffers; "teach a new board" calls from another thread
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        if mode == "emulator":
            from hailo_sdk_client import ClientRunner, InferenceContext
            runner = ClientRunner(har=str(path))
            self._ctx_manager = runner.infer_context(InferenceContext.SDK_QUANTIZED)
            ctx = self._ctx_manager.__enter__()  # kept open for the embedder's lifetime
            self._run = lambda x: runner.infer(ctx, x.astype(np.float32), batch_size=len(x))
        elif mode == "hailort":
            self._run = self._hailort(path)
        else:
            raise ValueError(f"unknown Hailo mode: {mode}")

    def _hailort(self, hef_path):
        from hailo_platform import FormatType, VDevice
        self._vdevice = VDevice()
        model = self._vdevice.create_infer_model(str(hef_path))
        model.input().set_format_type(FormatType.UINT8)       # raw camera pixels, NHWC
        for o in model.outputs:
            o.set_format_type(FormatType.FLOAT32)              # de-quantised on the device
        self._configured = model.configure()
        bindings = self._configured.create_bindings()
        outs = {o.name: np.empty(o.shape, np.float32) for o in model.outputs}
        for name, buf in outs.items():
            bindings.output(name).set_buffer(buf)

        def run(x):
            res = []
            for img in x:
                bindings.input().set_buffer(np.ascontiguousarray(img, dtype=np.uint8))
                self._configured.run([bindings], 1000)
                res.append([b.copy() for b in outs.values()])
            return [np.stack(m) for m in zip(*res)]
        return run

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raw = ((x.cpu() * self.std + self.mean) * 255).round().clamp(0, 255).permute(0, 2, 3, 1).numpy()
        with self._lock:
            maps = sorted(self._run(raw), key=lambda m: -m.shape[1])  # NHWC per layer, finest grid first
        return PatchEmbedder.merge([torch.from_numpy(np.ascontiguousarray(m)).permute(0, 3, 1, 2) for m in maps])


class PatchCore:
    def __init__(self, config: DetectorConfig | None = None, device: str = "cpu", pretrained: bool = True):
        self.cfg = config or DetectorConfig()
        self.device = torch.device(device)
        self.embedder = PatchEmbedder(self.cfg.backbone, self.cfg.layers, pretrained).eval().to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None
        self.defect_types: DefectTypeClassifier | None = None

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
    def _raw_map(self, x: torch.Tensor, with_descriptor: bool = False):
        assert self.memory_bank is not None, "call fit() or load() first"
        emb, (h, w) = self.embed(x)
        b = emb.shape[0]
        dists = torch.cdist(emb.reshape(-1, emb.shape[-1]), self.memory_bank)
        min_dist, nearest = dists.min(dim=1)
        patch_scores = min_dist.reshape(b, 1, h, w)
        amap = F.interpolate(patch_scores, size=(self.cfg.image_size,) * 2, mode="bilinear", align_corners=False)
        k = int(4 * self.cfg.blur_sigma) * 2 + 1
        amap = TF.gaussian_blur(amap, kernel_size=k, sigma=self.cfg.blur_sigma).squeeze(1)
        if not with_descriptor:
            return amap
        # Defect descriptor: mean residual (patch feature - its nearest normal patch) over the
        # most anomalous patches. Reuses this step's features, so it is almost free.
        top = min_dist.reshape(b, h * w).topk(min(TOP_PATCHES, h * w), dim=1).indices
        feats = torch.gather(emb, 1, top[..., None].expand(-1, -1, emb.shape[-1]))
        normal = self.memory_bank[torch.gather(nearest.reshape(b, h * w), 1, top)]
        return amap, (feats - normal).mean(dim=1)

    def predict(self, image: np.ndarray) -> Prediction:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        maps, desc = self._raw_map(self.preprocess(images), with_descriptor=True)
        maps = (maps / self.threshold).cpu().numpy().astype(np.float32)
        out = []
        for m, d in zip(maps, desc.cpu().numpy()):
            score = float(m.max())
            pred = Prediction(score=score, is_defect=score > 1.0, anomaly_map=m, descriptor=d)
            if pred.is_defect and self.defect_types is not None:
                pred.defect_type, pred.type_confidence = self.defect_types.predict(d)
            out.append(pred)
        return out

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {"config": asdict(self.cfg), "memory_bank": self.memory_bank.cpu(), "threshold": self.threshold}
        if self.defect_types is not None:
            state["defect_types"] = self.defect_types.to_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu", backend: str = "torch") -> "PatchCore":
        """backend: "torch" (default), "onnx" (<model>_embedder.onnx next to the model file),
        "hailo" (HailoRT, hailo/<model>_embedder.hef) or "hailo-emu" (DFC INT8 emulator,
        hailo/<model>_embedder_optimized.har); scripts/compile_hailo.sh builds the last two."""
        state = torch.load(path, map_location=device)
        cfg = state["config"]
        cfg["layers"] = tuple(cfg["layers"])
        model = cls(DetectorConfig(**cfg), device=device)
        model.memory_bank = state["memory_bank"].to(device)
        model.threshold = state["threshold"]
        if state.get("defect_types"):
            model.defect_types = DefectTypeClassifier.from_dict(state["defect_types"])
        if backend == "onnx":
            model.embedder = OnnxEmbedder(Path(path).with_suffix("").as_posix() + "_embedder.onnx",
                                          threads=torch.get_num_threads())
        elif backend in ("hailo", "hailo-emu"):
            stem = Path(path).parent / "hailo" / f"{Path(path).stem}_embedder"
            model.embedder = (HailoEmbedder(f"{stem}.hef", "hailort") if backend == "hailo"
                              else HailoEmbedder(f"{stem}_optimized.har", "emulator"))
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
