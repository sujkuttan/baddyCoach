"""
MMAction2 adapter for stroke classification.

Provides a drop-in classifier with the same interface as BSTClassifier,
supporting PoseC3D (skeleton-based) and SlowFast (RGB-based) modes.

Gracefully handles missing dependencies — returns None from the getter
when MMAction2 is not installed, allowing the pipeline to fall back to BST-only.
"""

import logging
import numpy as np
from typing import Optional

from app.config.settings import settings

logger = logging.getLogger("mmaction_adapter")


# ── Lazy import helpers ─────────────────────────────────────────────────────

def _mmaction2_available() -> bool:
    """Check if MMAction2 and its dependencies can be imported."""
    try:
        import mmcv  # noqa: F401
        import mmengine  # noqa: F401
        from mmaction.apis import init_recognizer, inference_recognizer  # noqa: F401
        return True
    except ImportError:
        return False


def _pytorchvideo_available() -> bool:
    """Check if PyTorchVideo is available (lighter alternative)."""
    try:
        import pytorchvideo  # noqa: F401
        return True
    except ImportError:
        return False


# ── Adapter Class ────────────────────────────────────────────────────────────


class MMActionClassifier:
    """Stroke classifier backed by MMAction2 models.

    Supports three modes:
      - "posec3d": skeleton-based (uses existing pose keypoints from clip data)
      - "slowfast": RGB-based (requires video clip files saved to disk)
      - "pytorchvideo": lightweight RGB alternative via torch.hub

    The adapter normalizes outputs to match the BST 25-class ShuttleSet format
    so the post-processing pipeline (context fusion, physics, etc.) works unchanged.

    When MMAction2/PyTorchVideo is unavailable, all methods log a warning
    and return fallback values matching BST's "unknown" pattern.
    """

    def __init__(self, mode: str = "posec3d", device: str = "cpu",
                 num_classes: int = 25, seq_len: int = 48):
        self.mode = mode
        self.device = device
        self.num_classes = num_classes
        self.seq_len = seq_len  # PoseC3D expects 48-frame clips
        self.model = None
        self._available = False
        self._init_model()

    def _init_model(self):
        """Lazy-init the MMAction2 or PyTorchVideo model."""
        if self.mode == "pytorchvideo":
            self._init_pytorchvideo()
        elif self.mode in ("posec3d", "slowfast"):
            self._init_mmaction2()
        else:
            logger.warning("Unknown MMActionAdapter mode: %s", self.mode)

    def _init_pytorchvideo(self):
        """Load a SlowFast model via PyTorchVideo torch.hub."""
        if not _pytorchvideo_available():
            logger.warning(
                "PyTorchVideo not installed. Install with: pip install pytorchvideo"
            )
            return
        try:
            import torch
            model = torch.hub.load(
                "facebookresearch/pytorchvideo:main",
                model="slowfast_r50",
                pretrained=True,
            )
            # Replace classification head for our number of classes
            in_features = model.blocks[6].proj.in_features
            model.blocks[6].proj = torch.nn.Linear(in_features, self.num_classes)
            self.model = model.eval().to(self.device)
            self._available = True
            logger.info("PyTorchVideo SlowFast loaded (head replaced for %d classes)", self.num_classes)
        except Exception as e:
            logger.warning("Failed to load PyTorchVideo model: %s", e)

    def _init_mmaction2(self):
        """Load an MMAction2 model (PoseC3D or SlowFast)."""
        if not _mmaction2_available():
            logger.warning(
                "MMAction2 not installed. Install with: "
                "pip install openmim && mim install mmengine mmcv && "
                "pip install mmaction2"
            )
            return
        try:
            from mmengine import Config
            from mmaction.apis import init_recognizer

            if self.mode == "posec3d":
                self.model = self._build_posec3d()
            elif self.mode == "slowfast":
                self.model = self._build_slowfast()

            if self.model is not None:
                self._available = True
                logger.info("MMAction2 %s model loaded", self.mode)
        except Exception as e:
            logger.warning("Failed to load MMAction2 %s model: %s", self.mode, e)

    def _build_posec3d(self):
        """Build a PoseC3D model with skeleton-based config."""
        from mmengine import Config
        from mmaction.apis import init_recognizer

        cfg = Config(dict(
            model=dict(
                type='Recognizer3D',
                backbone=dict(
                    type='ResNet3dSlowOnly',
                    depth=50,
                    pretrained=None,
                    in_channels=17,
                    base_channels=32,
                    num_stages=3,
                    out_indices=(2,),
                    stage_blocks=(4, 6, 3),
                    conv1_stride_s=1,
                    pool1_stride_s=1,
                    inflate=(0, 1, 1),
                    spatial_strides=(2, 2, 2),
                    temporal_strides=(1, 1, 2),
                ),
                cls_head=dict(
                    type='I3DHead',
                    in_channels=512,
                    num_classes=self.num_classes,
                    dropout_ratio=0.5,
                    average_clips='prob',
                ),
                data_preprocessor=None,
            ),
            test_pipeline=[
                dict(type='UniformSampleFrames', clip_len=self.seq_len),
                dict(type='PoseDecode'),
                dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(-1, 64)),
                dict(type='CenterCrop', crop_size=(56, 56)),
                dict(type='FormatShape', input_format='NCHW_3D'),
                dict(type='PackActionInputs'),
            ],
        ))
        # PoseC3D is typically trained from scratch (no pretrained on Kinetics
        # for skeleton data). We init with random weights.
        return init_recognizer(cfg, checkpoint=None, device=self.device)

    def _build_slowfast(self):
        """Build a SlowFast model pre-trained on Kinetics-400."""
        from mmengine import Config
        from mmaction.apis import init_recognizer

        cfg = Config(dict(
            model=dict(
                type='Recognizer3D',
                backbone=dict(
                    type='ResNet3dSlowFast',
                    pretrained=None,
                    resample_rate=8,
                    speed_ratio=8,
                    channel_ratio=8,
                    slow_pathway=dict(
                        type='resnet3d', depth=50, pretrained=None,
                        lateral=True, conv1_kernel=(1, 7, 7),
                        dilations=(1, 1, 1, 1), inflate=(0, 0, 1, 1),
                    ),
                    fast_pathway=dict(
                        type='resnet3d', depth=50, pretrained=None,
                        lateral=False, base_channels=8,
                        conv1_kernel=(5, 7, 7),
                    ),
                ),
                cls_head=dict(
                    type='SlowFastHead',
                    in_channels=2304,
                    num_classes=self.num_classes,
                    average_clips='prob',
                ),
                data_preprocessor=dict(
                    type='ActionDataPreprocessor',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    format_shape='NCTHW',
                ),
            ),
        ))
        # Load Kinetics-400 pretrained weights
        ckpt_url = (
            "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/"
            "slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/"
            "slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
        )
        return init_recognizer(cfg, ckpt_url, device=self.device)

    # ── Public API (matches BSTClassifier interface) ──────────────────────

    def predict_from_clips(self, clips: list[dict], batch_size: int = 16,
                           return_probs: bool = True, **kwargs) -> list | tuple:
        """Predict stroke types from pre-built clip dicts.

        Args:
            clips: List of clip dicts with keys matching _build_clip() output.
            batch_size: Batch size for inference.
            return_probs: If True, return (results, probs_matrix).

        Returns:
            If return_probs:
                (list of 6-tuples, np.ndarray of softmax probs)
            Else:
                list of 6-tuples
            Each 6-tuple: (stroke_type, confidence, raw_class_id,
                           alpha, aim_attention_p0, aim_attention_p1)
            (alpha is 0.5 for MMAction2, aim_attention values are 0.0)
        """
        if not self._available or self.model is None:
            return self._fallback_all(clips, return_probs)

        if self.mode in ("posec3d",):
            return self._predict_posec3d(clips, batch_size, return_probs)
        elif self.mode in ("slowfast", "pytorchvideo"):
            return self._predict_rgb(clips, batch_size, return_probs)
        else:
            return self._fallback_all(clips, return_probs)

    def _predict_posec3d(self, clips: list[dict], batch_size: int,
                         return_probs: bool) -> list | tuple:
        """PoseC3D inference: extracts keypoints from clip JnB data.

        PoseC3D expects single-person keypoint heatmaps. We run inference
        on the "near" player (p_idx=1) per clip, which typically has better
        pose coverage.
        """
        from mmaction.apis import inference_recognizer
        import torch

        n_clips = len(clips)
        results = []
        probs_list = []

        for i, clip in enumerate(clips):
            try:
                # Extract keypoints for the near player (p_idx=1) from JnB
                # JnB shape: (seq_len, 2, 72) → first 34 channels are joint coords
                jnb = clip['JnB']  # (T, 2, 72)
                joints_near = jnb[:, 1, :34].reshape(-1, 17, 2)  # (T, 17, 2)

                # Resample to PoseC3D's expected seq_len (48) if needed
                T = joints_near.shape[0]
                if T != self.seq_len:
                    from scipy import interpolate
                    orig = np.linspace(0, T - 1, T)
                    target = np.linspace(0, T - 1, self.seq_len)
                    joints_resampled = np.zeros((self.seq_len, 17, 2))
                    for j in range(17):
                        for k in range(2):
                            joints_resampled[:, j, k] = np.interp(
                                target, orig, joints_near[:, j, k]
                            )
                    joints_near = joints_resampled

                # Build pseudo heatmap volume for PoseC3D
                # Shape: (17, T, 56, 56) → 17 keypoint channels
                # We create simple gaussian blobs
                H = W = 56
                heatmaps = np.zeros((17, self.seq_len, H, W), dtype=np.float32)
                for j in range(17):
                    for t in range(self.seq_len):
                        x = int(np.clip(joints_near[t, j, 0] * W, 0, W - 1))
                        y = int(np.clip(joints_near[t, j, 1] * H, 0, H - 1))
                        heatmaps[j, t, y, x] = 1.0

                # Pack as expected by PoseC3D inference API
                # We bypass the full pipeline and run the backbone directly
                # since our heatmap format is already pre-processed
                inp = torch.from_numpy(heatmaps).float().unsqueeze(0).to(self.device)
                # inp shape: (1, 17, 48, 56, 56)
                with torch.no_grad():
                    logits = self.model.backbone(inp)
                    # logits shape: (1, 512, 12, 28, 28) after stage 3
                    logits = logits.mean(dim=[2, 3, 4])  # global avg pool
                    logits = self.model.cls_head(logits)  # (1, num_classes)
                    logits_np = logits.float().cpu().numpy()[0]

            except Exception as e:
                logger.warning("PoseC3D inference error on clip %d: %s", i, e)
                logits_np = np.zeros(self.num_classes)

            probs = np.exp(logits_np - logits_np.max())
            probs = probs / (probs.sum() + 1e-8)
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            from app.models.bst import map_to_coach_class
            stroke_type = map_to_coach_class(pred_idx)

            results.append((stroke_type, confidence, pred_idx,
                            0.5, 0.0, 0.0))
            probs_list.append(probs)

        if return_probs:
            probs_matrix = np.stack(probs_list) if probs_list else np.zeros((0, self.num_classes))
            return results, probs_matrix
        return results

    def _predict_rgb(self, clips: list[dict], batch_size: int,
                     return_probs: bool) -> list | tuple:
        """RGB-based inference (SlowFast).

        Requires video clip files saved to disk. Falls back to unknown
        if video clips are unavailable.
        """
        logger.warning(
            "RGB-based MMAction inference requires video clip files. "
            "Use posec3d mode or provide clips with 'video_path' key."
        )
        return self._fallback_all(clips, return_probs)

    def _fallback_all(self, clips: list[dict], return_probs: bool):
        """Return all-unknown results (adapter unavailable or mode unsupported)."""
        n = len(clips)
        results = [("unknown", 0.0, 0, 0.5, 0.0, 0.0)] * n
        probs = np.zeros((n, self.num_classes))
        if return_probs:
            return results, probs
        return results

    def predict_single(self, clip: dict) -> tuple:
        """Predict for a single clip."""
        results, _ = self.predict_from_clips([clip], return_probs=True)
        return results[0]


# ── Factory ──────────────────────────────────────────────────────────────────


def create_mmaction_classifier(mode: str = "posec3d",
                                device: str = "cpu") -> MMActionClassifier | None:
    """Factory: create an MMActionClassifier if dependencies are available.

    Args:
        mode: "posec3d", "slowfast", or "pytorchvideo"
        device: "cpu" or "cuda"

    Returns:
        MMActionClassifier instance, or None if dependencies unavailable.
    """
    if mode in ("slowfast", "posec3d"):
        if not _mmaction2_available():
            logger.warning(
                "MMAction2 not installed — cannot create %s classifier. "
                "Install: pip install openmim && mim install mmengine mmcv && pip install mmaction2",
                mode,
            )
            return None
    elif mode == "pytorchvideo":
        if not _pytorchvideo_available():
            logger.warning(
                "PyTorchVideo not installed — cannot create pytorchvideo classifier. "
                "Install: pip install pytorchvideo",
            )
            return None
    else:
        logger.warning("Unknown MMAction mode: %s", mode)
        return None

    return MMActionClassifier(mode=mode, device=device,
                               num_classes=settings.bst_n_classes if hasattr(settings, 'bst_n_classes') else 25)
