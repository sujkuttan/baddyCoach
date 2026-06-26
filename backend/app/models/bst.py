"""BST classifier for stroke classification.

Integrates the official BST_CG / BST_CG_AP model from:
https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger("bst")


COACH_STROKE_CLASSES = [
    "net_shot", "block", "smash", "lift", "clear", "drive",
    "drop", "push", "rush", "cross_court", "short_serve", "long_serve"
]

SHUTTLESET_CLASSES = [
    'unknown', 'Top_net_shot', 'Top_block', 'Top_smash', 'Top_lift',
    'Top_clear', 'Top_drive', 'Top_drop', 'Top_push', 'Top_rush',
    'Top_cross_court', 'Top_short_serve', 'Top_long_serve',
    'Bottom_net_shot', 'Bottom_block', 'Bottom_smash', 'Bottom_lift',
    'Bottom_clear', 'Bottom_drive', 'Bottom_drop', 'Bottom_push',
    'Bottom_rush', 'Bottom_cross_court', 'Bottom_short_serve', 'Bottom_long_serve'
]


def map_to_coach_class(shuttleset_class_id: int) -> str:
    """Map ShuttleSet class ID to simplified coaching class."""
    if shuttleset_class_id == 0:
        return "unknown"
    elif 1 <= shuttleset_class_id <= 12:
        return COACH_STROKE_CLASSES[shuttleset_class_id - 1]
    elif 13 <= shuttleset_class_id <= 24:
        base_class = COACH_STROKE_CLASSES[shuttleset_class_id - 13]
        return base_class
    return "unknown"


def get_shuttleset_class_info(class_id: int) -> tuple:
    """Return (stroke_type, side) from ShuttleSet class ID.

    side is 'top', 'bottom', or None for unknown.
    """
    if class_id == 0:
        return "unknown", None
    elif 1 <= class_id <= 12:
        return COACH_STROKE_CLASSES[class_id - 1], "top"
    elif 13 <= class_id <= 24:
        return COACH_STROKE_CLASSES[class_id - 13], "bottom"
    return "unknown", None


class BSTClassifier:
    """BST classifier using the official BST_CG / BST_CG_AP model.

    Supports loading from the official weight files and inference
    with proper preprocessing. Batch size is auto-detected from
    available GPU VRAM via gpu_batch config.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda",
                 default_seq_len: int = 30, batch_size: Optional[int] = None,
                 temperature: Optional[float] = None, adapt_batchnorm: bool = False):
        self.device = device
        self.model = None
        self.seq_len = default_seq_len
        self.n_classes = 25
        self.classes = COACH_STROKE_CLASSES
        self.batch_size = batch_size if batch_size is not None else self._default_batch_size()
        self.temperature = temperature if temperature is not None else 1.0
        self.adapt_batchnorm = adapt_batchnorm

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    @staticmethod
    def _default_batch_size() -> int:
        try:
            from app.config.gpu_batch import get_gpu_batch_config
            cfg = get_gpu_batch_config()
            return int(cfg.get("bst_batch", 32))
        except Exception:
            return 32

    def _load_model(self, path: str):
        """Load BST_CG or BST_CG_AP model from checkpoint."""
        from app.pipeline.shared.models import _checked_load, record_model_health
        try:
            import torch
            from app.models.bst_model import BST_CG, BST_CG_AP

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if not isinstance(checkpoint, dict):
                logger.error("BST checkpoint format not recognized: %s", type(checkpoint))
                record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                            "n_missing": 0, "n_unexpected": 0,
                                            "core_missing": ["checkpoint not a dict"]})
                return

            state_dict = checkpoint

            # Detect dimensions from state_dict
            in_dim = 72
            seq_len = None
            n_classes = 25

            for k, v in state_dict.items():
                if 'tcn_pose.net.0.weight' in k:
                    in_dim = v.shape[1]
                if 'mlp_head.mlp.mlp.3.weight' in k:
                    n_classes = v.shape[0]
                if 'embedding_tem' in k:
                    seq_len = v.shape[1] - 1

            if n_classes != 25:
                logger.warning("BST checkpoint has %d output classes, expected 25. "
                               "Predictions will be misaligned with SHUTTLESET_CLASSES mapping.", n_classes)

            if seq_len is None:
                logger.error("BST checkpoint missing embedding_tem, cannot determine seq_len")
                record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                            "n_missing": 0, "n_unexpected": 0,
                                            "core_missing": ["missing embedding_tem"]})
                return

            has_positions = any('mlp_positions' in k for k in state_dict)
            if not has_positions:
                logger.error("BST checkpoint missing mlp_positions, cannot load")
                record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                            "n_missing": 0, "n_unexpected": 0,
                                            "core_missing": ["missing mlp_positions"]})
                return

            # Always use BST_CG_AP (AimPlayer) — all 94-key checkpoints (CG, CG_AP,
            # ckpts/bst) have identical state dicts since nn.CosineSimilarity has no
            # trainable params. BST_CG_AP is a strict superset of BST_CG with added
            # AimPlayer cosine-similarity weighting that improves player-aware
            # classification. No extra parameters needed beyond BST_CG's.
            model_class = BST_CG_AP
            model = model_class(
                in_dim=in_dim,
                seq_len=seq_len,
                n_class=n_classes,
                d_model=100,
                d_head=128,
                n_head=6,
                depth_tem=2,
                depth_inter=1,
            )

            status = _checked_load(model, state_dict,
                                   core_prefixes=("tcn_pose", "mlp_head", "embedding_tem"))
            record_model_health("bst", status)

            if not status["loaded"]:
                logger.warning("BST core layers missing (%s). Model set to None.", status['core_missing'])
                return

            model.to(self.device).eval()

            self.model = model
            self.seq_len = seq_len
            self.n_classes = n_classes
            self._load_temperature()
            logger.info("BST loaded: class=BST_CG_AP, in_dim=%d, seq_len=%d, n_classes=%d, temperature=%.3f",
                        in_dim, seq_len, n_classes, self.temperature)
        except Exception as e:
            logger.error("BST load error: %s", e)
            import traceback
            traceback.print_exc()
            record_model_health("bst", {"loaded": False, "missing_frac": 1.0,
                                        "n_missing": 0, "n_unexpected": 0,
                                        "core_missing": [str(e)]})

    TEMPERATURE_CACHE = None

    @classmethod
    def _temperature_cache_path(cls) -> Optional[Path]:
        try:
            from app.pipeline.shared.models import CKPT_DIR
            return CKPT_DIR / "bst" / "bst_temperature.json"
        except Exception:
            return None

    def _load_temperature(self):
        """Load cached temperature, unless overridden by constructor param.

        NOTE: The cached temperature (T=1.4224 as of 2025-06-26) was computed
        from 12-class calibration data with broken InpaintNet features. After
        the InpaintNet/homography fix, features are in proper court-space and
        the logit distribution has changed significantly. Re-run calibration:

            python -c "
            import json, pandas as pd, numpy as np
            from app.models.bst import BSTClassifier
            df = pd.read_parquet('results/mmpose_results/debug/debug_bst_outputs.parquet')
            logits = np.array([json.loads(s) for s in df['logits_all']])
            labels = df['pred_class_id'].values
            T = BSTClassifier.compute_optimal_temperature(logits, labels)
            BSTClassifier._save_temperature(T)
            "
        """
        if self.temperature != 1.0:
            return
        cache_path = self._temperature_cache_path()
        if cache_path and cache_path.exists():
            try:
                import json
                with open(cache_path) as f:
                    data = json.load(f)
                cached = float(data.get("temperature", 1.0))
                self.temperature = cached
                logger.info("Loaded cached temperature: T=%.3f", cached)
                logger.info("NOTE: This temperature may be stale after InpaintNet fix. "
                           "Re-calibrate via scripts/calibrate_bst.py or the inline recipe in _load_temperature docstring.")
            except Exception as e:
                logger.warning("Could not load cached temperature: %s", e)

    @staticmethod
    def _save_temperature(temperature: float):
        """Persist calibrated temperature to cache file."""
        cache_path = BSTClassifier._temperature_cache_path()
        if cache_path is None:
            return
        try:
            import json
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"temperature": float(temperature)}, f)
            logger.info("Saved temperature T=%.3f -> %s", temperature, cache_path)
        except Exception as e:
            logger.warning("Could not save temperature: %s", e)

    @staticmethod
    def compute_optimal_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal temperature via NLL minimization (LBFGS).

        Args:
            logits: (N, n_classes) pre-softmax logits.
            labels: (N,) integer class labels.

        Returns:
            Optimal temperature scalar (T > 0). Returns 1.0 if optimization fails.
        """
        try:
            import torch
        except ImportError:
            logger.warning("Torch not available for temperature calibration")
            return 1.0

        try:
            logits_t = torch.from_numpy(logits).float()
            labels_t = torch.from_numpy(labels).long()
            nll = torch.nn.CrossEntropyLoss()

            T = torch.ones(1, requires_grad=True)
            optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

            def closure():
                optimizer.zero_grad()
                loss = nll(logits_t / T, labels_t)
                loss.backward()
                return loss

            optimizer.step(closure)

            T_opt = float(T.detach().item())
            if T_opt <= 0:
                return 1.0
            return max(0.01, min(T_opt, 100.0))
        except Exception as e:
            logger.warning("Temperature optimization failed: %s", e)
            return 1.0

    def predict_from_clips(self, clips: list, batch_size: Optional[int] = None,
                           debug_collector: Optional[list] = None) -> list:
        """Predict stroke types from prepared BST clips.

        Args:
            clips: List of dicts with keys: JnB, shuttle, pos, video_len
            batch_size: Number of clips to process in parallel.
                       Defaults to self.batch_size (auto-detected from GPU VRAM).
            debug_collector: If provided, append per-shot debug dicts with
                             full softmax distribution and feature stats.

        Returns:
            List of (stroke_type, confidence, raw_class_id) tuples
        """
        batch_size = batch_size or self.batch_size
        if self.model is None:
            return [(self._rule_based_predict(clip), 0.5, 0) for clip in clips]

        import torch

        # Adapt BatchNorm to input distribution when using non-bbox normalization
        # (e.g., court-space). Uses batch statistics instead of running stats
        # so the TCN's BatchNorm layers normalize the shifted feature distribution.
        _bn_restore = []
        if self.adapt_batchnorm and self.model is not None:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    _bn_restore.append((m, m.track_running_stats))
                    m.track_running_stats = False

        results = [None] * len(clips)

        for batch_start in range(0, len(clips), batch_size):
            batch_end = min(batch_start + batch_size, len(clips))
            batch_clips = clips[batch_start:batch_end]

            try:
                JnB_np = np.stack([c['JnB'] for c in batch_clips])
                shuttle_np = np.stack([c['shuttle'] for c in batch_clips])
                pos_np = np.stack([c['pos'] for c in batch_clips])

                JnB = torch.from_numpy(JnB_np).float().to(self.device)
                shuttle = torch.from_numpy(shuttle_np).float().to(self.device)
                pos = torch.from_numpy(pos_np).float().to(self.device)
                video_len = torch.tensor(
                    [c['video_len'] for c in batch_clips], dtype=torch.long
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(JnB, shuttle, pos, video_len)
                    logits_np = logits.float().cpu().numpy()
                    probs = torch.softmax(logits.float() / self.temperature, dim=1).cpu().numpy()

                for j in range(len(batch_clips)):
                    prob_dist = probs[j]
                    pred_idx = int(np.argmax(prob_dist))
                    confidence = float(prob_dist[pred_idx])

                    # Per-clip feature stats (NOT batch-level — each clip has
                    # different joint positions and shuttle trajectories)
                    clip_jnb = batch_clips[j]['JnB']
                    jnb_min = float(clip_jnb.min())
                    jnb_max = float(clip_jnb.max())
                    jnb_zero_frac = float((clip_jnb == 0.0).mean())

                    debug_info = None
                    if debug_collector is not None:
                        logit_class_0 = float(logits_np[j, 0])
                        logit_max = float(logits_np[j].max())
                        sorted_idxs = np.argsort(prob_dist)[::-1]
                        top5 = [(int(sorted_idxs[k]), float(prob_dist[sorted_idxs[k]]))
                                for k in range(5)]

                        debug_info = {
                            "pred_class_id": pred_idx,
                            "pred_confidence": confidence,
                            "logit_class_0": logit_class_0,
                            "logit_max": logit_max,
                            "top5": top5,
                            "logits_all": json.dumps([float(v) for v in logits_np[j]]),
                            "jnb_zero_frac": jnb_zero_frac,
                            "jnb_min": jnb_min,
                            "jnb_max": jnb_max,
                        }

                    if pred_idx == 0:
                        second_idx = int(np.argsort(prob_dist)[-2])
                        second_conf = float(prob_dist[second_idx])
                        if second_conf > 0.3:
                            pred_idx = second_idx
                            confidence = second_conf
                            if debug_info:
                                debug_info["is_second_best_override"] = True
                                debug_info["second_best_class_id"] = second_idx
                                debug_info["second_best_confidence"] = second_conf
                        else:
                            fallback = self._rule_based_predict(batch_clips[j])
                            rule_conf = min(confidence, 0.3)
                            if debug_info:
                                debug_info["is_rule_based"] = True
                                debug_info["fallback_stroke_type"] = fallback
                            if debug_collector is not None:
                                debug_collector.append(debug_info)
                            results[batch_start + j] = (fallback, rule_conf, 0)
                            continue

                    stroke_type = map_to_coach_class(pred_idx)
                    if debug_info:
                        debug_info["stroke_type"] = stroke_type
                    if debug_collector is not None:
                        debug_collector.append(debug_info)
                    results[batch_start + j] = (stroke_type, confidence, pred_idx)

            except Exception as e:
                logger.error("BST batch inference error: %s", e)
                for j in range(len(batch_clips)):
                    if results[batch_start + j] is None:
                        fallback = self._rule_based_predict(batch_clips[j])
                        if debug_collector is not None:
                            debug_collector.append({
                                "pred_class_id": 0,
                                "pred_confidence": 0.0,
                                "is_rule_based": True,
                                "fallback_stroke_type": fallback,
                                "error": str(e),
                            })
                        results[batch_start + j] = (fallback, 0.5, 0)

        # Restore BatchNorm running stats
        for m, prev in _bn_restore:
            m.track_running_stats = prev

        # Log class activation warning (Fix 3: detect ordering mismatch)
        if self.model is not None and hasattr(self, 'n_classes'):
            activated = set(r[2] for r in results if r[2] != 0)
            all_classes = set(range(self.n_classes))
            never_activated = all_classes - activated
            if never_activated and len(never_activated) > len(all_classes) * 0.5:
                logger.warning(
                    "BST: %d/%d classes never activated (%s). Possible class ordering mismatch.",
                    len(never_activated), len(all_classes),
                    sorted(never_activated),
                )

        return results

    def predict_single(self, clip: dict) -> tuple:
        """Predict stroke type for a single clip.

        Returns:
            (stroke_type, confidence, raw_class_id)
        """
        results = self.predict_from_clips([clip])
        return results[0] if results else ("unknown", 0.0, 0)

    def _rule_based_predict(self, clip: dict) -> str:
        """Fallback rule-based prediction using shuttle trajectory.

        The clip shuttle is already in court-normalized [0,1] range:
          channel 0 = court_x / court_length  (0=far end, 1=near end)
          channel 1 = court_y / court_width   (0=left, 1=right)

        These map approximately 1:1 to pixel-normalized coordinates:
          court_x (far→near) ≈ pixel_y (top→bottom)
          court_y (left→right) ≈ pixel_x (left→right)

        So the thresholds (designed for pixel-space [0,1]) work directly
        on shuttle without any extra conversion.

        Only the POST-HIT half of the trajectory is analyzed to avoid the
        V-shaped averaging problem from between-2-hits clips (the trajectory
        reverses direction at the hit point).
        """
        seq_len = self.seq_len if self.seq_len is not None else 30
        shuttle = clip.get('shuttle', np.zeros((seq_len, 2)))

        if len(shuttle) < 2:
            return "unknown"

        # Use only post-hit half of the trajectory
        mid = len(shuttle) // 2
        post_hit = shuttle[mid:]

        valid = (post_hit[:, 0] != 0) | (post_hit[:, 1] != 0)
        if valid.sum() < 2:
            return "unknown"
        valid_traj = post_hit[valid]

        dy = np.diff(valid_traj[:, 1])
        dx = np.diff(valid_traj[:, 0])
        speed = np.sqrt(dx**2 + dy**2)

        mean_speed = float(np.mean(speed))
        max_speed = float(np.max(speed))
        mean_dy = float(np.mean(dy))
        end_y = float(valid_traj[-1, 1])

        if max_speed > 0.15 and mean_dy > 0.05:
            return "smash"
        elif mean_speed < 0.03:
            return "net_shot"
        elif mean_dy < -0.03 and mean_speed > 0.05:
            return "clear"
        elif mean_speed > 0.08 and abs(mean_dy) < 0.02:
            return "drive"
        elif mean_dy > 0.04 and mean_speed > 0.05 and end_y > 0.5:
            return "lift"
        elif end_y > 0.7 and mean_speed < 0.06:
            return "drop"
        else:
            return "unknown"


def normalize_shuttlecock(arr: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    """Normalize shuttlecock position by video resolution."""
    from app.pipeline.shared.bst_preproc import normalize_shuttlecock as _ns
    return _ns(arr, v_width, v_height)


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    center_align: bool = True
) -> np.ndarray:
    """Normalize joints by bounding box diagonal distance (batched)."""
    from app.pipeline.shared.bst_preproc import normalize_joints_batched
    return normalize_joints_batched(arr, bbox, center_align)
