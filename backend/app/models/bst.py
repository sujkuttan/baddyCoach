"""BST classifier for stroke classification.

Integrates the official BST_CG / BST_CG_AP model from:
https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.pipeline.shared.stroke_features import (
    extract_clip_features, classify_family, classify_by_family,
    estimate_confidence, _build_evidence, top3_alternatives,
)
from app.pipeline.shared.bst_validator import BSTInputValidator

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

        Calibration is now exclusively via scripts/calibrate_bst.py
        with ground-truth labels from the labeling UI. The degenerate
        self-label path (fitting T against pred_class_id) has been
        retired — it drives T→0 and measures nothing.

        See _load_temperature docstring for the old auto-fit recipe
        (removed 2025-06-27).
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
    def _load_logit_bias(n_classes: int) -> Optional[np.ndarray]:
        """Load precomputed per-class logit bias from JSON, mean-centered.

        Uses settings.bst_logit_bias_path. Returns a (n_classes,) bias
        vector (mean-centered), or None if the file doesn't exist or
        has wrong shape.
        """
        path = settings.bst_logit_bias_path
        if path is None or not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            bias = np.array(data["bias"], dtype=np.float64)
            if bias.shape != (n_classes,):
                logger.warning(
                    "BST logit bias shape mismatch: got %s, expected (%d,). Ignoring.",
                    bias.shape, n_classes,
                )
                return None
            n_clips = data.get("n_clips", "?")
            source = data.get("source", "?")
            logger.info("Loaded BST logit bias (%d clips, source=%s)", n_clips, source)
            return bias - bias.mean()  # mean-center so overall logit scale is preserved
        except Exception as e:
            logger.warning("Could not load BST logit bias: %s", e)
            return None

    def _apply_prior_correction(self, logits: np.ndarray) -> np.ndarray:
        """Remove constant per-class logit bias from all clips.

        Two strategies (in priority order):
        1. Precomputed bias file → use it.
        2. Self-calibrate from current run if enough clips.

        Returns corrected logits (same shape). When neither applies,
        returns logits unchanged.
        """
        if not settings.bst_prior_correction_enabled:
            return logits

        n_classes = logits.shape[1]
        bias = self._load_logit_bias(n_classes)

        if bias is None:
            n_clips = len(logits)
            if n_clips >= settings.bst_prior_min_clips:
                bias = logits.mean(axis=0)
                bias = bias - bias.mean()
                logger.info(
                    "BST self-calibrated bias from %d clips (%.4f range)",
                    n_clips, float(bias.max() - bias.min()),
                )
            else:
                logger.warning(
                    "BST prior-correction skipped: %d clips < %d and no bias file",
                    n_clips, settings.bst_prior_min_clips,
                )
                return logits

        a = settings.bst_prior_correction_strength
        if a == 0.0:
            return logits

        corrected = logits - a * bias[np.newaxis, :]
        return corrected

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
                           debug_collector: Optional[list] = None,
                           return_probs: bool = False) -> list:
        """Predict stroke types from prepared BST clips.

        Two-pass design for prior correction:
        1. Collect raw logits across all batches
        2. Apply per-class bias correction + softmax(/T) + argmax + overrides

        Args:
            clips: List of dicts with keys: JnB, shuttle, pos, video_len
            batch_size: Number of clips to process in parallel.
                       Defaults to self.batch_size (auto-detected from GPU VRAM).
            debug_collector: If provided, append per-shot debug dicts with
                             full softmax distribution and feature stats.
            return_probs: If True, also return corrected probability matrix
                          of shape (n_clips, n_classes) for downstream ensemble.

        Returns:
            List of (stroke_type, confidence, raw_class_id, alpha,
                     aim_attention_p0, aim_attention_p1) tuples.
            alpha ∈ [0, 1]: >0.5 = far player (p1), <0.5 = near player (p2).
            0.5 = uncertain / no model available.
            aim_attention_p0/p1: cos(p0/p1_shuttle_CLS, shuttle_CLS), raw attention.
            If return_probs=True, returns (results, probs_matrix).
        """
        batch_size = batch_size or self.batch_size
        if self.model is None:
            results = []
            for clip in clips:
                st, conf, ev, top3 = self._rule_based_predict(clip)
                results.append((st, conf, 0, 0.5, 0.0, 0.0))
                if debug_collector is not None:
                    debug_collector.append({
                        "pred_class_id": 0,
                        "pred_confidence": conf,
                        "is_rule_based": True,
                        "fallback_stroke_type": st,
                        "rule_evidence": ev,
                        "rule_top3": top3,
                    })
            if return_probs:
                n_classes = getattr(self, 'n_classes', 25)
                return results, np.zeros((len(clips), n_classes))
            return results

        import torch

        # Adapt BatchNorm to input distribution when using non-bbox normalization
        _bn_restore = []
        if self.adapt_batchnorm and self.model is not None:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    _bn_restore.append((m, m.track_running_stats))
                    m.track_running_stats = False

        # ── Lazy-create the input validator ───────────────────────────
        if not hasattr(self, '_validator') or self._validator is None:
            self._validator = BSTInputValidator(
                seq_len=self.seq_len,
                n_classes=getattr(self, 'n_classes', 25),
                shuttle_norm=settings.bst_shuttle_norm,
                joint_norm=settings.bst_joint_norm,
                level=settings.bst_validation_level,
                clip_boundary=settings.bst_clip_boundary,
            )

        # ── Per-clip validation (before batching) ─────────────────────
        for i, clip in enumerate(clips):
            debug = clip.get("_debug_clip", {})
            frame_range = f"frames {debug.get('frame_start', '?')}-{debug.get('frame_end', '?')}"
            try:
                clip_result = self._validator.validate_clip(clip)
                if clip_result.n_warnings > 0 or clip_result.n_errors > 0:
                    msg = clip_result.warnings[0] if clip_result.warnings else clip_result.errors[0]
                    logger.warning(
                        "BST clip %d %s: %s",
                        i, frame_range, msg.replace("BST VALIDATION: ", "").replace("BST VALIDATION FAIL: ", ""),
                    )
            except Exception as e:
                logger.warning("BST clip %d %s validation error: %s", i, frame_range, e)

        n_clips = len(clips)
        raw_logits_list = [None] * n_clips
        clip_data = [None] * n_clips  # per-clip metadata for the second pass
        alpha_list = [0.5] * n_clips  # AimPlayer alpha per clip
        p1_sim_list = [0.0] * n_clips  # cos(p0_shuttle_CLS, shuttle_CLS)
        p2_sim_list = [0.0] * n_clips  # cos(p1_shuttle_CLS, shuttle_CLS)

        # ── Pass 1: collect raw logits + alpha ────────────────────────
        for batch_start in range(0, n_clips, batch_size):
            batch_end = min(batch_start + batch_size, n_clips)
            batch_clips = clips[batch_start:batch_end]

            try:
                JnB_np = np.stack([c['JnB'] for c in batch_clips])
                shuttle_np = np.stack([c['shuttle'] for c in batch_clips])
                pos_np = np.stack([c['pos'] for c in batch_clips])

                # ── Batch-level validation just before model call ─────
                try:
                    self._validator.validate_batch(JnB_np, shuttle_np, pos_np)
                except Exception as e:
                    logger.warning(
                        "BST batch %d-%d pre-inference validation error: %s",
                        batch_start, batch_end - 1, e,
                    )

                JnB = torch.from_numpy(JnB_np).float().to(self.device)
                shuttle = torch.from_numpy(shuttle_np).float().to(self.device)
                pos = torch.from_numpy(pos_np).float().to(self.device)
                video_len = torch.tensor(
                    [c['video_len'] for c in batch_clips], dtype=torch.long
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(JnB, shuttle, pos, video_len)
                    logits_np = logits.float().cpu().numpy()
                    if hasattr(self.model, '_last_alpha') and self.model._last_alpha is not None:
                        alpha_np = self.model._last_alpha.float().cpu().numpy()
                    else:
                        alpha_np = np.full(len(batch_clips), 0.5)

                    if hasattr(self.model, '_last_p1_sim') and self.model._last_p1_sim is not None:
                        p1_sim_np = self.model._last_p1_sim.float().cpu().numpy()
                    else:
                        p1_sim_np = np.zeros(len(batch_clips))
                    if hasattr(self.model, '_last_p2_sim') and self.model._last_p2_sim is not None:
                        p2_sim_np = self.model._last_p2_sim.float().cpu().numpy()
                    else:
                        p2_sim_np = np.zeros(len(batch_clips))

                for j in range(len(batch_clips)):
                    idx = batch_start + j
                    raw_logits_list[idx] = logits_np[j]
                    clip_data[idx] = batch_clips[j]
                    alpha_list[idx] = float(alpha_np[j])
                    p1_sim_list[idx] = float(p1_sim_np[j])
                    p2_sim_list[idx] = float(p2_sim_np[j])

            except Exception as e:
                logger.error("BST batch inference error at clip %d: %s", batch_start, e)
                for j in range(len(batch_clips)):
                    idx = batch_start + j
                    raw_logits_list[idx] = "error"
                    alpha_list[idx] = 0.5
                    p1_sim_list[idx] = 0.0
                    p2_sim_list[idx] = 0.0

        # ── Apply prior correction to all collected logits ─────────────
        valid_mask = [r is not None and isinstance(r, np.ndarray) for r in raw_logits_list]
        if any(valid_mask):
            all_logits = np.stack([raw_logits_list[i] for i in range(n_clips) if valid_mask[i]])
            corrected = self._apply_prior_correction(all_logits)
        else:
            corrected = None

        # ── Pass 2: argmax / softmax / second-best / fallback ─────────
        n_classes = getattr(self, 'n_classes', 25)
        results = [None] * n_clips
        probs_list = [None] * n_clips  # for return_probs
        corr_idx = 0
        for i in range(n_clips):
            if not valid_mask[i]:
                fallback, rb_conf, rb_ev, rb_top3 = self._rule_based_predict(clips[i])
                if debug_collector is not None:
                    debug_collector.append({
                        "pred_class_id": 0,
                        "pred_confidence": rb_conf,
                        "is_rule_based": True,
                        "fallback_stroke_type": fallback,
                        "rule_evidence": rb_ev,
                        "rule_top3": rb_top3,
                    })
                results[i] = (fallback, rb_conf, 0, alpha_list[i], p1_sim_list[i], p2_sim_list[i])
                probs_list[i] = np.zeros(n_classes)
                continue

            logits_np = corrected[corr_idx] if corrected is not None else raw_logits_list[i]
            corr_idx += 1

            probs = np.exp(logits_np / self.temperature)
            probs = probs / probs.sum()

            probs_list[i] = probs

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            clip_jnb = clip_data[i]['JnB']
            jnb_min = float(clip_jnb.min())
            jnb_max = float(clip_jnb.max())
            jnb_zero_frac = float((clip_jnb == 0.0).mean())

            debug_info = None
            if debug_collector is not None:
                logit_class_0 = float(logits_np[0])
                logit_max = float(logits_np.max())
                sorted_idxs = np.argsort(probs)[::-1]
                top5 = [(int(sorted_idxs[k]), float(probs[sorted_idxs[k]]))
                        for k in range(5)]

                debug_info = {
                    "pred_class_id": pred_idx,
                    "pred_confidence": confidence,
                    "logit_class_0": logit_class_0,
                    "logit_max": logit_max,
                    "top5": top5,
                    "logits_all": json.dumps([float(v) for v in logits_np]),
                    "jnb_zero_frac": jnb_zero_frac,
                    "jnb_min": jnb_min,
                    "jnb_max": jnb_max,
                }

            if pred_idx == 0:
                second_idx = int(np.argsort(probs)[-2])
                second_conf = float(probs[second_idx])
                if second_conf > 0.3:
                    pred_idx = second_idx
                    confidence = second_conf
                    if debug_info:
                        debug_info["is_second_best_override"] = True
                        debug_info["second_best_class_id"] = second_idx
                        debug_info["second_best_confidence"] = second_conf
                else:
                    fallback, rule_conf, ev, top3 = self._rule_based_predict(clip_data[i])
                    rule_conf = min(rule_conf, 0.3)
                    if debug_info:
                        debug_info["is_rule_based"] = True
                        debug_info["fallback_stroke_type"] = fallback
                        debug_info["rule_evidence"] = ev
                        debug_info["rule_top3"] = top3
                    if debug_collector is not None:
                        debug_collector.append(debug_info)
                    results[i] = (fallback, rule_conf, 0, alpha_list[i], p1_sim_list[i], p2_sim_list[i])
                    continue

            stroke_type = map_to_coach_class(pred_idx)
            if debug_info:
                debug_info["stroke_type"] = stroke_type
                debug_info["aimplayer_alpha"] = alpha_list[i]
            if debug_collector is not None:
                debug_collector.append(debug_info)
            results[i] = (stroke_type, confidence, pred_idx, alpha_list[i], p1_sim_list[i], p2_sim_list[i])

        # Restore BatchNorm running stats
        for m, prev in _bn_restore:
            m.track_running_stats = prev

        # Log class activation warning
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

        if return_probs:
            probs_matrix = np.stack([p if p is not None else np.zeros(n_classes) for p in probs_list])
            return results, probs_matrix
        return results

    def predict_single(self, clip: dict) -> tuple:
        """Predict stroke type for a single clip.

        Returns:
            (stroke_type, confidence, raw_class_id, alpha,
             aim_attention_p0, aim_attention_p1)
        """
        results = self.predict_from_clips([clip])
        return results[0] if results else ("unknown", 0.0, 0, 0.5, 0.0, 0.0)

    def _rule_based_predict(self, clip: dict) -> tuple:
        """Hierarchical rule-based prediction using stroke_features module.

        Implements the spec's two-level classifier:
          1. classify_family → family (overhead/underhand/net/mid_height/serve)
          2. classify_by_family → specific stroke within that family

        Returns:
            (stroke_type, confidence, evidence_dict, top3_list)
        """
        feats = extract_clip_features(clip)
        if not feats.get('usable', False):
            return ('unknown', 0.10, {}, [])

        family = classify_family(feats)
        stroke = classify_by_family(family, feats)
        confidence = estimate_confidence(stroke, feats)
        evidence = _build_evidence(stroke, feats)
        top3 = top3_alternatives(feats, stroke)

        return (stroke, confidence, evidence, top3)


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
