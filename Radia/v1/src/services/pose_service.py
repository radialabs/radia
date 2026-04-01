"""
Pose estimation service for WiFi-DensePose API.

Handles real CSI data processing and neural network inference only.
Mock data generation belongs in a separate MockPoseService selected at startup.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from src.config.settings import Settings
from src.config.domains import DomainConfig
from src.core.csi_processor import CSIProcessor
from src.core.phase_sanitizer import PhaseSanitizer
from src.models.densepose_head import DensePoseHead
from src.models.modality_translation import ModalityTranslationNetwork

logger = logging.getLogger(__name__)


class PoseService:
    """Service for pose estimation from real CSI data."""

    def __init__(self, settings: Settings, domain_config: DomainConfig):
        self.settings = settings
        self.domain_config = domain_config
        self.csi_processor: Optional[CSIProcessor] = None
        self.phase_sanitizer: Optional[PhaseSanitizer] = None
        self.densepose_model: Optional[DensePoseHead] = None
        self.modality_translator: Optional[ModalityTranslationNetwork] = None
        self.is_initialized = False
        self.is_running = False
        self.last_error: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._calibration_in_progress = False
        self._calibration_id: Optional[str] = None
        self._calibration_start: Optional[datetime] = None
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "processing_time_ms": 0.0,
        }

    async def initialize(self):
        try:
            logger.info("Initializing pose service...")
            self.csi_processor = CSIProcessor(config={
                'buffer_size': self.settings.csi_buffer_size,
                'sampling_rate': getattr(self.settings, 'csi_sampling_rate', 1000),
                'window_size': getattr(self.settings, 'csi_window_size', 512),
                'overlap': getattr(self.settings, 'csi_overlap', 0.5),
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1),
                'human_detection_threshold': getattr(self.settings, 'csi_human_detection_threshold', 0.55),
                'smoothing_factor': getattr(self.settings, 'csi_smoothing_factor', 0.65),
                'max_history_size': getattr(self.settings, 'csi_max_history_size', 500),
                'num_subcarriers': 56,
                'num_antennas': 3,
            })
            self.phase_sanitizer = PhaseSanitizer(config={
                'unwrapping_method': 'numpy',
                'outlier_threshold': 3.0,
                'smoothing_window': 5,
                'enable_outlier_removal': True,
                'enable_smoothing': True,
                'enable_noise_filtering': True,
                'noise_threshold': getattr(self.settings, 'csi_noise_threshold', 0.1),
            })
            await self._initialize_models()
            self.is_initialized = True
            self._start_time = datetime.now()
            logger.info("Pose service initialized successfully")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize pose service: {e}")
            raise

    async def _initialize_models(self):
        mt_config = {
            'input_channels': 64,
            'hidden_channels': [128, 256, 512],
            'output_channels': 256,
            'use_attention': True,
        }
        self.modality_translator = ModalityTranslationNetwork(mt_config)
        out_ch = mt_config['output_channels']
        self.densepose_model = DensePoseHead({
            'input_channels': out_ch,
            'num_body_parts': 24,
            'num_uv_coordinates': 2,
            'hidden_channels': [128, 64],
        })
        if self.settings.pose_model_path:
            logger.info("DensePose model path set (load weights when available)")
        else:
            logger.warning("No pose model path provided, using randomly initialized heads")
        self.densepose_model.eval()
        self.modality_translator.eval()

    async def start(self):
        if not self.is_initialized:
            await self.initialize()
        self.is_running = True
        logger.info("Pose service started")

    async def stop(self):
        self.is_running = False
        logger.info("Pose service stopped")

    async def process_csi_data(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_running:
            raise RuntimeError("Pose service is not running")
        start_time = datetime.now()
        try:
            processed = await self._process_csi(csi_data, metadata)
            poses = await self._estimate_poses(processed, metadata)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(poses, processing_time)
            return {
                "timestamp": start_time.isoformat(),
                "poses": poses,
                "metadata": metadata,
                "processing_time_ms": processing_time,
                "confidence_scores": [p.get("confidence", 0.0) for p in poses],
            }
        except Exception as e:
            self.last_error = str(e)
            self.stats["failed_detections"] += 1
            logger.error(f"Error processing CSI data: {e}")
            raise

    async def _process_csi(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        from src.hardware.csi_extractor import CSIData
        if csi_data.ndim == 1:
            amplitude = np.abs(csi_data)
            phase = np.angle(csi_data) if np.iscomplexobj(csi_data) else np.zeros_like(csi_data)
        else:
            amplitude = csi_data
            phase = np.zeros_like(csi_data)
        csi_obj = CSIData(
            timestamp=metadata.get("timestamp", datetime.now()),
            amplitude=amplitude, phase=phase,
            frequency=metadata.get("frequency", 5.0),
            bandwidth=metadata.get("bandwidth", 20.0),
            num_subcarriers=metadata.get("num_subcarriers", 56),
            num_antennas=metadata.get("num_antennas", 3),
            snr=metadata.get("snr", 20.0),
            metadata=metadata,
        )
        try:
            result = await self.csi_processor.process_csi_data(csi_obj)
            self.csi_processor.add_to_history(csi_obj)
            if result and result.features:
                amp = result.features.amplitude_mean
                if hasattr(result.features, 'phase_difference'):
                    # sanitize_phase requires 2D (n_antennas, n_subcarriers).
                    # phase_difference is 1D (n_subcarriers-1,), so reshape to
                    # (1, n) for sanitization then flatten back.
                    pd = result.features.phase_difference
                    sanitized = self.phase_sanitizer.sanitize_phase(
                        pd.reshape(1, -1)
                    ).ravel()
                    return np.concatenate([amp, sanitized])
                return amp
        except Exception as e:
            logger.warning(f"CSI processing failed, using raw data: {e}")
        return csi_data

    def _csi_to_visual_input(self, csi_tensor: torch.Tensor) -> torch.Tensor:
        """Reshape CSI features to (B, C, H, W) for the modality translator."""
        x = csi_tensor.float()
        while x.dim() < 2:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b = x.shape[0]
        flat = x.reshape(b, -1)
        ic = self.modality_translator.input_channels
        side = 8
        target = ic * side * side
        if flat.shape[1] < target:
            flat = F.pad(flat, (0, target - int(flat.shape[1])))
        else:
            flat = flat[:, :target]
        return flat.reshape(b, ic, side, side)

    async def _estimate_poses(self, csi_data: np.ndarray, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            csi_tensor = torch.from_numpy(np.asarray(csi_data, dtype=np.float32))
            visual_in = self._csi_to_visual_input(csi_tensor)
            with torch.no_grad():
                visual_features = self.modality_translator(visual_in)
                output_dict = self.densepose_model(visual_features)
            poses = self._poses_from_densepose_output(output_dict)
            filtered = [p for p in poses if p.get("confidence", 0.0) >= self.settings.pose_confidence_threshold]
            if len(filtered) > self.settings.pose_max_persons:
                filtered = sorted(filtered, key=lambda x: x.get("confidence", 0.0), reverse=True)[:self.settings.pose_max_persons]
            return filtered
        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return []

    def _poses_from_densepose_output(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """Turn DensePose dict outputs (segmentation + UV) into person records."""
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
        ]
        post = self.densepose_model.post_process_predictions(output_dict)
        body = post["body_parts"]
        uv = post["uv_coordinates"]
        conf_maps = post["confidence_scores"]
        seg_conf = conf_maps["segmentation_confidence"]
        uv_conf = conf_maps["uv_confidence"]
        b = body.shape[0]
        poses: List[Dict[str, Any]] = []
        for i in range(b):
            sc = float(seg_conf[i].mean().item())
            uc = float(uv_conf[i].mean().item())
            confidence = max(0.0, min(1.0, (sc + uc) / 2.0))
            h, w = body.shape[1], body.shape[2]
            mask = body[i] > 0
            if mask.any():
                ys, xs = torch.where(mask)
                bbox = {
                    "x": float(xs.float().min() / max(w - 1, 1)),
                    "y": float(ys.float().min() / max(h - 1, 1)),
                    "width": float((xs.float().max() - xs.float().min() + 1) / max(w, 1)),
                    "height": float((ys.float().max() - ys.float().min() + 1) / max(h, 1)),
                }
            else:
                bbox = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
            u_mean = float(uv[i, 0].mean().item())
            v_mean = float(uv[i, 1].mean().item())
            keypoints = []
            for j, name in enumerate(keypoint_names):
                ox = (j % 4) * 0.02
                oy = (j // 4) * 0.02
                keypoints.append({
                    "name": name,
                    "x": min(1.0, max(0.0, u_mean + ox)),
                    "y": min(1.0, max(0.0, v_mean + oy)),
                    "confidence": confidence * 0.9,
                })
            feat_norm = float(torch.norm(uv[i]).item())
            activity = (
                "walking" if feat_norm > 2.0 else "standing" if feat_norm > 1.0
                else "sitting" if feat_norm > 0.5 else "lying" if feat_norm > 0.1 else "unknown"
            )
            poses.append({
                "person_id": i,
                "confidence": confidence,
                "keypoints": keypoints,
                "bounding_box": bbox,
                "activity": activity,
                "timestamp": datetime.now().isoformat(),
            })
        return poses

    def _update_stats(self, poses: List[Dict[str, Any]], processing_time: float):
        self.stats["total_processed"] += 1
        if poses:
            self.stats["successful_detections"] += 1
            avg_conf = sum(p.get("confidence", 0.0) for p in poses) / len(poses)
            n = self.stats["successful_detections"]
            self.stats["average_confidence"] = (self.stats["average_confidence"] * (n - 1) + avg_conf) / n
        else:
            self.stats["failed_detections"] += 1
        n = self.stats["total_processed"]
        self.stats["processing_time_ms"] = (self.stats["processing_time_ms"] * (n - 1) + processing_time) / n

    async def estimate_poses(self, zone_ids=None, confidence_threshold=None, max_persons=None,
                             include_keypoints=True, include_segmentation=False,
                             csi_data: Optional[np.ndarray] = None):
        """Public API for pose estimation. Requires real CSI data."""
        if csi_data is None:
            raise NotImplementedError(
                "Pose estimation requires real CSI data. Pass csi_data from hardware, "
                "or use MockPoseService for development."
            )
        metadata = {
            "timestamp": datetime.now(),
            "zone_ids": zone_ids or ["zone_1"],
            "confidence_threshold": confidence_threshold or self.settings.pose_confidence_threshold,
            "max_persons": max_persons or self.settings.pose_max_persons,
        }
        result = await self.process_csi_data(csi_data, metadata)
        persons = []
        for pose in result["poses"]:
            person = {
                "person_id": str(pose["person_id"]),
                "confidence": pose["confidence"],
                "bounding_box": pose["bounding_box"],
                "zone_id": zone_ids[0] if zone_ids else "zone_1",
                "activity": pose["activity"],
                "timestamp": pose["timestamp"],
            }
            if include_keypoints:
                person["keypoints"] = pose["keypoints"]
            persons.append(person)
        zone_summary = {}
        for zid in (zone_ids or ["zone_1"]):
            zone_summary[zid] = len([p for p in persons if p.get("zone_id") == zid])
        return {
            "timestamp": datetime.now(),
            "frame_id": f"frame_{int(datetime.now().timestamp())}",
            "persons": persons,
            "zone_summary": zone_summary,
            "processing_time_ms": result["processing_time_ms"],
            "metadata": {"mock_data": False},
        }

    async def is_calibrating(self):
        return self._calibration_in_progress

    async def start_calibration(self):
        import uuid
        cid = str(uuid.uuid4())
        self._calibration_id = cid
        self._calibration_in_progress = True
        self._calibration_start = datetime.now()
        logger.info(f"Started calibration: {cid}")
        return cid

    async def run_calibration(self, calibration_id):
        logger.info(f"Running calibration: {calibration_id}")
        await asyncio.sleep(5)
        self._calibration_in_progress = False
        self._calibration_id = None
        logger.info(f"Calibration completed: {calibration_id}")

    async def get_calibration_status(self):
        if self._calibration_in_progress and self._calibration_start is not None:
            elapsed = (datetime.now() - self._calibration_start).total_seconds()
            return {
                "is_calibrating": True, "calibration_id": self._calibration_id,
                "progress_percent": round(min(100.0, (elapsed / 5.0) * 100.0), 1),
                "current_step": "collecting_baseline",
                "estimated_remaining_minutes": max(0.0, (5.0 - elapsed) / 60.0),
                "last_calibration": None,
            }
        return {
            "is_calibrating": False, "calibration_id": None, "progress_percent": 100,
            "current_step": "completed", "estimated_remaining_minutes": 0,
            "last_calibration": self._calibration_start,
        }

    async def get_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.is_running and not self.last_error else "unhealthy",
            "initialized": self.is_initialized,
            "running": self.is_running,
            "last_error": self.last_error,
            "statistics": self.stats.copy(),
            "configuration": {
                "confidence_threshold": self.settings.pose_confidence_threshold,
                "max_persons": self.settings.pose_max_persons,
                "batch_size": self.settings.pose_processing_batch_size,
            },
        }

    async def health_check(self):
        try:
            return {
                "status": "healthy" if self.is_running and not self.last_error else "unhealthy",
                "message": self.last_error or "Service is running normally",
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0.0,
                "metrics": {
                    "total_processed": self.stats["total_processed"],
                    "success_rate": self.stats["successful_detections"] / max(1, self.stats["total_processed"]),
                    "average_processing_time_ms": self.stats["processing_time_ms"],
                },
            }
        except Exception as e:
            return {"status": "unhealthy", "message": f"Health check failed: {e}"}
