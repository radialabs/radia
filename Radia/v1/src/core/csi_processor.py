"""CSI data processor for WiFi-DensePose system using TDD approach."""

import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque
import scipy.signal
import scipy.fft

try:
    from ..hardware.csi_extractor import CSIData
except ImportError:
    # Handle import for testing
    from src.hardware.csi_extractor import CSIData


class CSIProcessingError(Exception):
    """Exception raised for CSI processing errors."""
    pass


@dataclass
class CSIFeatures:
    """Data structure for extracted CSI features."""
    amplitude_mean: np.ndarray
    amplitude_variance: np.ndarray
    phase_difference: np.ndarray
    correlation_matrix: np.ndarray
    doppler_shift: np.ndarray
    power_spectral_density: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class HumanDetectionResult:
    """Data structure for human detection results."""
    human_detected: bool
    confidence: float
    motion_score: float
    timestamp: datetime
    features: CSIFeatures
    metadata: Dict[str, Any]


class CSIProcessor:
    """Processes CSI data for human detection and pose estimation."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize CSI processor.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing parameters
        self.sampling_rate = config['sampling_rate']
        self.window_size = config['window_size']
        self.overlap = config['overlap']
        self.noise_threshold = config['noise_threshold']
        self.human_detection_threshold = config.get('human_detection_threshold', 0.55)
        self.smoothing_factor = config.get('smoothing_factor', 0.65)
        self.max_history_size = config.get('max_history_size', 500)
        
        # Feature extraction flags
        self.enable_preprocessing = config.get('enable_preprocessing', True)
        self.enable_feature_extraction = config.get('enable_feature_extraction', True)
        self.enable_human_detection = config.get('enable_human_detection', True)
        
        # Processing state
        self.csi_history = deque(maxlen=self.max_history_size)
        self.previous_detection_confidence = 0.0

        # Doppler cache: pre-computed mean phase per frame for O(1) append
        self._phase_cache = deque(maxlen=self.max_history_size)
        self._doppler_window = min(config.get('doppler_window', 64), self.max_history_size)
        # Slow EMA of frame std so normalization preserves activity vs empty-room contrast
        self._baseline_std: Optional[float] = None
        self._baseline_std_alpha = float(config.get("baseline_std_ema_alpha", 0.99))

        self.stats = {"total_processed": 0, "processing_errors": 0, "human_detections": 0}
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['sampling_rate', 'window_size', 'overlap', 'noise_threshold']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        if config['sampling_rate'] <= 0:
            raise ValueError("sampling_rate must be positive")
        
        if config['window_size'] <= 0:
            raise ValueError("window_size must be positive")
        
        if not 0 <= config['overlap'] < 1:
            raise ValueError("overlap must be between 0 and 1")
    
    def preprocess_csi_data(self, csi_data: CSIData) -> CSIData:
        """Preprocess CSI data for feature extraction.

        Creates a single working copy and mutates amplitude in-place
        through noise removal, windowing, and normalization.
        """
        if not self.enable_preprocessing:
            return csi_data

        try:
            working = CSIData(
                timestamp=csi_data.timestamp,
                amplitude=csi_data.amplitude.copy(),
                phase=csi_data.phase,
                frequency=csi_data.frequency,
                bandwidth=csi_data.bandwidth,
                num_subcarriers=csi_data.num_subcarriers,
                num_antennas=csi_data.num_antennas,
                snr=csi_data.snr,
                metadata={**csi_data.metadata},
            )

            # Noise removal
            amplitude_db = 20 * np.log10(np.abs(working.amplitude) + 1e-12)
            working.amplitude[amplitude_db <= self.noise_threshold] = 0

            # Hamming window
            window = scipy.signal.windows.hamming(working.num_subcarriers)
            working.amplitude *= window[np.newaxis, :]

            # Normalize by the *prior* baseline std, then update the EMA.
            # Dividing before updating prevents self-normalization bias: a
            # high-energy burst would otherwise inflate σ̂ before being
            # divided by it, masking its own presence signal.
            frame_std = float(np.std(working.amplitude) + 1e-12)
            if self._baseline_std is None:
                self._baseline_std = frame_std
            divisor = self._baseline_std + 1e-12
            working.amplitude /= divisor
            # Update EMA after dividing so this frame doesn't deflate itself
            a = self._baseline_std_alpha
            self._baseline_std = a * self._baseline_std + (1.0 - a) * frame_std

            return working

        except Exception as e:
            raise CSIProcessingError(f"Failed to preprocess CSI data: {e}")
    
    def extract_features(self, csi_data: CSIData) -> Optional[CSIFeatures]:
        """Extract features from CSI data.
        
        Args:
            csi_data: Preprocessed CSI data
            
        Returns:
            Extracted features or None if disabled
            
        Raises:
            CSIProcessingError: If feature extraction fails
        """
        if not self.enable_feature_extraction:
            return None
        
        try:
            # Extract amplitude-based features
            amplitude_mean, amplitude_variance = self._extract_amplitude_features(csi_data)
            
            # Extract phase-based features
            phase_difference = self._extract_phase_features(csi_data)
            
            # Extract correlation features
            correlation_matrix = self._extract_correlation_features(csi_data)
            
            # Extract Doppler and frequency features
            doppler_shift, power_spectral_density = self._extract_doppler_features(csi_data)
            
            return CSIFeatures(
                amplitude_mean=amplitude_mean,
                amplitude_variance=amplitude_variance,
                phase_difference=phase_difference,
                correlation_matrix=correlation_matrix,
                doppler_shift=doppler_shift,
                power_spectral_density=power_spectral_density,
                timestamp=datetime.now(timezone.utc),
                metadata={'processing_params': self.config}
            )
            
        except Exception as e:
            raise CSIProcessingError(f"Failed to extract features: {e}")
    
    def detect_human_presence(self, features: CSIFeatures) -> Optional[HumanDetectionResult]:
        """Detect human presence from CSI features.
        
        Args:
            features: Extracted CSI features
            
        Returns:
            Detection result or None if disabled
            
        Raises:
            CSIProcessingError: If detection fails
        """
        if not self.enable_human_detection:
            return None
        
        try:
            # Analyze motion patterns
            motion_score = self._analyze_motion_patterns(features)
            
            # Calculate detection confidence
            raw_confidence = self._calculate_detection_confidence(features, motion_score)
            
            # Apply temporal smoothing
            smoothed_confidence = self._apply_temporal_smoothing(raw_confidence)
            
            # Determine if human is detected
            human_detected = smoothed_confidence >= self.human_detection_threshold
            
            if human_detected:
                self.stats["human_detections"] += 1
            
            return HumanDetectionResult(
                human_detected=human_detected,
                confidence=smoothed_confidence,
                motion_score=motion_score,
                timestamp=datetime.now(timezone.utc),
                features=features,
                metadata={'threshold': self.human_detection_threshold}
            )
            
        except Exception as e:
            raise CSIProcessingError(f"Failed to detect human presence: {e}")
    
    async def process_csi_data(self, csi_data: CSIData) -> HumanDetectionResult:
        """Process CSI data through the complete pipeline.
        
        Args:
            csi_data: Raw CSI data
            
        Returns:
            Human detection result
            
        Raises:
            CSIProcessingError: If processing fails
        """
        try:
            self.stats["total_processed"] += 1
            
            # Preprocess the data
            preprocessed_data = self.preprocess_csi_data(csi_data)
            
            # Extract features
            features = self.extract_features(preprocessed_data)
            
            # Detect human presence
            detection_result = self.detect_human_presence(features)
            
            # Add to history
            self.add_to_history(csi_data)
            
            return detection_result
            
        except Exception as e:
            self.stats["processing_errors"] += 1
            raise CSIProcessingError(f"Pipeline processing failed: {e}")
    
    def add_to_history(self, csi_data: CSIData) -> None:
        """Add CSI data to processing history.

        Args:
            csi_data: CSI data to add to history
        """
        self.csi_history.append(csi_data)
        # Cache mean phase for fast Doppler extraction
        if csi_data.phase.ndim == 2:
            self._phase_cache.append(np.mean(csi_data.phase, axis=0))
        else:
            self._phase_cache.append(csi_data.phase.flatten())
    
    def clear_history(self) -> None:
        """Clear the CSI data history."""
        self.csi_history.clear()
        self._phase_cache.clear()
        self._baseline_std = None
    
    def get_recent_history(self, count: int) -> List[CSIData]:
        """Get the last *count* CSI data entries from history."""
        return list(self.csi_history)[-count:]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Processing statistics including derived rates."""
        total = self.stats["total_processed"]
        return {
            **self.stats,
            "error_rate": self.stats["processing_errors"] / total if total else 0,
            "detection_rate": self.stats["human_detections"] / total if total else 0,
            "history_size": len(self.csi_history),
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {"total_processed": 0, "processing_errors": 0, "human_detections": 0}
    
    def _extract_amplitude_features(self, csi_data: CSIData) -> tuple:
        """Extract amplitude-based features."""
        amplitude_mean = np.mean(csi_data.amplitude, axis=0)
        amplitude_variance = np.var(csi_data.amplitude, axis=0)
        return amplitude_mean, amplitude_variance
    
    def _extract_phase_features(self, csi_data: CSIData) -> np.ndarray:
        """Extract phase-based features."""
        # Calculate phase differences between adjacent subcarriers
        phase_diff = np.diff(csi_data.phase, axis=1)
        return np.mean(phase_diff, axis=0)
    
    def _extract_correlation_features(self, csi_data: CSIData) -> np.ndarray:
        """Extract correlation features between antennas."""
        # Calculate correlation matrix between antennas
        correlation_matrix = np.corrcoef(csi_data.amplitude)
        return correlation_matrix
    
    def _extract_doppler_features(self, csi_data: CSIData) -> tuple:
        """Extract Doppler and frequency domain features from temporal CSI history.

        Uses cached mean-phase values for O(1) access instead of recomputing
        from raw CSI frames. Only uses the last `doppler_window` frames
        (default 64) for bounded computation time.

        Returns:
            tuple: (doppler_shift, power_spectral_density) as numpy arrays
        """
        # rfft of a length-64 real signal → 33 unique bins (n//2 + 1).
        # The upper half is conjugate-symmetric and carries no new information,
        # so rfft is ~2× faster than fft for real-valued inputs.
        n_fft = 64
        n_doppler_bins = n_fft // 2 + 1  # 33

        if len(self._phase_cache) >= 2:
            # Use cached mean-phase values (pre-computed in add_to_history)
            # Only take the last doppler_window frames for bounded cost
            window = min(len(self._phase_cache), self._doppler_window)
            cache_list = list(self._phase_cache)
            phase_matrix = np.array(cache_list[-window:])

            # Temporal phase differences between consecutive frames
            phase_diffs = np.diff(phase_matrix, axis=0)

            # Average across subcarriers for each time step
            mean_phase_diff = np.mean(phase_diffs, axis=1)

            # Hann window before FFT to reduce spectral leakage vs rectangular window
            wlen = len(mean_phase_diff)
            if wlen >= 2:
                mean_phase_diff = mean_phase_diff * np.hanning(wlen)

            # rfft: real-valued input → only non-redundant bins
            doppler_spectrum = np.abs(scipy.fft.rfft(mean_phase_diff, n=n_fft)) ** 2

            # Normalize
            max_val = np.max(doppler_spectrum)
            if max_val > 0:
                doppler_spectrum = doppler_spectrum / max_val

            doppler_shift = doppler_spectrum
        else:
            doppler_shift = np.zeros(n_doppler_bins)

        # Power spectral density of the current frame (rfft: 65 unique bins from length-128)
        psd = np.abs(scipy.fft.rfft(csi_data.amplitude.flatten(), n=128)) ** 2

        return doppler_shift, psd
    
    def _analyze_motion_patterns(self, features: CSIFeatures) -> float:
        """Analyze motion patterns from features."""
        # Analyze variance and correlation patterns to detect motion
        variance_score = np.mean(features.amplitude_variance)
        correlation_score = np.mean(np.abs(features.correlation_matrix - np.eye(features.correlation_matrix.shape[0])))
        
        # Combine scores (simplified approach)
        motion_score = 0.6 * variance_score + 0.4 * correlation_score
        return np.clip(motion_score, 0.0, 1.0)
    
    @staticmethod
    def _soft_indicator(value: float, center: float, scale: float) -> float:
        """Map a raw feature to (0, 1) via logistic; avoids boolean dead-zones."""
        if scale <= 1e-12:
            return 0.5
        z = (value - center) / scale
        return float(1.0 / (1.0 + np.exp(-z)))

    def _calculate_detection_confidence(self, features: CSIFeatures, motion_score: float) -> float:
        """Continuous feature scores + logistic blend; EMA uses ``smoothing_factor``."""
        amp_mean = float(np.mean(features.amplitude_mean))
        phase_std = float(np.std(features.phase_difference))
        f_amp = self._soft_indicator(amp_mean, center=0.1, scale=0.05)
        f_phase = self._soft_indicator(phase_std, center=0.05, scale=0.02)
        f_motion = float(np.clip(motion_score, 0.0, 1.0))
        linear = 0.4 * f_amp + 0.3 * f_phase + 0.3 * f_motion + (-0.35)
        raw = float(1.0 / (1.0 + np.exp(-4.0 * linear)))
        return np.clip(raw, 0.0, 1.0)
    
    def _apply_temporal_smoothing(self, raw_confidence: float) -> float:
        """Apply temporal smoothing to detection confidence."""
        # Exponential moving average
        smoothed_confidence = (self.smoothing_factor * self.previous_detection_confidence + 
                             (1 - self.smoothing_factor) * raw_confidence)
        
        self.previous_detection_confidence = smoothed_confidence
        return smoothed_confidence