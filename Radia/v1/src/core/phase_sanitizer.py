"""Phase sanitization module for WiFi-DensePose system using TDD approach."""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from scipy import signal


class PhaseSanitizationError(Exception):
    """Exception raised for phase sanitization errors."""
    pass


class PhaseSanitizer:
    """Sanitizes phase data from CSI signals for reliable processing."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize phase sanitizer.
        
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
        self.unwrapping_method = config['unwrapping_method']
        self.outlier_threshold = config['outlier_threshold']
        self.smoothing_window = config['smoothing_window']
        
        # Optional parameters with defaults
        self.enable_outlier_removal = config.get('enable_outlier_removal', True)
        self.enable_smoothing = config.get('enable_smoothing', True)
        self.enable_noise_filtering = config.get('enable_noise_filtering', False)
        self.noise_threshold = config.get('noise_threshold', 0.05)
        self.phase_range = config.get('phase_range', (-np.pi, np.pi))

        # Pre-compute Butterworth coefficients once. butter() solves for
        # filter poles (O(order) trig + bilinear transform) — no need to
        # repeat that work on every frame since cutoff never changes.
        nyquist = 0.5
        cutoff = float(self.noise_threshold) * nyquist
        if self.enable_noise_filtering and 0.0 < cutoff < 1.0:
            self._butter_b, self._butter_a = signal.butter(4, cutoff, btype='low')
        else:
            self._butter_b, self._butter_a = None, None

        self.stats = {"total_processed": 0, "outliers_removed": 0, "sanitization_errors": 0}
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ['unwrapping_method', 'outlier_threshold', 'smoothing_window']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        # Validate unwrapping method
        valid_methods = ['numpy', 'scipy', 'custom']
        if config['unwrapping_method'] not in valid_methods:
            raise ValueError(f"Invalid unwrapping method: {config['unwrapping_method']}. Must be one of {valid_methods}")
        
        # Validate thresholds
        if config['outlier_threshold'] <= 0:
            raise ValueError("outlier_threshold must be positive")
        
        if config['smoothing_window'] <= 0:
            raise ValueError("smoothing_window must be positive")
    
    def unwrap_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """Unwrap phase data to remove discontinuities."""
        if phase_data.size == 0:
            raise PhaseSanitizationError("Cannot unwrap empty phase data")
        try:
            if self.unwrapping_method == 'custom':
                unwrapped = phase_data.copy()
                for i in range(phase_data.shape[0]):
                    unwrapped[i, :] = np.unwrap(phase_data[i, :])
                return unwrapped
            return np.unwrap(phase_data, axis=1)
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to unwrap phase: {e}")
    
    def remove_outliers(self, phase_data: np.ndarray) -> np.ndarray:
        """Remove outliers from phase data.
        
        Args:
            phase_data: Phase data (2D array)
            
        Returns:
            Phase data with outliers removed
            
        Raises:
            PhaseSanitizationError: If outlier removal fails
        """
        if not self.enable_outlier_removal:
            return phase_data
        
        try:
            # Detect outliers
            outlier_mask = self._detect_outliers(phase_data)
            
            # Interpolate outliers
            clean_data = self._interpolate_outliers(phase_data, outlier_mask)
            
            return clean_data
            
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to remove outliers: {e}")
    
    def _detect_outliers(self, phase_data: np.ndarray) -> np.ndarray:
        """Detect outliers using statistical methods."""
        # Use Z-score method to detect outliers
        z_scores = np.abs((phase_data - np.mean(phase_data, axis=1, keepdims=True)) / 
                         (np.std(phase_data, axis=1, keepdims=True) + 1e-8))
        outlier_mask = z_scores > self.outlier_threshold
        
        # Update statistics
        self.stats["outliers_removed"] += int(np.sum(outlier_mask))
        
        return outlier_mask
    
    def _interpolate_outliers(self, phase_data: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        """Interpolate outlier values."""
        clean_data = phase_data.copy()
        
        for i in range(phase_data.shape[0]):
            outliers = outlier_mask[i, :]
            if np.any(outliers):
                # Linear interpolation for outliers
                valid_indices = np.where(~outliers)[0]
                outlier_indices = np.where(outliers)[0]
                
                if len(valid_indices) > 1:
                    clean_data[i, outlier_indices] = np.interp(
                        outlier_indices, valid_indices, phase_data[i, valid_indices]
                    )
        
        return clean_data
    
    def smooth_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """Smooth phase data to reduce noise.
        
        Args:
            phase_data: Phase data (2D array)
            
        Returns:
            Smoothed phase data
            
        Raises:
            PhaseSanitizationError: If smoothing fails
        """
        if not self.enable_smoothing:
            return phase_data
        
        try:
            smoothed_data = self._apply_moving_average(phase_data, self.smoothing_window)
            return smoothed_data
            
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to smooth phase: {e}")
    
    def _apply_moving_average(self, phase_data: np.ndarray, window_size: int) -> np.ndarray:
        """Savitzky–Golay smoothing along subcarriers (preserves derivatives vs box filter)."""
        if window_size % 2 == 0:
            window_size += 1
        w = min(window_size, phase_data.shape[1])
        if w % 2 == 0:
            w -= 1
        if w < 5:
            return phase_data.copy()
        poly = min(3, w - 1)
        try:
            return signal.savgol_filter(
                phase_data, window_length=w, polyorder=poly, axis=1, mode="interp"
            )
        except ValueError:
            return phase_data.copy()
    
    def filter_noise(self, phase_data: np.ndarray) -> np.ndarray:
        """Filter noise from phase data.
        
        Args:
            phase_data: Phase data (2D array)
            
        Returns:
            Filtered phase data
            
        Raises:
            PhaseSanitizationError: If noise filtering fails
        """
        if not self.enable_noise_filtering:
            return phase_data
        
        try:
            filtered_data = self._apply_low_pass_filter(phase_data, self.noise_threshold)
            return filtered_data
            
        except Exception as e:
            raise PhaseSanitizationError(f"Failed to filter noise: {e}")
    
    def _apply_low_pass_filter(self, phase_data: np.ndarray, threshold: float) -> np.ndarray:
        """Apply low-pass filter to remove high-frequency noise.

        Uses pre-computed filter coefficients (see __init__) instead of
        re-designing the Butterworth filter on every call.
        """
        filtered_data = phase_data.copy()

        # Minimum length for filtfilt with order-4 filter
        min_filter_length = 18
        if phase_data.shape[1] < min_filter_length:
            return filtered_data

        b, a = self._butter_b, self._butter_a
        if b is None:
            return filtered_data

        # Apply pre-computed coefficients to each antenna row
        for i in range(phase_data.shape[0]):
            filtered_data[i, :] = signal.filtfilt(b, a, phase_data[i, :])

        return filtered_data
    
    def sanitize_phase(self, phase_data: np.ndarray) -> np.ndarray:
        """Sanitize phase data through complete pipeline.
        
        Args:
            phase_data: Raw phase data (2D array)
            
        Returns:
            Sanitized phase data
            
        Raises:
            PhaseSanitizationError: If sanitization fails
        """
        try:
            self.stats["total_processed"] += 1
            
            # Validate input data
            self.validate_phase_data(phase_data)
            
            # Apply complete sanitization pipeline
            sanitized_data = self.unwrap_phase(phase_data)
            sanitized_data = self.remove_outliers(sanitized_data)
            sanitized_data = self.smooth_phase(sanitized_data)
            sanitized_data = self.filter_noise(sanitized_data)
            
            return sanitized_data
            
        except PhaseSanitizationError:
            self.stats["sanitization_errors"] += 1
            raise
        except Exception as e:
            self.stats["sanitization_errors"] += 1
            raise PhaseSanitizationError(f"Sanitization pipeline failed: {e}")
    
    def validate_phase_data(self, phase_data: np.ndarray) -> bool:
        """Validate phase data format and values.
        
        Args:
            phase_data: Phase data to validate
            
        Returns:
            True if valid
            
        Raises:
            PhaseSanitizationError: If validation fails
        """
        # Check if data is 2D
        if phase_data.ndim != 2:
            raise PhaseSanitizationError("Phase data must be 2D array")
        
        # Check if data is not empty
        if phase_data.size == 0:
            raise PhaseSanitizationError("Phase data cannot be empty")
        
        # Check if values are within valid range
        min_val, max_val = self.phase_range
        if np.any(phase_data < min_val) or np.any(phase_data > max_val):
            raise PhaseSanitizationError(f"Phase values outside valid range [{min_val}, {max_val}]")
        
        return True
    
    def get_sanitization_statistics(self) -> Dict[str, Any]:
        """Get sanitization statistics including derived rates."""
        total = self.stats["total_processed"]
        return {
            **self.stats,
            "outlier_rate": self.stats["outliers_removed"] / total if total else 0,
            "error_rate": self.stats["sanitization_errors"] / total if total else 0,
        }

    def reset_statistics(self) -> None:
        """Reset sanitization statistics."""
        self.stats = {"total_processed": 0, "outliers_removed": 0, "sanitization_errors": 0}