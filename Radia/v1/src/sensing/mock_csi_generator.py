"""
Synthetic CSI generator for development and mock router mode.

Uses random data by design. Real deployments should use hardware-backed CSI.
"""

import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)

MOCK_MODE_BANNER = """
================================================================================
  WARNING: MOCK MODE ACTIVE - Using synthetic CSI data

  All CSI data is randomly generated and does NOT represent real WiFi signals.
  For real pose estimation, configure hardware per docs/hardware-setup.md.
================================================================================
"""


class MockCSIGenerator:
    """Generates complex-valued CSI tensors for mock / dev use."""

    def __init__(
        self,
        num_subcarriers: int = 64,
        num_antennas: int = 4,
        num_samples: int = 100,
        noise_level: float = 0.1,
        movement_freq: float = 0.5,
        movement_amplitude: float = 0.3,
    ):
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.movement_freq = movement_freq
        self.movement_amplitude = movement_amplitude
        self._phase = 0.0
        self._frequency = 0.1
        self._amplitude_base = 1.0
        self._banner_shown = False

    def show_banner(self) -> None:
        if not self._banner_shown:
            logger.warning(MOCK_MODE_BANNER)
            self._banner_shown = True

    def generate(self) -> np.ndarray:
        self.show_banner()
        self._phase += self._frequency
        time_axis = np.linspace(0, 1, self.num_samples)
        csi_data = np.zeros(
            (self.num_antennas, self.num_subcarriers, self.num_samples),
            dtype=complex,
        )
        for antenna in range(self.num_antennas):
            for subcarrier in range(self.num_subcarriers):
                amplitude = (
                    self._amplitude_base
                    * (1 + 0.2 * np.sin(2 * np.pi * subcarrier / self.num_subcarriers))
                    * (1 + 0.1 * antenna)
                )
                phase_offset = (
                    self._phase
                    + 2 * np.pi * subcarrier / self.num_subcarriers
                    + np.pi * antenna / self.num_antennas
                )
                movement = self.movement_amplitude * np.sin(
                    2 * np.pi * self.movement_freq * time_axis
                )
                signal_amplitude = amplitude * (1 + movement)
                signal_phase = phase_offset + movement * 0.5
                noise = np.random.normal(0, self.noise_level, self.num_samples) + 1j * np.random.normal(
                    0, self.noise_level, self.num_samples
                )
                csi_data[antenna, subcarrier, :] = (
                    signal_amplitude * np.exp(1j * signal_phase) + noise
                )
        return csi_data

    def configure(self, config: Dict[str, Any]) -> None:
        if "sampling_rate" in config:
            self._frequency = config["sampling_rate"] / 1000.0
        if "noise_level" in config:
            self.noise_level = config["noise_level"]
        if "num_subcarriers" in config:
            self.num_subcarriers = config["num_subcarriers"]
        if "num_antennas" in config:
            self.num_antennas = config["num_antennas"]
        if "movement_freq" in config:
            self.movement_freq = config["movement_freq"]
        if "movement_amplitude" in config:
            self.movement_amplitude = config["movement_amplitude"]

    def get_router_info(self) -> Dict[str, Any]:
        return {
            "model": "Mock Router",
            "firmware": "1.0.0-mock",
            "wifi_standard": "802.11ac",
            "antennas": self.num_antennas,
            "supported_bands": ["2.4GHz", "5GHz"],
            "csi_capabilities": {
                "max_subcarriers": self.num_subcarriers,
                "max_antennas": self.num_antennas,
                "sampling_rate": 1000,
            },
        }
