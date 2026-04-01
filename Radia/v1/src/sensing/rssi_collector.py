"""
RSSI data collection from WiFi interfaces.

Provides platform-specific collectors with a shared base class:
    - LinuxWifiCollector: reads RSSI from /proc/net/wireless and iw
    - WindowsWifiCollector: reads RSSI from netsh wlan
    - MacosWifiCollector: reads RSSI via CoreWLAN Swift utility
    - SimulatedCollector: deterministic synthetic signals for testing
"""

from __future__ import annotations

import logging
import math
import os
import platform
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Protocol, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WifiSample:
    """A single WiFi measurement sample."""
    timestamp: float
    rssi_dbm: float
    noise_dbm: float
    link_quality: float
    tx_bytes: int
    rx_bytes: int
    retry_count: int
    interface: str


class RingBuffer:
    """Thread-safe fixed-size ring buffer for WifiSample objects."""

    def __init__(self, max_size: int) -> None:
        self._buf: Deque[WifiSample] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, sample: WifiSample) -> None:
        with self._lock:
            self._buf.append(sample)

    def get_all(self) -> List[WifiSample]:
        with self._lock:
            return list(self._buf)

    def get_last_n(self, n: int) -> List[WifiSample]:
        with self._lock:
            items = list(self._buf)
            return items[-n:] if n < len(items) else items

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


class WifiCollector(Protocol):
    """Protocol that all WiFi collectors must satisfy."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_samples(self, n: Optional[int] = None) -> List[WifiSample]: ...
    @property
    def sample_rate_hz(self) -> float: ...


class BaseCollector(ABC):
    """Shared lifecycle for all WiFi collectors."""

    def __init__(self, sample_rate_hz: float, buffer_seconds: int, interface: str = "unknown") -> None:
        self._interface = interface
        self._rate = sample_rate_hz
        self._buffer = RingBuffer(max_size=int(sample_rate_hz * buffer_seconds))
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    def start(self) -> None:
        if self._running:
            return
        self._validate()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True, name=f"{type(self).__name__}-loop")
        self._thread.start()
        logger.info("%s started on %s at %.1f Hz", type(self).__name__, self._interface, self._rate)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("%s stopped", type(self).__name__)

    def get_samples(self, n: Optional[int] = None) -> List[WifiSample]:
        if n is not None:
            return self._buffer.get_last_n(n)
        return self._buffer.get_all()

    def _validate(self) -> None:
        """Override to check platform availability before starting."""

    def _sample_loop(self) -> None:
        interval = 1.0 / self._rate
        while self._running:
            t0 = time.monotonic()
            try:
                sample = self._collect_one_sample()
                self._buffer.append(sample)
            except Exception:
                logger.exception("Error reading WiFi sample")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    @abstractmethod
    def _collect_one_sample(self) -> WifiSample:
        """Platform-specific: collect a single WiFi sample."""


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

class LinuxWifiCollector(BaseCollector):
    def __init__(self, interface: str = "wlan0", sample_rate_hz: float = 10.0, buffer_seconds: int = 120) -> None:
        super().__init__(sample_rate_hz, buffer_seconds, interface)

    def _validate(self) -> None:
        available, reason = self.is_available(self._interface)
        if not available:
            raise RuntimeError(reason)

    @classmethod
    def is_available(cls, interface: str = "wlan0") -> tuple[bool, str]:
        if not os.path.exists("/proc/net/wireless"):
            return False, "/proc/net/wireless not found."
        try:
            with open("/proc/net/wireless", "r") as f:
                content = f.read()
        except OSError as exc:
            return False, f"Cannot read /proc/net/wireless: {exc}"
        if interface not in content:
            names = cls._parse_interface_names(content)
            return False, f"Interface '{interface}' not listed. Available: {names or '(none)'}."
        return True, "ok"

    @staticmethod
    def _parse_interface_names(proc_content: str) -> List[str]:
        names: List[str] = []
        for line in proc_content.splitlines()[2:]:
            parts = line.split(":")
            if len(parts) >= 2:
                names.append(parts[0].strip())
        return names

    def _collect_one_sample(self) -> WifiSample:
        rssi, noise, quality = self._read_proc_wireless()
        tx_bytes, rx_bytes, retries = self._read_iw_station()
        return WifiSample(
            timestamp=time.time(), rssi_dbm=rssi, noise_dbm=noise, link_quality=quality,
            tx_bytes=tx_bytes, rx_bytes=rx_bytes, retry_count=retries, interface=self._interface,
        )

    def collect_once(self) -> WifiSample:
        return self._collect_one_sample()

    def _read_proc_wireless(self) -> tuple[float, float, float]:
        try:
            with open("/proc/net/wireless", "r") as f:
                for line in f:
                    if self._interface in line:
                        parts = line.split()
                        quality_raw = float(parts[2].rstrip("."))
                        signal_raw = float(parts[3].rstrip("."))
                        noise_raw = float(parts[4].rstrip("."))
                        quality = min(1.0, max(0.0, quality_raw / 70.0))
                        return signal_raw, noise_raw, quality
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Failed to read /proc/net/wireless for {self._interface}: {exc}") from exc
        raise RuntimeError(f"Interface {self._interface} not found in /proc/net/wireless")

    def _read_iw_station(self) -> tuple[int, int, int]:
        try:
            result = subprocess.run(
                ["iw", "dev", self._interface, "station", "dump"],
                capture_output=True, text=True, timeout=2.0,
            )
            text = result.stdout
            tx = self._extract_int(text, r"tx bytes:\s*(\d+)")
            rx = self._extract_int(text, r"rx bytes:\s*(\d+)")
            retries = self._extract_int(text, r"tx retries:\s*(\d+)")
            return tx, rx, retries
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0, 0, 0

    @staticmethod
    def _extract_int(text: str, pattern: str) -> int:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

class WindowsWifiCollector(BaseCollector):
    def __init__(self, interface: str = "Wi-Fi", sample_rate_hz: float = 2.0, buffer_seconds: int = 120) -> None:
        super().__init__(sample_rate_hz, buffer_seconds, interface)
        self._cumulative_tx = 0
        self._cumulative_rx = 0

    def _validate(self) -> None:
        try:
            result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5.0)
            if self._interface not in result.stdout:
                raise RuntimeError(f"WiFi interface '{self._interface}' not found.")
        except FileNotFoundError:
            raise RuntimeError("netsh not found. This collector requires Windows.")

    def _collect_one_sample(self) -> WifiSample:
        result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5.0)
        rssi = -80.0
        signal_pct = 0.0
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("rssi"):
                try:
                    rssi = float(stripped.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif stripped.lower().startswith("signal"):
                try:
                    signal_pct = float(stripped.split(":")[1].strip().rstrip("%"))
                except (IndexError, ValueError):
                    pass
        self._cumulative_tx += 1500
        self._cumulative_rx += 3000
        return WifiSample(
            timestamp=time.time(), rssi_dbm=rssi, noise_dbm=-95.0, link_quality=signal_pct / 100.0,
            tx_bytes=self._cumulative_tx, rx_bytes=self._cumulative_rx, retry_count=0, interface=self._interface,
        )

    def collect_once(self) -> WifiSample:
        return self._collect_one_sample()


# ---------------------------------------------------------------------------
# macOS (persistent subprocess, overrides _sample_loop)
# ---------------------------------------------------------------------------

class MacosWifiCollector(BaseCollector):
    def __init__(self, sample_rate_hz: float = 10.0, buffer_seconds: int = 120) -> None:
        super().__init__(sample_rate_hz, buffer_seconds, "en0")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.swift_src = os.path.join(base_dir, "mac_wifi.swift")
        self.swift_bin = os.path.join(base_dir, "mac_wifi")
        self._process: Optional[subprocess.Popen] = None

    def _validate(self) -> None:
        if not os.path.exists(self.swift_bin):
            logger.info("Compiling mac_wifi.swift to %s", self.swift_bin)
            try:
                subprocess.run(["swiftc", "-O", "-o", self.swift_bin, self.swift_src], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to compile macOS WiFi utility: {e.stderr.decode('utf-8')}")
            except FileNotFoundError:
                raise RuntimeError("swiftc not found. Install Xcode Command Line Tools.")

    def stop(self) -> None:
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        super().stop()

    def _sample_loop(self) -> None:
        import json
        self._process = subprocess.Popen(
            [self.swift_bin], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        while self._running and self._process and self._process.poll() is None:
            try:
                line = self._process.stdout.readline().strip()
                if not line or not line.startswith("{"):
                    continue
                data = json.loads(line)
                if "error" in data:
                    logger.error("macOS WiFi utility error: %s", data["error"])
                    continue
                rssi = float(data.get("rssi", -80.0))
                noise = float(data.get("noise", -95.0))
                self._buffer.append(WifiSample(
                    timestamp=time.time(), rssi_dbm=rssi, noise_dbm=noise,
                    link_quality=max(0.0, min(1.0, (rssi + 100.0) / 60.0)),
                    tx_bytes=0, rx_bytes=0, retry_count=0, interface=self._interface,
                ))
            except Exception as e:
                logger.error("Error reading macOS WiFi stream: %s", e)
                time.sleep(1.0)
        if self._running:
            logger.error("macOS WiFi utility exited unexpectedly.")
            self._running = False

    def _collect_one_sample(self) -> WifiSample:
        raise NotImplementedError("MacosWifiCollector uses a persistent subprocess via _sample_loop")


# ---------------------------------------------------------------------------
# Simulated collector
# ---------------------------------------------------------------------------

class SimulatedCollector(BaseCollector):
    """Deterministic simulated WiFi collector.

    Accepts an optional ``signal_generator`` callable for injecting
    deterministic test signals.  When not provided, generates a default
    sine + noise signal.
    """

    def __init__(
        self,
        seed: int = 42,
        sample_rate_hz: float = 10.0,
        buffer_seconds: int = 120,
        baseline_dbm: float = -50.0,
        sine_freq_hz: float = 0.3,
        sine_amplitude_dbm: float = 2.0,
        noise_std_dbm: float = 0.5,
        step_change_at: Optional[float] = None,
        step_change_dbm: float = -10.0,
        signal_generator: Optional[Callable[[float, int], float]] = None,
    ) -> None:
        super().__init__(sample_rate_hz, buffer_seconds, "sim0")
        self._rng = np.random.default_rng(seed)
        self._baseline = baseline_dbm
        self._sine_freq = sine_freq_hz
        self._sine_amp = sine_amplitude_dbm
        self._noise_std = noise_std_dbm
        self._step_at = step_change_at
        self._step_dbm = step_change_dbm
        self._signal_generator = signal_generator
        self._start_time: float = 0.0
        self._sample_index: int = 0

    def start(self) -> None:
        self._start_time = time.time()
        self._sample_index = 0
        super().start()

    def _collect_one_sample(self) -> WifiSample:
        now = time.time()
        t_offset = now - self._start_time
        sample = self._make_sample(now, t_offset, self._sample_index)
        self._sample_index += 1
        return sample

    def generate_samples(self, duration_seconds: float) -> List[WifiSample]:
        """Generate a batch of samples without the background thread."""
        n_samples = int(duration_seconds * self._rate)
        base_time = time.time()
        return [self._make_sample(base_time + i / self._rate, i / self._rate, i) for i in range(n_samples)]

    def _make_sample(self, timestamp: float, t_offset: float, index: int) -> WifiSample:
        if self._signal_generator is not None:
            rssi = self._signal_generator(t_offset, index)
        else:
            sine = self._sine_amp * math.sin(2.0 * math.pi * self._sine_freq * t_offset)
            noise = self._rng.normal(0.0, self._noise_std)
            step = self._step_dbm if (self._step_at is not None and t_offset >= self._step_at) else 0.0
            rssi = self._baseline + sine + noise + step
        return WifiSample(
            timestamp=timestamp, rssi_dbm=float(rssi), noise_dbm=-95.0,
            link_quality=max(0.0, min(1.0, (rssi + 100.0) / 60.0)),
            tx_bytes=index * 1500, rx_bytes=index * 3000,
            retry_count=max(0, index // 100), interface="sim0",
        )


# ---------------------------------------------------------------------------
# Collector factory
# ---------------------------------------------------------------------------

CollectorType = Union[LinuxWifiCollector, WindowsWifiCollector, MacosWifiCollector, SimulatedCollector]


def create_collector(
    preferred: str = "auto",
    interface: str = "wlan0",
    sample_rate_hz: float = 10.0,
) -> CollectorType:
    """Create the best available WiFi collector for the current platform."""
    _VALID = {"auto", "linux", "windows", "macos", "simulated"}
    if preferred not in _VALID:
        logger.warning("Unknown preferred=%r, falling back to auto.", preferred)
        preferred = "auto"

    system = platform.system()

    if preferred == "auto":
        if system == "Linux":
            available, reason = LinuxWifiCollector.is_available(interface)
            if available:
                logger.info("Using LinuxWifiCollector on %s", interface)
                return LinuxWifiCollector(interface=interface, sample_rate_hz=sample_rate_hz)
            logger.warning("LinuxWifiCollector unavailable: %s", reason)
        elif system == "Windows":
            try:
                win_iface = interface if interface != "wlan0" else "Wi-Fi"
                c = WindowsWifiCollector(interface=win_iface, sample_rate_hz=min(sample_rate_hz, 2.0))
                c.collect_once()
                return c
            except Exception as exc:
                logger.warning("WindowsWifiCollector unavailable: %s", exc)
        elif system == "Darwin":
            try:
                return MacosWifiCollector(sample_rate_hz=sample_rate_hz)
            except Exception as exc:
                logger.warning("MacosWifiCollector unavailable: %s", exc)
    elif preferred == "linux":
        return LinuxWifiCollector(interface=interface, sample_rate_hz=sample_rate_hz)
    elif preferred == "windows":
        return WindowsWifiCollector(interface=interface, sample_rate_hz=min(sample_rate_hz, 2.0))
    elif preferred == "macos":
        return MacosWifiCollector(sample_rate_hz=sample_rate_hz)
    elif preferred == "simulated":
        return SimulatedCollector(seed=42, sample_rate_hz=sample_rate_hz)

    logger.info("Falling back to SimulatedCollector.")
    return SimulatedCollector(seed=42, sample_rate_hz=sample_rate_hz)
