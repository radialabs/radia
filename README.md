# Radia

**WiFi-based human pose estimation** — detects where people are and how they're moving using ordinary WiFi signals, no camera required.

---

## Some basic sources to get you started

If you're new to WiFi sensing and CSI-based pose estimation, these are worth watching before diving into the code and the rest of the readme:

- [WiFi Sensing — how CSI captures human motion](https://youtu.be/fGZzNZnYIHo?si=rJjVlFxhNFKnpX0v)
- [Through-wall human detection with WiFi signals](https://www.youtube.com/watch?v=sXwDrcd1t-E)
- [CSI-based pose estimation explained](https://www.youtube.com/watch?v=u-Tv4PKZioI)

---

## What's in this repo

```
Radia/
├── v1/                        # Python server (start here)
│   ├── src/
│   │   ├── main.py            # Entry point — run this
│   │   ├── api/               # REST endpoints + WebSocket
│   │   ├── services/          # pose_service, hardware_service, etc.
│   │   ├── core/              # CSI processing, phase sanitization
│   │   └── sensing/           # RSSI, feature extraction, backend
│   └── tests/
│       └── mocks/             # Fake hardware for development
├── rust-port/wifi-densepose-rs/  # Rust sensing server (faster, optional)
├── firmware/esp32-csi-node/      # ESP32 firmware (flash to hardware)
├── ui/                           # Web dashboard
└── docs/adr/                     # Architecture decisions (43 ADRs)
```

> **Tip:** If a README inside a subdirectory disagrees with what you see on disk, trust the disk.

---

## Quick start

**Python server** (no hardware needed for dev):

```bash
cd Radia/v1
pip install -r requirements.txt
python src/main.py
# API runs at http://localhost:8000
```

**With real hardware** (ESP32-S3 on USB):
1. Flash `firmware/esp32-csi-node/` to your ESP32-S3
2. Provision WiFi: `python firmware/esp32-csi-node/provision.py --port COM7 --ssid "YourWiFi" --password "secret"`
3. Start the server — it auto-detects the ESP32 over UDP

**Rust server** (optional, ~10x faster):
```bash
cd Radia/rust-port/wifi-densepose-rs
cargo run --release -- --source esp32
```

---

## What changed from RuView

| Area | Before | After |
|------|--------|-------|
| App startup | Several competing entry points | One: `v1/src/main.py` |
| `pose_service` | Layered logic with mock paths mixed in | Handles real CSI only; fails clearly without hardware |
| CLI | Separate `v1/src/commands/` directory | Single `v1/src/cli.py` |
| Database | DB code referenced everywhere but never wired up | Explicit stubs that print "not configured" |
| Mocks | Scattered in `src/` | Collected under `v1/tests/mocks/` |
| Signal pipelines | Duplicate filter helpers in Rust | One shared implementation |
| Network tokens | Generic strings | `RADIA_BEACON`, `._radia._udp`, `RADIA_PROV` — peers must match |

None of this improves the sensing model or accuracy. It's maintenance work.

---

## Key concepts (plain English)

**CSI (Channel State Information)** — when WiFi transmits, the signal bounces off walls and people. CSI is the record of how each signal path changed. A person moving shifts the bounces in a detectable pattern.

**Subcarriers** — WiFi spreads data across ~56 frequency channels simultanously. Each one sees the environment slightly differently, which gives the model more signal to work with.

**Phase vs amplitude** — CSI has two parts: how strong the signal is (amplitude) and where it is in its cycle (phase). Phase is more sensitive to small movements like breathing.

**Pose estimation** — the system maps CSI patterns to body positions (17 keypoints: nose, shoulders, elbows, etc.) using a neural network trained on paired WiFi+camera data.

---

## Supported hardware

| Device | Role | Notes |
|--------|------|-------|
| ESP32-S3 (8 MB flash) | WiFi CSI node | Recommended, ~$9 |
| ESP32-S3 SuperMini (4 MB) | WiFi CSI node (compact) | Needs `sdkconfig.defaults.4mb` |
| ESP32-C6 + MR60BHA2 | 60 GHz mmWave (heart rate, breathing) | Fuses with CSI for better vitals |

**Not supported:** original ESP32 or ESP32-C3 — single-core, can't run the DSP pipeline.

---

## Running tests

```bash
# Python
cd Radia/v1 && python -m pytest tests/ -x -q

# Rust
cd Radia/rust-port/wifi-densepose-rs
cargo test --workspace --no-default-features
```

---

## Caveats

This README was written by reading and editing the code, not by running a full test suite or connecting hardware. Treat the changes table as a guide for where to look in the code, not as verified outcomes. Architecture Decision Records in `docs/adr/` can read like marketing — use them as history.

---

## The Inspiration

We used RuView as well as other wifi CSI's as inspiration and then applied a set of signal processing and machine learning formulas to better it. The original RF ideas gave us a solid foundation — clear CSI collection logic and a working neural network pipeline. What we built on top of that is where Radia diverges: more rigorous phase handling, proper vital sign isolation, multi-node and a continuous room-calibration loop that keeps the model honest as the environment drifts (this is actually a lot more important than people think).

### Below are some important improvements that I made. (Feel free to cross check them and verify them)

**1. Circular Mean Phase Sanitization**
`φ_ref = arg( Σ exp(jφᵢ) )`
Extracts a stable phase reference across antennas by taking the complex circular mean of raw CSI phase values. Removes per-antenna local-oscillator offset drift that would otherwise corrupt the motion signal before it reaches the neural network.

**2. Doppler Shift Detection via STFT**
`M(f) = | STFT{ Δφ(t) } |²`
Applies a Short-Time Fourier Transform to the temporal sequence of inter-frame phase differences. Peaks in the 0.1–2.0 Hz band confirm body motion; the power spectrum feeds directly into the vital sign pipeline and the pose confidence scorer.

**3. Vital Sign Extraction (Bandpass + FFT Peak)**
`BR = argmax_{ f ∈ [0.1, 0.5 Hz] } |X(f)|` → 6–30 BPM
`HR = argmax_{ f ∈ [0.8, 2.0 Hz] } |X(f)|` → 40–120 BPM
Splits the same CSI phase signal into two narrow frequency windows and finds the dominant peak in each. A 3-stage smoother (outlier gate → 21-frame trimmed mean → EMA α=0.02) keeps the output stable under real-world noise.

**4. Welford EMA Baseline Normalization**
`σ_baseline ← α · σ_baseline + (1−α) · |A(t)|`, α = 0.99
Continuously tracks the empty-room amplitude deviation using an exponential moving average. All incoming frames are normalized against this baseline, making motion detection robust to slow environmental drift — temperature changes, furniture shifts, or a router rebooting.

**5. CrossViewpoint Attention Fusion (RuVector)**
`A = softmax( QKᵀ / √d_k + G_bias ) · V`
Fuses CSI feature vectors from multiple ESP32 nodes using scaled dot-product attention with a geometry-aware bias term. `G_bias` encodes the Cramér-Rao lower bound of the physical antenna geometry — so nodes that are better placed to observe a particular body region naturally receive higher attention weight, without any manual tuning.

---

## License

Check the `LICENSE` file in each crate, the Python package root, the firmware tree, and any vendor subtrees you use.
