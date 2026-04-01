# Radia

**WiFi-based human pose estimation** — detects where people are and how they're moving using ordinary WiFi signals, no camera required.

Forked from [RuView](https://github.com/ruvnet/RuView). The core RF ideas and most of the heavy lifting are theirs. What we did here was clean house: one clear entry point, less duplicated logic, honest stubs where the database was removed, and Radia-specific wire tokens so nodes can discover each other.

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

**Subcarriers** — WiFi spreads data across ~56 frequency channels simultaneously. Each one sees the environment slightly differently, which gives the model more signal to work with.

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

## License

Check the `LICENSE` file in each crate, the Python package root, the firmware tree, and any vendor subtrees you use.
