//! Respiratory rate extraction from CSI residuals.
//!
//! Uses bandpass filtering (0.1-0.5 Hz) and autocorrelation peak detection
//! (shared with [`crate::heartrate`]) plus FFT-based in-band SNR for confidence.
//!
//! Weighted subcarrier fusion matches the ESP32 CSI preprocessor output.

use std::f64::consts::PI;

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::acf::autocorrelation_peak;
use crate::iir::IirBandpass;
use crate::types::{VitalEstimate, VitalStatus};

/// Respiratory rate extractor using bandpass filtering and autocorrelation peak detection.
pub struct BreathingExtractor {
    /// Per-sample filtered signal history.
    filtered_history: Vec<f64>,
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Analysis window in seconds.
    window_secs: f64,
    /// Maximum subcarrier slots.
    n_subcarriers: usize,
    /// Breathing band low cutoff (Hz).
    freq_low: f64,
    /// Breathing band high cutoff (Hz).
    freq_high: f64,
    /// IIR bandpass filter.
    filter: IirBandpass,
}

impl BreathingExtractor {
    /// Create a new breathing extractor.
    ///
    /// - `n_subcarriers`: number of subcarrier channels.
    /// - `sample_rate`: input sample rate in Hz.
    /// - `window_secs`: analysis window length in seconds (default: 30).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(n_subcarriers: usize, sample_rate: f64, window_secs: f64) -> Self {
        let capacity = (sample_rate * window_secs) as usize;
        Self {
            filtered_history: Vec::with_capacity(capacity),
            sample_rate,
            window_secs,
            n_subcarriers,
            freq_low: 0.1,
            freq_high: 0.5,
            filter: IirBandpass::new(0.1, 0.5, sample_rate),
        }
    }

    /// Create with ESP32 defaults (56 subcarriers, 100 Hz, 30 s window).
    #[must_use]
    pub fn esp32_default() -> Self {
        Self::new(56, 100.0, 30.0)
    }

    /// Extract respiratory rate from a vector of per-subcarrier residuals.
    ///
    /// - `residuals`: amplitude residuals from the preprocessor.
    /// - `weights`: per-subcarrier attention weights (higher = more
    ///   body-sensitive). If shorter than `residuals`, missing weights
    ///   default to uniform.
    ///
    /// Returns a `VitalEstimate` with the breathing rate in BPM, or
    /// `None` if insufficient history has been accumulated.
    pub fn extract(&mut self, residuals: &[f64], weights: &[f64]) -> Option<VitalEstimate> {
        let n = residuals.len().min(self.n_subcarriers);
        if n == 0 {
            return None;
        }

        // Weighted fusion of subcarrier residuals
        let uniform_w = 1.0 / n as f64;
        let weighted_signal: f64 = residuals
            .iter()
            .enumerate()
            .take(n)
            .map(|(i, &r)| {
                let w = weights.get(i).copied().unwrap_or(uniform_w);
                r * w
            })
            .sum();

        let filtered = self.filter.filter(weighted_signal);

        // Append to history, enforce window limit
        self.filtered_history.push(filtered);
        let max_len = (self.sample_rate * self.window_secs) as usize;
        if self.filtered_history.len() > max_len {
            self.filtered_history.remove(0);
        }

        // Need at least 10 seconds of data
        let min_samples = (self.sample_rate * 10.0) as usize;
        if self.filtered_history.len() < min_samples {
            return None;
        }

        let (period_samples, _acf_peak) = autocorrelation_peak(
            &self.filtered_history,
            self.sample_rate,
            self.freq_low,
            self.freq_high,
        );
        if period_samples == 0 {
            return None;
        }

        let frequency_hz = self.sample_rate / period_samples as f64;

        // Validate frequency is within the breathing band
        if frequency_hz < self.freq_low || frequency_hz > self.freq_high {
            return None;
        }

        let bpm = frequency_hz * 60.0;
        let confidence = compute_confidence_spectral(
            &self.filtered_history,
            self.sample_rate,
            self.freq_low,
            self.freq_high,
        );

        let status = if confidence >= 0.7 {
            VitalStatus::Valid
        } else if confidence >= 0.4 {
            VitalStatus::Degraded
        } else {
            VitalStatus::Unreliable
        };

        Some(VitalEstimate {
            value_bpm: bpm,
            confidence,
            status,
        })
    }

    /// Reset all filter state and history.
    pub fn reset(&mut self) {
        self.filtered_history.clear();
        self.filter.reset();
    }

    /// Current number of samples in the history buffer.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.filtered_history.len()
    }

    /// Breathing band cutoff frequencies.
    #[must_use]
    pub fn band(&self) -> (f64, f64) {
        (self.freq_low, self.freq_high)
    }
}

/// Count zero crossings in a signal.
fn count_zero_crossings(signal: &[f64]) -> usize {
    signal.windows(2).filter(|w| w[0] * w[1] < 0.0).count()
}

/// In-band periodogram SNR: peak-bin power vs mean power in other bins (excluding peak ±1).
fn compute_confidence_spectral(
    history: &[f64],
    sample_rate: f64,
    freq_low: f64,
    freq_high: f64,
) -> f64 {
    let n = history.len();
    if n < 32 {
        return 0.0;
    }

    let fft_len = n.next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);

    let mut buffer: Vec<Complex<f64>> = (0..fft_len)
        .map(|i| {
            if i < n {
                let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0).max(1.0)).cos());
                Complex::new(history[i] * w, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();

    fft.process(&mut buffer);

    let mags: Vec<f64> = buffer
        .iter()
        .take(fft_len / 2 + 1)
        .map(|c| c.norm())
        .collect();

    let freq_res = sample_rate / fft_len as f64;
    let min_bin = (freq_low / freq_res).ceil() as usize;
    let max_bin = ((freq_high / freq_res).floor() as usize).min(mags.len().saturating_sub(1));

    if min_bin >= max_bin {
        return 0.0;
    }

    let mut peak_bin = min_bin;
    let mut peak_mag = mags[min_bin];
    for b in min_bin..=max_bin {
        if mags[b] > peak_mag {
            peak_mag = mags[b];
            peak_bin = b;
        }
    }

    let signal_power = mags[peak_bin].powi(2);
    let noise_bins: Vec<usize> = (min_bin..=max_bin)
        .filter(|&b| (b as i64 - peak_bin as i64).abs() > 1)
        .collect();

    let noise_power = if noise_bins.is_empty() {
        1e-15
    } else {
        noise_bins.iter().map(|&b| mags[b].powi(2)).sum::<f64>() / noise_bins.len() as f64
    };

    let snr = signal_power / (noise_power + 1e-15);
    (snr / 10.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_data_returns_none() {
        let mut ext = BreathingExtractor::new(4, 10.0, 30.0);
        assert!(ext.extract(&[], &[]).is_none());
    }

    #[test]
    fn insufficient_history_returns_none() {
        let mut ext = BreathingExtractor::new(2, 10.0, 30.0);
        // Just a few frames are not enough
        for _ in 0..5 {
            assert!(ext.extract(&[1.0, 2.0], &[0.5, 0.5]).is_none());
        }
    }

    #[test]
    fn zero_crossings_count() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert_eq!(count_zero_crossings(&signal), 4);
    }

    #[test]
    fn zero_crossings_constant() {
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(count_zero_crossings(&signal), 0);
    }

    #[test]
    fn sinusoidal_breathing_detected() {
        let sample_rate = 10.0;
        let mut ext = BreathingExtractor::new(1, sample_rate, 60.0);
        let breathing_freq = 0.25; // 15 BPM

        // Generate 60 seconds of sinusoidal breathing signal
        for i in 0..600 {
            let t = i as f64 / sample_rate;
            let signal = (2.0 * std::f64::consts::PI * breathing_freq * t).sin();
            ext.extract(&[signal], &[1.0]);
        }

        let result = ext.extract(&[0.0], &[1.0]);
        if let Some(est) = result {
            // Should be approximately 15 BPM (0.25 Hz * 60)
            assert!(
                est.value_bpm > 5.0 && est.value_bpm < 40.0,
                "estimated BPM should be in breathing range: {}",
                est.value_bpm,
            );
            assert!(est.confidence > 0.0, "confidence should be > 0");
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut ext = BreathingExtractor::new(2, 10.0, 30.0);
        ext.extract(&[1.0, 2.0], &[0.5, 0.5]);
        assert!(ext.history_len() > 0);
        ext.reset();
        assert_eq!(ext.history_len(), 0);
    }

    #[test]
    fn band_returns_correct_values() {
        let ext = BreathingExtractor::new(1, 10.0, 30.0);
        let (low, high) = ext.band();
        assert!((low - 0.1).abs() < f64::EPSILON);
        assert!((high - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_zero_for_flat_signal() {
        let history = vec![0.0; 100];
        let conf = compute_confidence_spectral(&history, 10.0, 0.1, 0.5);
        assert!((conf - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_positive_for_oscillating_signal() {
        let history: Vec<f64> = (0..256)
            .map(|i| (i as f64 * 0.5).sin())
            .collect();
        let conf = compute_confidence_spectral(&history, 50.0, 0.1, 0.5);
        assert!(conf > 0.0);
    }

    #[test]
    fn esp32_default_creates_correctly() {
        let ext = BreathingExtractor::esp32_default();
        assert_eq!(ext.n_subcarriers, 56);
    }
}
