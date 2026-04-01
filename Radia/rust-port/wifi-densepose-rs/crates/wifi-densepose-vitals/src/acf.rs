//! Autocorrelation peak detection shared by breathing and heart-rate extractors.

/// Find the dominant periodicity via normalized autocorrelation in ``freq_low``..``freq_high``.
///
/// Returns `(period_in_samples, peak_normalized_acf)`. If no peak is found, returns `(0, 0.0)`.
pub fn autocorrelation_peak(
    signal: &[f64],
    sample_rate: f64,
    freq_low: f64,
    freq_high: f64,
) -> (usize, f64) {
    let n = signal.len();
    if n < 4 {
        return (0, 0.0);
    }

    let min_lag = (sample_rate / freq_high).floor() as usize;
    let max_lag = (sample_rate / freq_low).ceil() as usize;
    let max_lag = max_lag.min(n / 2);

    if min_lag >= max_lag || min_lag >= n {
        return (0, 0.0);
    }

    let mean: f64 = signal.iter().sum::<f64>() / n as f64;
    let acf0: f64 = signal.iter().map(|&x| (x - mean) * (x - mean)).sum();
    if acf0 < 1e-15 {
        return (0, 0.0);
    }

    let mut best_lag = 0;
    let mut best_acf = f64::MIN;

    for lag in min_lag..=max_lag {
        let acf: f64 = signal
            .iter()
            .take(n - lag)
            .enumerate()
            .map(|(i, &x)| (x - mean) * (signal[i + lag] - mean))
            .sum();

        let normalized = acf / acf0;
        if normalized > best_acf {
            best_acf = normalized;
            best_lag = lag;
        }
    }

    if best_acf > 0.0 {
        (best_lag, best_acf)
    } else {
        (0, 0.0)
    }
}
