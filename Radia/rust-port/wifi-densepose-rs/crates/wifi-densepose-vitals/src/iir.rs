//! Shared IIR bandpass filter (2nd-order resonator).

/// 2nd-order IIR bandpass filter using a resonator topology.
///
/// Transfer function: `y[n] = (1-r)(x[n]-x[n-2]) + 2r·cos(ω₀)·y[n-1] − r²·y[n-2]`
#[derive(Clone, Debug)]
pub struct IirBandpass {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    r: f64,
    cos_w0: f64,
}

impl IirBandpass {
    /// Create a new bandpass filter for the given frequency band and sample rate.
    #[must_use]
    pub fn new(freq_low: f64, freq_high: f64, sample_rate: f64) -> Self {
        let omega_low = 2.0 * std::f64::consts::PI * freq_low / sample_rate;
        let omega_high = 2.0 * std::f64::consts::PI * freq_high / sample_rate;
        let bw = omega_high - omega_low;
        let center = f64::midpoint(omega_low, omega_high);
        Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
            r: 1.0 - bw / 2.0,
            cos_w0: center.cos(),
        }
    }

    /// Filter one input sample and return the output.
    pub fn filter(&mut self, input: f64) -> f64 {
        let output = (1.0 - self.r) * (input - self.x2)
            + 2.0 * self.r * self.cos_w0 * self.y1
            - self.r * self.r * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    /// Reset filter state to zero.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_produces_output() {
        let mut f = IirBandpass::new(0.1, 0.5, 10.0);
        let out = f.filter(1.0);
        assert!(out.is_finite());
    }

    #[test]
    fn reset_zeroes_state() {
        let mut f = IirBandpass::new(0.8, 2.0, 100.0);
        f.filter(1.0);
        f.filter(2.0);
        f.reset();
        assert_eq!(f.x1, 0.0);
        assert_eq!(f.y1, 0.0);
    }
}
