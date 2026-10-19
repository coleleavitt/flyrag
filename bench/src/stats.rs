//! Summary statistics and confidence intervals.

use kenyon::Rng;

/// Mean (0 for empty).
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

/// Sample standard deviation (n - 1).
pub fn sd(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = mean(x);
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

/// Two-sided 97.5% quantile of Student's t with `df` degrees of freedom.
pub fn t975(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::NAN,
        1..=30 => T[df - 1],
        31..=60 => 2.000 + (2.042 - 2.000) * (60 - df) as f64 / 30.0,
        _ => 1.96,
    }
}

/// 95% t-interval for the mean.
pub fn t_ci(x: &[f64]) -> (f64, f64) {
    let m = mean(x);
    let h = t975(x.len().saturating_sub(1)) * sd(x) / (x.len() as f64).sqrt();
    (m - h, m + h)
}

/// 95% percentile-bootstrap interval for the mean (`b` resamples, seeded).
pub fn bootstrap_ci(x: &[f64], b: usize, seed: u64) -> (f64, f64) {
    if x.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut r = Rng::new(seed);
    let mut means: Vec<f64> = (0..b)
        .map(|_| (0..x.len()).map(|_| x[r.below(x.len())]).sum::<f64>() / x.len() as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let q = |p: f64| means[((p * (b - 1) as f64).round() as usize).min(b - 1)];
    (q(0.025), q(0.975))
}

/// Paired-difference summary used by the wiring study.
#[derive(Debug, Clone, Copy)]
pub struct Paired {
    /// Mean of `a - b`.
    pub mean_diff: f64,
    /// 95% t-interval.
    pub t_lo: f64,
    /// 95% t-interval.
    pub t_hi: f64,
    /// 95% bootstrap interval.
    pub boot_lo: f64,
    /// 95% bootstrap interval.
    pub boot_hi: f64,
    /// Cohen's d_z (mean / sd of differences).
    pub dz: f64,
}

/// Summarise paired samples `a[i] - b[i]`.
pub fn paired(a: &[f64], b: &[f64], seed: u64) -> Paired {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let (t_lo, t_hi) = t_ci(&d);
    let (boot_lo, boot_hi) = bootstrap_ci(&d, 10_000, seed);
    let s = sd(&d);
    Paired {
        mean_diff: mean(&d),
        t_lo,
        t_hi,
        boot_lo,
        boot_hi,
        dz: if s > 0.0 { mean(&d) / s } else { f64::NAN },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_contains_mean() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (lo, hi) = t_ci(&x);
        assert!(lo < 3.0 && hi > 3.0);
        let (bl, bh) = bootstrap_ci(&x, 2000, 1);
        assert!(bl < 3.0 && bh > 3.0);
        assert!((t975(19) - 2.093).abs() < 1e-9);
    }
}
