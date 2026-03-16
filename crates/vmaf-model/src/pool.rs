//! Score pooling — spec §5.5

/// Pooling method.
pub enum PoolMethod {
    Mean,
    HarmonicMean,
    Min,
    Max,
}

/// Select the scores that participate in pooling — spec §5.5.
///
/// Iterates `index_low..=index_high`; skips frame `i` when
/// `n_subsample > 1 && i % n_subsample != 0`.
pub fn collect_scores(
    scores: &[f64],
    index_low: usize,
    index_high: usize,
    n_subsample: usize,
) -> Vec<f64> {
    (index_low..=index_high)
        .filter(|&i| n_subsample <= 1 || i % n_subsample == 0)
        .map(|i| scores[i])
        .collect()
}

/// Pool a non-empty slice of scores using the given method.
///
/// Panics if `scores` is empty (spec precondition).
pub fn pool(scores: &[f64], method: PoolMethod) -> f64 {
    assert!(!scores.is_empty(), "pool: score slice must be non-empty");
    match method {
        PoolMethod::Mean => mean(scores),
        PoolMethod::HarmonicMean => harmonic_mean(scores),
        PoolMethod::Min => scores.iter().cloned().fold(f64::INFINITY, f64::min),
        PoolMethod::Max => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn mean(scores: &[f64]) -> f64 {
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// `(N / Σ 1/(sᵢ + 1)) − 1` — spec §5.5.
fn harmonic_mean(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    let sum_recip: f64 = scores.iter().map(|&s| 1.0 / (s + 1.0)).sum();
    (n / sum_recip) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_simple() {
        assert_eq!(pool(&[10.0, 20.0, 30.0], PoolMethod::Mean), 20.0);
    }

    #[test]
    fn mean_single() {
        assert_eq!(pool(&[42.0], PoolMethod::Mean), 42.0);
    }

    /// Harmonic mean of identical values equals those values.
    #[test]
    fn harmonic_mean_identical() {
        let v = pool(&[3.0, 3.0], PoolMethod::HarmonicMean);
        assert!((v - 3.0).abs() < 1e-10, "expected 3.0, got {v}");
    }

    /// (2 / (1/2 + 1/3)) − 1 = 12/5 − 1 = 7/5 = 1.4
    #[test]
    fn harmonic_mean_two_values() {
        let v = pool(&[1.0, 2.0], PoolMethod::HarmonicMean);
        assert!((v - 1.4).abs() < 1e-10, "expected 1.4, got {v}");
    }

    #[test]
    fn min_max() {
        assert_eq!(pool(&[5.0, 1.0, 3.0], PoolMethod::Min), 1.0);
        assert_eq!(pool(&[5.0, 1.0, 3.0], PoolMethod::Max), 5.0);
    }

    /// n_subsample=1: all frames included.
    #[test]
    fn collect_all() {
        let scores = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(collect_scores(&scores, 0, 3, 1), vec![0.0, 1.0, 2.0, 3.0]);
    }

    /// n_subsample=2: only even-indexed frames (0, 2, 4, …).
    #[test]
    fn collect_subsample_2() {
        let scores = vec![0.0, 1.0, 2.0, 3.0];
        assert_eq!(collect_scores(&scores, 0, 3, 2), vec![0.0, 2.0]);
    }

    /// Subrange selection.
    #[test]
    fn collect_subrange() {
        let scores = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(collect_scores(&scores, 1, 3, 1), vec![20.0, 30.0, 40.0]);
    }
}
