//! Feature normalization — spec §5.1
//!
//! `f_normalized[i] = slopes[i+1] * f_raw[i] + intercepts[i+1]`
//! (1-based indexing in the JSON array, hence `[i+1]` maps to `slopes[i]` here)

/// Apply per-feature linear rescaling.
///
/// `slopes` and `intercepts` are the 6 feature-level values
/// (JSON `slopes[1..6]` / `intercepts[1..6]`).
pub fn normalize_features(raw: &[f64; 6], slopes: &[f64; 6], intercepts: &[f64; 6]) -> [f64; 6] {
    std::array::from_fn(|i| slopes[i] * raw[i] + intercepts[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    // Model values from spec §3.3
    const SLOPES: [f64; 6] = [
        2.8098077503,
        0.0626440747,
        1.2227634563,
        1.5360318811,
        1.7620864996,
        2.0865646829,
    ];
    const INTERCEPTS: [f64; 6] = [
        -1.7993969, -0.0030172, -0.1728125, -0.5294309, -0.7577186, -1.0834286,
    ];

    /// All-zero input → output equals intercepts.
    #[test]
    fn normalize_zeros() {
        let out = normalize_features(&[0.0; 6], &SLOPES, &INTERCEPTS);
        for (i, (&got, &exp)) in out.iter().zip(INTERCEPTS.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "index {i}: got {got}, expected {exp}"
            );
        }
    }

    /// All-ones input → output = slope + intercept per feature.
    #[test]
    fn normalize_ones() {
        let out = normalize_features(&[1.0; 6], &SLOPES, &INTERCEPTS);
        for i in 0..6 {
            let exp = SLOPES[i] + INTERCEPTS[i];
            assert!(
                (out[i] - exp).abs() < 1e-10,
                "index {i}: got {}, expected {exp}",
                out[i]
            );
        }
    }

    /// Spot-check: adm2 feature (index 0) at value 0.8.
    /// 2.8098077503 * 0.8 + (−1.7993969) = 2.2478462002 − 1.7993969 = 0.4484493002
    #[test]
    fn normalize_adm2_spot() {
        let mut raw = [0.0; 6];
        raw[0] = 0.8;
        let out = normalize_features(&raw, &SLOPES, &INTERCEPTS);
        let exp = 2.8098077503 * 0.8 + (-1.7993969);
        assert!((out[0] - exp).abs() < 1e-10, "got {}", out[0]);
    }
}
