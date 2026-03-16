//! Nu-SVR prediction with RBF kernel — spec §5.2

use crate::model::SvmModel;

/// RBF kernel: `exp(−γ · Σ(xₖ − sₖ)²)` — spec §5.2.
pub fn rbf_kernel(x: &[f64; 6], sv: &[f64; 6], gamma: f64) -> f64 {
    let sq_diff: f64 = x
        .iter()
        .zip(sv.iter())
        .map(|(&xi, &si)| (xi - si) * (xi - si))
        .sum();
    libm::exp(-gamma * sq_diff)
}

/// Nu-SVR decision: `Σ αᵢ·K(x, SVᵢ) − ρ` — spec §5.2.
pub fn svm_predict(model: &SvmModel, x: &[f64; 6]) -> f64 {
    let result: f64 = model
        .support_vectors
        .iter()
        .map(|sv| sv.coef * rbf_kernel(x, &sv.values, model.gamma))
        .sum();
    result - model.rho
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SupportVector, SvmModel};

    fn zeros() -> [f64; 6] {
        [0.0; 6]
    }

    /// K(x, x) = exp(0) = 1.0 for any x.
    #[test]
    fn rbf_identical_vectors() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(rbf_kernel(&x, &x, 0.04), 1.0);
    }

    /// Unit difference in one dimension: K = exp(−0.04 · 1) = exp(−0.04).
    #[test]
    fn rbf_unit_diff() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = zeros();
        let k = rbf_kernel(&a, &b, 0.04);
        let exp = libm::exp(-0.04_f64);
        assert!((k - exp).abs() < 1e-15, "got {k}, expected {exp}");
    }

    /// §8 conformance: exp(−1.0) must equal 0.36787944117144233.
    #[test]
    fn exp_conformance_vector() {
        // sq_diff = 25, gamma = 0.04 → −0.04 * 25 = −1.0
        let a = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = zeros();
        let k = rbf_kernel(&a, &b, 0.04);
        assert_eq!(k, 0.36787944117144233_f64);
    }

    /// Single SV with coef=1, sv=zeros, rho=0.5, query=zeros → 1·1 − 0.5 = 0.5.
    #[test]
    fn svm_predict_trivial() {
        let model = SvmModel {
            gamma: 0.04,
            rho: 0.5,
            support_vectors: vec![SupportVector {
                coef: 1.0,
                values: zeros(),
            }],
        };
        let result = svm_predict(&model, &zeros());
        assert!((result - 0.5).abs() < 1e-15, "got {result}");
    }

    /// Two SVs that cancel: coef=+1 and coef=−1, same SV, rho=0 → result=0.
    #[test]
    fn svm_predict_cancels() {
        let model = SvmModel {
            gamma: 0.04,
            rho: 0.0,
            support_vectors: vec![
                SupportVector {
                    coef: 1.0,
                    values: zeros(),
                },
                SupportVector {
                    coef: -1.0,
                    values: zeros(),
                },
            ],
        };
        let result = svm_predict(&model, &zeros());
        assert!(result.abs() < 1e-15, "got {result}");
    }
}
