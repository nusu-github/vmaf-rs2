//! Score denormalization and transformation pipeline — spec §5.3, §5.4

use crate::model::ScoreTransform;

/// Denormalize the raw SVM output — spec §5.3.
///
/// `raw = svm_predict(...)`
/// `denormalized = (raw − intercept) / slope`
pub fn denormalize(raw: f64, slope: f64, intercept: f64) -> f64 {
    (raw - intercept) / slope
}

/// Apply the full score transformation pipeline — spec §5.4.
///
/// Steps (in order):
/// 1. Polynomial  `p0 + p1·x + p2·x²`
/// 2. Piecewise-linear mapping via knots
/// 3. Rectification (`out_gte_in` / `out_lte_in`)
/// 4. Clip to `[clip[0], clip[1]]`
pub fn score_transform(score_in: f64, st: Option<&ScoreTransform>, clip: [f64; 2]) -> f64 {
    let mut score = score_in;

    if let Some(st) = st {
        // Step 1: polynomial
        if st.p0.is_some() || st.p1.is_some() || st.p2.is_some() {
            let mut poly = 0.0_f64;
            if let Some(p0) = st.p0 {
                poly += p0;
            }
            if let Some(p1) = st.p1 {
                poly += p1 * score;
            }
            if let Some(p2) = st.p2 {
                poly += p2 * score * score;
            }
            score = poly;
        }

        // Step 2: piecewise-linear knots
        if let Some(knots) = &st.knots {
            if knots.len() >= 2 {
                score = piecewise_linear(score, knots);
            }
        }

        // Step 3: rectification (compared against original score_in, not post-poly)
        if st.out_lte_in {
            score = score.min(score_in);
        }
        if st.out_gte_in {
            score = score.max(score_in);
        }
    }

    // Step 4: clip
    score.clamp(clip[0], clip[1])
}

/// Piecewise-linear interpolation with first/last-segment extrapolation — spec §5.4.
fn piecewise_linear(x: f64, knots: &[[f64; 2]]) -> f64 {
    let n_seg = knots.len() - 1;
    for i in 0..=n_seg {
        let [x0, y0] = knots[i];

        // Last-segment extrapolation (x > x1 of last segment)
        if i == n_seg {
            let [xi, yi] = knots[i - 1];
            let slope = if (x0 - xi).abs() < f64::EPSILON {
                0.0
            } else {
                (y0 - yi) / (x0 - xi)
            };
            return yi + slope * (x - xi);
        }

        let [x1, y1] = knots[i + 1];

        // First-segment extrapolation
        if i == 0 && x < x0 {
            let slope = if (x1 - x0).abs() < f64::EPSILON {
                0.0
            } else {
                (y1 - y0) / (x1 - x0)
            };
            return y0 + slope * (x - x0);
        }

        if x <= x1 {
            if (y1 - y0).abs() < f64::EPSILON {
                return y0; // horizontal segment
            }
            let slope = (y1 - y0) / (x1 - x0);
            return y0 + slope * (x - x0);
        }
    }
    x // unreachable for well-formed knots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ScoreTransform;

    // --- denormalize ---

    /// (0.0 − (−0.3092981928)) / 0.012020766 ≈ 25.73
    #[test]
    fn denormalize_model_values() {
        let v = denormalize(0.0, 0.012020766, -0.3092981928);
        assert!((v - 25.731).abs() < 0.001, "got {v}");
    }

    #[test]
    fn denormalize_roundtrip() {
        // If raw = intercept + slope * x, then denormalize gives back x.
        let slope = 0.012020766_f64;
        let intercept = -0.3092981928_f64;
        let x = 73.42_f64;
        let raw = intercept + slope * x;
        let back = denormalize(raw, slope, intercept);
        assert!((back - x).abs() < 1e-9, "got {back}");
    }

    // --- piecewise_linear ---

    #[test]
    fn piecewise_identity_knots() {
        // [[0,0],[100,100]] → identity
        let knots = [[0.0, 0.0], [100.0, 100.0]];
        assert!((piecewise_linear(50.0, &knots) - 50.0).abs() < 1e-10);
        assert!((piecewise_linear(0.0, &knots) - 0.0).abs() < 1e-10);
        assert!((piecewise_linear(100.0, &knots) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn piecewise_slope2() {
        // [[0,0],[50,100]] → slope = 2
        let knots = [[0.0, 0.0], [50.0, 100.0]];
        assert!((piecewise_linear(25.0, &knots) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn piecewise_extrapolate_left() {
        let knots = [[0.0, 0.0], [100.0, 100.0]];
        // slope=1, x=-5 → -5
        assert!((piecewise_linear(-5.0, &knots) - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn piecewise_extrapolate_right() {
        let knots = [[0.0, 0.0], [100.0, 100.0]];
        // slope=1, x=110 → 110
        assert!((piecewise_linear(110.0, &knots) - 110.0).abs() < 1e-10);
    }

    #[test]
    fn piecewise_horizontal_segment() {
        let knots = [[0.0, 5.0], [100.0, 5.0]];
        assert!((piecewise_linear(50.0, &knots) - 5.0).abs() < 1e-10);
    }

    // --- score_transform (full pipeline) ---

    #[test]
    fn v0_6_1_reference_compat_vs_transform() {
        // Values taken from the reference `./vmaf` JSON for:
        //   video/original.y4m vs video/noise.y4m, model/vmaf_v0.6.1.json
        // Frame 85 is a good canary: denorm≈61.2293, transform≈80.9733.
        let model = crate::load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap();

        // Feature order matches the model: [adm2, motion2, vif0..3]
        let raw: [f64; 6] = [0.874602, 3.447131, 0.195294, 0.629553, 0.797059, 0.880153];

        let normed =
            crate::normalize_features(&raw, &model.feature_slopes, &model.feature_intercepts);
        let raw_svm = crate::svm_predict(&model.svm, &normed);
        let denorm = denormalize(raw_svm, model.score_slope, model.score_intercept);

        // Libvmaf (as used by ./vmaf here) behaves like "denorm + clip".
        let compat = score_transform(denorm, None, model.score_clip);
        assert!((compat - 61.229368).abs() < 1e-3, "compat={compat}");

        // Spec behavior: apply polynomial + out_gte_in rectification.
        let transformed = score_transform(denorm, model.score_transform.as_ref(), model.score_clip);
        assert!(
            (transformed - 80.973290).abs() < 1e-3,
            "transformed={transformed}"
        );
    }

    fn model_transform() -> ScoreTransform {
        ScoreTransform {
            p0: Some(1.70674692),
            p1: Some(1.72643844),
            p2: Some(-0.00705305),
            knots: Some(vec![[0.0, 0.0], [100.0, 100.0]]),
            out_gte_in: true,
            out_lte_in: false,
        }
    }

    /// score_in=50 → poly≈70.4 → piecewise identity → out_gte_in: max(70.4,50)=70.4 → clip ok
    #[test]
    fn transform_typical_score() {
        let st = model_transform();
        let v = score_transform(50.0, Some(&st), [0.0, 100.0]);
        // poly = 1.70674692 + 1.72643844*50 - 0.00705305*2500
        //      = 1.70674692 + 86.321922 - 17.632625 ≈ 70.3961
        assert!((v - 70.396).abs() < 0.001, "got {v}");
    }

    /// score_in=0.0 → poly=p0≈1.707 → out_gte_in: max(1.707,0.0)=1.707
    #[test]
    fn transform_zero_score() {
        let st = model_transform();
        let v = score_transform(0.0, Some(&st), [0.0, 100.0]);
        assert!((v - 1.70674692).abs() < 1e-6, "got {v}");
    }

    /// out_gte_in ensures transformed score never goes below score_in.
    #[test]
    fn transform_out_gte_in_clamps_down() {
        // Make a transform whose polynomial would give a value lower than score_in.
        let st = ScoreTransform {
            p0: Some(-100.0), // forces output low
            p1: None,
            p2: None,
            knots: None,
            out_gte_in: true,
            out_lte_in: false,
        };
        let v = score_transform(90.0, Some(&st), [0.0, 100.0]);
        // poly = -100, out_gte_in → max(-100, 90) = 90
        assert!((v - 90.0).abs() < 1e-10, "got {v}");
    }

    /// Clip is applied after rectification.
    #[test]
    fn transform_clip() {
        let st = ScoreTransform {
            p0: Some(150.0),
            p1: None,
            p2: None,
            knots: None,
            out_gte_in: false,
            out_lte_in: false,
        };
        let v = score_transform(0.0, Some(&st), [0.0, 100.0]);
        assert_eq!(v, 100.0);
    }
}
