//! Core data structures — spec §3.3, §5

use vmaf_cpu::GainLimit;

/// One support vector with its dual coefficient.
pub struct SupportVector {
    pub coef: f64,
    pub values: [f64; 6],
}

/// Parsed LIBSVM Nu-SVR model — spec §3.3.
pub struct SvmModel {
    pub gamma: f64,
    pub rho: f64,
    pub support_vectors: Vec<SupportVector>,
}

/// Optional score transformation parameters — spec §5.4.
pub struct ScoreTransform {
    pub p0: Option<f64>,
    pub p1: Option<f64>,
    pub p2: Option<f64>,
    /// Knots for piecewise-linear mapping: `[[x0,y0],[x1,y1],...]`.
    pub knots: Option<Vec<[f64; 2]>>,
    pub out_gte_in: bool,
    pub out_lte_in: bool,
}

/// Complete parsed VMAF model — spec §3.3 + §5.
pub struct VmafModel {
    pub svm: SvmModel,

    /// Feature names in SVM index order (length 6) — spec §3.3.
    pub feature_names: [String; 6],

    /// Per-feature normalization slopes (indices 1–6 from JSON).
    pub feature_slopes: [f64; 6],
    /// Per-feature normalization intercepts (indices 1–6 from JSON).
    pub feature_intercepts: [f64; 6],

    /// Top-level score denormalization slope (`slopes[0]` / `slope` field).
    pub score_slope: f64,
    /// Top-level score denormalization intercept (`intercepts[0]` / `intercept` field).
    pub score_intercept: f64,

    /// Clip range; default `[0.0, 100.0]`.
    pub score_clip: [f64; 2],
    pub score_transform: Option<ScoreTransform>,

    /// ADM enhancement gain limit (default 100.0) — spec §3.3 / §4.3.
    pub adm_enhn_gain_limit: GainLimit,
    /// VIF enhancement gain clamp (default 100.0) — spec §3.3 / §4.2.
    pub vif_enhn_gain_limit: GainLimit,
}
