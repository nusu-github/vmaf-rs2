//! Error types for model loading and LIBSVM parsing.

use std::num::{ParseFloatError, ParseIntError};

use thiserror::Error;

/// Semantic validation failures while parsing a VMAF model.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ModelValidationError {
    #[error("unsupported feature_names[{idx}] = {name}")]
    UnsupportedFeatureName { idx: usize, name: String },
    #[error("slopes must have ≥ 7 entries, got {len}")]
    SlopesLen { len: usize },
    #[error("intercepts must have ≥ 7 entries, got {len}")]
    InterceptsLen { len: usize },
    #[error("feature_names must have length 6, got {len}")]
    FeatureNamesLen { len: usize },
    #[error("feature_opts_dicts length must equal feature_names length (6), got {len}")]
    FeatureOptsLen { len: usize },
    #[error("feature_opts_dicts[{idx}].{key} must be number/bool/string, got {found}")]
    FeatureOptionType {
        idx: usize,
        key: String,
        found: &'static str,
    },
    #[error("feature_opts_dicts[{idx}].{key} is not representable as f64")]
    GainLimitNotRepresentable { idx: usize, key: &'static str },
    #[error("feature_opts_dicts[{idx}].{key} must be a JSON number, got {found}")]
    GainLimitType {
        idx: usize,
        key: &'static str,
        found: &'static str,
    },
    #[error("feature_opts_dicts[{idx}].{key} must be finite")]
    GainLimitNonFinite { idx: usize, key: &'static str },
    #[error("feature_opts_dicts[{idx}].{key} must be >= 1.0")]
    GainLimitTooSmall { idx: usize, key: &'static str },
    #[error("vif_enhn_gain_limit must be specified for all 4 VIF scale features")]
    MissingVifGainLimit,
    #[error("vif_enhn_gain_limit must match across all VIF scales")]
    VifGainLimitMismatch,
    #[error("score_clip lower bound ({lower}) must be <= upper bound ({upper})")]
    InvalidScoreClipBounds { lower: f64, upper: f64 },
    #[error("score_transform.knots must have at least 2 points, got {len}")]
    TooFewScoreTransformKnots { len: usize },
    #[error(
        "score_transform.knots x values must be strictly increasing between points {idx} and {next_idx}"
    )]
    NonIncreasingScoreTransformX { idx: usize, next_idx: usize },
    #[error(
        "score_transform.knots y values must be nondecreasing between points {idx} and {next_idx}"
    )]
    DecreasingScoreTransformY { idx: usize, next_idx: usize },
}

/// Parsing failures while decoding embedded LIBSVM text.
#[derive(Debug, Error)]
pub enum LibsvmParseError {
    #[error("gamma: {source}")]
    GammaParse {
        #[source]
        source: ParseFloatError,
    },
    #[error("rho: {source}")]
    RhoParse {
        #[source]
        source: ParseFloatError,
    },
    #[error("coef: {source}")]
    CoefficientParse {
        #[source]
        source: ParseFloatError,
    },
    #[error("index: {source}")]
    FeatureIndexParse {
        #[source]
        source: ParseIntError,
    },
    #[error("value: {source}")]
    FeatureValueParse {
        #[source]
        source: ParseFloatError,
    },
    #[error("missing gamma")]
    MissingGamma,
    #[error("missing rho")]
    MissingRho,
    #[error("empty SV line")]
    EmptySupportVectorLine,
    #[error("bad token: {token}")]
    InvalidSupportVectorToken { token: String },
    #[error("feature index {idx} out of range 1–6")]
    FeatureIndexOutOfRange { idx: usize },
}

/// Top-level model loading failures.
#[derive(Debug, Error)]
pub enum LoadModelError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Validation(#[from] ModelValidationError),
    #[error(transparent)]
    Libsvm(#[from] LibsvmParseError),
}
