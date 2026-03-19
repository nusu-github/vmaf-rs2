//! VMAF model: JSON/LIBSVM loading, SVM Nu-SVR inference, score pipeline — spec §3.3, §5
#![deny(unsafe_code)]

mod error;
mod json;
mod libsvm;
mod model;
mod normalize;
mod pool;
mod predict;
mod svm;

pub use error::{LibsvmParseError, LoadModelError, ModelValidationError};
pub use json::load_model;
pub use model::{ScoreTransform, SupportVector, SvmModel, VmafModel};
pub use normalize::normalize_features;
pub use pool::{PoolMethod, collect_scores, pool};
pub use predict::{denormalize, score_transform};
pub use svm::svm_predict;
pub use vmaf_cpu::{GainLimit, GainLimitError};
