//! VMAF model: JSON/LIBSVM loading, SVM Nu-SVR inference, score pipeline — spec §3.3, §5
#![deny(unsafe_code)]

mod json;
mod libsvm;
mod model;
mod normalize;
mod pool;
mod predict;
mod svm;

pub use json::load_model;
pub use model::{ScoreTransform, SupportVector, SvmModel, VmafModel};
pub use normalize::normalize_features;
pub use pool::{collect_scores, pool, PoolMethod};
pub use predict::{denormalize, score_transform};
pub use svm::svm_predict;
