use crate::SimdBackend;

pub(crate) fn best_available_backend() -> SimdBackend {
    if backend_available() {
        SimdBackend::Aarch64Neon
    } else {
        SimdBackend::Scalar
    }
}

pub(crate) fn backend_available() -> bool {
    // `cpufeatures` does not currently expose `neon` probing on `aarch64`.
    // Advanced SIMD is part of the baseline architecture for mainstream
    // `aarch64` targets, so this module acts as the future NEON hook while the
    // shared backend enum and buffer APIs remain target-neutral.
    cfg!(target_feature = "neon") || cfg!(target_arch = "aarch64")
}
