use std::sync::OnceLock;

use vmaf_cpu::SimdBackend;

pub(crate) fn default_backend() -> SimdBackend {
    static BACKEND: OnceLock<SimdBackend> = OnceLock::new();

    *BACKEND.get_or_init(SimdBackend::detect_effective)
}
