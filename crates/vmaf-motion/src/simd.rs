use std::sync::OnceLock;

use vmaf_cpu::SimdBackend;

pub(crate) fn default_backend() -> SimdBackend {
    static BACKEND: OnceLock<SimdBackend> = OnceLock::new();

    *BACKEND.get_or_init(select_backend)
}

pub(crate) fn select_backend() -> SimdBackend {
    effective_backend(SimdBackend::detect())
}

pub(crate) fn effective_backend(backend: SimdBackend) -> SimdBackend {
    if !backend.is_available() {
        return SimdBackend::Scalar;
    }

    match backend {
        SimdBackend::X86Avx512 => {
            if SimdBackend::X86Avx2Fma.is_available() {
                SimdBackend::X86Avx2Fma
            } else if SimdBackend::X86Sse2.is_available() {
                SimdBackend::X86Sse2
            } else {
                SimdBackend::Scalar
            }
        }
        other => other,
    }
}
