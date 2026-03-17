use crate::SimdBackend;

cpufeatures::new!(x86_sse2, "sse2");
cpufeatures::new!(x86_avx2_fma, "avx2", "fma");
cpufeatures::new!(x86_avx512f, "avx512f");

pub(crate) fn best_available_backend() -> SimdBackend {
    if x86_avx512f::get() {
        SimdBackend::X86Avx512
    } else if x86_avx2_fma::get() {
        SimdBackend::X86Avx2Fma
    } else if x86_sse2::get() {
        SimdBackend::X86Sse2
    } else {
        SimdBackend::Scalar
    }
}

pub(crate) fn backend_available(backend: SimdBackend) -> bool {
    match backend {
        SimdBackend::Scalar => true,
        SimdBackend::X86Sse2 => x86_sse2::get(),
        SimdBackend::X86Avx2Fma => x86_avx2_fma::get(),
        SimdBackend::X86Avx512 => x86_avx512f::get(),
        SimdBackend::Aarch64Neon => false,
    }
}
