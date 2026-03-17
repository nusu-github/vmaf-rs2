use crate::SimdBackend;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

pub(crate) fn best_available_backend() -> SimdBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::best_available_backend()
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        target_arch = "aarch64"
    ))]
    {
        aarch64::best_available_backend()
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        not(target_arch = "aarch64")
    ))]
    {
        SimdBackend::Scalar
    }
}

pub(crate) fn backend_available(backend: SimdBackend) -> bool {
    match backend {
        SimdBackend::Scalar => true,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Sse2 | SimdBackend::X86Avx2Fma | SimdBackend::X86Avx512 => {
            x86::backend_available(backend)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        SimdBackend::X86Sse2 | SimdBackend::X86Avx2Fma | SimdBackend::X86Avx512 => false,
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Aarch64Neon => aarch64::backend_available(),
        #[cfg(not(target_arch = "aarch64"))]
        SimdBackend::Aarch64Neon => false,
    }
}
