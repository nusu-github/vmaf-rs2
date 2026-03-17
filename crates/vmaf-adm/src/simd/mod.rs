use crate::dwt::{dwt_s123 as scalar_dwt_s123, dwt_scale0 as scalar_dwt_scale0, Bands16, Bands32};
use vmaf_cpu::SimdBackend;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

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

pub(crate) fn dwt_scale0(
    backend: SimdBackend,
    src: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
) -> Bands16 {
    match effective_backend(backend) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Avx2Fma => x86::dwt_scale0_avx2(src, width, height, bpc),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Sse2 => x86::dwt_scale0_sse2(src, width, height, bpc),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Aarch64Neon => aarch64::dwt_scale0(src, width, height, bpc),
        _ => scalar_dwt_scale0(src, width, height, bpc),
    }
}

pub(crate) fn dwt_s123(
    backend: SimdBackend,
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
) -> Bands32 {
    match effective_backend(backend) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Avx2Fma => x86::dwt_s123_avx2(ll, width, height, scale),
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Aarch64Neon => aarch64::dwt_s123(ll, width, height, scale),
        _ => scalar_dwt_s123(ll, width, height, scale),
    }
}
