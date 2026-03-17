use crate::dwt::{
    dwt_s123_into as scalar_dwt_s123_into, dwt_scale0_into as scalar_dwt_scale0_into, Bands16,
    Bands16Buffer, Bands32, Bands32Buffer, Scale0DwtWorkspace, Scale123DwtWorkspace,
};
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
    let mut workspace = Scale0DwtWorkspace::new(width, height);
    let mut bands = Bands16Buffer::with_capacity(width.div_ceil(2) * height.div_ceil(2));
    dwt_scale0_into(backend, src, width, height, bpc, &mut workspace, &mut bands);
    bands.into_owned()
}

pub(crate) fn dwt_scale0_into(
    backend: SimdBackend,
    src: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    workspace: &mut Scale0DwtWorkspace,
    bands: &mut Bands16Buffer,
) {
    match effective_backend(backend) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Avx2Fma => {
            x86::dwt_scale0_avx2_into(src, width, height, bpc, workspace, bands)
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Sse2 => {
            x86::dwt_scale0_sse2_into(src, width, height, bpc, workspace, bands)
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Aarch64Neon => {
            aarch64::dwt_scale0_into(src, width, height, bpc, workspace, bands)
        }
        _ => scalar_dwt_scale0_into(src, width, height, bpc, workspace, bands),
    }
}

pub(crate) fn dwt_s123(
    backend: SimdBackend,
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
) -> Bands32 {
    let mut workspace = Scale123DwtWorkspace::new(width, height);
    let mut bands = Bands32Buffer::with_capacity(width.div_ceil(2) * height.div_ceil(2));
    dwt_s123_into(
        backend,
        ll,
        width,
        height,
        scale,
        &mut workspace,
        &mut bands,
    );
    bands.into_owned()
}

pub(crate) fn dwt_s123_into(
    backend: SimdBackend,
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
    workspace: &mut Scale123DwtWorkspace,
    bands: &mut Bands32Buffer,
) {
    match effective_backend(backend) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Avx2Fma => {
            x86::dwt_s123_avx2_into(ll, width, height, scale, workspace, bands)
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Aarch64Neon => {
            aarch64::dwt_s123_into(ll, width, height, scale, workspace, bands)
        }
        _ => scalar_dwt_s123_into(ll, width, height, scale, workspace, bands),
    }
}
