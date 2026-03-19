//! Sum of Absolute Differences for motion score — spec §4.4.2

use vmaf_cpu::SimdBackend;

type SadImpl = fn(&[u16], &[u16], usize, usize) -> f32;

/// Compute the normalised SAD between two blurred frames.
///
/// Both buffers are flat row-major `[u16]` with layout `[row * width + col]`.
///
/// Formula (spec §4.4.2):
/// ```text
/// sad = Σ |buf_a[i][j] - buf_b[i][j]|   (u64 accumulator)
/// return f32(sad) / 256.0_f32 / f32(width * height)
/// ```
/// CRITICAL: cast to f32 **before** dividing by `width * height` — spec note.
#[cfg(test)]
pub(crate) fn compute_sad(buf_a: &[u16], buf_b: &[u16], width: usize, height: usize) -> f32 {
    compute_sad_with_backend(crate::simd::default_backend(), buf_a, buf_b, width, height)
}

pub(crate) fn compute_sad_with_backend(
    backend: SimdBackend,
    buf_a: &[u16],
    buf_b: &[u16],
    width: usize,
    height: usize,
) -> f32 {
    select_impl(backend.effective())(buf_a, buf_b, width, height)
}

fn select_impl(backend: SimdBackend) -> SadImpl {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(kernel) = x86::select(backend) {
        return kernel;
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(kernel) = aarch64::select(backend) {
        return kernel;
    }

    compute_sad_scalar
}

#[inline]
fn normalize_sad(sad: u64, width: usize, height: usize) -> f32 {
    (sad as f32 / 256.0_f32) / (width * height) as f32
}

fn compute_sad_scalar(buf_a: &[u16], buf_b: &[u16], width: usize, height: usize) -> f32 {
    let mut sad = 0u64;
    for (&a, &b) in buf_a.iter().zip(buf_b.iter()) {
        sad += (a as i32 - b as i32).unsigned_abs() as u64;
    }
    normalize_sad(sad, width, height)
}

#[cfg(test)]
pub(crate) fn compute_sad_scalar_reference(
    buf_a: &[u16],
    buf_b: &[u16],
    width: usize,
    height: usize,
) -> f32 {
    compute_sad_scalar(buf_a, buf_b, width, height)
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use vmaf_cpu::SimdBackend;

    use super::SadImpl;

    pub(super) fn select(_backend: SimdBackend) -> Option<SadImpl> {
        None
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use vmaf_cpu::SimdBackend;

    use super::{SadImpl, normalize_sad};

    const AVX2_LANES: usize = 16;
    const SSE2_LANES: usize = 8;
    const FLUSH_ITERS: usize = 32_768;

    pub(super) fn select(backend: SimdBackend) -> Option<SadImpl> {
        match backend {
            SimdBackend::X86Avx2Fma => Some(compute_sad_avx2),
            SimdBackend::X86Avx512 if SimdBackend::X86Avx2Fma.is_available() => {
                Some(compute_sad_avx2)
            }
            SimdBackend::X86Sse2 => Some(compute_sad_sse2),
            SimdBackend::X86Avx512 if SimdBackend::X86Sse2.is_available() => Some(compute_sad_sse2),
            _ => None,
        }
    }

    fn compute_sad_sse2(buf_a: &[u16], buf_b: &[u16], width: usize, height: usize) -> f32 {
        // SAFETY: this wrapper is installed only after runtime SSE2 detection.
        let sad = unsafe { sad_sum_sse2(buf_a, buf_b) };
        normalize_sad(sad, width, height)
    }

    fn compute_sad_avx2(buf_a: &[u16], buf_b: &[u16], width: usize, height: usize) -> f32 {
        // SAFETY: this wrapper is installed only after runtime AVX2 detection.
        let sad = unsafe { sad_sum_avx2(buf_a, buf_b) };
        normalize_sad(sad, width, height)
    }

    /// SAFETY: caller must ensure SSE2 is available for the current process.
    #[target_feature(enable = "sse2")]
    unsafe fn sad_sum_sse2(buf_a: &[u16], buf_b: &[u16]) -> u64 {
        let len = buf_a.len().min(buf_b.len());
        let mut sad = 0u64;
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let chunk_len = SSE2_LANES * FLUSH_ITERS;

        while i + chunk_len <= len {
            let end = i + chunk_len;
            let mut acc_lo = _mm_setzero_si128();
            let mut acc_hi = _mm_setzero_si128();

            while i + SSE2_LANES <= end {
                let a = _mm_loadu_si128(buf_a.as_ptr().add(i).cast());
                let b = _mm_loadu_si128(buf_b.as_ptr().add(i).cast());
                let diff = _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
                acc_lo = _mm_add_epi32(acc_lo, _mm_unpacklo_epi16(diff, zero));
                acc_hi = _mm_add_epi32(acc_hi, _mm_unpackhi_epi16(diff, zero));
                i += SSE2_LANES;
            }

            sad += flush_u32x4_sse2(acc_lo) + flush_u32x4_sse2(acc_hi);
        }

        let mut acc_lo = _mm_setzero_si128();
        let mut acc_hi = _mm_setzero_si128();
        while i + SSE2_LANES <= len {
            let a = _mm_loadu_si128(buf_a.as_ptr().add(i).cast());
            let b = _mm_loadu_si128(buf_b.as_ptr().add(i).cast());
            let diff = _mm_or_si128(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
            acc_lo = _mm_add_epi32(acc_lo, _mm_unpacklo_epi16(diff, zero));
            acc_hi = _mm_add_epi32(acc_hi, _mm_unpackhi_epi16(diff, zero));
            i += SSE2_LANES;
        }
        sad += flush_u32x4_sse2(acc_lo) + flush_u32x4_sse2(acc_hi);

        for (&a, &b) in buf_a[i..len].iter().zip(&buf_b[i..len]) {
            sad += (a as i32 - b as i32).unsigned_abs() as u64;
        }

        sad
    }

    /// SAFETY: caller must ensure AVX2 is available for the current process.
    #[target_feature(enable = "avx2")]
    unsafe fn sad_sum_avx2(buf_a: &[u16], buf_b: &[u16]) -> u64 {
        let len = buf_a.len().min(buf_b.len());
        let mut sad = 0u64;
        let mut i = 0usize;
        let zero = _mm256_setzero_si256();
        let chunk_len = AVX2_LANES * FLUSH_ITERS;

        while i + chunk_len <= len {
            let end = i + chunk_len;
            let mut acc_lo = _mm256_setzero_si256();
            let mut acc_hi = _mm256_setzero_si256();

            while i + AVX2_LANES <= end {
                let a = _mm256_loadu_si256(buf_a.as_ptr().add(i).cast());
                let b = _mm256_loadu_si256(buf_b.as_ptr().add(i).cast());
                let diff = _mm256_or_si256(_mm256_subs_epu16(a, b), _mm256_subs_epu16(b, a));
                acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(diff, zero));
                acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(diff, zero));
                i += AVX2_LANES;
            }

            sad += flush_u32x8_avx2(acc_lo) + flush_u32x8_avx2(acc_hi);
        }

        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        while i + AVX2_LANES <= len {
            let a = _mm256_loadu_si256(buf_a.as_ptr().add(i).cast());
            let b = _mm256_loadu_si256(buf_b.as_ptr().add(i).cast());
            let diff = _mm256_or_si256(_mm256_subs_epu16(a, b), _mm256_subs_epu16(b, a));
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(diff, zero));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(diff, zero));
            i += AVX2_LANES;
        }
        sad += flush_u32x8_avx2(acc_lo) + flush_u32x8_avx2(acc_hi);

        for (&a, &b) in buf_a[i..len].iter().zip(&buf_b[i..len]) {
            sad += (a as i32 - b as i32).unsigned_abs() as u64;
        }

        sad
    }

    #[target_feature(enable = "sse2")]
    unsafe fn flush_u32x4_sse2(acc: __m128i) -> u64 {
        let mut values = [0u32; 4];
        _mm_storeu_si128(values.as_mut_ptr().cast(), acc);
        values.into_iter().map(u64::from).sum()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn flush_u32x8_avx2(acc: __m256i) -> u64 {
        let mut values = [0u32; 8];
        _mm256_storeu_si256(values.as_mut_ptr().cast(), acc);
        values.into_iter().map(u64::from).sum()
    }
}
