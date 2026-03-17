//! 5-tap Gaussian blur for motion estimation — spec §4.4.1

use vmaf_cpu::SimdBackend;

/// Coefficients of the 5-tap Gaussian filter (sums to 65536 = 2^16).
const MOTION_FILTER: [u32; 5] = [3571, 16004, 26386, 16004, 3571];

type BlurImpl = fn(&[u16], usize, usize, usize, u8) -> Vec<u16>;

/// Reflective (mirror) padding — spec §4.1.1.
#[inline]
fn reflect(i: i32, len: i32) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= len {
        (2 * len - 2 - i) as usize
    } else {
        i as usize
    }
}

/// Gaussian-blur one luma plane and return the blurred frame as a flat `Vec<u16>`.
///
/// - `src`: flat luma samples in row-major order, `src[row * stride + col]`
/// - `stride`: row stride **in samples** (not bytes)
/// - `bpc`: bits per component (8, 10, or 12)
///
/// Output layout: `blurred[row * width + col]` (stride = width).
pub fn blur_frame(src: &[u16], stride: usize, width: usize, height: usize, bpc: u8) -> Vec<u16> {
    blur_frame_with_backend(
        crate::simd::default_backend(),
        src,
        stride,
        width,
        height,
        bpc,
    )
}

pub(crate) fn blur_frame_with_backend(
    backend: SimdBackend,
    src: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    bpc: u8,
) -> Vec<u16> {
    select_impl(crate::simd::effective_backend(backend))(src, stride, width, height, bpc)
}

fn select_impl(backend: SimdBackend) -> BlurImpl {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(kernel) = x86::select(backend) {
        return kernel;
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(kernel) = aarch64::select(backend) {
        return kernel;
    }

    blur_frame_scalar
}

fn blur_frame_scalar(src: &[u16], stride: usize, width: usize, height: usize, bpc: u8) -> Vec<u16> {
    let n = width * height;
    let mut tmp = vec![0u16; n];
    let mut out = vec![0u16; n];

    vertical_pass_scalar(src, stride, width, height, bpc, &mut tmp);
    horizontal_pass_scalar(&tmp, width, height, &mut out);

    out
}

#[cfg(test)]
pub(crate) fn blur_frame_scalar_reference(
    src: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    bpc: u8,
) -> Vec<u16> {
    blur_frame_scalar(src, stride, width, height, bpc)
}

fn vertical_pass_scalar(
    src: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    bpc: u8,
    tmp: &mut [u16],
) {
    let round_v = 1u32 << (bpc - 1);
    let shift_v = bpc;

    for row in 0..height {
        let out_row = &mut tmp[row * width..(row + 1) * width];
        for (col, out) in out_row.iter_mut().enumerate() {
            *out = vertical_pixel(src, stride, row, col, height, round_v, shift_v);
        }
    }
}

fn horizontal_pass_scalar(tmp: &[u16], width: usize, height: usize, out: &mut [u16]) {
    for row in 0..height {
        let src_row = &tmp[row * width..(row + 1) * width];
        let dst_row = &mut out[row * width..(row + 1) * width];
        horizontal_row_scalar_range(src_row, width, dst_row, 0, width);
    }
}

#[inline]
fn vertical_pixel(
    src: &[u16],
    stride: usize,
    row: usize,
    col: usize,
    height: usize,
    round_v: u32,
    shift_v: u8,
) -> u16 {
    let mut accum = 0u32;
    for (k, &filter_val) in MOTION_FILTER.iter().enumerate() {
        let ii = reflect(row as i32 - 2 + k as i32, height as i32);
        accum = accum.wrapping_add(filter_val * src[ii * stride + col] as u32);
    }
    ((accum + round_v) >> shift_v) as u16
}

#[inline]
fn horizontal_pixel(row: &[u16], width: usize, col: usize) -> u16 {
    let mut accum = 0u32;
    for (k, &filter_val) in MOTION_FILTER.iter().enumerate() {
        let jj = reflect(col as i32 - 2 + k as i32, width as i32);
        accum = accum.wrapping_add(filter_val * row[jj] as u32);
    }
    ((accum + 32768) >> 16) as u16
}

#[inline]
fn horizontal_row_scalar_range(
    row: &[u16],
    width: usize,
    out_row: &mut [u16],
    start: usize,
    end: usize,
) {
    for col in start..end {
        out_row[col] = horizontal_pixel(row, width, col);
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::BlurImpl;
    use vmaf_cpu::SimdBackend;

    pub(super) fn select(_backend: SimdBackend) -> Option<BlurImpl> {
        None
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{
        horizontal_pass_scalar, horizontal_row_scalar_range, reflect, vertical_pass_scalar,
        vertical_pixel, BlurImpl, MOTION_FILTER,
    };
    use vmaf_cpu::{Align32, AlignedScratch, SimdBackend};

    pub(super) fn select(backend: SimdBackend) -> Option<BlurImpl> {
        match backend {
            SimdBackend::X86Avx2Fma => Some(blur_frame_avx2),
            SimdBackend::X86Avx512 if SimdBackend::X86Avx2Fma.is_available() => {
                Some(blur_frame_avx2)
            }
            SimdBackend::X86Sse2 => Some(blur_frame_sse2),
            SimdBackend::X86Avx512 if SimdBackend::X86Sse2.is_available() => Some(blur_frame_sse2),
            _ => None,
        }
    }

    fn blur_frame_sse2(
        src: &[u16],
        stride: usize,
        width: usize,
        height: usize,
        bpc: u8,
    ) -> Vec<u16> {
        let n = width * height;
        let mut tmp = AlignedScratch::<u16, Align32>::zeroed(n);
        let mut out = vec![0u16; n];

        // SAFETY: this wrapper is installed only after runtime SSE2 detection.
        unsafe { vertical_pass_sse2(src, stride, width, height, bpc, tmp.as_mut_slice()) };
        horizontal_pass_scalar(tmp.as_slice(), width, height, &mut out);

        out
    }

    fn blur_frame_avx2(
        src: &[u16],
        stride: usize,
        width: usize,
        height: usize,
        bpc: u8,
    ) -> Vec<u16> {
        let n = width * height;
        let mut tmp = AlignedScratch::<u16, Align32>::zeroed(n);
        let mut out = vec![0u16; n];

        // SAFETY: this wrapper is installed only after runtime AVX2 detection.
        unsafe { vertical_pass_avx2(src, stride, width, height, bpc, tmp.as_mut_slice()) };
        // SAFETY: this wrapper is installed only after runtime AVX2 detection.
        unsafe { horizontal_pass_avx2(tmp.as_slice(), width, height, &mut out) };

        out
    }

    /// SAFETY: caller must ensure SSE2 is available for the current process.
    #[target_feature(enable = "sse2")]
    unsafe fn vertical_pass_sse2(
        src: &[u16],
        stride: usize,
        width: usize,
        height: usize,
        bpc: u8,
        tmp: &mut [u16],
    ) {
        if width < 8 {
            vertical_pass_scalar(src, stride, width, height, bpc, tmp);
            return;
        }

        let round_v = 1u32 << (bpc - 1);
        let shift = _mm_cvtsi32_si128(i32::from(bpc));
        let round = _mm_set1_epi32(round_v as i32);
        let zero = _mm_setzero_si128();
        let bias = _mm_set1_epi32(0x8000);
        let sign = _mm_set1_epi16(0x8000u16 as i16);
        let coeff0 = _mm_set1_epi32(MOTION_FILTER[0] as i32);
        let coeff1 = _mm_set1_epi32(MOTION_FILTER[1] as i32);
        let coeff2 = _mm_set1_epi32(MOTION_FILTER[2] as i32);
        let coeff3 = _mm_set1_epi32(MOTION_FILTER[3] as i32);
        let coeff4 = _mm_set1_epi32(MOTION_FILTER[4] as i32);

        for row in 0..height {
            let rows = row_indices(row, height);
            let r0 = src.as_ptr().add(rows[0] * stride);
            let r1 = src.as_ptr().add(rows[1] * stride);
            let r2 = src.as_ptr().add(rows[2] * stride);
            let r3 = src.as_ptr().add(rows[3] * stride);
            let r4 = src.as_ptr().add(rows[4] * stride);
            let out_row = tmp.as_mut_ptr().add(row * width);
            let mut col = 0usize;

            while col + 8 <= width {
                let s0 = _mm_loadu_si128(r0.add(col).cast());
                let s1 = _mm_loadu_si128(r1.add(col).cast());
                let s2 = _mm_loadu_si128(r2.add(col).cast());
                let s3 = _mm_loadu_si128(r3.add(col).cast());
                let s4 = _mm_loadu_si128(r4.add(col).cast());

                let (mut acc_lo, mut acc_hi) = mul_u16_u16_to_u32_pair_sse2(s0, coeff0, zero);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_sse2(s1, coeff1, zero);
                acc_lo = _mm_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm_add_epi32(acc_hi, prod_hi);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_sse2(s2, coeff2, zero);
                acc_lo = _mm_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm_add_epi32(acc_hi, prod_hi);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_sse2(s3, coeff3, zero);
                acc_lo = _mm_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm_add_epi32(acc_hi, prod_hi);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_sse2(s4, coeff4, zero);
                acc_lo = _mm_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm_add_epi32(acc_hi, prod_hi);

                let values_lo = _mm_srl_epi32(_mm_add_epi32(acc_lo, round), shift);
                let values_hi = _mm_srl_epi32(_mm_add_epi32(acc_hi, round), shift);
                let packed = pack_u32_pair_to_u16_sse2(values_lo, values_hi, bias, sign);
                _mm_storeu_si128(out_row.add(col).cast(), packed);
                col += 8;
            }

            while col < width {
                *out_row.add(col) = vertical_pixel(src, stride, row, col, height, round_v, bpc);
                col += 1;
            }
        }
    }

    /// SAFETY: caller must ensure AVX2 is available for the current process.
    #[target_feature(enable = "avx2")]
    unsafe fn vertical_pass_avx2(
        src: &[u16],
        stride: usize,
        width: usize,
        height: usize,
        bpc: u8,
        tmp: &mut [u16],
    ) {
        if width < 16 {
            vertical_pass_scalar(src, stride, width, height, bpc, tmp);
            return;
        }

        let round_v = 1u32 << (bpc - 1);
        let shift = _mm_cvtsi32_si128(i32::from(bpc));
        let round = _mm256_set1_epi32(round_v as i32);
        let coeff0 = _mm256_set1_epi32(MOTION_FILTER[0] as i32);
        let coeff1 = _mm256_set1_epi32(MOTION_FILTER[1] as i32);
        let coeff2 = _mm256_set1_epi32(MOTION_FILTER[2] as i32);
        let coeff3 = _mm256_set1_epi32(MOTION_FILTER[3] as i32);
        let coeff4 = _mm256_set1_epi32(MOTION_FILTER[4] as i32);

        for row in 0..height {
            let rows = row_indices(row, height);
            let r0 = src.as_ptr().add(rows[0] * stride);
            let r1 = src.as_ptr().add(rows[1] * stride);
            let r2 = src.as_ptr().add(rows[2] * stride);
            let r3 = src.as_ptr().add(rows[3] * stride);
            let r4 = src.as_ptr().add(rows[4] * stride);
            let out_row = tmp.as_mut_ptr().add(row * width);
            let mut col = 0usize;

            while col + 16 <= width {
                let s0 = _mm256_loadu_si256(r0.add(col).cast());
                let s1 = _mm256_loadu_si256(r1.add(col).cast());
                let s2 = _mm256_loadu_si256(r2.add(col).cast());
                let s3 = _mm256_loadu_si256(r3.add(col).cast());
                let s4 = _mm256_loadu_si256(r4.add(col).cast());

                let (mut acc_lo, mut acc_hi) = mul_u16_u16_to_u32_pair_avx2(s0, coeff0);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s1, coeff1);
                acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm256_add_epi32(acc_hi, prod_hi);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s2, coeff2);
                acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm256_add_epi32(acc_hi, prod_hi);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s3, coeff3);
                acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm256_add_epi32(acc_hi, prod_hi);
                let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s4, coeff4);
                acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                acc_hi = _mm256_add_epi32(acc_hi, prod_hi);

                let values_lo = _mm256_srl_epi32(_mm256_add_epi32(acc_lo, round), shift);
                let values_hi = _mm256_srl_epi32(_mm256_add_epi32(acc_hi, round), shift);
                let packed = pack_u32_pair_to_u16_avx2(values_lo, values_hi);
                _mm256_storeu_si256(out_row.add(col).cast(), packed);
                col += 16;
            }

            while col < width {
                *out_row.add(col) = vertical_pixel(src, stride, row, col, height, round_v, bpc);
                col += 1;
            }
        }
    }

    /// SAFETY: caller must ensure AVX2 is available for the current process.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_pass_avx2(tmp: &[u16], width: usize, height: usize, out: &mut [u16]) {
        let round = _mm256_set1_epi32(32768);
        let shift = _mm_cvtsi32_si128(16);
        let coeff0 = _mm256_set1_epi32(MOTION_FILTER[0] as i32);
        let coeff1 = _mm256_set1_epi32(MOTION_FILTER[1] as i32);
        let coeff2 = _mm256_set1_epi32(MOTION_FILTER[2] as i32);
        let coeff3 = _mm256_set1_epi32(MOTION_FILTER[3] as i32);
        let coeff4 = _mm256_set1_epi32(MOTION_FILTER[4] as i32);

        for row in 0..height {
            let src_row = &tmp[row * width..(row + 1) * width];
            let dst_row = &mut out[row * width..(row + 1) * width];
            let prefix_end = width.min(2);
            horizontal_row_scalar_range(src_row, width, dst_row, 0, prefix_end);

            let mut col = prefix_end;
            if width >= 18 {
                col = 2;
                while col + 18 <= width {
                    let s0 = _mm256_loadu_si256(src_row.as_ptr().add(col - 2).cast());
                    let s1 = _mm256_loadu_si256(src_row.as_ptr().add(col - 1).cast());
                    let s2 = _mm256_loadu_si256(src_row.as_ptr().add(col).cast());
                    let s3 = _mm256_loadu_si256(src_row.as_ptr().add(col + 1).cast());
                    let s4 = _mm256_loadu_si256(src_row.as_ptr().add(col + 2).cast());

                    let (mut acc_lo, mut acc_hi) = mul_u16_u16_to_u32_pair_avx2(s0, coeff0);
                    let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s1, coeff1);
                    acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                    acc_hi = _mm256_add_epi32(acc_hi, prod_hi);
                    let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s2, coeff2);
                    acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                    acc_hi = _mm256_add_epi32(acc_hi, prod_hi);
                    let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s3, coeff3);
                    acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                    acc_hi = _mm256_add_epi32(acc_hi, prod_hi);
                    let (prod_lo, prod_hi) = mul_u16_u16_to_u32_pair_avx2(s4, coeff4);
                    acc_lo = _mm256_add_epi32(acc_lo, prod_lo);
                    acc_hi = _mm256_add_epi32(acc_hi, prod_hi);

                    let values_lo = _mm256_srl_epi32(_mm256_add_epi32(acc_lo, round), shift);
                    let values_hi = _mm256_srl_epi32(_mm256_add_epi32(acc_hi, round), shift);
                    let packed = pack_u32_pair_to_u16_avx2(values_lo, values_hi);
                    _mm256_storeu_si256(dst_row.as_mut_ptr().add(col).cast(), packed);
                    col += 16;
                }
            }

            horizontal_row_scalar_range(src_row, width, dst_row, col, width);
        }
    }

    #[inline]
    fn row_indices(row: usize, height: usize) -> [usize; 5] {
        [
            reflect(row as i32 - 2, height as i32),
            reflect(row as i32 - 1, height as i32),
            row,
            reflect(row as i32 + 1, height as i32),
            reflect(row as i32 + 2, height as i32),
        ]
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul_u16_u16_to_u32_pair_sse2(
        samples: __m128i,
        coeff: __m128i,
        zero: __m128i,
    ) -> (__m128i, __m128i) {
        let lo = _mm_unpacklo_epi16(samples, zero);
        let hi = _mm_unpackhi_epi16(samples, zero);
        (_mm_madd_epi16(lo, coeff), _mm_madd_epi16(hi, coeff))
    }

    #[target_feature(enable = "sse2")]
    unsafe fn pack_u32_pair_to_u16_sse2(
        values_lo: __m128i,
        values_hi: __m128i,
        bias: __m128i,
        sign: __m128i,
    ) -> __m128i {
        let biased_lo = _mm_sub_epi32(values_lo, bias);
        let biased_hi = _mm_sub_epi32(values_hi, bias);
        let packed = _mm_packs_epi32(biased_lo, biased_hi);
        _mm_xor_si128(packed, sign)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_u16_u16_to_u32_pair_avx2(samples: __m256i, coeff: __m256i) -> (__m256i, __m256i) {
        let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(samples));
        let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(samples, 1));
        (_mm256_mullo_epi32(lo, coeff), _mm256_mullo_epi32(hi, coeff))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pack_u32_pair_to_u16_avx2(values_lo: __m256i, values_hi: __m256i) -> __m256i {
        let packed = _mm256_packus_epi32(values_lo, values_hi);
        _mm256_permute4x64_epi64(packed, 0xD8)
    }
}
