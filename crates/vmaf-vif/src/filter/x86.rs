#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use vmaf_cpu::SimdBackend;

use super::{
    FILTER_TAP_CAP, HORIZONTAL_ROUND, HORIZONTAL_SHIFT, SubsampleWorkspace,
    decimate_filtered_row_into, horizontal_scalar_range, horizontal_simd_body_range,
    reflected_row_offsets, vertical_scalar_range,
};
use crate::tables::{FILTER, FILTER_WIDTH};

enum X86FilterKernel {
    Sse2,
    Avx2,
}

pub(super) fn subsample_into(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    backend: SimdBackend,
    workspace: &mut SubsampleWorkspace,
    out_ref: &mut Vec<u16>,
    out_dis: &mut Vec<u16>,
) -> (usize, usize) {
    let filt = &FILTER[scale + 1][..FILTER_WIDTH[scale + 1]];
    let half = filt.len() / 2;
    let (shift_v, round_v) = if scale == 0 {
        (bpc as u32, 1u32 << (bpc - 1))
    } else {
        (16u32, 32768u32)
    };
    let out_w = width / 2;
    let out_h = height / 2;
    let out_len = out_w * out_h;

    out_ref.resize(out_len, 0);
    out_dis.resize(out_len, 0);

    workspace.prepare_rows(width);
    let tmp_ref_row = &mut workspace.tmp_ref_row.as_mut_slice()[..width];
    let tmp_dis_row = &mut workspace.tmp_dis_row.as_mut_slice()[..width];
    let filt_ref_row = &mut workspace.filt_ref_row.as_mut_slice()[..width];
    let filt_dis_row = &mut workspace.filt_dis_row.as_mut_slice()[..width];

    match select_kernel(backend) {
        X86FilterKernel::Avx2 => {
            // SAFETY: `select_kernel` only returns `Avx2` when runtime detection
            // has already proved the current process supports AVX2.
            unsafe {
                let mut coeff32_vecs = [_mm256_setzero_si256(); FILTER_TAP_CAP];
                for tap in 0..filt.len() {
                    coeff32_vecs[tap] = _mm256_set1_epi32(filt[tap] as i32);
                }

                for out_i in 0..out_h {
                    let src_i = out_i * 2;
                    let row_offsets = reflected_row_offsets(src_i, height, width, half, filt.len());
                    vertical_row_avx2(
                        ref_in,
                        dis_in,
                        &row_offsets[..filt.len()],
                        filt,
                        &coeff32_vecs,
                        shift_v,
                        round_v,
                        tmp_ref_row,
                        tmp_dis_row,
                    );
                    horizontal_row_avx2(
                        tmp_ref_row,
                        tmp_dis_row,
                        filt,
                        &coeff32_vecs,
                        half,
                        filt_ref_row,
                        filt_dis_row,
                    );

                    let dst = out_i * out_w;
                    decimate_filtered_row_into(
                        filt_ref_row,
                        filt_dis_row,
                        &mut out_ref[dst..dst + out_w],
                        &mut out_dis[dst..dst + out_w],
                    );
                }
            }
        }
        X86FilterKernel::Sse2 => {
            // SAFETY: x86 dispatch only reaches this module after runtime
            // detection confirmed at least SSE2 support.
            unsafe {
                let mut coeff16_vecs = [_mm_setzero_si128(); FILTER_TAP_CAP];
                for tap in 0..filt.len() {
                    coeff16_vecs[tap] = _mm_set1_epi16(filt[tap] as i16);
                }

                for out_i in 0..out_h {
                    let src_i = out_i * 2;
                    let row_offsets = reflected_row_offsets(src_i, height, width, half, filt.len());
                    vertical_row_sse2(
                        ref_in,
                        dis_in,
                        &row_offsets[..filt.len()],
                        filt,
                        &coeff16_vecs,
                        shift_v,
                        round_v,
                        tmp_ref_row,
                        tmp_dis_row,
                    );
                    horizontal_row_sse2(
                        tmp_ref_row,
                        tmp_dis_row,
                        filt,
                        &coeff16_vecs,
                        half,
                        filt_ref_row,
                        filt_dis_row,
                    );

                    let dst = out_i * out_w;
                    decimate_filtered_row_into(
                        filt_ref_row,
                        filt_dis_row,
                        &mut out_ref[dst..dst + out_w],
                        &mut out_dis[dst..dst + out_w],
                    );
                }
            }
        }
    }

    (out_w, out_h)
}

#[inline]
fn select_kernel(backend: SimdBackend) -> X86FilterKernel {
    match backend {
        SimdBackend::X86Avx2Fma => X86FilterKernel::Avx2,
        SimdBackend::X86Avx512 if SimdBackend::X86Avx2Fma.is_available() => X86FilterKernel::Avx2,
        _ => X86FilterKernel::Sse2,
    }
}

/// SAFETY: the caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
unsafe fn vertical_row_sse2(
    ref_in: &[u16],
    dis_in: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    coeff16_vecs: &[__m128i; FILTER_TAP_CAP],
    shift: u32,
    round: u32,
    tmp_ref_row: &mut [u16],
    tmp_dis_row: &mut [u16],
) {
    let simd_end = (tmp_ref_row.len() / 8) * 8;
    let round_vec = _mm_set1_epi32(round as i32);
    let shift_vec = _mm_cvtsi32_si128(shift as i32);
    for j in (0..simd_end).step_by(8) {
        let mut acc_ref_lo = _mm_setzero_si128();
        let mut acc_ref_hi = _mm_setzero_si128();
        let mut acc_dis_lo = _mm_setzero_si128();
        let mut acc_dis_hi = _mm_setzero_si128();

        for tap in 0..coeffs.len() {
            let coeff16 = coeff16_vecs[tap];
            let ref_values = _mm_loadu_si128(ref_in.as_ptr().add(row_offsets[tap] + j).cast());
            let dis_values = _mm_loadu_si128(dis_in.as_ptr().add(row_offsets[tap] + j).cast());
            let (ref_lo, ref_hi) = mul_u16x8_u32x8(ref_values, coeff16);
            let (dis_lo, dis_hi) = mul_u16x8_u32x8(dis_values, coeff16);
            acc_ref_lo = _mm_add_epi32(acc_ref_lo, ref_lo);
            acc_ref_hi = _mm_add_epi32(acc_ref_hi, ref_hi);
            acc_dis_lo = _mm_add_epi32(acc_dis_lo, dis_lo);
            acc_dis_hi = _mm_add_epi32(acc_dis_hi, dis_hi);
        }

        let acc_ref_lo = _mm_srl_epi32(_mm_add_epi32(acc_ref_lo, round_vec), shift_vec);
        let acc_ref_hi = _mm_srl_epi32(_mm_add_epi32(acc_ref_hi, round_vec), shift_vec);
        let acc_dis_lo = _mm_srl_epi32(_mm_add_epi32(acc_dis_lo, round_vec), shift_vec);
        let acc_dis_hi = _mm_srl_epi32(_mm_add_epi32(acc_dis_hi, round_vec), shift_vec);
        store_u32x8_as_u16(tmp_ref_row, j, acc_ref_lo, acc_ref_hi);
        store_u32x8_as_u16(tmp_dis_row, j, acc_dis_lo, acc_dis_hi);
    }

    vertical_scalar_range(
        ref_in,
        dis_in,
        row_offsets,
        coeffs,
        shift,
        round,
        simd_end,
        tmp_ref_row.len(),
        tmp_ref_row,
        tmp_dis_row,
    );
}

/// SAFETY: the caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
unsafe fn horizontal_row_sse2(
    tmp_ref: &[u16],
    tmp_dis: &[u16],
    coeffs: &[u16],
    coeff16_vecs: &[__m128i; FILTER_TAP_CAP],
    half: usize,
    filt_ref: &mut [u16],
    filt_dis: &mut [u16],
) {
    let round_vec = _mm_set1_epi32(HORIZONTAL_ROUND as i32);
    let width = tmp_ref.len();
    let (body_start, simd_end) = horizontal_simd_body_range(width, half, 8);

    horizontal_scalar_range(
        tmp_ref, tmp_dis, coeffs, half, 0, body_start, filt_ref, filt_dis,
    );

    for j in (body_start..simd_end).step_by(8) {
        let base_start = j - half;
        let mut acc_ref_lo = _mm_setzero_si128();
        let mut acc_ref_hi = _mm_setzero_si128();
        let mut acc_dis_lo = _mm_setzero_si128();
        let mut acc_dis_hi = _mm_setzero_si128();

        for tap in 0..coeffs.len() {
            let coeff16 = coeff16_vecs[tap];
            let base = base_start + tap;
            let ref_values = _mm_loadu_si128(tmp_ref.as_ptr().add(base).cast());
            let dis_values = _mm_loadu_si128(tmp_dis.as_ptr().add(base).cast());
            let (ref_lo, ref_hi) = mul_u16x8_u32x8(ref_values, coeff16);
            let (dis_lo, dis_hi) = mul_u16x8_u32x8(dis_values, coeff16);
            acc_ref_lo = _mm_add_epi32(acc_ref_lo, ref_lo);
            acc_ref_hi = _mm_add_epi32(acc_ref_hi, ref_hi);
            acc_dis_lo = _mm_add_epi32(acc_dis_lo, dis_lo);
            acc_dis_hi = _mm_add_epi32(acc_dis_hi, dis_hi);
        }

        let acc_ref_lo = _mm_srli_epi32(
            _mm_add_epi32(acc_ref_lo, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        let acc_ref_hi = _mm_srli_epi32(
            _mm_add_epi32(acc_ref_hi, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        let acc_dis_lo = _mm_srli_epi32(
            _mm_add_epi32(acc_dis_lo, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        let acc_dis_hi = _mm_srli_epi32(
            _mm_add_epi32(acc_dis_hi, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        store_u32x8_as_u16(filt_ref, j, acc_ref_lo, acc_ref_hi);
        store_u32x8_as_u16(filt_dis, j, acc_dis_lo, acc_dis_hi);
    }

    horizontal_scalar_range(
        tmp_ref, tmp_dis, coeffs, half, simd_end, width, filt_ref, filt_dis,
    );
}

/// SAFETY: the caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
unsafe fn vertical_row_avx2(
    ref_in: &[u16],
    dis_in: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    coeff32_vecs: &[__m256i; FILTER_TAP_CAP],
    shift: u32,
    round: u32,
    tmp_ref_row: &mut [u16],
    tmp_dis_row: &mut [u16],
) {
    let simd_end = (tmp_ref_row.len() / 16) * 16;
    let round_vec = _mm256_set1_epi32(round as i32);
    let shift_vec = _mm_cvtsi32_si128(shift as i32);
    for j in (0..simd_end).step_by(16) {
        let mut acc_ref_lo = _mm256_setzero_si256();
        let mut acc_ref_hi = _mm256_setzero_si256();
        let mut acc_dis_lo = _mm256_setzero_si256();
        let mut acc_dis_hi = _mm256_setzero_si256();

        for tap in 0..coeffs.len() {
            let coeff32 = coeff32_vecs[tap];
            let ref_values = _mm256_loadu_si256(ref_in.as_ptr().add(row_offsets[tap] + j).cast());
            let dis_values = _mm256_loadu_si256(dis_in.as_ptr().add(row_offsets[tap] + j).cast());
            let (ref_lo, ref_hi) = mul_u16x16_u32x16(ref_values, coeff32);
            let (dis_lo, dis_hi) = mul_u16x16_u32x16(dis_values, coeff32);
            acc_ref_lo = _mm256_add_epi32(acc_ref_lo, ref_lo);
            acc_ref_hi = _mm256_add_epi32(acc_ref_hi, ref_hi);
            acc_dis_lo = _mm256_add_epi32(acc_dis_lo, dis_lo);
            acc_dis_hi = _mm256_add_epi32(acc_dis_hi, dis_hi);
        }

        let acc_ref_lo = _mm256_srl_epi32(_mm256_add_epi32(acc_ref_lo, round_vec), shift_vec);
        let acc_ref_hi = _mm256_srl_epi32(_mm256_add_epi32(acc_ref_hi, round_vec), shift_vec);
        let acc_dis_lo = _mm256_srl_epi32(_mm256_add_epi32(acc_dis_lo, round_vec), shift_vec);
        let acc_dis_hi = _mm256_srl_epi32(_mm256_add_epi32(acc_dis_hi, round_vec), shift_vec);
        store_u32x16_as_u16(tmp_ref_row, j, acc_ref_lo, acc_ref_hi);
        store_u32x16_as_u16(tmp_dis_row, j, acc_dis_lo, acc_dis_hi);
    }

    vertical_scalar_range(
        ref_in,
        dis_in,
        row_offsets,
        coeffs,
        shift,
        round,
        simd_end,
        tmp_ref_row.len(),
        tmp_ref_row,
        tmp_dis_row,
    );
}

/// SAFETY: the caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
unsafe fn horizontal_row_avx2(
    tmp_ref: &[u16],
    tmp_dis: &[u16],
    coeffs: &[u16],
    coeff32_vecs: &[__m256i; FILTER_TAP_CAP],
    half: usize,
    filt_ref: &mut [u16],
    filt_dis: &mut [u16],
) {
    let round_vec = _mm256_set1_epi32(HORIZONTAL_ROUND as i32);
    let width = tmp_ref.len();
    let (body_start, simd_end) = horizontal_simd_body_range(width, half, 16);

    horizontal_scalar_range(
        tmp_ref, tmp_dis, coeffs, half, 0, body_start, filt_ref, filt_dis,
    );

    for j in (body_start..simd_end).step_by(16) {
        let base_start = j - half;
        let mut acc_ref_lo = _mm256_setzero_si256();
        let mut acc_ref_hi = _mm256_setzero_si256();
        let mut acc_dis_lo = _mm256_setzero_si256();
        let mut acc_dis_hi = _mm256_setzero_si256();

        for tap in 0..coeffs.len() {
            let coeff32 = coeff32_vecs[tap];
            let base = base_start + tap;
            let ref_values = _mm256_loadu_si256(tmp_ref.as_ptr().add(base).cast());
            let dis_values = _mm256_loadu_si256(tmp_dis.as_ptr().add(base).cast());
            let (ref_lo, ref_hi) = mul_u16x16_u32x16(ref_values, coeff32);
            let (dis_lo, dis_hi) = mul_u16x16_u32x16(dis_values, coeff32);
            acc_ref_lo = _mm256_add_epi32(acc_ref_lo, ref_lo);
            acc_ref_hi = _mm256_add_epi32(acc_ref_hi, ref_hi);
            acc_dis_lo = _mm256_add_epi32(acc_dis_lo, dis_lo);
            acc_dis_hi = _mm256_add_epi32(acc_dis_hi, dis_hi);
        }

        let acc_ref_lo = _mm256_srli_epi32(
            _mm256_add_epi32(acc_ref_lo, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        let acc_ref_hi = _mm256_srli_epi32(
            _mm256_add_epi32(acc_ref_hi, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        let acc_dis_lo = _mm256_srli_epi32(
            _mm256_add_epi32(acc_dis_lo, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        let acc_dis_hi = _mm256_srli_epi32(
            _mm256_add_epi32(acc_dis_hi, round_vec),
            HORIZONTAL_SHIFT as i32,
        );
        store_u32x16_as_u16(filt_ref, j, acc_ref_lo, acc_ref_hi);
        store_u32x16_as_u16(filt_dis, j, acc_dis_lo, acc_dis_hi);
    }

    horizontal_scalar_range(
        tmp_ref, tmp_dis, coeffs, half, simd_end, width, filt_ref, filt_dis,
    );
}

#[inline(always)]
unsafe fn mul_u16x8_u32x8(values: __m128i, coeff: __m128i) -> (__m128i, __m128i) {
    let low = _mm_mullo_epi16(values, coeff);
    let high = _mm_mulhi_epu16(values, coeff);
    (_mm_unpacklo_epi16(low, high), _mm_unpackhi_epi16(low, high))
}

#[inline(always)]
unsafe fn mul_u16x16_u32x16(values: __m256i, coeff: __m256i) -> (__m256i, __m256i) {
    let low16 = _mm256_castsi256_si128(values);
    let high16 = _mm256_extracti128_si256(values, 1);
    let low32 = _mm256_cvtepu16_epi32(low16);
    let high32 = _mm256_cvtepu16_epi32(high16);
    (
        _mm256_mullo_epi32(low32, coeff),
        _mm256_mullo_epi32(high32, coeff),
    )
}

#[inline(always)]
unsafe fn store_u32x8_as_u16(out: &mut [u16], start: usize, low: __m128i, high: __m128i) {
    let mut tmp = [0u32; 8];
    _mm_storeu_si128(tmp.as_mut_ptr().cast(), low);
    _mm_storeu_si128(tmp.as_mut_ptr().add(4).cast(), high);

    for (dst, value) in out[start..start + 8].iter_mut().zip(tmp) {
        *dst = value as u16;
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn store_u32x16_as_u16(out: &mut [u16], start: usize, low: __m256i, high: __m256i) {
    let low_lo = _mm256_castsi256_si128(low);
    let low_hi = _mm256_extracti128_si256(low, 1);
    let high_lo = _mm256_castsi256_si128(high);
    let high_hi = _mm256_extracti128_si256(high, 1);
    let packed_lo = _mm_packus_epi32(low_lo, low_hi);
    let packed_hi = _mm_packus_epi32(high_lo, high_hi);
    _mm_storeu_si128(out.as_mut_ptr().add(start).cast(), packed_lo);
    _mm_storeu_si128(out.as_mut_ptr().add(start + 8).cast(), packed_hi);
}
