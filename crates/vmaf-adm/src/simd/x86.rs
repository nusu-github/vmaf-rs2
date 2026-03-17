use crate::dwt::{
    dwt_s123_horizontal_scalar_at, dwt_scale0_horizontal_scalar_at, Bands16, Bands32, DWT_LO_SUM,
    FILTER_HI, FILTER_LO, SCALE_PARAMS,
};
use crate::math::reflect_index;
use vmaf_cpu::{Align32, AlignedScratch};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
fn scale0_params(bpc: u8) -> (u32, i32) {
    let shift_vp = if bpc == 8 { 8u32 } else { bpc as u32 };
    let round_vp = 1i32 << (shift_vp - 1);
    (shift_vp, round_vp)
}

#[inline]
fn write_i32_as_i16<const N: usize>(dst: &mut [i16], start: usize, values: [i32; N]) {
    for (lane, value) in values.into_iter().enumerate() {
        dst[start + lane] = value as i16;
    }
}

#[inline]
fn write_i64_as_i32<const N: usize>(dst: &mut [i32], start: usize, values: [i64; N]) {
    for (lane, value) in values.into_iter().enumerate() {
        dst[start + lane] = value as i32;
    }
}

#[inline]
fn write_i64_pairs_as_i32<const N: usize>(
    dst: &mut [i32],
    start: usize,
    even: [i64; N],
    odd: [i64; N],
) {
    for lane in 0..N {
        dst[start + lane * 2] = even[lane] as i32;
        dst[start + lane * 2 + 1] = odd[lane] as i32;
    }
}

pub(crate) fn dwt_scale0_sse2(src: &[u16], width: usize, height: usize, bpc: u8) -> Bands16 {
    let (shift_vp, round_vp) = scale0_params(bpc);
    let h_half = height.div_ceil(2);
    let w_half = width.div_ceil(2);
    let mut tmplo = AlignedScratch::<i16, Align32>::zeroed(h_half * width);
    let mut tmphi = AlignedScratch::<i16, Align32>::zeroed(h_half * width);

    {
        let tmplo = tmplo.as_mut_slice();
        let tmphi = tmphi.as_mut_slice();
        for i in 0..h_half {
            let row_start = i * width;
            let row_end = row_start + width;
            // SAFETY: the dispatcher only selects this kernel when SSE2 is available.
            unsafe {
                dwt_scale0_vertical_row_sse2(
                    src,
                    width,
                    height,
                    i,
                    shift_vp,
                    round_vp,
                    &mut tmplo[row_start..row_end],
                    &mut tmphi[row_start..row_end],
                );
            }
        }
    }

    let mut band_a = vec![0i16; h_half * w_half];
    let mut band_v = vec![0i16; h_half * w_half];
    let mut band_h = vec![0i16; h_half * w_half];
    let mut band_d = vec![0i16; h_half * w_half];
    let tmplo = tmplo.as_slice();
    let tmphi = tmphi.as_slice();

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let dst_row_start = i * w_half;
        let dst_row_end = dst_row_start + w_half;
        // SAFETY: the dispatcher only selects this kernel when SSE2 is available.
        unsafe {
            dwt_scale0_horizontal_row_sse2(
                &tmplo[src_row_start..src_row_end],
                &tmphi[src_row_start..src_row_end],
                width,
                &mut band_a[dst_row_start..dst_row_end],
                &mut band_v[dst_row_start..dst_row_end],
                &mut band_h[dst_row_start..dst_row_end],
                &mut band_d[dst_row_start..dst_row_end],
            );
        }
    }

    Bands16 {
        a: band_a,
        v: band_v,
        h: band_h,
        d: band_d,
        width: w_half,
        height: h_half,
    }
}

pub(crate) fn dwt_scale0_avx2(src: &[u16], width: usize, height: usize, bpc: u8) -> Bands16 {
    let (shift_vp, round_vp) = scale0_params(bpc);
    let h_half = height.div_ceil(2);
    let w_half = width.div_ceil(2);
    let mut tmplo = AlignedScratch::<i16, Align32>::zeroed(h_half * width);
    let mut tmphi = AlignedScratch::<i16, Align32>::zeroed(h_half * width);

    {
        let tmplo = tmplo.as_mut_slice();
        let tmphi = tmphi.as_mut_slice();
        for i in 0..h_half {
            let row_start = i * width;
            let row_end = row_start + width;
            // SAFETY: the dispatcher only selects this kernel when AVX2 is available.
            unsafe {
                dwt_scale0_vertical_row_avx2(
                    src,
                    width,
                    height,
                    i,
                    shift_vp,
                    round_vp,
                    &mut tmplo[row_start..row_end],
                    &mut tmphi[row_start..row_end],
                );
            }
        }
    }

    let mut band_a = vec![0i16; h_half * w_half];
    let mut band_v = vec![0i16; h_half * w_half];
    let mut band_h = vec![0i16; h_half * w_half];
    let mut band_d = vec![0i16; h_half * w_half];
    let tmplo = tmplo.as_slice();
    let tmphi = tmphi.as_slice();

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let dst_row_start = i * w_half;
        let dst_row_end = dst_row_start + w_half;
        // SAFETY: the dispatcher only selects this kernel when AVX2 is available.
        unsafe {
            dwt_scale0_horizontal_row_avx2(
                &tmplo[src_row_start..src_row_end],
                &tmphi[src_row_start..src_row_end],
                width,
                &mut band_a[dst_row_start..dst_row_end],
                &mut band_v[dst_row_start..dst_row_end],
                &mut band_h[dst_row_start..dst_row_end],
                &mut band_d[dst_row_start..dst_row_end],
            );
        }
    }

    Bands16 {
        a: band_a,
        v: band_v,
        h: band_h,
        d: band_d,
        width: w_half,
        height: h_half,
    }
}

pub(crate) fn dwt_s123_avx2(ll: &[i32], width: usize, height: usize, scale: usize) -> Bands32 {
    let (round_vp, shift_vp, round_hp, shift_hp) = SCALE_PARAMS[scale];
    let h_half = height.div_ceil(2);
    let w_half = width.div_ceil(2);
    let mut tmplo = AlignedScratch::<i32, Align32>::zeroed(h_half * width);
    let mut tmphi = AlignedScratch::<i32, Align32>::zeroed(h_half * width);

    {
        let tmplo = tmplo.as_mut_slice();
        let tmphi = tmphi.as_mut_slice();
        for i in 0..h_half {
            let row_start = i * width;
            let row_end = row_start + width;
            // SAFETY: the dispatcher only selects this kernel when AVX2 is available.
            unsafe {
                dwt_s123_vertical_row_avx2(
                    ll,
                    width,
                    height,
                    i,
                    round_vp,
                    shift_vp,
                    &mut tmplo[row_start..row_end],
                    &mut tmphi[row_start..row_end],
                );
            }
        }
    }

    let mut band_a = vec![0i32; h_half * w_half];
    let mut band_v = vec![0i32; h_half * w_half];
    let mut band_h = vec![0i32; h_half * w_half];
    let mut band_d = vec![0i32; h_half * w_half];
    let tmplo = tmplo.as_slice();
    let tmphi = tmphi.as_slice();

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let dst_row_start = i * w_half;
        let dst_row_end = dst_row_start + w_half;
        // SAFETY: the dispatcher only selects this kernel when AVX2 is available.
        unsafe {
            dwt_s123_horizontal_row_avx2(
                &tmplo[src_row_start..src_row_end],
                &tmphi[src_row_start..src_row_end],
                width,
                round_hp,
                shift_hp,
                &mut band_a[dst_row_start..dst_row_end],
                &mut band_v[dst_row_start..dst_row_end],
                &mut band_h[dst_row_start..dst_row_end],
                &mut band_d[dst_row_start..dst_row_end],
            );
        }
    }

    Bands32 {
        a: band_a,
        v: band_v,
        h: band_h,
        d: band_d,
        width: w_half,
        height: h_half,
    }
}

#[target_feature(enable = "sse2")]
unsafe fn mul_u16_const_to_i32_sse2(v: __m128i, coeff: i32) -> (__m128i, __m128i) {
    let coeff_abs = _mm_set1_epi16(coeff.unsigned_abs() as i16);
    let lo = _mm_mullo_epi16(v, coeff_abs);
    let hi = _mm_mulhi_epu16(v, coeff_abs);
    let mut prod_lo = _mm_unpacklo_epi16(lo, hi);
    let mut prod_hi = _mm_unpackhi_epi16(lo, hi);
    if coeff < 0 {
        let zero = _mm_setzero_si128();
        prod_lo = _mm_sub_epi32(zero, prod_lo);
        prod_hi = _mm_sub_epi32(zero, prod_hi);
    }
    (prod_lo, prod_hi)
}

#[target_feature(enable = "sse2")]
unsafe fn dwt_scale0_vertical_row_sse2(
    src: &[u16],
    width: usize,
    height: usize,
    i: usize,
    shift_vp: u32,
    round_vp: i32,
    tmplo_row: &mut [i16],
    tmphi_row: &mut [i16],
) {
    debug_assert_eq!(tmplo_row.len(), width);
    debug_assert_eq!(tmphi_row.len(), width);

    let base = 2 * i as i32;
    let r0 = reflect_index(base - 1, height as i32);
    let r1 = i * 2;
    let r2 = reflect_index(base + 1, height as i32);
    let r3 = reflect_index(base + 2, height as i32);
    let row0 = src.as_ptr().wrapping_add(r0 * width);
    let row1 = src.as_ptr().wrapping_add(r1 * width);
    let row2 = src.as_ptr().wrapping_add(r2 * width);
    let row3 = src.as_ptr().wrapping_add(r3 * width);
    let lo_bias = _mm_set1_epi32(round_vp.wrapping_sub(DWT_LO_SUM.wrapping_mul(round_vp)));
    let hi_bias = _mm_set1_epi32(round_vp);
    let shift = _mm_cvtsi32_si128(shift_vp as i32);
    let mut j = 0;

    while j + 8 <= width {
        let s0 = _mm_loadu_si128(row0.add(j) as *const __m128i);
        let s1 = _mm_loadu_si128(row1.add(j) as *const __m128i);
        let s2 = _mm_loadu_si128(row2.add(j) as *const __m128i);
        let s3 = _mm_loadu_si128(row3.add(j) as *const __m128i);

        let (lo0, hi0) = mul_u16_const_to_i32_sse2(s0, FILTER_LO[0]);
        let (lo1, hi1) = mul_u16_const_to_i32_sse2(s1, FILTER_LO[1]);
        let (lo2, hi2) = mul_u16_const_to_i32_sse2(s2, FILTER_LO[2]);
        let (lo3, hi3) = mul_u16_const_to_i32_sse2(s3, FILTER_LO[3]);
        let al_lo = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_add_epi32(lo0, lo1), _mm_add_epi32(lo2, lo3)),
                lo_bias,
            ),
            shift,
        );
        let al_hi = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_add_epi32(hi0, hi1), _mm_add_epi32(hi2, hi3)),
                lo_bias,
            ),
            shift,
        );

        let (lo0, hi0) = mul_u16_const_to_i32_sse2(s0, FILTER_HI[0]);
        let (lo1, hi1) = mul_u16_const_to_i32_sse2(s1, FILTER_HI[1]);
        let (lo2, hi2) = mul_u16_const_to_i32_sse2(s2, FILTER_HI[2]);
        let (lo3, hi3) = mul_u16_const_to_i32_sse2(s3, FILTER_HI[3]);
        let ah_lo = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_add_epi32(lo0, lo1), _mm_add_epi32(lo2, lo3)),
                hi_bias,
            ),
            shift,
        );
        let ah_hi = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_add_epi32(hi0, hi1), _mm_add_epi32(hi2, hi3)),
                hi_bias,
            ),
            shift,
        );

        let mut al_buf_lo = [0i32; 4];
        let mut al_buf_hi = [0i32; 4];
        let mut ah_buf_lo = [0i32; 4];
        let mut ah_buf_hi = [0i32; 4];
        _mm_storeu_si128(al_buf_lo.as_mut_ptr() as *mut __m128i, al_lo);
        _mm_storeu_si128(al_buf_hi.as_mut_ptr() as *mut __m128i, al_hi);
        _mm_storeu_si128(ah_buf_lo.as_mut_ptr() as *mut __m128i, ah_lo);
        _mm_storeu_si128(ah_buf_hi.as_mut_ptr() as *mut __m128i, ah_hi);
        write_i32_as_i16(tmplo_row, j, al_buf_lo);
        write_i32_as_i16(tmplo_row, j + 4, al_buf_hi);
        write_i32_as_i16(tmphi_row, j, ah_buf_lo);
        write_i32_as_i16(tmphi_row, j + 4, ah_buf_hi);
        j += 8;
    }

    let lo_bias_scalar = round_vp.wrapping_sub(DWT_LO_SUM.wrapping_mul(round_vp));
    while j < width {
        let s0 = *src.get_unchecked(r0 * width + j) as i32;
        let s1 = *src.get_unchecked(r1 * width + j) as i32;
        let s2 = *src.get_unchecked(r2 * width + j) as i32;
        let s3 = *src.get_unchecked(r3 * width + j) as i32;
        let al = FILTER_LO[0]
            .wrapping_mul(s0)
            .wrapping_add(FILTER_LO[1].wrapping_mul(s1))
            .wrapping_add(FILTER_LO[2].wrapping_mul(s2))
            .wrapping_add(FILTER_LO[3].wrapping_mul(s3));
        let ah = FILTER_HI[0]
            .wrapping_mul(s0)
            .wrapping_add(FILTER_HI[1].wrapping_mul(s1))
            .wrapping_add(FILTER_HI[2].wrapping_mul(s2))
            .wrapping_add(FILTER_HI[3].wrapping_mul(s3));
        *tmplo_row.get_unchecked_mut(j) = (al.wrapping_add(lo_bias_scalar) >> shift_vp) as i16;
        *tmphi_row.get_unchecked_mut(j) = (ah.wrapping_add(round_vp) >> shift_vp) as i16;
        j += 1;
    }
}

#[target_feature(enable = "sse2")]
unsafe fn dwt_scale0_horizontal_row_sse2(
    tmplo_row: &[i16],
    tmphi_row: &[i16],
    width: usize,
    band_a_row: &mut [i16],
    band_v_row: &mut [i16],
    band_h_row: &mut [i16],
    band_d_row: &mut [i16],
) {
    let w_half = band_a_row.len();
    debug_assert_eq!(band_v_row.len(), w_half);
    debug_assert_eq!(band_h_row.len(), w_half);
    debug_assert_eq!(band_d_row.len(), w_half);

    let lo01 = _mm_setr_epi16(
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
    );
    let lo23 = _mm_setr_epi16(
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
    );
    let hi01 = _mm_setr_epi16(
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
    );
    let hi23 = _mm_setr_epi16(
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
    );
    let round = _mm_set1_epi32(32768);
    let mut j = 0;

    if j < w_half {
        let (a, v, h, d) = dwt_scale0_horizontal_scalar_at(tmplo_row, tmphi_row, width, j);
        band_a_row[j] = a;
        band_v_row[j] = v;
        band_h_row[j] = h;
        band_d_row[j] = d;
        j += 1;
    }

    while j + 4 <= w_half && 2 * j + 8 < width {
        let lo_pairs01 = _mm_loadu_si128(tmplo_row.as_ptr().add(2 * j - 1) as *const __m128i);
        let lo_pairs23 = _mm_loadu_si128(tmplo_row.as_ptr().add(2 * j + 1) as *const __m128i);
        let hi_pairs01 = _mm_loadu_si128(tmphi_row.as_ptr().add(2 * j - 1) as *const __m128i);
        let hi_pairs23 = _mm_loadu_si128(tmphi_row.as_ptr().add(2 * j + 1) as *const __m128i);

        let band_a = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_madd_epi16(lo_pairs01, lo01),
                    _mm_madd_epi16(lo_pairs23, lo23),
                ),
                round,
            ),
            16,
        );
        let band_v = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_madd_epi16(lo_pairs01, hi01),
                    _mm_madd_epi16(lo_pairs23, hi23),
                ),
                round,
            ),
            16,
        );
        let band_h = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_madd_epi16(hi_pairs01, lo01),
                    _mm_madd_epi16(hi_pairs23, lo23),
                ),
                round,
            ),
            16,
        );
        let band_d = _mm_srai_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_madd_epi16(hi_pairs01, hi01),
                    _mm_madd_epi16(hi_pairs23, hi23),
                ),
                round,
            ),
            16,
        );

        let mut a_buf = [0i32; 4];
        let mut v_buf = [0i32; 4];
        let mut h_buf = [0i32; 4];
        let mut d_buf = [0i32; 4];
        _mm_storeu_si128(a_buf.as_mut_ptr() as *mut __m128i, band_a);
        _mm_storeu_si128(v_buf.as_mut_ptr() as *mut __m128i, band_v);
        _mm_storeu_si128(h_buf.as_mut_ptr() as *mut __m128i, band_h);
        _mm_storeu_si128(d_buf.as_mut_ptr() as *mut __m128i, band_d);
        write_i32_as_i16(band_a_row, j, a_buf);
        write_i32_as_i16(band_v_row, j, v_buf);
        write_i32_as_i16(band_h_row, j, h_buf);
        write_i32_as_i16(band_d_row, j, d_buf);
        j += 4;
    }

    while j < w_half {
        let (a, v, h, d) = dwt_scale0_horizontal_scalar_at(tmplo_row, tmphi_row, width, j);
        band_a_row[j] = a;
        band_v_row[j] = v;
        band_h_row[j] = h;
        band_d_row[j] = d;
        j += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dwt_scale0_vertical_row_avx2(
    src: &[u16],
    width: usize,
    height: usize,
    i: usize,
    shift_vp: u32,
    round_vp: i32,
    tmplo_row: &mut [i16],
    tmphi_row: &mut [i16],
) {
    debug_assert_eq!(tmplo_row.len(), width);
    debug_assert_eq!(tmphi_row.len(), width);

    let base = 2 * i as i32;
    let r0 = reflect_index(base - 1, height as i32);
    let r1 = i * 2;
    let r2 = reflect_index(base + 1, height as i32);
    let r3 = reflect_index(base + 2, height as i32);
    let row0 = src.as_ptr().wrapping_add(r0 * width);
    let row1 = src.as_ptr().wrapping_add(r1 * width);
    let row2 = src.as_ptr().wrapping_add(r2 * width);
    let row3 = src.as_ptr().wrapping_add(r3 * width);

    let lo_bias = _mm256_set1_epi32(round_vp.wrapping_sub(DWT_LO_SUM.wrapping_mul(round_vp)));
    let hi_bias = _mm256_set1_epi32(round_vp);
    let shift = _mm_cvtsi32_si128(shift_vp as i32);
    let lo0 = _mm256_set1_epi32(FILTER_LO[0]);
    let lo1 = _mm256_set1_epi32(FILTER_LO[1]);
    let lo2 = _mm256_set1_epi32(FILTER_LO[2]);
    let lo3 = _mm256_set1_epi32(FILTER_LO[3]);
    let hi0 = _mm256_set1_epi32(FILTER_HI[0]);
    let hi1 = _mm256_set1_epi32(FILTER_HI[1]);
    let hi2 = _mm256_set1_epi32(FILTER_HI[2]);
    let hi3 = _mm256_set1_epi32(FILTER_HI[3]);
    let mut j = 0;

    while j + 16 <= width {
        let s0 = _mm256_loadu_si256(row0.add(j) as *const __m256i);
        let s1 = _mm256_loadu_si256(row1.add(j) as *const __m256i);
        let s2 = _mm256_loadu_si256(row2.add(j) as *const __m256i);
        let s3 = _mm256_loadu_si256(row3.add(j) as *const __m256i);

        let s0_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s0));
        let s0_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s0, 1));
        let s1_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s1));
        let s1_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s1, 1));
        let s2_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s2));
        let s2_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s2, 1));
        let s3_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s3));
        let s3_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s3, 1));

        let al_lo = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s0_lo, lo0),
                        _mm256_mullo_epi32(s1_lo, lo1),
                    ),
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s2_lo, lo2),
                        _mm256_mullo_epi32(s3_lo, lo3),
                    ),
                ),
                lo_bias,
            ),
            shift,
        );
        let al_hi = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s0_hi, lo0),
                        _mm256_mullo_epi32(s1_hi, lo1),
                    ),
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s2_hi, lo2),
                        _mm256_mullo_epi32(s3_hi, lo3),
                    ),
                ),
                lo_bias,
            ),
            shift,
        );
        let ah_lo = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s0_lo, hi0),
                        _mm256_mullo_epi32(s1_lo, hi1),
                    ),
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s2_lo, hi2),
                        _mm256_mullo_epi32(s3_lo, hi3),
                    ),
                ),
                hi_bias,
            ),
            shift,
        );
        let ah_hi = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s0_hi, hi0),
                        _mm256_mullo_epi32(s1_hi, hi1),
                    ),
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(s2_hi, hi2),
                        _mm256_mullo_epi32(s3_hi, hi3),
                    ),
                ),
                hi_bias,
            ),
            shift,
        );

        let mut al_buf_lo = [0i32; 8];
        let mut al_buf_hi = [0i32; 8];
        let mut ah_buf_lo = [0i32; 8];
        let mut ah_buf_hi = [0i32; 8];
        _mm256_storeu_si256(al_buf_lo.as_mut_ptr() as *mut __m256i, al_lo);
        _mm256_storeu_si256(al_buf_hi.as_mut_ptr() as *mut __m256i, al_hi);
        _mm256_storeu_si256(ah_buf_lo.as_mut_ptr() as *mut __m256i, ah_lo);
        _mm256_storeu_si256(ah_buf_hi.as_mut_ptr() as *mut __m256i, ah_hi);
        write_i32_as_i16(tmplo_row, j, al_buf_lo);
        write_i32_as_i16(tmplo_row, j + 8, al_buf_hi);
        write_i32_as_i16(tmphi_row, j, ah_buf_lo);
        write_i32_as_i16(tmphi_row, j + 8, ah_buf_hi);
        j += 16;
    }

    let lo_bias_scalar = round_vp.wrapping_sub(DWT_LO_SUM.wrapping_mul(round_vp));
    while j < width {
        let s0 = *src.get_unchecked(r0 * width + j) as i32;
        let s1 = *src.get_unchecked(r1 * width + j) as i32;
        let s2 = *src.get_unchecked(r2 * width + j) as i32;
        let s3 = *src.get_unchecked(r3 * width + j) as i32;
        let al = FILTER_LO[0]
            .wrapping_mul(s0)
            .wrapping_add(FILTER_LO[1].wrapping_mul(s1))
            .wrapping_add(FILTER_LO[2].wrapping_mul(s2))
            .wrapping_add(FILTER_LO[3].wrapping_mul(s3));
        let ah = FILTER_HI[0]
            .wrapping_mul(s0)
            .wrapping_add(FILTER_HI[1].wrapping_mul(s1))
            .wrapping_add(FILTER_HI[2].wrapping_mul(s2))
            .wrapping_add(FILTER_HI[3].wrapping_mul(s3));
        *tmplo_row.get_unchecked_mut(j) = (al.wrapping_add(lo_bias_scalar) >> shift_vp) as i16;
        *tmphi_row.get_unchecked_mut(j) = (ah.wrapping_add(round_vp) >> shift_vp) as i16;
        j += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dwt_scale0_horizontal_row_avx2(
    tmplo_row: &[i16],
    tmphi_row: &[i16],
    width: usize,
    band_a_row: &mut [i16],
    band_v_row: &mut [i16],
    band_h_row: &mut [i16],
    band_d_row: &mut [i16],
) {
    let w_half = band_a_row.len();
    debug_assert_eq!(band_v_row.len(), w_half);
    debug_assert_eq!(band_h_row.len(), w_half);
    debug_assert_eq!(band_d_row.len(), w_half);

    let lo01 = _mm256_setr_epi16(
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
        FILTER_LO[0] as i16,
        FILTER_LO[1] as i16,
    );
    let lo23 = _mm256_setr_epi16(
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
        FILTER_LO[2] as i16,
        FILTER_LO[3] as i16,
    );
    let hi01 = _mm256_setr_epi16(
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
        FILTER_HI[0] as i16,
        FILTER_HI[1] as i16,
    );
    let hi23 = _mm256_setr_epi16(
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
        FILTER_HI[2] as i16,
        FILTER_HI[3] as i16,
    );
    let round = _mm256_set1_epi32(32768);
    let mut j = 0;

    if j < w_half {
        let (a, v, h, d) = dwt_scale0_horizontal_scalar_at(tmplo_row, tmphi_row, width, j);
        band_a_row[j] = a;
        band_v_row[j] = v;
        band_h_row[j] = h;
        band_d_row[j] = d;
        j += 1;
    }

    while j + 8 <= w_half && 2 * j + 16 < width {
        let lo_pairs01 = _mm256_loadu_si256(tmplo_row.as_ptr().add(2 * j - 1) as *const __m256i);
        let lo_pairs23 = _mm256_loadu_si256(tmplo_row.as_ptr().add(2 * j + 1) as *const __m256i);
        let hi_pairs01 = _mm256_loadu_si256(tmphi_row.as_ptr().add(2 * j - 1) as *const __m256i);
        let hi_pairs23 = _mm256_loadu_si256(tmphi_row.as_ptr().add(2 * j + 1) as *const __m256i);

        let band_a = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(lo_pairs01, lo01),
                    _mm256_madd_epi16(lo_pairs23, lo23),
                ),
                round,
            ),
            16,
        );
        let band_v = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(lo_pairs01, hi01),
                    _mm256_madd_epi16(lo_pairs23, hi23),
                ),
                round,
            ),
            16,
        );
        let band_h = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(hi_pairs01, lo01),
                    _mm256_madd_epi16(hi_pairs23, lo23),
                ),
                round,
            ),
            16,
        );
        let band_d = _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_madd_epi16(hi_pairs01, hi01),
                    _mm256_madd_epi16(hi_pairs23, hi23),
                ),
                round,
            ),
            16,
        );

        let mut a_buf = [0i32; 8];
        let mut v_buf = [0i32; 8];
        let mut h_buf = [0i32; 8];
        let mut d_buf = [0i32; 8];
        _mm256_storeu_si256(a_buf.as_mut_ptr() as *mut __m256i, band_a);
        _mm256_storeu_si256(v_buf.as_mut_ptr() as *mut __m256i, band_v);
        _mm256_storeu_si256(h_buf.as_mut_ptr() as *mut __m256i, band_h);
        _mm256_storeu_si256(d_buf.as_mut_ptr() as *mut __m256i, band_d);
        write_i32_as_i16(band_a_row, j, a_buf);
        write_i32_as_i16(band_v_row, j, v_buf);
        write_i32_as_i16(band_h_row, j, h_buf);
        write_i32_as_i16(band_d_row, j, d_buf);
        j += 8;
    }

    while j < w_half {
        let (a, v, h, d) = dwt_scale0_horizontal_scalar_at(tmplo_row, tmphi_row, width, j);
        band_a_row[j] = a;
        band_v_row[j] = v;
        band_h_row[j] = h;
        band_d_row[j] = d;
        j += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mul_i32_const_pairs_avx2(v: __m256i, coeff: i32) -> (__m256i, __m256i) {
    let coeffs = _mm256_set1_epi32(coeff);
    let even = _mm256_mul_epi32(v, coeffs);
    let odd = _mm256_mul_epi32(_mm256_srli_epi64(v, 32), coeffs);
    (even, odd)
}

#[target_feature(enable = "avx2")]
unsafe fn pair_sum_i32_avx2(v: __m256i, coeff0: i32, coeff1: i32) -> __m256i {
    let coeffs0 = _mm256_set1_epi32(coeff0);
    let coeffs1 = _mm256_set1_epi32(coeff1);
    let even = _mm256_mul_epi32(v, coeffs0);
    let odd = _mm256_mul_epi32(_mm256_srli_epi64(v, 32), coeffs1);
    _mm256_add_epi64(even, odd)
}

#[target_feature(enable = "avx2")]
unsafe fn srai_epi64_avx2(v: __m256i, shift: u32) -> __m256i {
    match shift {
        0 => v,
        15 => {
            let shifted = _mm256_srli_epi64(v, 15);
            let negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
            let fill = _mm256_slli_epi64(negative, 49);
            _mm256_or_si256(shifted, fill)
        }
        16 => {
            let shifted = _mm256_srli_epi64(v, 16);
            let negative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
            let fill = _mm256_slli_epi64(negative, 48);
            _mm256_or_si256(shifted, fill)
        }
        _ => unreachable!("unsupported 64-bit shift: {shift}"),
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dwt_s123_vertical_row_avx2(
    ll: &[i32],
    width: usize,
    height: usize,
    i: usize,
    round_vp: i64,
    shift_vp: u32,
    tmplo_row: &mut [i32],
    tmphi_row: &mut [i32],
) {
    debug_assert_eq!(tmplo_row.len(), width);
    debug_assert_eq!(tmphi_row.len(), width);

    let base = 2 * i as i32;
    let r0 = reflect_index(base - 1, height as i32);
    let r1 = i * 2;
    let r2 = reflect_index(base + 1, height as i32);
    let r3 = reflect_index(base + 2, height as i32);
    let row0 = ll.as_ptr().wrapping_add(r0 * width);
    let row1 = ll.as_ptr().wrapping_add(r1 * width);
    let row2 = ll.as_ptr().wrapping_add(r2 * width);
    let row3 = ll.as_ptr().wrapping_add(r3 * width);
    let round = _mm256_set1_epi64x(round_vp);
    let mut j = 0;

    while j + 8 <= width {
        let s0 = _mm256_loadu_si256(row0.add(j) as *const __m256i);
        let s1 = _mm256_loadu_si256(row1.add(j) as *const __m256i);
        let s2 = _mm256_loadu_si256(row2.add(j) as *const __m256i);
        let s3 = _mm256_loadu_si256(row3.add(j) as *const __m256i);

        let (al_even0, al_odd0) = mul_i32_const_pairs_avx2(s0, FILTER_LO[0]);
        let (al_even1, al_odd1) = mul_i32_const_pairs_avx2(s1, FILTER_LO[1]);
        let (al_even2, al_odd2) = mul_i32_const_pairs_avx2(s2, FILTER_LO[2]);
        let (al_even3, al_odd3) = mul_i32_const_pairs_avx2(s3, FILTER_LO[3]);
        let al_even = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(al_even0, al_even1),
                    _mm256_add_epi64(al_even2, al_even3),
                ),
                round,
            ),
            shift_vp,
        );
        let al_odd = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(al_odd0, al_odd1),
                    _mm256_add_epi64(al_odd2, al_odd3),
                ),
                round,
            ),
            shift_vp,
        );

        let (ah_even0, ah_odd0) = mul_i32_const_pairs_avx2(s0, FILTER_HI[0]);
        let (ah_even1, ah_odd1) = mul_i32_const_pairs_avx2(s1, FILTER_HI[1]);
        let (ah_even2, ah_odd2) = mul_i32_const_pairs_avx2(s2, FILTER_HI[2]);
        let (ah_even3, ah_odd3) = mul_i32_const_pairs_avx2(s3, FILTER_HI[3]);
        let ah_even = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(ah_even0, ah_even1),
                    _mm256_add_epi64(ah_even2, ah_even3),
                ),
                round,
            ),
            shift_vp,
        );
        let ah_odd = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(ah_odd0, ah_odd1),
                    _mm256_add_epi64(ah_odd2, ah_odd3),
                ),
                round,
            ),
            shift_vp,
        );

        let mut al_even_buf = [0i64; 4];
        let mut al_odd_buf = [0i64; 4];
        let mut ah_even_buf = [0i64; 4];
        let mut ah_odd_buf = [0i64; 4];
        _mm256_storeu_si256(al_even_buf.as_mut_ptr() as *mut __m256i, al_even);
        _mm256_storeu_si256(al_odd_buf.as_mut_ptr() as *mut __m256i, al_odd);
        _mm256_storeu_si256(ah_even_buf.as_mut_ptr() as *mut __m256i, ah_even);
        _mm256_storeu_si256(ah_odd_buf.as_mut_ptr() as *mut __m256i, ah_odd);
        write_i64_pairs_as_i32(tmplo_row, j, al_even_buf, al_odd_buf);
        write_i64_pairs_as_i32(tmphi_row, j, ah_even_buf, ah_odd_buf);
        j += 8;
    }

    while j < width {
        let s0 = *ll.get_unchecked(r0 * width + j) as i64;
        let s1 = *ll.get_unchecked(r1 * width + j) as i64;
        let s2 = *ll.get_unchecked(r2 * width + j) as i64;
        let s3 = *ll.get_unchecked(r3 * width + j) as i64;
        let al = FILTER_LO[0] as i64 * s0
            + FILTER_LO[1] as i64 * s1
            + FILTER_LO[2] as i64 * s2
            + FILTER_LO[3] as i64 * s3;
        let ah = FILTER_HI[0] as i64 * s0
            + FILTER_HI[1] as i64 * s1
            + FILTER_HI[2] as i64 * s2
            + FILTER_HI[3] as i64 * s3;
        *tmplo_row.get_unchecked_mut(j) = ((al + round_vp) >> shift_vp) as i32;
        *tmphi_row.get_unchecked_mut(j) = ((ah + round_vp) >> shift_vp) as i32;
        j += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dwt_s123_horizontal_row_avx2(
    tmplo_row: &[i32],
    tmphi_row: &[i32],
    width: usize,
    round_hp: i64,
    shift_hp: u32,
    band_a_row: &mut [i32],
    band_v_row: &mut [i32],
    band_h_row: &mut [i32],
    band_d_row: &mut [i32],
) {
    let w_half = band_a_row.len();
    debug_assert_eq!(band_v_row.len(), w_half);
    debug_assert_eq!(band_h_row.len(), w_half);
    debug_assert_eq!(band_d_row.len(), w_half);

    let round = _mm256_set1_epi64x(round_hp);
    let mut j = 0;

    if j < w_half {
        let (a, v, h, d) =
            dwt_s123_horizontal_scalar_at(tmplo_row, tmphi_row, width, j, round_hp, shift_hp);
        band_a_row[j] = a;
        band_v_row[j] = v;
        band_h_row[j] = h;
        band_d_row[j] = d;
        j += 1;
    }

    while j + 4 <= w_half && 2 * j + 8 < width {
        let lo_pairs01 = _mm256_loadu_si256(tmplo_row.as_ptr().add(2 * j - 1) as *const __m256i);
        let lo_pairs23 = _mm256_loadu_si256(tmplo_row.as_ptr().add(2 * j + 1) as *const __m256i);
        let hi_pairs01 = _mm256_loadu_si256(tmphi_row.as_ptr().add(2 * j - 1) as *const __m256i);
        let hi_pairs23 = _mm256_loadu_si256(tmphi_row.as_ptr().add(2 * j + 1) as *const __m256i);

        let band_a = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    pair_sum_i32_avx2(lo_pairs01, FILTER_LO[0], FILTER_LO[1]),
                    pair_sum_i32_avx2(lo_pairs23, FILTER_LO[2], FILTER_LO[3]),
                ),
                round,
            ),
            shift_hp,
        );
        let band_v = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    pair_sum_i32_avx2(lo_pairs01, FILTER_HI[0], FILTER_HI[1]),
                    pair_sum_i32_avx2(lo_pairs23, FILTER_HI[2], FILTER_HI[3]),
                ),
                round,
            ),
            shift_hp,
        );
        let band_h = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    pair_sum_i32_avx2(hi_pairs01, FILTER_LO[0], FILTER_LO[1]),
                    pair_sum_i32_avx2(hi_pairs23, FILTER_LO[2], FILTER_LO[3]),
                ),
                round,
            ),
            shift_hp,
        );
        let band_d = srai_epi64_avx2(
            _mm256_add_epi64(
                _mm256_add_epi64(
                    pair_sum_i32_avx2(hi_pairs01, FILTER_HI[0], FILTER_HI[1]),
                    pair_sum_i32_avx2(hi_pairs23, FILTER_HI[2], FILTER_HI[3]),
                ),
                round,
            ),
            shift_hp,
        );

        let mut a_buf = [0i64; 4];
        let mut v_buf = [0i64; 4];
        let mut h_buf = [0i64; 4];
        let mut d_buf = [0i64; 4];
        _mm256_storeu_si256(a_buf.as_mut_ptr() as *mut __m256i, band_a);
        _mm256_storeu_si256(v_buf.as_mut_ptr() as *mut __m256i, band_v);
        _mm256_storeu_si256(h_buf.as_mut_ptr() as *mut __m256i, band_h);
        _mm256_storeu_si256(d_buf.as_mut_ptr() as *mut __m256i, band_d);
        write_i64_as_i32(band_a_row, j, a_buf);
        write_i64_as_i32(band_v_row, j, v_buf);
        write_i64_as_i32(band_h_row, j, h_buf);
        write_i64_as_i32(band_d_row, j, d_buf);
        j += 4;
    }

    while j < w_half {
        let (a, v, h, d) =
            dwt_s123_horizontal_scalar_at(tmplo_row, tmphi_row, width, j, round_hp, shift_hp);
        band_a_row[j] = a;
        band_v_row[j] = v;
        band_h_row[j] = h;
        band_d_row[j] = d;
        j += 1;
    }
}
