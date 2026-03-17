use vmaf_cpu::SimdBackend;

use crate::tables::{FILTER, FILTER_WIDTH};

use super::{
    finalize_scale_stat, horizontal_scalar_range, horizontal_simd_body_range,
    process_filtered_pixel, process_sigma_values, reflected_row_offsets, stat_params,
    vertical_scalar_range_non_wrapping, vertical_scalar_range_wrapping, RunningStatAccumulators,
    ScaleStat, FILTER_TAP_CAP,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
enum X86StatKernel {
    Sse2,
    Avx2,
}

pub(super) fn vif_statistic(
    ref_plane: &[u16],
    dis_plane: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    vif_enhn_gain_limit: f64,
    workspace: &mut super::VifStatWorkspace,
    backend: SimdBackend,
) -> ScaleStat {
    let filt = &FILTER[scale][..FILTER_WIDTH[scale]];
    let half = filt.len() / 2;
    let (shift_mu, round_mu, sq_shift, sq_round64) = stat_params(bpc, scale);
    let uses_wrapping_sq = bpc == 8 && scale == 0;
    let kernel = select_kernel(backend);
    let mut accum = RunningStatAccumulators::default();
    workspace.prepare(width);
    let tmp_mu1 = &mut workspace.tmp_mu1.as_mut_slice()[..width];
    let tmp_mu2 = &mut workspace.tmp_mu2.as_mut_slice()[..width];
    let tmp_ref_sq = &mut workspace.tmp_ref_sq.as_mut_slice()[..width];
    let tmp_dis_sq = &mut workspace.tmp_dis_sq.as_mut_slice()[..width];
    let tmp_ref_dis = &mut workspace.tmp_ref_dis.as_mut_slice()[..width];

    for i in 0..height {
        let row_offsets = reflected_row_offsets(i, height, width, half, filt.len());

        match (kernel, uses_wrapping_sq) {
            (X86StatKernel::Avx2, true) => {
                // SAFETY: `select_kernel` only returns `Avx2` when runtime
                // detection has already proved the current process supports AVX2.
                unsafe {
                    vertical_row_wrapping_avx2(
                        ref_plane,
                        dis_plane,
                        &row_offsets[..filt.len()],
                        filt,
                        shift_mu,
                        round_mu,
                        tmp_mu1,
                        tmp_mu2,
                        tmp_ref_sq,
                        tmp_dis_sq,
                        tmp_ref_dis,
                    );
                }
            }
            (X86StatKernel::Avx2, false) => {
                // SAFETY: `select_kernel` only returns `Avx2` when runtime
                // detection has already proved the current process supports AVX2.
                unsafe {
                    vertical_row_non_wrapping_avx2(
                        ref_plane,
                        dis_plane,
                        &row_offsets[..filt.len()],
                        filt,
                        shift_mu,
                        round_mu,
                        sq_shift,
                        sq_round64,
                        tmp_mu1,
                        tmp_mu2,
                        tmp_ref_sq,
                        tmp_dis_sq,
                        tmp_ref_dis,
                    );
                }
            }
            (_, true) => {
                vertical_scalar_range_wrapping(
                    ref_plane,
                    dis_plane,
                    &row_offsets[..filt.len()],
                    filt,
                    shift_mu,
                    round_mu,
                    0,
                    width,
                    tmp_mu1,
                    tmp_mu2,
                    tmp_ref_sq,
                    tmp_dis_sq,
                    tmp_ref_dis,
                );
            }
            (_, false) => {
                vertical_scalar_range_non_wrapping(
                    ref_plane,
                    dis_plane,
                    &row_offsets[..filt.len()],
                    filt,
                    shift_mu,
                    round_mu,
                    sq_shift,
                    sq_round64,
                    0,
                    width,
                    tmp_mu1,
                    tmp_mu2,
                    tmp_ref_sq,
                    tmp_dis_sq,
                    tmp_ref_dis,
                );
            }
        }

        match kernel {
            X86StatKernel::Avx2 => {
                // SAFETY: `select_kernel` only returns `Avx2` when runtime
                // detection has already proved the current process supports AVX2.
                unsafe {
                    horizontal_row_avx2(
                        tmp_mu1,
                        tmp_mu2,
                        tmp_ref_sq,
                        tmp_dis_sq,
                        tmp_ref_dis,
                        filt,
                        half,
                        vif_enhn_gain_limit,
                        &mut accum,
                    );
                }
            }
            X86StatKernel::Sse2 => {
                // SAFETY: x86 dispatch only reaches this module after runtime
                // detection confirmed at least SSE2 support.
                unsafe {
                    horizontal_row_sse2(
                        tmp_mu1,
                        tmp_mu2,
                        tmp_ref_sq,
                        tmp_dis_sq,
                        tmp_ref_dis,
                        filt,
                        half,
                        vif_enhn_gain_limit,
                        &mut accum,
                    );
                }
            }
        }
    }

    finalize_scale_stat(accum)
}

#[inline]
fn select_kernel(backend: SimdBackend) -> X86StatKernel {
    match backend {
        SimdBackend::X86Avx2Fma => X86StatKernel::Avx2,
        SimdBackend::X86Avx512 if SimdBackend::X86Avx2Fma.is_available() => X86StatKernel::Avx2,
        _ => X86StatKernel::Sse2,
    }
}

/// SAFETY: the caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
unsafe fn vertical_row_wrapping_avx2(
    ref_plane: &[u16],
    dis_plane: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    shift_mu: u32,
    round_mu: u32,
    tmp_mu1: &mut [u16],
    tmp_mu2: &mut [u16],
    tmp_ref_sq: &mut [u32],
    tmp_dis_sq: &mut [u32],
    tmp_ref_dis: &mut [u32],
) {
    debug_assert_eq!(row_offsets.len(), coeffs.len());

    let simd_end = tmp_mu1.len() / 8 * 8;
    let round_mu_vec = _mm256_set1_epi32(round_mu as i32);
    let shift_mu_vec = _mm_cvtsi32_si128(shift_mu as i32);
    let mut ref_rows = [std::ptr::null::<u16>(); FILTER_TAP_CAP];
    let mut dis_rows = [std::ptr::null::<u16>(); FILTER_TAP_CAP];
    let mut coeff32_vecs = [_mm256_setzero_si256(); FILTER_TAP_CAP];

    for tap in 0..coeffs.len() {
        ref_rows[tap] = ref_plane.as_ptr().add(row_offsets[tap]);
        dis_rows[tap] = dis_plane.as_ptr().add(row_offsets[tap]);
        coeff32_vecs[tap] = _mm256_set1_epi32(coeffs[tap] as i32);
    }

    for j in (0..simd_end).step_by(8) {
        let mut acc_mu1 = _mm256_setzero_si256();
        let mut acc_mu2 = _mm256_setzero_si256();
        let mut acc_rsq = _mm256_setzero_si256();
        let mut acc_dsq = _mm256_setzero_si256();
        let mut acc_rdi = _mm256_setzero_si256();

        for tap in 0..coeffs.len() {
            let coeff32 = coeff32_vecs[tap];
            let ref_values = _mm256_cvtepu16_epi32(_mm_loadu_si128(ref_rows[tap].add(j).cast()));
            let dis_values = _mm256_cvtepu16_epi32(_mm_loadu_si128(dis_rows[tap].add(j).cast()));
            let ref_coeff = _mm256_mullo_epi32(ref_values, coeff32);
            let dis_coeff = _mm256_mullo_epi32(dis_values, coeff32);

            acc_mu1 = _mm256_add_epi32(acc_mu1, ref_coeff);
            acc_mu2 = _mm256_add_epi32(acc_mu2, dis_coeff);
            acc_rsq = _mm256_add_epi32(acc_rsq, _mm256_mullo_epi32(ref_coeff, ref_values));
            acc_dsq = _mm256_add_epi32(acc_dsq, _mm256_mullo_epi32(dis_coeff, dis_values));
            acc_rdi = _mm256_add_epi32(acc_rdi, _mm256_mullo_epi32(ref_coeff, dis_values));
        }

        store_u32x8_as_u16_avx2(
            tmp_mu1,
            j,
            _mm256_srl_epi32(_mm256_add_epi32(acc_mu1, round_mu_vec), shift_mu_vec),
        );
        store_u32x8_as_u16_avx2(
            tmp_mu2,
            j,
            _mm256_srl_epi32(_mm256_add_epi32(acc_mu2, round_mu_vec), shift_mu_vec),
        );
        _mm256_storeu_si256(tmp_ref_sq.as_mut_ptr().add(j).cast(), acc_rsq);
        _mm256_storeu_si256(tmp_dis_sq.as_mut_ptr().add(j).cast(), acc_dsq);
        _mm256_storeu_si256(tmp_ref_dis.as_mut_ptr().add(j).cast(), acc_rdi);
    }

    vertical_scalar_range_wrapping(
        ref_plane,
        dis_plane,
        row_offsets,
        coeffs,
        shift_mu,
        round_mu,
        simd_end,
        tmp_mu1.len(),
        tmp_mu1,
        tmp_mu2,
        tmp_ref_sq,
        tmp_dis_sq,
        tmp_ref_dis,
    );
}

/// SAFETY: the caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
unsafe fn vertical_row_non_wrapping_avx2(
    ref_plane: &[u16],
    dis_plane: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    shift_mu: u32,
    round_mu: u32,
    sq_shift: u32,
    sq_round64: u64,
    tmp_mu1: &mut [u16],
    tmp_mu2: &mut [u16],
    tmp_ref_sq: &mut [u32],
    tmp_dis_sq: &mut [u32],
    tmp_ref_dis: &mut [u32],
) {
    debug_assert_eq!(row_offsets.len(), coeffs.len());

    let simd_end = tmp_mu1.len() / 8 * 8;
    let round_mu_vec = _mm256_set1_epi32(round_mu as i32);
    let shift_mu_vec = _mm_cvtsi32_si128(shift_mu as i32);
    let sq_round_vec = _mm256_set1_epi64x(sq_round64 as i64);
    let sq_shift_vec = _mm_cvtsi32_si128(sq_shift as i32);
    let pack_indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let mut ref_rows = [std::ptr::null::<u16>(); FILTER_TAP_CAP];
    let mut dis_rows = [std::ptr::null::<u16>(); FILTER_TAP_CAP];
    let mut coeff32_vecs = [_mm256_setzero_si256(); FILTER_TAP_CAP];

    for tap in 0..coeffs.len() {
        ref_rows[tap] = ref_plane.as_ptr().add(row_offsets[tap]);
        dis_rows[tap] = dis_plane.as_ptr().add(row_offsets[tap]);
        coeff32_vecs[tap] = _mm256_set1_epi32(coeffs[tap] as i32);
    }

    for j in (0..simd_end).step_by(8) {
        let mut acc_mu1 = _mm256_setzero_si256();
        let mut acc_mu2 = _mm256_setzero_si256();
        let mut acc_ref_even = _mm256_setzero_si256();
        let mut acc_ref_odd = _mm256_setzero_si256();
        let mut acc_dis_even = _mm256_setzero_si256();
        let mut acc_dis_odd = _mm256_setzero_si256();
        let mut acc_rdi_even = _mm256_setzero_si256();
        let mut acc_rdi_odd = _mm256_setzero_si256();

        for tap in 0..coeffs.len() {
            let coeff32 = coeff32_vecs[tap];
            let ref_values = _mm256_cvtepu16_epi32(_mm_loadu_si128(ref_rows[tap].add(j).cast()));
            let dis_values = _mm256_cvtepu16_epi32(_mm_loadu_si128(dis_rows[tap].add(j).cast()));
            let ref_coeff = _mm256_mullo_epi32(ref_values, coeff32);
            let dis_coeff = _mm256_mullo_epi32(dis_values, coeff32);
            let ref_values_odd = _mm256_srli_epi64(ref_values, 32);
            let dis_values_odd = _mm256_srli_epi64(dis_values, 32);
            let ref_coeff_odd = _mm256_srli_epi64(ref_coeff, 32);
            let dis_coeff_odd = _mm256_srli_epi64(dis_coeff, 32);

            acc_mu1 = _mm256_add_epi32(acc_mu1, ref_coeff);
            acc_mu2 = _mm256_add_epi32(acc_mu2, dis_coeff);
            acc_ref_even = _mm256_add_epi64(acc_ref_even, _mm256_mul_epu32(ref_values, ref_coeff));
            acc_ref_odd =
                _mm256_add_epi64(acc_ref_odd, _mm256_mul_epu32(ref_values_odd, ref_coeff_odd));
            acc_dis_even = _mm256_add_epi64(acc_dis_even, _mm256_mul_epu32(dis_values, dis_coeff));
            acc_dis_odd =
                _mm256_add_epi64(acc_dis_odd, _mm256_mul_epu32(dis_values_odd, dis_coeff_odd));
            acc_rdi_even = _mm256_add_epi64(acc_rdi_even, _mm256_mul_epu32(ref_coeff, dis_values));
            acc_rdi_odd =
                _mm256_add_epi64(acc_rdi_odd, _mm256_mul_epu32(ref_coeff_odd, dis_values_odd));
        }

        store_u32x8_as_u16_avx2(
            tmp_mu1,
            j,
            _mm256_srl_epi32(_mm256_add_epi32(acc_mu1, round_mu_vec), shift_mu_vec),
        );
        store_u32x8_as_u16_avx2(
            tmp_mu2,
            j,
            _mm256_srl_epi32(_mm256_add_epi32(acc_mu2, round_mu_vec), shift_mu_vec),
        );

        let ref_even = _mm256_srl_epi64(_mm256_add_epi64(acc_ref_even, sq_round_vec), sq_shift_vec);
        let ref_odd = _mm256_srl_epi64(_mm256_add_epi64(acc_ref_odd, sq_round_vec), sq_shift_vec);
        let dis_even = _mm256_srl_epi64(_mm256_add_epi64(acc_dis_even, sq_round_vec), sq_shift_vec);
        let dis_odd = _mm256_srl_epi64(_mm256_add_epi64(acc_dis_odd, sq_round_vec), sq_shift_vec);
        let rdi_even = _mm256_srl_epi64(_mm256_add_epi64(acc_rdi_even, sq_round_vec), sq_shift_vec);
        let rdi_odd = _mm256_srl_epi64(_mm256_add_epi64(acc_rdi_odd, sq_round_vec), sq_shift_vec);

        store_interleaved_u64x4_as_u32_avx2(tmp_ref_sq, j, ref_even, ref_odd, pack_indices);
        store_interleaved_u64x4_as_u32_avx2(tmp_dis_sq, j, dis_even, dis_odd, pack_indices);
        store_interleaved_u64x4_as_u32_avx2(tmp_ref_dis, j, rdi_even, rdi_odd, pack_indices);
    }

    vertical_scalar_range_non_wrapping(
        ref_plane,
        dis_plane,
        row_offsets,
        coeffs,
        shift_mu,
        round_mu,
        sq_shift,
        sq_round64,
        simd_end,
        tmp_mu1.len(),
        tmp_mu1,
        tmp_mu2,
        tmp_ref_sq,
        tmp_dis_sq,
        tmp_ref_dis,
    );
}

/// SAFETY: the caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
unsafe fn horizontal_row_sse2(
    tmp_mu1: &[u16],
    tmp_mu2: &[u16],
    tmp_ref_sq: &[u32],
    tmp_dis_sq: &[u32],
    tmp_ref_dis: &[u32],
    coeffs: &[u16],
    half: usize,
    vif_enhn_gain_limit: f64,
    accum: &mut RunningStatAccumulators,
) {
    let (body_start, simd_end) = horizontal_simd_body_range(tmp_mu1.len(), half, 4);
    let mut coeff16_vecs = [_mm_setzero_si128(); FILTER_TAP_CAP];
    let mut coeff32_vecs = [_mm_setzero_si128(); FILTER_TAP_CAP];

    for tap in 0..coeffs.len() {
        coeff16_vecs[tap] = _mm_set1_epi16(coeffs[tap] as i16);
        coeff32_vecs[tap] = _mm_set1_epi32(coeffs[tap] as i32);
    }

    horizontal_scalar_range(
        tmp_mu1,
        tmp_mu2,
        tmp_ref_sq,
        tmp_dis_sq,
        tmp_ref_dis,
        coeffs,
        half,
        0,
        body_start,
        vif_enhn_gain_limit,
        accum,
    );

    for j in (body_start..simd_end).step_by(4) {
        let base_start = j - half;
        let mut acc_mu1 = _mm_setzero_si128();
        let mut acc_mu2 = _mm_setzero_si128();
        let mut acc_ref_even = _mm_setzero_si128();
        let mut acc_ref_odd = _mm_setzero_si128();
        let mut acc_dis_even = _mm_setzero_si128();
        let mut acc_dis_odd = _mm_setzero_si128();
        let mut acc_rdi_even = _mm_setzero_si128();
        let mut acc_rdi_odd = _mm_setzero_si128();

        for tap in 0..coeffs.len() {
            let base = base_start + tap;
            let coeff16 = coeff16_vecs[tap];
            let coeff32 = coeff32_vecs[tap];

            acc_mu1 = _mm_add_epi32(
                acc_mu1,
                mul_u16x4_to_u32(tmp_mu1.as_ptr().add(base), coeff16),
            );
            acc_mu2 = _mm_add_epi32(
                acc_mu2,
                mul_u16x4_to_u32(tmp_mu2.as_ptr().add(base), coeff16),
            );

            let ref_values = _mm_loadu_si128(tmp_ref_sq.as_ptr().add(base).cast());
            let dis_values = _mm_loadu_si128(tmp_dis_sq.as_ptr().add(base).cast());
            let rdi_values = _mm_loadu_si128(tmp_ref_dis.as_ptr().add(base).cast());
            acc_ref_even = _mm_add_epi64(acc_ref_even, _mm_mul_epu32(ref_values, coeff32));
            acc_ref_odd = _mm_add_epi64(
                acc_ref_odd,
                _mm_mul_epu32(_mm_srli_si128(ref_values, 4), coeff32),
            );
            acc_dis_even = _mm_add_epi64(acc_dis_even, _mm_mul_epu32(dis_values, coeff32));
            acc_dis_odd = _mm_add_epi64(
                acc_dis_odd,
                _mm_mul_epu32(_mm_srli_si128(dis_values, 4), coeff32),
            );
            acc_rdi_even = _mm_add_epi64(acc_rdi_even, _mm_mul_epu32(rdi_values, coeff32));
            acc_rdi_odd = _mm_add_epi64(
                acc_rdi_odd,
                _mm_mul_epu32(_mm_srli_si128(rdi_values, 4), coeff32),
            );
        }

        let mut mu1_values = [0u32; 4];
        let mut mu2_values = [0u32; 4];
        let mut ref_even = [0u64; 2];
        let mut ref_odd = [0u64; 2];
        let mut dis_even = [0u64; 2];
        let mut dis_odd = [0u64; 2];
        let mut rdi_even = [0u64; 2];
        let mut rdi_odd = [0u64; 2];
        _mm_storeu_si128(mu1_values.as_mut_ptr().cast(), acc_mu1);
        _mm_storeu_si128(mu2_values.as_mut_ptr().cast(), acc_mu2);
        _mm_storeu_si128(ref_even.as_mut_ptr().cast(), acc_ref_even);
        _mm_storeu_si128(ref_odd.as_mut_ptr().cast(), acc_ref_odd);
        _mm_storeu_si128(dis_even.as_mut_ptr().cast(), acc_dis_even);
        _mm_storeu_si128(dis_odd.as_mut_ptr().cast(), acc_dis_odd);
        _mm_storeu_si128(rdi_even.as_mut_ptr().cast(), acc_rdi_even);
        _mm_storeu_si128(rdi_odd.as_mut_ptr().cast(), acc_rdi_odd);

        for lane in 0..4 {
            let even = lane / 2;
            let acc_ref = if lane % 2 == 0 {
                ref_even[even]
            } else {
                ref_odd[even]
            };
            let acc_dis = if lane % 2 == 0 {
                dis_even[even]
            } else {
                dis_odd[even]
            };
            let acc_rdi = if lane % 2 == 0 {
                rdi_even[even]
            } else {
                rdi_odd[even]
            };
            process_filtered_pixel(
                mu1_values[lane],
                mu2_values[lane],
                acc_ref,
                acc_dis,
                acc_rdi,
                vif_enhn_gain_limit,
                accum,
            );
        }
    }

    horizontal_scalar_range(
        tmp_mu1,
        tmp_mu2,
        tmp_ref_sq,
        tmp_dis_sq,
        tmp_ref_dis,
        coeffs,
        half,
        simd_end,
        tmp_mu1.len(),
        vif_enhn_gain_limit,
        accum,
    );
}

/// SAFETY: the caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
unsafe fn horizontal_row_avx2(
    tmp_mu1: &[u16],
    tmp_mu2: &[u16],
    tmp_ref_sq: &[u32],
    tmp_dis_sq: &[u32],
    tmp_ref_dis: &[u32],
    coeffs: &[u16],
    half: usize,
    vif_enhn_gain_limit: f64,
    accum: &mut RunningStatAccumulators,
) {
    let (body_start, simd_end) = horizontal_simd_body_range(tmp_mu1.len(), half, 8);
    let mul_round = _mm256_set1_epi64x(1i64 << 31);
    let filt_round = _mm256_set1_epi64x(32768);
    let zero = _mm256_setzero_si256();
    let mut coeff32_vecs = [_mm256_setzero_si256(); FILTER_TAP_CAP];

    for tap in 0..coeffs.len() {
        coeff32_vecs[tap] = _mm256_set1_epi32(coeffs[tap] as i32);
    }

    horizontal_scalar_range(
        tmp_mu1,
        tmp_mu2,
        tmp_ref_sq,
        tmp_dis_sq,
        tmp_ref_dis,
        coeffs,
        half,
        0,
        body_start,
        vif_enhn_gain_limit,
        accum,
    );

    for j in (body_start..simd_end).step_by(8) {
        let base_start = j - half;
        let mut acc_mu1 = _mm256_setzero_si256();
        let mut acc_mu2 = _mm256_setzero_si256();
        let mut acc_ref_even = _mm256_setzero_si256();
        let mut acc_ref_odd = _mm256_setzero_si256();
        let mut acc_dis_even = _mm256_setzero_si256();
        let mut acc_dis_odd = _mm256_setzero_si256();
        let mut acc_rdi_even = _mm256_setzero_si256();
        let mut acc_rdi_odd = _mm256_setzero_si256();

        for tap in 0..coeffs.len() {
            let base = base_start + tap;
            let coeff32 = coeff32_vecs[tap];
            let mu1_values =
                _mm256_cvtepu16_epi32(_mm_loadu_si128(tmp_mu1.as_ptr().add(base).cast()));
            let mu2_values =
                _mm256_cvtepu16_epi32(_mm_loadu_si128(tmp_mu2.as_ptr().add(base).cast()));
            acc_mu1 = _mm256_add_epi32(acc_mu1, _mm256_mullo_epi32(mu1_values, coeff32));
            acc_mu2 = _mm256_add_epi32(acc_mu2, _mm256_mullo_epi32(mu2_values, coeff32));

            let ref_values = _mm256_loadu_si256(tmp_ref_sq.as_ptr().add(base).cast());
            let dis_values = _mm256_loadu_si256(tmp_dis_sq.as_ptr().add(base).cast());
            let rdi_values = _mm256_loadu_si256(tmp_ref_dis.as_ptr().add(base).cast());
            acc_ref_even = _mm256_add_epi64(acc_ref_even, _mm256_mul_epu32(ref_values, coeff32));
            acc_ref_odd = _mm256_add_epi64(
                acc_ref_odd,
                _mm256_mul_epu32(_mm256_srli_epi64(ref_values, 32), coeff32),
            );
            acc_dis_even = _mm256_add_epi64(acc_dis_even, _mm256_mul_epu32(dis_values, coeff32));
            acc_dis_odd = _mm256_add_epi64(
                acc_dis_odd,
                _mm256_mul_epu32(_mm256_srli_epi64(dis_values, 32), coeff32),
            );
            acc_rdi_even = _mm256_add_epi64(acc_rdi_even, _mm256_mul_epu32(rdi_values, coeff32));
            acc_rdi_odd = _mm256_add_epi64(
                acc_rdi_odd,
                _mm256_mul_epu32(_mm256_srli_epi64(rdi_values, 32), coeff32),
            );
        }

        let (mu1_sq_even, mu1_sq_odd) = rounded_mulhi_epu32_avx2(acc_mu1, acc_mu1, mul_round);
        let (mu2_sq_even, mu2_sq_odd) = rounded_mulhi_epu32_avx2(acc_mu2, acc_mu2, mul_round);
        let (mu1_mu2_even, mu1_mu2_odd) = rounded_mulhi_epu32_avx2(acc_mu1, acc_mu2, mul_round);

        let ref_filt_even = rounded_shift16_epu64_avx2(acc_ref_even, filt_round);
        let ref_filt_odd = rounded_shift16_epu64_avx2(acc_ref_odd, filt_round);
        let dis_filt_even = rounded_shift16_epu64_avx2(acc_dis_even, filt_round);
        let dis_filt_odd = rounded_shift16_epu64_avx2(acc_dis_odd, filt_round);
        let rdi_filt_even = rounded_shift16_epu64_avx2(acc_rdi_even, filt_round);
        let rdi_filt_odd = rounded_shift16_epu64_avx2(acc_rdi_odd, filt_round);

        let sigma1_even = _mm256_sub_epi64(ref_filt_even, mu1_sq_even);
        let sigma1_odd = _mm256_sub_epi64(ref_filt_odd, mu1_sq_odd);
        let sigma2_even = _mm256_sub_epi64(dis_filt_even, mu2_sq_even);
        let sigma2_odd = _mm256_sub_epi64(dis_filt_odd, mu2_sq_odd);
        let sigma2_even = _mm256_and_si256(sigma2_even, _mm256_cmpgt_epi64(sigma2_even, zero));
        let sigma2_odd = _mm256_and_si256(sigma2_odd, _mm256_cmpgt_epi64(sigma2_odd, zero));
        let sigma12_even = _mm256_sub_epi64(rdi_filt_even, mu1_mu2_even);
        let sigma12_odd = _mm256_sub_epi64(rdi_filt_odd, mu1_mu2_odd);

        let mut sigma1_even_values = [0i64; 4];
        let mut sigma1_odd_values = [0i64; 4];
        let mut sigma2_even_values = [0i64; 4];
        let mut sigma2_odd_values = [0i64; 4];
        let mut sigma12_even_values = [0i64; 4];
        let mut sigma12_odd_values = [0i64; 4];
        _mm256_storeu_si256(sigma1_even_values.as_mut_ptr().cast(), sigma1_even);
        _mm256_storeu_si256(sigma1_odd_values.as_mut_ptr().cast(), sigma1_odd);
        _mm256_storeu_si256(sigma2_even_values.as_mut_ptr().cast(), sigma2_even);
        _mm256_storeu_si256(sigma2_odd_values.as_mut_ptr().cast(), sigma2_odd);
        _mm256_storeu_si256(sigma12_even_values.as_mut_ptr().cast(), sigma12_even);
        _mm256_storeu_si256(sigma12_odd_values.as_mut_ptr().cast(), sigma12_odd);

        for pair in 0..4 {
            process_sigma_values(
                sigma1_even_values[pair],
                sigma2_even_values[pair],
                sigma12_even_values[pair],
                vif_enhn_gain_limit,
                accum,
            );
            process_sigma_values(
                sigma1_odd_values[pair],
                sigma2_odd_values[pair],
                sigma12_odd_values[pair],
                vif_enhn_gain_limit,
                accum,
            );
        }
    }

    horizontal_scalar_range(
        tmp_mu1,
        tmp_mu2,
        tmp_ref_sq,
        tmp_dis_sq,
        tmp_ref_dis,
        coeffs,
        half,
        simd_end,
        tmp_mu1.len(),
        vif_enhn_gain_limit,
        accum,
    );
}

#[inline(always)]
unsafe fn mul_u16x4_to_u32(values: *const u16, coeff: __m128i) -> __m128i {
    let packed = _mm_loadl_epi64(values.cast());
    let low = _mm_mullo_epi16(packed, coeff);
    let high = _mm_mulhi_epu16(packed, coeff);
    _mm_unpacklo_epi16(low, high)
}

#[target_feature(enable = "avx2")]
unsafe fn rounded_mulhi_epu32_avx2(
    lhs: __m256i,
    rhs: __m256i,
    round: __m256i,
) -> (__m256i, __m256i) {
    let lhs_odd = _mm256_srli_epi64(lhs, 32);
    let rhs_odd = _mm256_srli_epi64(rhs, 32);
    let even = _mm256_srli_epi64(_mm256_add_epi64(_mm256_mul_epu32(lhs, rhs), round), 32);
    let odd = _mm256_srli_epi64(
        _mm256_add_epi64(_mm256_mul_epu32(lhs_odd, rhs_odd), round),
        32,
    );
    (even, odd)
}

#[target_feature(enable = "avx2")]
unsafe fn rounded_shift16_epu64_avx2(values: __m256i, round: __m256i) -> __m256i {
    _mm256_srli_epi64(_mm256_add_epi64(values, round), 16)
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn store_u32x8_as_u16_avx2(out: &mut [u16], start: usize, values: __m256i) {
    let low = _mm256_castsi256_si128(values);
    let high = _mm256_extracti128_si256(values, 1);
    let packed = _mm_packus_epi32(low, high);
    _mm_storeu_si128(out.as_mut_ptr().add(start).cast(), packed);
}

#[target_feature(enable = "avx2")]
unsafe fn store_interleaved_u64x4_as_u32_avx2(
    out: &mut [u32],
    start: usize,
    even: __m256i,
    odd: __m256i,
    pack_indices: __m256i,
) {
    let even = _mm256_permutevar8x32_epi32(even, pack_indices);
    let odd = _mm256_permutevar8x32_epi32(odd, pack_indices);
    let even = _mm256_castsi256_si128(even);
    let odd = _mm256_castsi256_si128(odd);
    let lo = _mm_unpacklo_epi32(even, odd);
    let hi = _mm_unpackhi_epi32(even, odd);
    _mm_storeu_si128(out.as_mut_ptr().add(start).cast(), lo);
    _mm_storeu_si128(out.as_mut_ptr().add(start + 4).cast(), hi);
}

#[cfg(test)]
mod tests {
    use super::{vertical_row_non_wrapping_avx2, vertical_row_wrapping_avx2, FILTER, FILTER_WIDTH};
    use crate::stat::{
        horizontal_scalar_range, reflected_row_offsets, stat_params,
        vertical_scalar_range_non_wrapping, vertical_scalar_range_wrapping,
        RunningStatAccumulators,
    };
    use vmaf_cpu::SimdBackend;

    fn patterned_plane(width: usize, height: usize, modulus: u16, bias: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    ((x * 29 + y * 31 + ((x ^ y) * 13) + (x * y * 7) + bias) % modulus as usize)
                        as u16
                })
            })
            .collect()
    }

    #[test]
    fn vertical_avx2_matches_scalar_non_wrapping_on_misaligned_odd_width() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let width = 29;
        let height = 13;
        let scale = 0;
        let len = width * height;
        let mut ref_storage = vec![0u16; len + 3];
        let mut dis_storage = vec![0u16; len + 5];
        let ref_plane = &mut ref_storage[1..1 + len];
        let dis_plane = &mut dis_storage[3..3 + len];
        ref_plane.copy_from_slice(&patterned_plane(width, height, 1 << 12, 19));
        dis_plane.copy_from_slice(&patterned_plane(width, height, 1 << 12, 71));

        let filt = &FILTER[scale][..FILTER_WIDTH[scale]];
        let half = filt.len() / 2;
        let (shift_mu, round_mu, sq_shift, sq_round64) = stat_params(12, scale);

        let mut scalar_mu1 = vec![0u16; width];
        let mut scalar_mu2 = vec![0u16; width];
        let mut scalar_ref_sq = vec![0u32; width];
        let mut scalar_dis_sq = vec![0u32; width];
        let mut scalar_ref_dis = vec![0u32; width];
        let mut simd_mu1 = vec![0u16; width];
        let mut simd_mu2 = vec![0u16; width];
        let mut simd_ref_sq = vec![0u32; width];
        let mut simd_dis_sq = vec![0u32; width];
        let mut simd_ref_dis = vec![0u32; width];

        for i in 0..height {
            let row_offsets = reflected_row_offsets(i, height, width, half, filt.len());

            vertical_scalar_range_non_wrapping(
                ref_plane,
                dis_plane,
                &row_offsets[..filt.len()],
                filt,
                shift_mu,
                round_mu,
                sq_shift,
                sq_round64,
                0,
                width,
                &mut scalar_mu1,
                &mut scalar_mu2,
                &mut scalar_ref_sq,
                &mut scalar_dis_sq,
                &mut scalar_ref_dis,
            );

            // SAFETY: the test checks runtime AVX2 support above.
            unsafe {
                vertical_row_non_wrapping_avx2(
                    ref_plane,
                    dis_plane,
                    &row_offsets[..filt.len()],
                    filt,
                    shift_mu,
                    round_mu,
                    sq_shift,
                    sq_round64,
                    &mut simd_mu1,
                    &mut simd_mu2,
                    &mut simd_ref_sq,
                    &mut simd_dis_sq,
                    &mut simd_ref_dis,
                );
            }

            assert_eq!(scalar_mu1, simd_mu1, "row {i} mu1");
            assert_eq!(scalar_mu2, simd_mu2, "row {i} mu2");
            assert_eq!(scalar_ref_sq, simd_ref_sq, "row {i} ref_sq");
            assert_eq!(scalar_dis_sq, simd_dis_sq, "row {i} dis_sq");
            assert_eq!(scalar_ref_dis, simd_ref_dis, "row {i} ref_dis");
        }
    }

    #[test]
    fn vertical_avx2_matches_scalar_wrapping_on_misaligned_odd_width() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let width = 27;
        let height = 17;
        let scale = 0;
        let len = width * height;
        let mut ref_storage = vec![0u16; len + 5];
        let mut dis_storage = vec![0u16; len + 7];
        let ref_plane = &mut ref_storage[3..3 + len];
        let dis_plane = &mut dis_storage[1..1 + len];
        ref_plane.copy_from_slice(&patterned_plane(width, height, 256, 5));
        dis_plane.copy_from_slice(&patterned_plane(width, height, 256, 133));

        let filt = &FILTER[scale][..FILTER_WIDTH[scale]];
        let half = filt.len() / 2;
        let (shift_mu, round_mu, _, _) = stat_params(8, scale);

        let mut scalar_mu1 = vec![0u16; width];
        let mut scalar_mu2 = vec![0u16; width];
        let mut scalar_ref_sq = vec![0u32; width];
        let mut scalar_dis_sq = vec![0u32; width];
        let mut scalar_ref_dis = vec![0u32; width];
        let mut simd_mu1 = vec![0u16; width];
        let mut simd_mu2 = vec![0u16; width];
        let mut simd_ref_sq = vec![0u32; width];
        let mut simd_dis_sq = vec![0u32; width];
        let mut simd_ref_dis = vec![0u32; width];

        for i in 0..height {
            let row_offsets = reflected_row_offsets(i, height, width, half, filt.len());

            vertical_scalar_range_wrapping(
                ref_plane,
                dis_plane,
                &row_offsets[..filt.len()],
                filt,
                shift_mu,
                round_mu,
                0,
                width,
                &mut scalar_mu1,
                &mut scalar_mu2,
                &mut scalar_ref_sq,
                &mut scalar_dis_sq,
                &mut scalar_ref_dis,
            );

            // SAFETY: the test checks runtime AVX2 support above.
            unsafe {
                vertical_row_wrapping_avx2(
                    ref_plane,
                    dis_plane,
                    &row_offsets[..filt.len()],
                    filt,
                    shift_mu,
                    round_mu,
                    &mut simd_mu1,
                    &mut simd_mu2,
                    &mut simd_ref_sq,
                    &mut simd_dis_sq,
                    &mut simd_ref_dis,
                );
            }

            assert_eq!(scalar_mu1, simd_mu1, "row {i} mu1");
            assert_eq!(scalar_mu2, simd_mu2, "row {i} mu2");
            assert_eq!(scalar_ref_sq, simd_ref_sq, "row {i} ref_sq");
            assert_eq!(scalar_dis_sq, simd_dis_sq, "row {i} dis_sq");
            assert_eq!(scalar_ref_dis, simd_ref_dis, "row {i} ref_dis");
        }
    }

    #[test]
    fn horizontal_avx2_matches_scalar_non_wrapping_on_misaligned_odd_width() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let width = 29;
        let height = 13;
        let scale = 0;
        let row = 7;
        let len = width * height;
        let mut ref_storage = vec![0u16; len + 3];
        let mut dis_storage = vec![0u16; len + 5];
        let ref_plane = &mut ref_storage[1..1 + len];
        let dis_plane = &mut dis_storage[3..3 + len];
        ref_plane.copy_from_slice(&patterned_plane(width, height, 1 << 12, 19));
        dis_plane.copy_from_slice(&patterned_plane(width, height, 1 << 12, 71));

        let filt = &FILTER[scale][..FILTER_WIDTH[scale]];
        let half = filt.len() / 2;
        let (shift_mu, round_mu, sq_shift, sq_round64) = stat_params(12, scale);
        let row_offsets = reflected_row_offsets(row, height, width, half, filt.len());

        let mut tmp_mu1 = vec![0u16; width];
        let mut tmp_mu2 = vec![0u16; width];
        let mut tmp_ref_sq = vec![0u32; width];
        let mut tmp_dis_sq = vec![0u32; width];
        let mut tmp_ref_dis = vec![0u32; width];
        vertical_scalar_range_non_wrapping(
            ref_plane,
            dis_plane,
            &row_offsets[..filt.len()],
            filt,
            shift_mu,
            round_mu,
            sq_shift,
            sq_round64,
            0,
            width,
            &mut tmp_mu1,
            &mut tmp_mu2,
            &mut tmp_ref_sq,
            &mut tmp_dis_sq,
            &mut tmp_ref_dis,
        );

        let mut scalar = RunningStatAccumulators::default();
        horizontal_scalar_range(
            &tmp_mu1,
            &tmp_mu2,
            &tmp_ref_sq,
            &tmp_dis_sq,
            &tmp_ref_dis,
            filt,
            half,
            0,
            width,
            100.0,
            &mut scalar,
        );

        let mut simd = RunningStatAccumulators::default();
        unsafe {
            super::horizontal_row_avx2(
                &tmp_mu1,
                &tmp_mu2,
                &tmp_ref_sq,
                &tmp_dis_sq,
                &tmp_ref_dis,
                filt,
                half,
                100.0,
                &mut simd,
            );
        }

        assert_eq!(scalar.accum_num_log, simd.accum_num_log);
        assert_eq!(scalar.accum_den_log, simd.accum_den_log);
        assert_eq!(scalar.accum_num_non_log, simd.accum_num_non_log);
        assert_eq!(scalar.accum_den_non_log, simd.accum_den_non_log);
    }

    #[test]
    fn horizontal_avx2_matches_scalar_wrapping_on_misaligned_odd_width() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let width = 27;
        let height = 17;
        let scale = 0;
        let row = 9;
        let len = width * height;
        let mut ref_storage = vec![0u16; len + 5];
        let mut dis_storage = vec![0u16; len + 7];
        let ref_plane = &mut ref_storage[3..3 + len];
        let dis_plane = &mut dis_storage[1..1 + len];
        ref_plane.copy_from_slice(&patterned_plane(width, height, 256, 5));
        dis_plane.copy_from_slice(&patterned_plane(width, height, 256, 133));

        let filt = &FILTER[scale][..FILTER_WIDTH[scale]];
        let half = filt.len() / 2;
        let (shift_mu, round_mu, _, _) = stat_params(8, scale);
        let row_offsets = reflected_row_offsets(row, height, width, half, filt.len());

        let mut tmp_mu1 = vec![0u16; width];
        let mut tmp_mu2 = vec![0u16; width];
        let mut tmp_ref_sq = vec![0u32; width];
        let mut tmp_dis_sq = vec![0u32; width];
        let mut tmp_ref_dis = vec![0u32; width];
        vertical_scalar_range_wrapping(
            ref_plane,
            dis_plane,
            &row_offsets[..filt.len()],
            filt,
            shift_mu,
            round_mu,
            0,
            width,
            &mut tmp_mu1,
            &mut tmp_mu2,
            &mut tmp_ref_sq,
            &mut tmp_dis_sq,
            &mut tmp_ref_dis,
        );

        let mut scalar = RunningStatAccumulators::default();
        horizontal_scalar_range(
            &tmp_mu1,
            &tmp_mu2,
            &tmp_ref_sq,
            &tmp_dis_sq,
            &tmp_ref_dis,
            filt,
            half,
            0,
            width,
            100.0,
            &mut scalar,
        );

        let mut simd = RunningStatAccumulators::default();
        unsafe {
            super::horizontal_row_avx2(
                &tmp_mu1,
                &tmp_mu2,
                &tmp_ref_sq,
                &tmp_dis_sq,
                &tmp_ref_dis,
                filt,
                half,
                100.0,
                &mut simd,
            );
        }

        assert_eq!(scalar.accum_num_log, simd.accum_num_log);
        assert_eq!(scalar.accum_den_log, simd.accum_den_log);
        assert_eq!(scalar.accum_num_non_log, simd.accum_num_non_log);
        assert_eq!(scalar.accum_den_non_log, simd.accum_den_non_log);
    }
}
