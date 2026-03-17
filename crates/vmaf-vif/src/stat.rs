//! VIF statistic core — spec §4.2.6–4.2.8

use crate::math::{log2_32, log2_64, reflect_index};
use crate::tables::{FILTER, FILTER_WIDTH, LOG2_TABLE};
use vmaf_cpu::{Align32, AlignedScratch, SimdBackend};

mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

const FILTER_TAP_CAP: usize = 18;
const SIGMA_NSQ: u32 = 131072; // 2^17
const EPSILON: f64 = 65536.0 * 1e-10; // ≈ 6.5536e-6

/// Per-scale VIF numerator and denominator — spec §4.2.8.
pub(crate) struct ScaleStat {
    pub num: f64,
    pub den: f64,
}

/// Reusable aligned row buffers for one VIF statistic pass.
#[derive(Debug)]
pub(crate) struct VifStatWorkspace {
    tmp_mu1: AlignedScratch<u16, Align32>,
    tmp_mu2: AlignedScratch<u16, Align32>,
    tmp_ref_sq: AlignedScratch<u32, Align32>,
    tmp_dis_sq: AlignedScratch<u32, Align32>,
    tmp_ref_dis: AlignedScratch<u32, Align32>,
}

impl Default for VifStatWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}

impl VifStatWorkspace {
    pub(crate) fn new(max_width: usize) -> Self {
        Self {
            tmp_mu1: AlignedScratch::zeroed(max_width),
            tmp_mu2: AlignedScratch::zeroed(max_width),
            tmp_ref_sq: AlignedScratch::zeroed(max_width),
            tmp_dis_sq: AlignedScratch::zeroed(max_width),
            tmp_ref_dis: AlignedScratch::zeroed(max_width),
        }
    }

    fn prepare(&mut self, width: usize) {
        if self.tmp_mu1.len() < width {
            self.tmp_mu1 = AlignedScratch::zeroed(width);
        }
        if self.tmp_mu2.len() < width {
            self.tmp_mu2 = AlignedScratch::zeroed(width);
        }
        if self.tmp_ref_sq.len() < width {
            self.tmp_ref_sq = AlignedScratch::zeroed(width);
        }
        if self.tmp_dis_sq.len() < width {
            self.tmp_dis_sq = AlignedScratch::zeroed(width);
        }
        if self.tmp_ref_dis.len() < width {
            self.tmp_ref_dis = AlignedScratch::zeroed(width);
        }
    }
}

#[derive(Default)]
struct RunningStatAccumulators {
    accum_num_log: i64,
    accum_den_log: i64,
    accum_num_non_log: i64,
    accum_den_non_log: i64,
}

/// Compute VIF statistics for one scale — spec §4.2.6–4.2.8.
///
/// CRITICAL: for `bpc == 8 && scale == 0`, the squared accumulators in the
/// vertical pass use **uint32** (wrapping) intentionally — spec §8.
/// All other paths use uint64.
pub(crate) fn vif_statistic(
    ref_plane: &[u16],
    dis_plane: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    vif_enhn_gain_limit: f64,
    backend: SimdBackend,
) -> ScaleStat {
    let mut workspace = VifStatWorkspace::new(width);
    vif_statistic_with_workspace(
        ref_plane,
        dis_plane,
        width,
        height,
        bpc,
        scale,
        vif_enhn_gain_limit,
        &mut workspace,
        backend,
    )
}

pub(crate) fn vif_statistic_with_workspace(
    ref_plane: &[u16],
    dis_plane: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    vif_enhn_gain_limit: f64,
    workspace: &mut VifStatWorkspace,
    backend: SimdBackend,
) -> ScaleStat {
    match backend {
        SimdBackend::Scalar => vif_statistic_scalar_with_workspace(
            ref_plane,
            dis_plane,
            width,
            height,
            bpc,
            scale,
            vif_enhn_gain_limit,
            workspace,
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Sse2 | SimdBackend::X86Avx2Fma | SimdBackend::X86Avx512 => {
            x86::vif_statistic(
                ref_plane,
                dis_plane,
                width,
                height,
                bpc,
                scale,
                vif_enhn_gain_limit,
                workspace,
                backend,
            )
        }
        SimdBackend::Aarch64Neon => aarch64::vif_statistic(
            ref_plane,
            dis_plane,
            width,
            height,
            bpc,
            scale,
            vif_enhn_gain_limit,
            workspace,
            backend,
        ),
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        _ => vif_statistic_scalar_with_workspace(
            ref_plane,
            dis_plane,
            width,
            height,
            bpc,
            scale,
            vif_enhn_gain_limit,
            workspace,
        ),
    }
}

fn vif_statistic_scalar_with_workspace(
    ref_plane: &[u16],
    dis_plane: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    vif_enhn_gain_limit: f64,
    workspace: &mut VifStatWorkspace,
) -> ScaleStat {
    let filt = &FILTER[scale][..FILTER_WIDTH[scale]];
    let half = filt.len() / 2;
    let (shift_mu, round_mu, sq_shift, sq_round64) = stat_params(bpc, scale);
    let uses_wrapping_sq = bpc == 8 && scale == 0;
    let mut accum = RunningStatAccumulators::default();

    workspace.prepare(width);
    let tmp_mu1 = &mut workspace.tmp_mu1.as_mut_slice()[..width];
    let tmp_mu2 = &mut workspace.tmp_mu2.as_mut_slice()[..width];
    let tmp_ref_sq = &mut workspace.tmp_ref_sq.as_mut_slice()[..width];
    let tmp_dis_sq = &mut workspace.tmp_dis_sq.as_mut_slice()[..width];
    let tmp_ref_dis = &mut workspace.tmp_ref_dis.as_mut_slice()[..width];

    for i in 0..height {
        let row_offsets = reflected_row_offsets(i, height, width, half, filt.len());

        if uses_wrapping_sq {
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
        } else {
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

        horizontal_scalar_range(
            tmp_mu1,
            tmp_mu2,
            tmp_ref_sq,
            tmp_dis_sq,
            tmp_ref_dis,
            filt,
            half,
            0,
            width,
            vif_enhn_gain_limit,
            &mut accum,
        );
    }

    finalize_scale_stat(accum)
}

#[inline]
fn stat_params(bpc: u8, scale: usize) -> (u32, u32, u32, u64) {
    let (shift_mu, round_mu) = if scale == 0 {
        (bpc as u32, 1u32 << (bpc - 1))
    } else {
        (16u32, 32768u32)
    };

    let (sq_shift, sq_round64) = if scale == 0 && bpc == 8 {
        (0, 0)
    } else if scale == 0 {
        let sh = (bpc as u32 - 8) * 2;
        (sh, 1u64 << (sh - 1))
    } else {
        (16, 32768)
    };

    (shift_mu, round_mu, sq_shift, sq_round64)
}

#[inline]
fn reflected_row_offsets(
    i: usize,
    height: usize,
    width: usize,
    half: usize,
    taps: usize,
) -> [usize; FILTER_TAP_CAP] {
    let mut row_offsets = [0usize; FILTER_TAP_CAP];

    for k in 0..taps {
        let ii = reflect_index(i as i32 - half as i32 + k as i32, height as i32);
        row_offsets[k] = ii * width;
    }

    row_offsets
}

#[inline]
fn vertical_scalar_range_wrapping(
    ref_plane: &[u16],
    dis_plane: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    shift_mu: u32,
    round_mu: u32,
    start: usize,
    end: usize,
    tmp_mu1: &mut [u16],
    tmp_mu2: &mut [u16],
    tmp_ref_sq: &mut [u32],
    tmp_dis_sq: &mut [u32],
    tmp_ref_dis: &mut [u32],
) {
    for j in start..end {
        let mut acc_mu1 = 0u32;
        let mut acc_mu2 = 0u32;
        let mut acc_rsq = 0u32;
        let mut acc_dsq = 0u32;
        let mut acc_rdi = 0u32;

        for (tap, &coeff) in coeffs.iter().enumerate() {
            let idx = row_offsets[tap] + j;
            let c = coeff as u32;
            let r = ref_plane[idx] as u32;
            let d = dis_plane[idx] as u32;
            let cr = c * r;
            let cd = c * d;
            acc_mu1 += cr;
            acc_mu2 += cd;
            acc_rsq = acc_rsq.wrapping_add(cr * r);
            acc_dsq = acc_dsq.wrapping_add(cd * d);
            acc_rdi = acc_rdi.wrapping_add(cr * d);
        }

        tmp_mu1[j] = ((acc_mu1 + round_mu) >> shift_mu) as u16;
        tmp_mu2[j] = ((acc_mu2 + round_mu) >> shift_mu) as u16;
        tmp_ref_sq[j] = acc_rsq;
        tmp_dis_sq[j] = acc_dsq;
        tmp_ref_dis[j] = acc_rdi;
    }
}

#[inline]
fn vertical_scalar_range_non_wrapping(
    ref_plane: &[u16],
    dis_plane: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    shift_mu: u32,
    round_mu: u32,
    sq_shift: u32,
    sq_round64: u64,
    start: usize,
    end: usize,
    tmp_mu1: &mut [u16],
    tmp_mu2: &mut [u16],
    tmp_ref_sq: &mut [u32],
    tmp_dis_sq: &mut [u32],
    tmp_ref_dis: &mut [u32],
) {
    for j in start..end {
        let mut acc_mu1 = 0u32;
        let mut acc_mu2 = 0u32;
        let mut acc_rsq = 0u64;
        let mut acc_dsq = 0u64;
        let mut acc_rdi = 0u64;

        for (tap, &coeff) in coeffs.iter().enumerate() {
            let idx = row_offsets[tap] + j;
            let c = coeff as u32;
            let r = ref_plane[idx] as u32;
            let d = dis_plane[idx] as u32;
            let cr = c as u64 * r as u64;
            let cd = c as u64 * d as u64;
            acc_mu1 += c * r;
            acc_mu2 += c * d;
            acc_rsq += cr * r as u64;
            acc_dsq += cd * d as u64;
            acc_rdi += cr * d as u64;
        }

        tmp_mu1[j] = ((acc_mu1 + round_mu) >> shift_mu) as u16;
        tmp_mu2[j] = ((acc_mu2 + round_mu) >> shift_mu) as u16;
        tmp_ref_sq[j] = ((acc_rsq + sq_round64) >> sq_shift) as u32;
        tmp_dis_sq[j] = ((acc_dsq + sq_round64) >> sq_shift) as u32;
        tmp_ref_dis[j] = ((acc_rdi + sq_round64) >> sq_shift) as u32;
    }
}

#[inline]
fn horizontal_scalar_range(
    tmp_mu1: &[u16],
    tmp_mu2: &[u16],
    tmp_ref_sq: &[u32],
    tmp_dis_sq: &[u32],
    tmp_ref_dis: &[u32],
    coeffs: &[u16],
    half: usize,
    start: usize,
    end: usize,
    vif_enhn_gain_limit: f64,
    accum: &mut RunningStatAccumulators,
) {
    let width = tmp_mu1.len();

    for j in start..end {
        let mut acc_mu1 = 0u32;
        let mut acc_mu2 = 0u32;
        let mut acc_ref = 0u64;
        let mut acc_dis = 0u64;
        let mut acc_rdi = 0u64;

        for (tap, &coeff) in coeffs.iter().enumerate() {
            let jj = reflect_index(j as i32 - half as i32 + tap as i32, width as i32);
            let c = coeff as u32;
            acc_mu1 += c * tmp_mu1[jj] as u32;
            acc_mu2 += c * tmp_mu2[jj] as u32;
            acc_ref += c as u64 * tmp_ref_sq[jj] as u64;
            acc_dis += c as u64 * tmp_dis_sq[jj] as u64;
            acc_rdi += c as u64 * tmp_ref_dis[jj] as u64;
        }

        process_filtered_pixel(
            acc_mu1,
            acc_mu2,
            acc_ref,
            acc_dis,
            acc_rdi,
            vif_enhn_gain_limit,
            accum,
        );
    }
}

#[inline]
fn horizontal_simd_body_range(width: usize, half: usize, lanes: usize) -> (usize, usize) {
    let start = half.min(width);
    let interior_end = width.saturating_sub(half);

    if interior_end <= start {
        return (start, start);
    }

    let simd_end = start + ((interior_end - start) / lanes) * lanes;
    (start, simd_end)
}

#[inline]
fn process_filtered_pixel(
    acc_mu1: u32,
    acc_mu2: u32,
    acc_ref: u64,
    acc_dis: u64,
    acc_rdi: u64,
    vif_enhn_gain_limit: f64,
    accum: &mut RunningStatAccumulators,
) {
    let mu1_sq = ((acc_mu1 as u64 * acc_mu1 as u64 + 2147483648) >> 32) as u32;
    let mu2_sq = ((acc_mu2 as u64 * acc_mu2 as u64 + 2147483648) >> 32) as u32;
    let mu1_mu2 = ((acc_mu1 as u64 * acc_mu2 as u64 + 2147483648) >> 32) as u32;

    let ref_filt = ((acc_ref + 32768) >> 16) as u32;
    let dis_filt = ((acc_dis + 32768) >> 16) as u32;
    let rdi_filt = ((acc_rdi + 32768) >> 16) as u32;

    let sigma1_sq = ref_filt as i64 - mu1_sq as i64;
    let sigma2_sq = (dis_filt as i64 - mu2_sq as i64).max(0);
    let sigma12 = rdi_filt as i64 - mu1_mu2 as i64;

    if sigma1_sq >= SIGMA_NSQ as i64 {
        let sigma1_u32 = sigma1_sq.min(u32::MAX as i64) as u32;
        accum.accum_den_log +=
            log2_32(&LOG2_TABLE, SIGMA_NSQ.saturating_add(sigma1_u32)) as i64 - 2048 * 17;

        if sigma12 > 0 && sigma2_sq > 0 {
            let g = sigma12 as f64 / (sigma1_sq as f64 + EPSILON);
            let sv_sq = (sigma2_sq - (g * sigma12 as f64) as i64).max(0);
            let g = g.min(vif_enhn_gain_limit);

            let sv_u32 = sv_sq.min(u32::MAX as i64) as u32;
            let numer1 = sv_u32.saturating_add(SIGMA_NSQ);
            let numer1_tmp = (g * g * sigma1_sq as f64) as i64 + numer1 as i64;
            let numer1_tmp = numer1_tmp.max(numer1 as i64) as u64;

            accum.accum_num_log += log2_64(&LOG2_TABLE, numer1_tmp) as i64
                - log2_64(&LOG2_TABLE, numer1 as u64) as i64;
        }
    } else {
        accum.accum_num_non_log += sigma2_sq;
        accum.accum_den_non_log += 1;
    }
}

#[inline]
fn finalize_scale_stat(accum: RunningStatAccumulators) -> ScaleStat {
    let non_log_penalty = accum.accum_num_non_log as f64 / 16384.0 / 65025.0;
    let num =
        accum.accum_num_log as f64 / 2048.0 + (accum.accum_den_non_log as f64 - non_log_penalty);
    let den = accum.accum_den_log as f64 / 2048.0 + accum.accum_den_non_log as f64;

    ScaleStat { num, den }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmaf_cpu::SimdBackend;

    fn patterned_plane(width: usize, height: usize, modulus: u16, bias: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    ((x * 13 + y * 19 + (x ^ y) * 11 + x * y * 3 + bias) % modulus as usize) as u16
                })
            })
            .collect()
    }

    #[test]
    fn workspace_path_matches_owned_wrapper() {
        let width = 27;
        let height = 17;
        let reference = patterned_plane(width, height, 1024, 5);
        let distorted = patterned_plane(width, height, 1024, 23);
        let expected = vif_statistic(
            &reference,
            &distorted,
            width,
            height,
            10,
            0,
            100.0,
            SimdBackend::Scalar,
        );

        let mut workspace = VifStatWorkspace::new(width);
        let actual = vif_statistic_with_workspace(
            &reference,
            &distorted,
            width,
            height,
            10,
            0,
            100.0,
            &mut workspace,
            SimdBackend::Scalar,
        );
        assert_eq!(expected.num.to_bits(), actual.num.to_bits());
        assert_eq!(expected.den.to_bits(), actual.den.to_bits());

        let repeated = vif_statistic_with_workspace(
            &reference,
            &distorted,
            width,
            height,
            10,
            0,
            100.0,
            &mut workspace,
            SimdBackend::Scalar,
        );
        assert_eq!(expected.num.to_bits(), repeated.num.to_bits());
        assert_eq!(expected.den.to_bits(), repeated.den.to_bits());
    }
}
