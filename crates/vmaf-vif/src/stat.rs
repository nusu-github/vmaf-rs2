//! VIF statistic core — spec §4.2.6–4.2.8

use crate::math::{log2_32, log2_64, reflect_index};
use crate::tables::{FILTER, FILTER_WIDTH, LOG2_TABLE};

const SIGMA_NSQ: u32 = 131072; // 2^17
const EPSILON: f64 = 65536.0 * 1e-10; // ≈ 6.5536e-6

/// Per-scale VIF numerator and denominator — spec §4.2.8.
pub(crate) struct ScaleStat {
    pub num: f64,
    pub den: f64,
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
) -> ScaleStat {
    let filt = &FILTER[scale];
    let fw = FILTER_WIDTH[scale];
    let half = fw / 2;

    // Shift/round for mu accumulators
    let (shift_mu, round_mu) = if scale == 0 {
        (bpc as u32, 1u32 << (bpc - 1))
    } else {
        (16u32, 32768u32)
    };

    // Shift/round for squared accumulators (64-bit path only)
    let (sq_shift, sq_round64): (u32, u64) = if scale == 0 && bpc == 8 {
        (0, 0) // unused — 8-bit scale-0 path uses u32 with no shift
    } else if scale == 0 {
        let sh = (bpc as u32 - 8) * 2;
        (sh, 1u64 << (sh - 1))
    } else {
        (16, 32768)
    };

    let mut accum_num_log: i64 = 0;
    let mut accum_den_log: i64 = 0;
    let mut accum_num_non_log: i64 = 0;
    let mut accum_den_non_log: i64 = 0;

    // Per-row scratch buffers (un-padded)
    let mut tmp_mu1 = vec![0u16; width];
    let mut tmp_mu2 = vec![0u16; width];
    let mut tmp_ref_sq = vec![0u32; width];
    let mut tmp_dis_sq = vec![0u32; width];
    let mut tmp_ref_dis = vec![0u32; width];

    for i in 0..height {
        // ── Step 1: Vertical pass ──────────────────────────────────────────────

        for j in 0..width {
            let mut acc_mu1 = 0u32;
            let mut acc_mu2 = 0u32;

            if bpc == 8 && scale == 0 {
                // 8-bit scale-0: uint32 squared accumulators (wrapping — intentional)
                let mut acc_rsq = 0u32;
                let mut acc_dsq = 0u32;
                let mut acc_rdi = 0u32;

                for fi in 0..fw {
                    let ii = reflect_index(i as i32 - half as i32 + fi as i32, height as i32);
                    let c = FILTER[0][fi] as u32;
                    let r = ref_plane[ii * width + j] as u32;
                    let d = dis_plane[ii * width + j] as u32;

                    let cr = c * r; // c ≤ 7784, r ≤ 255 → product ≤ 1.98M, no overflow
                    let cd = c * d;
                    acc_mu1 += cr;
                    acc_mu2 += cd;
                    acc_rsq = acc_rsq.wrapping_add(cr * r); // sum wraps — intentional
                    acc_dsq = acc_dsq.wrapping_add(cd * d);
                    acc_rdi = acc_rdi.wrapping_add(cr * d);
                }

                tmp_ref_sq[j] = acc_rsq; // sq_shift = 0
                tmp_dis_sq[j] = acc_dsq;
                tmp_ref_dis[j] = acc_rdi;
            } else {
                // uint64 squared accumulators
                let mut acc_rsq = 0u64;
                let mut acc_dsq = 0u64;
                let mut acc_rdi = 0u64;

                for fi in 0..fw {
                    let ii = reflect_index(i as i32 - half as i32 + fi as i32, height as i32);
                    let c = filt[fi] as u32;
                    let r = ref_plane[ii * width + j] as u32;
                    let d = dis_plane[ii * width + j] as u32;

                    let cr = c as u64 * r as u64;
                    let cd = c as u64 * d as u64;
                    acc_mu1 += c * r;
                    acc_mu2 += c * d;
                    acc_rsq += cr * r as u64;
                    acc_dsq += cd * d as u64;
                    acc_rdi += cr * d as u64;
                }

                tmp_ref_sq[j] = ((acc_rsq + sq_round64) >> sq_shift) as u32;
                tmp_dis_sq[j] = ((acc_dsq + sq_round64) >> sq_shift) as u32;
                tmp_ref_dis[j] = ((acc_rdi + sq_round64) >> sq_shift) as u32;
            }

            tmp_mu1[j] = ((acc_mu1 + round_mu) >> shift_mu) as u16;
            tmp_mu2[j] = ((acc_mu2 + round_mu) >> shift_mu) as u16;
        }

        // ── Step 2: Horizontal pass (uses reflect_index for boundary) ──────────

        for j in 0..width {
            let mut acc_mu1 = 0u32;
            let mut acc_mu2 = 0u32;
            let mut acc_ref = 0u64;
            let mut acc_dis = 0u64;
            let mut acc_rdi = 0u64;

            for fj in 0..fw {
                // reflect_index gives the same result as the spec's padding approach
                let jj = reflect_index(j as i32 - half as i32 + fj as i32, width as i32);
                let c = filt[fj] as u32;

                acc_mu1 += c * tmp_mu1[jj] as u32;
                acc_mu2 += c * tmp_mu2[jj] as u32;
                acc_ref += c as u64 * tmp_ref_sq[jj] as u64;
                acc_dis += c as u64 * tmp_dis_sq[jj] as u64;
                acc_rdi += c as u64 * tmp_ref_dis[jj] as u64;
            }

            let mu1_val = acc_mu1;
            let mu2_val = acc_mu2;

            // Square and cross-product of mu values in Q32
            let mu1_sq = ((mu1_val as u64 * mu1_val as u64 + 2147483648) >> 32) as u32;
            let mu2_sq = ((mu2_val as u64 * mu2_val as u64 + 2147483648) >> 32) as u32;
            let mu1_mu2 = ((mu1_val as u64 * mu2_val as u64 + 2147483648) >> 32) as u32;

            let ref_filt = ((acc_ref + 32768) >> 16) as u32;
            let dis_filt = ((acc_dis + 32768) >> 16) as u32;
            let rdi_filt = ((acc_rdi + 32768) >> 16) as u32;

            // Compute variances/covariance in a wider signed type to avoid
            // debug overflow when intermediate u32 values exceed i32::MAX.
            let sigma1_sq = ref_filt as i64 - mu1_sq as i64;
            let sigma2_sq = (dis_filt as i64 - mu2_sq as i64).max(0);
            let sigma12 = rdi_filt as i64 - mu1_mu2 as i64;

            // ── Step 3: VIF accumulator logic — spec §4.2.7 ──────────────────
            if sigma1_sq >= SIGMA_NSQ as i64 {
                let sigma1_u32 = sigma1_sq.min(u32::MAX as i64) as u32;

                accum_den_log +=
                    log2_32(&LOG2_TABLE, SIGMA_NSQ.saturating_add(sigma1_u32)) as i64 - 2048 * 17;

                if sigma12 > 0 && sigma2_sq > 0 {
                    let g = sigma12 as f64 / (sigma1_sq as f64 + EPSILON);

                    // Clamp sv_sq in integer domain (not float) — spec §4.2.7
                    let sv_sq = (sigma2_sq - (g * sigma12 as f64) as i64).max(0);

                    let g = g.min(vif_enhn_gain_limit);

                    let sv_u32 = sv_sq.min(u32::MAX as i64) as u32;
                    let numer1 = sv_u32.saturating_add(SIGMA_NSQ);

                    let numer1_tmp = (g * g * sigma1_sq as f64) as i64 + numer1 as i64;
                    let numer1_tmp = numer1_tmp.max(numer1 as i64) as u64;

                    accum_num_log += log2_64(&LOG2_TABLE, numer1_tmp) as i64
                        - log2_64(&LOG2_TABLE, numer1 as u64) as i64;
                }
            } else {
                accum_num_non_log += sigma2_sq;
                accum_den_non_log += 1;
            }
        }
    }

    // ── Step 4: Per-scale score extraction — spec §4.2.8 ──────────────────────
    // Evaluate left-to-right; do not reorder the divisions.
    let non_log_penalty = accum_num_non_log as f64 / 16384.0 / 65025.0;
    let num = accum_num_log as f64 / 2048.0 + (accum_den_non_log as f64 - non_log_penalty);
    let den = accum_den_log as f64 / 2048.0 + accum_den_non_log as f64;

    ScaleStat { num, den }
}
