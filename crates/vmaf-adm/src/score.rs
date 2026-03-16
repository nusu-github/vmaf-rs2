//! Border exclusion and integer score accumulation — libvmaf `integer_adm` compatibility.
//!
//! This module matches the fixed-point CSF/CM logic and the shifted cube accumulators
//! from libvmaf `integer_adm.c`.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use crate::noise_floor;

const ADM_BORDER_FACTOR: f64 = 0.1;

const ONE_BY_15: i32 = 8738;
const I4_ONE_BY_15: i64 = 286_331_153;

#[inline]
fn ceil_log2_u32(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// Score accumulation border — spec §4.3.7.
///
/// Returns `(left, top, right, bottom)` as exclusive bounds.
pub(crate) fn accum_border(w: usize, h: usize) -> (usize, usize, usize, usize) {
    let l = (w as f64 * ADM_BORDER_FACTOR - 0.5).floor() as i64;
    let t = (h as f64 * ADM_BORDER_FACTOR - 0.5).floor() as i64;
    let left = l.max(0) as usize;
    let top = t.max(0) as usize;
    let right = w - left;
    let bottom = h - top;
    (left, top, right, bottom)
}

fn csf_scale0(
    art_h: &[i16],
    art_v: &[i16],
    art_d: &[i16],
) -> (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>) {
    // Default-viewing-distance fixed-point factors from libvmaf.
    let i_rfactor: [i32; 3] = [36453, 36453, 49417];
    let i_shifts: [i32; 3] = [15, 15, 17];
    let i_shiftsadd: [i32; 3] = [16384, 16384, 65535];
    let fix_one_by_30: i32 = 4369; // (1/30) * 2^17

    let n = art_h.len();
    debug_assert_eq!(art_v.len(), n);
    debug_assert_eq!(art_d.len(), n);

    let mut csf_a_h = vec![0i16; n];
    let mut csf_a_v = vec![0i16; n];
    let mut csf_a_d = vec![0i16; n];
    let mut csf_f_h = vec![0i16; n];
    let mut csf_f_v = vec![0i16; n];
    let mut csf_f_d = vec![0i16; n];

    for k in 0..n {
        let srcs = [art_h[k] as i32, art_v[k] as i32, art_d[k] as i32];
        let mut dsts = [0i16; 3];
        let mut flts = [0i16; 3];
        for theta in 0..3 {
            let dst_val = i_rfactor[theta] * srcs[theta];
            let i16_dst = ((dst_val + i_shiftsadd[theta]) >> i_shifts[theta]) as i16;
            dsts[theta] = i16_dst;
            let flt = (((fix_one_by_30 * (i16_dst as i32).abs()) + 2048) >> 12) as i16;
            flts[theta] = flt;
        }
        csf_a_h[k] = dsts[0];
        csf_a_v[k] = dsts[1];
        csf_a_d[k] = dsts[2];
        csf_f_h[k] = flts[0];
        csf_f_v[k] = flts[1];
        csf_f_d[k] = flts[2];
    }

    (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d)
}

fn csf_s123(
    art_h: &[i32],
    art_v: &[i32],
    art_d: &[i32],
    scale: usize,
    rfactor: [f32; 3],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = art_h.len();
    debug_assert_eq!(art_v.len(), n);
    debug_assert_eq!(art_d.len(), n);

    // Fixed-point rfactor (Q32) as in libvmaf.
    let pow2_32 = 4294967296.0f64;
    let i_rfactor: [u32; 3] = [
        (rfactor[0] as f64 * pow2_32) as u32,
        (rfactor[1] as f64 * pow2_32) as u32,
        (rfactor[2] as f64 * pow2_32) as u32,
    ];

    const FIX_ONE_BY_30: i64 = 143_165_577;
    const SHIFT_DST: u32 = 28;
    const SHIFT_FLT: u32 = 32;

    let add_bef_shift_dst: i64 = 1i64 << (SHIFT_DST - 1);
    let add_bef_shift_flt: i64 = 1i64 << (SHIFT_FLT - 1);

    let mut csf_a_h = vec![0i32; n];
    let mut csf_a_v = vec![0i32; n];
    let mut csf_a_d = vec![0i32; n];
    let mut csf_f_h = vec![0i32; n];
    let mut csf_f_v = vec![0i32; n];
    let mut csf_f_d = vec![0i32; n];

    // `scale` is 1..=3; arrays in libvmaf index by (scale-1).
    let _ = scale; // same SHIFT_* for all scales.

    for k in 0..n {
        let srcs = [art_h[k] as i64, art_v[k] as i64, art_d[k] as i64];
        let mut dsts = [0i32; 3];
        let mut flts = [0i32; 3];
        for theta in 0..3 {
            let dst_val =
                (((i_rfactor[theta] as i64) * srcs[theta]) + add_bef_shift_dst) >> SHIFT_DST;
            dsts[theta] = dst_val as i32;
            let flt = (((FIX_ONE_BY_30 * (dst_val as i32).abs() as i64) + add_bef_shift_flt)
                >> SHIFT_FLT) as i32;
            flts[theta] = flt;
        }
        csf_a_h[k] = dsts[0];
        csf_a_v[k] = dsts[1];
        csf_a_d[k] = dsts[2];
        csf_f_h[k] = flts[0];
        csf_f_v[k] = flts[1];
        csf_f_d[k] = flts[2];
    }

    (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d)
}

fn adm_csf_den_scale0(
    ref_h: &[i16],
    ref_v: &[i16],
    ref_d: &[i16],
    w: usize,
    h: usize,
    rfactor: [f32; 3],
) -> f32 {
    let (left, top, right, bottom) = accum_border(w, h);
    let area = (bottom - top) * (right - left);

    let shift_accum_i32 = (ceil_log2_u32(area as u32) as i32 - 20).max(0);
    let shift_accum = shift_accum_i32 as u32;
    let add_shift_accum: u64 = if shift_accum > 0 {
        1u64 << (shift_accum - 1)
    } else {
        0
    };

    let mut accum_h: u64 = 0;
    let mut accum_v: u64 = 0;
    let mut accum_d: u64 = 0;

    for i in top..bottom {
        let mut inner_h: u64 = 0;
        let mut inner_v: u64 = 0;
        let mut inner_d: u64 = 0;
        let row = i * w;
        for j in left..right {
            let idx = row + j;
            let hh = (ref_h[idx] as i32).unsigned_abs() as u64;
            let vv = (ref_v[idx] as i32).unsigned_abs() as u64;
            let dd = (ref_d[idx] as i32).unsigned_abs() as u64;
            inner_h += hh * hh * hh;
            inner_v += vv * vv * vv;
            inner_d += dd * dd * dd;
        }
        accum_h += (inner_h + add_shift_accum) >> shift_accum;
        accum_v += (inner_v + add_shift_accum) >> shift_accum;
        accum_d += (inner_d + add_shift_accum) >> shift_accum;
    }

    let shift_csf = 2.0f64.powi((18 - shift_accum_i32) as i32);
    let rf0 = rfactor[0] as f64;
    let rf1 = rfactor[1] as f64;
    let rf2 = rfactor[2] as f64;
    let csf_h = (accum_h as f64 / shift_csf) * rf0.powi(3);
    let csf_v = (accum_v as f64 / shift_csf) * rf1.powi(3);
    let csf_d = (accum_d as f64 / shift_csf) * rf2.powi(3);

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    (csf_h as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_v as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_d as f32).powf(1.0 / 3.0)
        + powf_add
}

fn adm_csf_den_s123(
    ref_h: &[i32],
    ref_v: &[i32],
    ref_d: &[i32],
    scale: usize,
    w: usize,
    h: usize,
    rfactor: [f32; 3],
) -> f32 {
    let (left, top, right, bottom) = accum_border(w, h);
    let area_w = right - left;
    let area_h = bottom - top;
    let area = area_w * area_h;

    const SHIFT_SQ: [u32; 3] = [31, 30, 31];
    const ACCUM_CONVERT_FLOAT: [i32; 3] = [32, 27, 23];

    let shift_sq = SHIFT_SQ[scale - 1];
    let add_shift_sq: u64 = 1u64 << shift_sq;

    let shift_cub = ceil_log2_u32(area_w as u32);
    let add_shift_cub: u64 = if shift_cub > 0 {
        1u64 << (shift_cub - 1)
    } else {
        0
    };
    let shift_accum = ceil_log2_u32(area_h as u32);
    let add_shift_accum: u64 = if shift_accum > 0 {
        1u64 << (shift_accum - 1)
    } else {
        0
    };

    let mut accum_h: u64 = 0;
    let mut accum_v: u64 = 0;
    let mut accum_d: u64 = 0;

    for i in top..bottom {
        let mut inner_h: u64 = 0;
        let mut inner_v: u64 = 0;
        let mut inner_d: u64 = 0;
        let row = i * w;
        for j in left..right {
            let idx = row + j;
            let hh = (ref_h[idx] as i64).unsigned_abs();
            let vv = (ref_v[idx] as i64).unsigned_abs();
            let dd = (ref_d[idx] as i64).unsigned_abs();

            let h2 = ((hh * hh + add_shift_sq) >> shift_sq) * hh;
            inner_h += (h2 + add_shift_cub) >> shift_cub;
            let v2 = ((vv * vv + add_shift_sq) >> shift_sq) * vv;
            inner_v += (v2 + add_shift_cub) >> shift_cub;
            let d2 = ((dd * dd + add_shift_sq) >> shift_sq) * dd;
            inner_d += (d2 + add_shift_cub) >> shift_cub;
        }
        accum_h += (inner_h + add_shift_accum) >> shift_accum;
        accum_v += (inner_v + add_shift_accum) >> shift_accum;
        accum_d += (inner_d + add_shift_accum) >> shift_accum;
    }

    let shift_csf_exp = ACCUM_CONVERT_FLOAT[scale - 1] - shift_accum as i32 - shift_cub as i32;
    let shift_csf = 2.0f64.powi(shift_csf_exp);
    let rf0 = rfactor[0] as f64;
    let rf1 = rfactor[1] as f64;
    let rf2 = rfactor[2] as f64;
    let csf_h = (accum_h as f64 / shift_csf) * rf0.powi(3);
    let csf_v = (accum_v as f64 / shift_csf) * rf1.powi(3);
    let csf_d = (accum_d as f64 / shift_csf) * rf2.powi(3);

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    (csf_h as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_v as f32).powf(1.0 / 3.0)
        + powf_add
        + (csf_d as f32).powf(1.0 / 3.0)
        + powf_add
}

fn adm_cm_scale0(
    rst_h: &[i16],
    rst_v: &[i16],
    rst_d: &[i16],
    csf_a_h: &[i16],
    csf_a_v: &[i16],
    csf_a_d: &[i16],
    csf_f_h: &[i16],
    csf_f_v: &[i16],
    csf_f_d: &[i16],
    w: usize,
    h: usize,
) -> f32 {
    let (left, top, right, bottom) = accum_border(w, h);

    let start_col = if left > 1 { left } else { 1 };
    let end_col = if right < (w - 1) { right } else { w - 1 };
    let start_row = if top > 1 { top } else { 1 };
    let end_row = if bottom < (h - 1) { bottom } else { h - 1 };

    let shift_xhsq: i32 = 29;
    let shift_xvsq: i32 = 29;
    let shift_xdsq: i32 = 30;
    let add_shift_xhsq: i64 = 268_435_456; // 2^28
    let add_shift_xvsq: i64 = 268_435_456;
    let add_shift_xdsq: i64 = 536_870_912; // 2^29

    let shift_xhcub = ceil_log2_u32(w as u32).saturating_sub(4);
    let shift_xvcub = ceil_log2_u32(w as u32).saturating_sub(4);
    let shift_xdcub = ceil_log2_u32(w as u32).saturating_sub(3);

    let add_shift_xhcub: i64 = if shift_xhcub > 0 {
        1i64 << (shift_xhcub - 1)
    } else {
        0
    };
    let add_shift_xvcub: i64 = if shift_xvcub > 0 {
        1i64 << (shift_xvcub - 1)
    } else {
        0
    };
    let add_shift_xdcub: i64 = if shift_xdcub > 0 {
        1i64 << (shift_xdcub - 1)
    } else {
        0
    };

    let shift_inner_accum = ceil_log2_u32(h as u32);
    let add_shift_inner_accum: i64 = if shift_inner_accum > 0 {
        1i64 << (shift_inner_accum - 1)
    } else {
        0
    };

    let shift_xhsub: i32 = 10;
    let shift_xvsub: i32 = 10;
    let shift_xdsub: i32 = 12;

    // Default-viewing-distance fixed-point factors from libvmaf.
    let i_rfactor: [i32; 3] = [36453, 36453, 49417];

    let mut accum_h: i64 = 0;
    let mut accum_v: i64 = 0;
    let mut accum_d: i64 = 0;

    for i in start_row..end_row {
        let mut inner_h: i64 = 0;
        let mut inner_v: i64 = 0;
        let mut inner_d: i64 = 0;

        for j in start_col..end_col {
            let idx = i * w + j;

            let thr = {
                let mut thr_total: i32 = 0;
                for (csf_a, csf_f) in [(csf_a_h, csf_f_h), (csf_a_v, csf_f_v), (csf_a_d, csf_f_d)] {
                    let mut sum: i32 = 0;
                    let r0 = (i - 1) * w;
                    let r1 = i * w;
                    let r2 = (i + 1) * w;

                    sum += csf_f[r0 + j - 1] as i32;
                    sum += csf_f[r0 + j] as i32;
                    sum += csf_f[r0 + j + 1] as i32;

                    sum += csf_f[r1 + j - 1] as i32;
                    sum += ((ONE_BY_15 * (csf_a[r1 + j] as i32).abs()) + 2048) >> 12;
                    sum += csf_f[r1 + j + 1] as i32;

                    sum += csf_f[r2 + j - 1] as i32;
                    sum += csf_f[r2 + j] as i32;
                    sum += csf_f[r2 + j + 1] as i32;

                    thr_total += sum;
                }
                thr_total
            };

            let xh = (rst_h[idx] as i32) * i_rfactor[0];
            let xv = (rst_v[idx] as i32) * i_rfactor[1];
            let xd = (rst_d[idx] as i32) * i_rfactor[2];

            // ADM_CM_ACCUM_ROUND for H
            {
                let mut x = xh.abs() - (thr << shift_xhsub);
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_xhsq) >> shift_xhsq) as i32;
                let val = (((x_sq as i64 * x as i64) + add_shift_xhcub) >> shift_xhcub) as i64;
                inner_h += val;
            }
            // V
            {
                let mut x = xv.abs() - (thr << shift_xvsub);
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_xvsq) >> shift_xvsq) as i32;
                let val = (((x_sq as i64 * x as i64) + add_shift_xvcub) >> shift_xvcub) as i64;
                inner_v += val;
            }
            // D
            {
                let mut x = xd.abs() - (thr << shift_xdsub);
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_xdsq) >> shift_xdsq) as i32;
                let val = (((x_sq as i64 * x as i64) + add_shift_xdcub) >> shift_xdcub) as i64;
                inner_d += val;
            }
        }

        accum_h += (inner_h + add_shift_inner_accum) >> shift_inner_accum;
        accum_v += (inner_v + add_shift_inner_accum) >> shift_inner_accum;
        accum_d += (inner_d + add_shift_inner_accum) >> shift_inner_accum;
    }

    let area = (bottom - top) * (right - left);

    let f_accum_h = (accum_h as f64
        / 2.0f64.powi((52 - shift_xhcub as i32 - shift_inner_accum as i32) as i32))
        as f32;
    let f_accum_v = (accum_v as f64
        / 2.0f64.powi((52 - shift_xvcub as i32 - shift_inner_accum as i32) as i32))
        as f32;
    let f_accum_d = (accum_d as f64
        / 2.0f64.powi((57 - shift_xdcub as i32 - shift_inner_accum as i32) as i32))
        as f32;

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    f_accum_h.powf(1.0 / 3.0)
        + powf_add
        + f_accum_v.powf(1.0 / 3.0)
        + powf_add
        + f_accum_d.powf(1.0 / 3.0)
        + powf_add
}

fn adm_cm_s123(
    rst_h: &[i32],
    rst_v: &[i32],
    rst_d: &[i32],
    csf_a_h: &[i32],
    csf_a_v: &[i32],
    csf_a_d: &[i32],
    csf_f_h: &[i32],
    csf_f_v: &[i32],
    csf_f_d: &[i32],
    scale: usize,
    w: usize,
    h: usize,
    rfactor: [f32; 3],
) -> f32 {
    let (left, top, right, bottom) = accum_border(w, h);

    let start_col = if left > 1 { left } else { 1 };
    let end_col = if right < (w - 1) { right } else { w - 1 };
    let start_row = if top > 1 { top } else { 1 };
    let end_row = if bottom < (h - 1) { bottom } else { h - 1 };

    let pow2_32 = 4294967296.0f64;
    let rfactor_fp: [u32; 3] = [
        (rfactor[0] as f64 * pow2_32) as u32,
        (rfactor[1] as f64 * pow2_32) as u32,
        (rfactor[2] as f64 * pow2_32) as u32,
    ];

    const SHIFT_DST: u32 = 28;
    const SHIFT_FLT: u32 = 32;

    let add_bef_shift_dst: i64 = 1i64 << (SHIFT_DST - 1);
    let add_bef_shift_flt: i64 = 1i64 << (SHIFT_FLT - 1);

    let shift_cub = ceil_log2_u32(w as u32);
    let add_shift_cub: i64 = if shift_cub > 0 {
        1i64 << (shift_cub - 1)
    } else {
        0
    };

    let shift_inner_accum = ceil_log2_u32(h as u32);
    let add_shift_inner_accum: i64 = if shift_inner_accum > 0 {
        1i64 << (shift_inner_accum - 1)
    } else {
        0
    };

    let base_exp: i32 = match scale {
        1 => 45,
        2 => 39,
        3 => 36,
        _ => unreachable!(),
    };
    let final_shift: f32 =
        (2.0f64).powi(base_exp - shift_cub as i32 - shift_inner_accum as i32) as f32;

    let shift_sq: i32 = 30;
    let add_shift_sq: i64 = 536_870_912; // 2^29

    let mut accum_h: i64 = 0;
    let mut accum_v: i64 = 0;
    let mut accum_d: i64 = 0;

    for i in start_row..end_row {
        let mut inner_h: i64 = 0;
        let mut inner_v: i64 = 0;
        let mut inner_d: i64 = 0;

        for j in start_col..end_col {
            let idx = i * w + j;

            let xh = (((rst_h[idx] as i64) * (rfactor_fp[0] as i64) + add_bef_shift_dst)
                >> SHIFT_DST) as i32;
            let xv = (((rst_v[idx] as i64) * (rfactor_fp[1] as i64) + add_bef_shift_dst)
                >> SHIFT_DST) as i32;
            let xd = (((rst_d[idx] as i64) * (rfactor_fp[2] as i64) + add_bef_shift_dst)
                >> SHIFT_DST) as i32;

            let thr = {
                let mut thr_total: i32 = 0;
                for (csf_a, csf_f) in [(csf_a_h, csf_f_h), (csf_a_v, csf_f_v), (csf_a_d, csf_f_d)] {
                    let mut sum: i32 = 0;
                    let r0 = (i - 1) * w;
                    let r1 = i * w;
                    let r2 = (i + 1) * w;

                    sum += csf_f[r0 + j - 1];
                    sum += csf_f[r0 + j];
                    sum += csf_f[r0 + j + 1];

                    sum += csf_f[r1 + j - 1];
                    sum += (((I4_ONE_BY_15 * (csf_a[r1 + j] as i64).abs()) + add_bef_shift_flt)
                        >> SHIFT_FLT) as i32;
                    sum += csf_f[r1 + j + 1];

                    sum += csf_f[r2 + j - 1];
                    sum += csf_f[r2 + j];
                    sum += csf_f[r2 + j + 1];

                    thr_total += sum;
                }
                thr_total
            };

            // I4_ADM_CM_ACCUM_ROUND (shift_sub = 0)
            {
                let mut x = xh.abs() - thr;
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                let val = (((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub) as i64;
                inner_h += val;
            }
            {
                let mut x = xv.abs() - thr;
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                let val = (((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub) as i64;
                inner_v += val;
            }
            {
                let mut x = xd.abs() - thr;
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                let val = (((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub) as i64;
                inner_d += val;
            }
        }

        accum_h += (inner_h + add_shift_inner_accum) >> shift_inner_accum;
        accum_v += (inner_v + add_shift_inner_accum) >> shift_inner_accum;
        accum_d += (inner_d + add_shift_inner_accum) >> shift_inner_accum;
    }

    let area = (bottom - top) * (right - left);
    let f_accum_h = (accum_h as f32) / final_shift;
    let f_accum_v = (accum_v as f32) / final_shift;
    let f_accum_d = (accum_d as f32) / final_shift;

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    f_accum_h.powf(1.0 / 3.0)
        + powf_add
        + f_accum_v.powf(1.0 / 3.0)
        + powf_add
        + f_accum_d.powf(1.0 / 3.0)
        + powf_add
}

/// Score scale 0: returns `(num_scale, den_scale)`.
pub(crate) fn score_scale0(
    ref_h: &[i16],
    ref_v: &[i16],
    ref_d: &[i16],
    rst_h: &[i16],
    rst_v: &[i16],
    rst_d: &[i16],
    art_h: &[i16],
    art_v: &[i16],
    art_d: &[i16],
    width: usize,
    height: usize,
) -> (f32, f32) {
    let rfactor = noise_floor::rfactor(0);
    let (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d) = csf_scale0(art_h, art_v, art_d);
    let num = adm_cm_scale0(
        rst_h, rst_v, rst_d, &csf_a_h, &csf_a_v, &csf_a_d, &csf_f_h, &csf_f_v, &csf_f_d, width,
        height,
    );
    let den = adm_csf_den_scale0(ref_h, ref_v, ref_d, width, height, rfactor);
    (num, den)
}

/// Score scales 1–3: returns `(num_scale, den_scale)`.
pub(crate) fn score_scale_s123(
    ref_h: &[i32],
    ref_v: &[i32],
    ref_d: &[i32],
    rst_h: &[i32],
    rst_v: &[i32],
    rst_d: &[i32],
    art_h: &[i32],
    art_v: &[i32],
    art_d: &[i32],
    scale: usize,
    width: usize,
    height: usize,
) -> (f32, f32) {
    let rfactor = noise_floor::rfactor(scale);
    let (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d) =
        csf_s123(art_h, art_v, art_d, scale, rfactor);
    let num = adm_cm_s123(
        rst_h, rst_v, rst_d, &csf_a_h, &csf_a_v, &csf_a_d, &csf_f_h, &csf_f_v, &csf_f_d, scale,
        width, height, rfactor,
    );
    let den = adm_csf_den_s123(ref_h, ref_v, ref_d, scale, width, height, rfactor);
    (num, den)
}
