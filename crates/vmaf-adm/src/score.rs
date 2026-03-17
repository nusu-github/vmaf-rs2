//! Border exclusion and integer score accumulation — libvmaf `integer_adm` compatibility.
//!
//! This module matches the fixed-point CSF/CM logic and the shifted cube accumulators
//! from libvmaf `integer_adm.c`.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::mem::MaybeUninit;

use crate::decouple::{decouple_s123, decouple_scale0};
use crate::noise_floor;

const ADM_BORDER_FACTOR: f64 = 0.1;

const ONE_BY_15: i32 = 8738;
const I4_ONE_BY_15: i64 = 286_331_153;
const SCALE0_CSF_RFACTOR: [i32; 3] = [36453, 36453, 49417];
const SCALE0_CSF_SHIFTS: [u32; 3] = [15, 15, 17];
const SCALE0_CSF_SHIFT_ADDS: [i32; 3] = [16384, 16384, 65535];
const SCALE0_FIX_ONE_BY_30: i32 = 4369;
const SCALE0_FIX_ONE_BY_30_ADD: i32 = 2048;
const S123_FIX_ONE_BY_30: i64 = 143_165_577;
const S123_SHIFT_DST: u32 = 28;
const S123_SHIFT_FLT: u32 = 32;
const S123_ADD_BEFORE_SHIFT_DST: i64 = 1i64 << (S123_SHIFT_DST - 1);
const S123_ADD_BEFORE_SHIFT_FLT: i64 = 1i64 << (S123_SHIFT_FLT - 1);

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

#[inline]
fn row<T>(plane: &[T], width: usize, row: usize) -> &[T] {
    let start = row * width;
    &plane[start..start + width]
}

#[inline]
fn scale123_rfactor_fp(rfactor: [f32; 3]) -> [u32; 3] {
    let pow2_32 = 4294967296.0f64;
    [
        (rfactor[0] as f64 * pow2_32) as u32,
        (rfactor[1] as f64 * pow2_32) as u32,
        (rfactor[2] as f64 * pow2_32) as u32,
    ]
}

#[inline]
fn prepare_row_storage<T>(vec: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    vec.clear();
    if vec.capacity() < len {
        vec.reserve(len - vec.capacity());
    }
    &mut vec.spare_capacity_mut()[..len]
}

#[inline]
fn csf_scale0_triplet(art_h: i16, art_v: i16, art_d: i16) -> ([i16; 3], [i16; 3]) {
    let srcs = [art_h as i32, art_v as i32, art_d as i32];
    let mut dsts = [0i16; 3];
    let mut flts = [0i16; 3];

    for theta in 0..3 {
        let dst_val = SCALE0_CSF_RFACTOR[theta] * srcs[theta];
        let dst = ((dst_val + SCALE0_CSF_SHIFT_ADDS[theta]) >> SCALE0_CSF_SHIFTS[theta]) as i16;
        dsts[theta] = dst;
        flts[theta] =
            (((SCALE0_FIX_ONE_BY_30 * (dst as i32).abs()) + SCALE0_FIX_ONE_BY_30_ADD) >> 12) as i16;
    }

    (dsts, flts)
}

#[inline]
fn csf_s123_triplet(
    art_h: i32,
    art_v: i32,
    art_d: i32,
    i_rfactor: [u32; 3],
) -> ([i32; 3], [i32; 3]) {
    let srcs = [art_h as i64, art_v as i64, art_d as i64];
    let mut dsts = [0i32; 3];
    let mut flts = [0i32; 3];

    for theta in 0..3 {
        let dst_val = (((i_rfactor[theta] as i64) * srcs[theta]) + S123_ADD_BEFORE_SHIFT_DST)
            >> S123_SHIFT_DST;
        let dst = dst_val as i32;
        dsts[theta] = dst;
        flts[theta] = (((S123_FIX_ONE_BY_30 * (dst as i64).abs()) + S123_ADD_BEFORE_SHIFT_FLT)
            >> S123_SHIFT_FLT) as i32;
    }

    (dsts, flts)
}

struct AdmCmScale0Row {
    scaled_h: Vec<i32>,
    scaled_v: Vec<i32>,
    scaled_d: Vec<i32>,
    csf_a_h: Vec<i16>,
    csf_a_v: Vec<i16>,
    csf_a_d: Vec<i16>,
    csf_f_h: Vec<i16>,
    csf_f_v: Vec<i16>,
    csf_f_d: Vec<i16>,
}

impl AdmCmScale0Row {
    fn with_capacity(width: usize) -> Self {
        Self {
            scaled_h: Vec::with_capacity(width),
            scaled_v: Vec::with_capacity(width),
            scaled_d: Vec::with_capacity(width),
            csf_a_h: Vec::with_capacity(width),
            csf_a_v: Vec::with_capacity(width),
            csf_a_d: Vec::with_capacity(width),
            csf_f_h: Vec::with_capacity(width),
            csf_f_v: Vec::with_capacity(width),
            csf_f_d: Vec::with_capacity(width),
        }
    }

    fn refill(
        &mut self,
        ref_h: &[i16],
        ref_v: &[i16],
        ref_d: &[i16],
        dis_h: &[i16],
        dis_v: &[i16],
        dis_d: &[i16],
        adm_enhn_gain_limit: f64,
    ) {
        let width = ref_h.len();
        debug_assert_eq!(ref_v.len(), width);
        debug_assert_eq!(ref_d.len(), width);
        debug_assert_eq!(dis_h.len(), width);
        debug_assert_eq!(dis_v.len(), width);
        debug_assert_eq!(dis_d.len(), width);

        let Self {
            scaled_h,
            scaled_v,
            scaled_d,
            csf_a_h,
            csf_a_v,
            csf_a_d,
            csf_f_h,
            csf_f_v,
            csf_f_d,
        } = self;
        let scaled_h_out = prepare_row_storage(scaled_h, width);
        let scaled_v_out = prepare_row_storage(scaled_v, width);
        let scaled_d_out = prepare_row_storage(scaled_d, width);
        let csf_a_h_out = prepare_row_storage(csf_a_h, width);
        let csf_a_v_out = prepare_row_storage(csf_a_v, width);
        let csf_a_d_out = prepare_row_storage(csf_a_d, width);
        let csf_f_h_out = prepare_row_storage(csf_f_h, width);
        let csf_f_v_out = prepare_row_storage(csf_f_v, width);
        let csf_f_d_out = prepare_row_storage(csf_f_d, width);

        for k in 0..width {
            let (rst_h, rst_v, rst_d, art_h, art_v, art_d) = decouple_scale0(
                ref_h[k],
                ref_v[k],
                ref_d[k],
                dis_h[k],
                dis_v[k],
                dis_d[k],
                adm_enhn_gain_limit,
            );
            let (csf_a, csf_f) = csf_scale0_triplet(art_h, art_v, art_d);

            scaled_h_out[k].write(rst_h as i32 * SCALE0_CSF_RFACTOR[0]);
            scaled_v_out[k].write(rst_v as i32 * SCALE0_CSF_RFACTOR[1]);
            scaled_d_out[k].write(rst_d as i32 * SCALE0_CSF_RFACTOR[2]);
            csf_a_h_out[k].write(csf_a[0]);
            csf_a_v_out[k].write(csf_a[1]);
            csf_a_d_out[k].write(csf_a[2]);
            csf_f_h_out[k].write(csf_f[0]);
            csf_f_v_out[k].write(csf_f[1]);
            csf_f_d_out[k].write(csf_f[2]);
        }

        // SAFETY: every slot in each spare-capacity slice above is written exactly once.
        unsafe {
            scaled_h.set_len(width);
            scaled_v.set_len(width);
            scaled_d.set_len(width);
            csf_a_h.set_len(width);
            csf_a_v.set_len(width);
            csf_a_d.set_len(width);
            csf_f_h.set_len(width);
            csf_f_v.set_len(width);
            csf_f_d.set_len(width);
        }
    }
}

struct AdmCmScale123Row {
    scaled_h: Vec<i32>,
    scaled_v: Vec<i32>,
    scaled_d: Vec<i32>,
    csf_a_h: Vec<i32>,
    csf_a_v: Vec<i32>,
    csf_a_d: Vec<i32>,
    csf_f_h: Vec<i32>,
    csf_f_v: Vec<i32>,
    csf_f_d: Vec<i32>,
}

impl AdmCmScale123Row {
    fn with_capacity(width: usize) -> Self {
        Self {
            scaled_h: Vec::with_capacity(width),
            scaled_v: Vec::with_capacity(width),
            scaled_d: Vec::with_capacity(width),
            csf_a_h: Vec::with_capacity(width),
            csf_a_v: Vec::with_capacity(width),
            csf_a_d: Vec::with_capacity(width),
            csf_f_h: Vec::with_capacity(width),
            csf_f_v: Vec::with_capacity(width),
            csf_f_d: Vec::with_capacity(width),
        }
    }

    fn refill(
        &mut self,
        ref_h: &[i32],
        ref_v: &[i32],
        ref_d: &[i32],
        dis_h: &[i32],
        dis_v: &[i32],
        dis_d: &[i32],
        adm_enhn_gain_limit: f64,
        i_rfactor: [u32; 3],
    ) {
        let width = ref_h.len();
        debug_assert_eq!(ref_v.len(), width);
        debug_assert_eq!(ref_d.len(), width);
        debug_assert_eq!(dis_h.len(), width);
        debug_assert_eq!(dis_v.len(), width);
        debug_assert_eq!(dis_d.len(), width);

        let Self {
            scaled_h,
            scaled_v,
            scaled_d,
            csf_a_h,
            csf_a_v,
            csf_a_d,
            csf_f_h,
            csf_f_v,
            csf_f_d,
        } = self;
        let scaled_h_out = prepare_row_storage(scaled_h, width);
        let scaled_v_out = prepare_row_storage(scaled_v, width);
        let scaled_d_out = prepare_row_storage(scaled_d, width);
        let csf_a_h_out = prepare_row_storage(csf_a_h, width);
        let csf_a_v_out = prepare_row_storage(csf_a_v, width);
        let csf_a_d_out = prepare_row_storage(csf_a_d, width);
        let csf_f_h_out = prepare_row_storage(csf_f_h, width);
        let csf_f_v_out = prepare_row_storage(csf_f_v, width);
        let csf_f_d_out = prepare_row_storage(csf_f_d, width);

        for k in 0..width {
            let (rst_h, rst_v, rst_d, art_h, art_v, art_d) = decouple_s123(
                ref_h[k],
                ref_v[k],
                ref_d[k],
                dis_h[k],
                dis_v[k],
                dis_d[k],
                adm_enhn_gain_limit,
            );
            let (csf_a, csf_f) = csf_s123_triplet(art_h, art_v, art_d, i_rfactor);

            scaled_h_out[k].write(
                (((rst_h as i64) * (i_rfactor[0] as i64) + S123_ADD_BEFORE_SHIFT_DST)
                    >> S123_SHIFT_DST) as i32,
            );
            scaled_v_out[k].write(
                (((rst_v as i64) * (i_rfactor[1] as i64) + S123_ADD_BEFORE_SHIFT_DST)
                    >> S123_SHIFT_DST) as i32,
            );
            scaled_d_out[k].write(
                (((rst_d as i64) * (i_rfactor[2] as i64) + S123_ADD_BEFORE_SHIFT_DST)
                    >> S123_SHIFT_DST) as i32,
            );
            csf_a_h_out[k].write(csf_a[0]);
            csf_a_v_out[k].write(csf_a[1]);
            csf_a_d_out[k].write(csf_a[2]);
            csf_f_h_out[k].write(csf_f[0]);
            csf_f_v_out[k].write(csf_f[1]);
            csf_f_d_out[k].write(csf_f[2]);
        }

        // SAFETY: every slot in each spare-capacity slice above is written exactly once.
        unsafe {
            scaled_h.set_len(width);
            scaled_v.set_len(width);
            scaled_d.set_len(width);
            csf_a_h.set_len(width);
            csf_a_v.set_len(width);
            csf_a_d.set_len(width);
            csf_f_h.set_len(width);
            csf_f_v.set_len(width);
            csf_f_d.set_len(width);
        }
    }
}

#[inline]
fn threshold_scale0_component(
    prev: &[i16],
    cur: &[i16],
    next: &[i16],
    center: i16,
    j: usize,
) -> i32 {
    let center = ((ONE_BY_15 * (center as i32).abs()) + SCALE0_FIX_ONE_BY_30_ADD) >> 12;
    prev[j - 1] as i32
        + prev[j] as i32
        + prev[j + 1] as i32
        + cur[j - 1] as i32
        + center
        + cur[j + 1] as i32
        + next[j - 1] as i32
        + next[j] as i32
        + next[j + 1] as i32
}

#[inline]
fn threshold_scale0(
    prev: &AdmCmScale0Row,
    cur: &AdmCmScale0Row,
    next: &AdmCmScale0Row,
    j: usize,
) -> i32 {
    threshold_scale0_component(
        &prev.csf_f_h,
        &cur.csf_f_h,
        &next.csf_f_h,
        cur.csf_a_h[j],
        j,
    ) + threshold_scale0_component(
        &prev.csf_f_v,
        &cur.csf_f_v,
        &next.csf_f_v,
        cur.csf_a_v[j],
        j,
    ) + threshold_scale0_component(
        &prev.csf_f_d,
        &cur.csf_f_d,
        &next.csf_f_d,
        cur.csf_a_d[j],
        j,
    )
}

#[inline]
fn threshold_scale123_component(
    prev: &[i32],
    cur: &[i32],
    next: &[i32],
    center: i32,
    j: usize,
) -> i32 {
    let center = (((I4_ONE_BY_15 * (center as i64).abs()) + S123_ADD_BEFORE_SHIFT_FLT)
        >> S123_SHIFT_FLT) as i32;
    prev[j - 1]
        + prev[j]
        + prev[j + 1]
        + cur[j - 1]
        + center
        + cur[j + 1]
        + next[j - 1]
        + next[j]
        + next[j + 1]
}

#[inline]
fn threshold_scale123(
    prev: &AdmCmScale123Row,
    cur: &AdmCmScale123Row,
    next: &AdmCmScale123Row,
    j: usize,
) -> i32 {
    threshold_scale123_component(
        &prev.csf_f_h,
        &cur.csf_f_h,
        &next.csf_f_h,
        cur.csf_a_h[j],
        j,
    ) + threshold_scale123_component(
        &prev.csf_f_v,
        &cur.csf_f_v,
        &next.csf_f_v,
        cur.csf_a_v[j],
        j,
    ) + threshold_scale123_component(
        &prev.csf_f_d,
        &cur.csf_f_d,
        &next.csf_f_d,
        cur.csf_a_d[j],
        j,
    )
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
    ref_h: &[i16],
    ref_v: &[i16],
    ref_d: &[i16],
    dis_h: &[i16],
    dis_v: &[i16],
    dis_d: &[i16],
    w: usize,
    h: usize,
    adm_enhn_gain_limit: f64,
) -> f32 {
    let n = w * h;
    debug_assert_eq!(ref_h.len(), n);
    debug_assert_eq!(ref_v.len(), n);
    debug_assert_eq!(ref_d.len(), n);
    debug_assert_eq!(dis_h.len(), n);
    debug_assert_eq!(dis_v.len(), n);
    debug_assert_eq!(dis_d.len(), n);

    let (left, top, right, bottom) = accum_border(w, h);
    let start_col = left.max(1);
    let end_col = right.min(w.saturating_sub(1));
    let start_row = top.max(1);
    let end_row = bottom.min(h.saturating_sub(1));

    let shift_xhsq: i32 = 29;
    let shift_xvsq: i32 = 29;
    let shift_xdsq: i32 = 30;
    let add_shift_xhsq: i64 = 268_435_456;
    let add_shift_xvsq: i64 = 268_435_456;
    let add_shift_xdsq: i64 = 536_870_912;

    let shift_xhcub = ceil_log2_u32(w as u32).saturating_sub(4);
    let shift_xvcub = ceil_log2_u32(w as u32).saturating_sub(4);
    let shift_xdcub = ceil_log2_u32(w as u32).saturating_sub(3);
    let add_shift_xhcub = if shift_xhcub > 0 {
        1i64 << (shift_xhcub - 1)
    } else {
        0
    };
    let add_shift_xvcub = if shift_xvcub > 0 {
        1i64 << (shift_xvcub - 1)
    } else {
        0
    };
    let add_shift_xdcub = if shift_xdcub > 0 {
        1i64 << (shift_xdcub - 1)
    } else {
        0
    };

    let shift_inner_accum = ceil_log2_u32(h as u32);
    let add_shift_inner_accum = if shift_inner_accum > 0 {
        1i64 << (shift_inner_accum - 1)
    } else {
        0
    };

    let shift_xhsub: i32 = 10;
    let shift_xvsub: i32 = 10;
    let shift_xdsub: i32 = 12;

    let mut accum_h: i64 = 0;
    let mut accum_v: i64 = 0;
    let mut accum_d: i64 = 0;

    if start_row < end_row && start_col < end_col {
        let mut prev = AdmCmScale0Row::with_capacity(w);
        let mut cur = AdmCmScale0Row::with_capacity(w);
        let mut next = AdmCmScale0Row::with_capacity(w);

        prev.refill(
            row(ref_h, w, start_row - 1),
            row(ref_v, w, start_row - 1),
            row(ref_d, w, start_row - 1),
            row(dis_h, w, start_row - 1),
            row(dis_v, w, start_row - 1),
            row(dis_d, w, start_row - 1),
            adm_enhn_gain_limit,
        );
        cur.refill(
            row(ref_h, w, start_row),
            row(ref_v, w, start_row),
            row(ref_d, w, start_row),
            row(dis_h, w, start_row),
            row(dis_v, w, start_row),
            row(dis_d, w, start_row),
            adm_enhn_gain_limit,
        );
        next.refill(
            row(ref_h, w, start_row + 1),
            row(ref_v, w, start_row + 1),
            row(ref_d, w, start_row + 1),
            row(dis_h, w, start_row + 1),
            row(dis_v, w, start_row + 1),
            row(dis_d, w, start_row + 1),
            adm_enhn_gain_limit,
        );

        for i in start_row..end_row {
            let mut inner_h: i64 = 0;
            let mut inner_v: i64 = 0;
            let mut inner_d: i64 = 0;

            for j in start_col..end_col {
                let thr = threshold_scale0(&prev, &cur, &next, j);

                {
                    let mut x = cur.scaled_h[j].abs() - (thr << shift_xhsub);
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_xhsq) >> shift_xhsq) as i32;
                    inner_h += ((x_sq as i64 * x as i64) + add_shift_xhcub) >> shift_xhcub;
                }
                {
                    let mut x = cur.scaled_v[j].abs() - (thr << shift_xvsub);
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_xvsq) >> shift_xvsq) as i32;
                    inner_v += ((x_sq as i64 * x as i64) + add_shift_xvcub) >> shift_xvcub;
                }
                {
                    let mut x = cur.scaled_d[j].abs() - (thr << shift_xdsub);
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_xdsq) >> shift_xdsq) as i32;
                    inner_d += ((x_sq as i64 * x as i64) + add_shift_xdcub) >> shift_xdcub;
                }
            }

            accum_h += (inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (inner_d + add_shift_inner_accum) >> shift_inner_accum;

            if i + 1 < end_row {
                std::mem::swap(&mut prev, &mut cur);
                std::mem::swap(&mut cur, &mut next);
                let fill_row = i + 2;
                next.refill(
                    row(ref_h, w, fill_row),
                    row(ref_v, w, fill_row),
                    row(ref_d, w, fill_row),
                    row(dis_h, w, fill_row),
                    row(dis_v, w, fill_row),
                    row(dis_d, w, fill_row),
                    adm_enhn_gain_limit,
                );
            }
        }
    }

    let area = (bottom - top) * (right - left);
    let f_accum_h =
        (accum_h as f64 / 2.0f64.powi(52 - shift_xhcub as i32 - shift_inner_accum as i32)) as f32;
    let f_accum_v =
        (accum_v as f64 / 2.0f64.powi(52 - shift_xvcub as i32 - shift_inner_accum as i32)) as f32;
    let f_accum_d =
        (accum_d as f64 / 2.0f64.powi(57 - shift_xdcub as i32 - shift_inner_accum as i32)) as f32;

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    f_accum_h.powf(1.0 / 3.0)
        + powf_add
        + f_accum_v.powf(1.0 / 3.0)
        + powf_add
        + f_accum_d.powf(1.0 / 3.0)
        + powf_add
}

fn adm_cm_s123(
    ref_h: &[i32],
    ref_v: &[i32],
    ref_d: &[i32],
    dis_h: &[i32],
    dis_v: &[i32],
    dis_d: &[i32],
    scale: usize,
    w: usize,
    h: usize,
    i_rfactor: [u32; 3],
    adm_enhn_gain_limit: f64,
) -> f32 {
    let n = w * h;
    debug_assert_eq!(ref_h.len(), n);
    debug_assert_eq!(ref_v.len(), n);
    debug_assert_eq!(ref_d.len(), n);
    debug_assert_eq!(dis_h.len(), n);
    debug_assert_eq!(dis_v.len(), n);
    debug_assert_eq!(dis_d.len(), n);

    let (left, top, right, bottom) = accum_border(w, h);
    let start_col = left.max(1);
    let end_col = right.min(w.saturating_sub(1));
    let start_row = top.max(1);
    let end_row = bottom.min(h.saturating_sub(1));

    let shift_cub = ceil_log2_u32(w as u32);
    let add_shift_cub = if shift_cub > 0 {
        1i64 << (shift_cub - 1)
    } else {
        0
    };

    let shift_inner_accum = ceil_log2_u32(h as u32);
    let add_shift_inner_accum = if shift_inner_accum > 0 {
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
    let final_shift = 2.0f64.powi(base_exp - shift_cub as i32 - shift_inner_accum as i32) as f32;

    let shift_sq: i32 = 30;
    let add_shift_sq: i64 = 536_870_912;

    let mut accum_h: i64 = 0;
    let mut accum_v: i64 = 0;
    let mut accum_d: i64 = 0;

    if start_row < end_row && start_col < end_col {
        let mut prev = AdmCmScale123Row::with_capacity(w);
        let mut cur = AdmCmScale123Row::with_capacity(w);
        let mut next = AdmCmScale123Row::with_capacity(w);

        prev.refill(
            row(ref_h, w, start_row - 1),
            row(ref_v, w, start_row - 1),
            row(ref_d, w, start_row - 1),
            row(dis_h, w, start_row - 1),
            row(dis_v, w, start_row - 1),
            row(dis_d, w, start_row - 1),
            adm_enhn_gain_limit,
            i_rfactor,
        );
        cur.refill(
            row(ref_h, w, start_row),
            row(ref_v, w, start_row),
            row(ref_d, w, start_row),
            row(dis_h, w, start_row),
            row(dis_v, w, start_row),
            row(dis_d, w, start_row),
            adm_enhn_gain_limit,
            i_rfactor,
        );
        next.refill(
            row(ref_h, w, start_row + 1),
            row(ref_v, w, start_row + 1),
            row(ref_d, w, start_row + 1),
            row(dis_h, w, start_row + 1),
            row(dis_v, w, start_row + 1),
            row(dis_d, w, start_row + 1),
            adm_enhn_gain_limit,
            i_rfactor,
        );

        for i in start_row..end_row {
            let mut inner_h: i64 = 0;
            let mut inner_v: i64 = 0;
            let mut inner_d: i64 = 0;

            for j in start_col..end_col {
                let thr = threshold_scale123(&prev, &cur, &next, j);

                {
                    let mut x = cur.scaled_h[j].abs() - thr;
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                    inner_h += ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
                }
                {
                    let mut x = cur.scaled_v[j].abs() - thr;
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                    inner_v += ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
                }
                {
                    let mut x = cur.scaled_d[j].abs() - thr;
                    if x < 0 {
                        x = 0;
                    }
                    let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                    inner_d += ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
                }
            }

            accum_h += (inner_h + add_shift_inner_accum) >> shift_inner_accum;
            accum_v += (inner_v + add_shift_inner_accum) >> shift_inner_accum;
            accum_d += (inner_d + add_shift_inner_accum) >> shift_inner_accum;

            if i + 1 < end_row {
                std::mem::swap(&mut prev, &mut cur);
                std::mem::swap(&mut cur, &mut next);
                let fill_row = i + 2;
                next.refill(
                    row(ref_h, w, fill_row),
                    row(ref_v, w, fill_row),
                    row(ref_d, w, fill_row),
                    row(dis_h, w, fill_row),
                    row(dis_v, w, fill_row),
                    row(dis_d, w, fill_row),
                    adm_enhn_gain_limit,
                    i_rfactor,
                );
            }
        }
    }

    let area = (bottom - top) * (right - left);
    let f_accum_h = accum_h as f32 / final_shift;
    let f_accum_v = accum_v as f32 / final_shift;
    let f_accum_d = accum_d as f32 / final_shift;

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    f_accum_h.powf(1.0 / 3.0)
        + powf_add
        + f_accum_v.powf(1.0 / 3.0)
        + powf_add
        + f_accum_d.powf(1.0 / 3.0)
        + powf_add
}

#[cfg(test)]
fn csf_scale0_reference(
    art_h: &[i16],
    art_v: &[i16],
    art_d: &[i16],
) -> (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>) {
    let n = art_h.len();
    debug_assert_eq!(art_v.len(), n);
    debug_assert_eq!(art_d.len(), n);

    let mut csf_a_h = Vec::with_capacity(n);
    let mut csf_a_v = Vec::with_capacity(n);
    let mut csf_a_d = Vec::with_capacity(n);
    let mut csf_f_h = Vec::with_capacity(n);
    let mut csf_f_v = Vec::with_capacity(n);
    let mut csf_f_d = Vec::with_capacity(n);

    for k in 0..n {
        let (csf_a, csf_f) = csf_scale0_triplet(art_h[k], art_v[k], art_d[k]);
        csf_a_h.push(csf_a[0]);
        csf_a_v.push(csf_a[1]);
        csf_a_d.push(csf_a[2]);
        csf_f_h.push(csf_f[0]);
        csf_f_v.push(csf_f[1]);
        csf_f_d.push(csf_f[2]);
    }

    (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d)
}

#[cfg(test)]
fn csf_s123_reference(
    art_h: &[i32],
    art_v: &[i32],
    art_d: &[i32],
    i_rfactor: [u32; 3],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = art_h.len();
    debug_assert_eq!(art_v.len(), n);
    debug_assert_eq!(art_d.len(), n);

    let mut csf_a_h = Vec::with_capacity(n);
    let mut csf_a_v = Vec::with_capacity(n);
    let mut csf_a_d = Vec::with_capacity(n);
    let mut csf_f_h = Vec::with_capacity(n);
    let mut csf_f_v = Vec::with_capacity(n);
    let mut csf_f_d = Vec::with_capacity(n);

    for k in 0..n {
        let (csf_a, csf_f) = csf_s123_triplet(art_h[k], art_v[k], art_d[k], i_rfactor);
        csf_a_h.push(csf_a[0]);
        csf_a_v.push(csf_a[1]);
        csf_a_d.push(csf_a[2]);
        csf_f_h.push(csf_f[0]);
        csf_f_v.push(csf_f[1]);
        csf_f_d.push(csf_f[2]);
    }

    (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d)
}

#[cfg(test)]
fn adm_cm_scale0_reference(
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
    let start_col = left.max(1);
    let end_col = right.min(w.saturating_sub(1));
    let start_row = top.max(1);
    let end_row = bottom.min(h.saturating_sub(1));

    let shift_xhsq: i32 = 29;
    let shift_xvsq: i32 = 29;
    let shift_xdsq: i32 = 30;
    let add_shift_xhsq: i64 = 268_435_456;
    let add_shift_xvsq: i64 = 268_435_456;
    let add_shift_xdsq: i64 = 536_870_912;

    let shift_xhcub = ceil_log2_u32(w as u32).saturating_sub(4);
    let shift_xvcub = ceil_log2_u32(w as u32).saturating_sub(4);
    let shift_xdcub = ceil_log2_u32(w as u32).saturating_sub(3);
    let add_shift_xhcub = if shift_xhcub > 0 {
        1i64 << (shift_xhcub - 1)
    } else {
        0
    };
    let add_shift_xvcub = if shift_xvcub > 0 {
        1i64 << (shift_xvcub - 1)
    } else {
        0
    };
    let add_shift_xdcub = if shift_xdcub > 0 {
        1i64 << (shift_xdcub - 1)
    } else {
        0
    };

    let shift_inner_accum = ceil_log2_u32(h as u32);
    let add_shift_inner_accum = if shift_inner_accum > 0 {
        1i64 << (shift_inner_accum - 1)
    } else {
        0
    };

    let shift_xhsub: i32 = 10;
    let shift_xvsub: i32 = 10;
    let shift_xdsub: i32 = 12;

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
                    sum += ((ONE_BY_15 * (csf_a[r1 + j] as i32).abs()) + SCALE0_FIX_ONE_BY_30_ADD)
                        >> 12;
                    sum += csf_f[r1 + j + 1] as i32;
                    sum += csf_f[r2 + j - 1] as i32;
                    sum += csf_f[r2 + j] as i32;
                    sum += csf_f[r2 + j + 1] as i32;

                    thr_total += sum;
                }
                thr_total
            };

            {
                let mut x =
                    ((rst_h[idx] as i32) * SCALE0_CSF_RFACTOR[0]).abs() - (thr << shift_xhsub);
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_xhsq) >> shift_xhsq) as i32;
                inner_h += ((x_sq as i64 * x as i64) + add_shift_xhcub) >> shift_xhcub;
            }
            {
                let mut x =
                    ((rst_v[idx] as i32) * SCALE0_CSF_RFACTOR[1]).abs() - (thr << shift_xvsub);
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_xvsq) >> shift_xvsq) as i32;
                inner_v += ((x_sq as i64 * x as i64) + add_shift_xvcub) >> shift_xvcub;
            }
            {
                let mut x =
                    ((rst_d[idx] as i32) * SCALE0_CSF_RFACTOR[2]).abs() - (thr << shift_xdsub);
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_xdsq) >> shift_xdsq) as i32;
                inner_d += ((x_sq as i64 * x as i64) + add_shift_xdcub) >> shift_xdcub;
            }
        }

        accum_h += (inner_h + add_shift_inner_accum) >> shift_inner_accum;
        accum_v += (inner_v + add_shift_inner_accum) >> shift_inner_accum;
        accum_d += (inner_d + add_shift_inner_accum) >> shift_inner_accum;
    }

    let area = (bottom - top) * (right - left);
    let f_accum_h =
        (accum_h as f64 / 2.0f64.powi(52 - shift_xhcub as i32 - shift_inner_accum as i32)) as f32;
    let f_accum_v =
        (accum_v as f64 / 2.0f64.powi(52 - shift_xvcub as i32 - shift_inner_accum as i32)) as f32;
    let f_accum_d =
        (accum_d as f64 / 2.0f64.powi(57 - shift_xdcub as i32 - shift_inner_accum as i32)) as f32;

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    f_accum_h.powf(1.0 / 3.0)
        + powf_add
        + f_accum_v.powf(1.0 / 3.0)
        + powf_add
        + f_accum_d.powf(1.0 / 3.0)
        + powf_add
}

#[cfg(test)]
fn adm_cm_s123_reference(
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
    i_rfactor: [u32; 3],
) -> f32 {
    let (left, top, right, bottom) = accum_border(w, h);
    let start_col = left.max(1);
    let end_col = right.min(w.saturating_sub(1));
    let start_row = top.max(1);
    let end_row = bottom.min(h.saturating_sub(1));

    let shift_cub = ceil_log2_u32(w as u32);
    let add_shift_cub = if shift_cub > 0 {
        1i64 << (shift_cub - 1)
    } else {
        0
    };
    let shift_inner_accum = ceil_log2_u32(h as u32);
    let add_shift_inner_accum = if shift_inner_accum > 0 {
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
    let final_shift = 2.0f64.powi(base_exp - shift_cub as i32 - shift_inner_accum as i32) as f32;

    let shift_sq: i32 = 30;
    let add_shift_sq: i64 = 536_870_912;

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

                    sum += csf_f[r0 + j - 1];
                    sum += csf_f[r0 + j];
                    sum += csf_f[r0 + j + 1];
                    sum += csf_f[r1 + j - 1];
                    sum += (((I4_ONE_BY_15 * (csf_a[r1 + j] as i64).abs())
                        + S123_ADD_BEFORE_SHIFT_FLT)
                        >> S123_SHIFT_FLT) as i32;
                    sum += csf_f[r1 + j + 1];
                    sum += csf_f[r2 + j - 1];
                    sum += csf_f[r2 + j];
                    sum += csf_f[r2 + j + 1];

                    thr_total += sum;
                }
                thr_total
            };

            {
                let mut x = ((((rst_h[idx] as i64) * (i_rfactor[0] as i64)
                    + S123_ADD_BEFORE_SHIFT_DST)
                    >> S123_SHIFT_DST) as i32)
                    .abs()
                    - thr;
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                inner_h += ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
            }
            {
                let mut x = ((((rst_v[idx] as i64) * (i_rfactor[1] as i64)
                    + S123_ADD_BEFORE_SHIFT_DST)
                    >> S123_SHIFT_DST) as i32)
                    .abs()
                    - thr;
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                inner_v += ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
            }
            {
                let mut x = ((((rst_d[idx] as i64) * (i_rfactor[2] as i64)
                    + S123_ADD_BEFORE_SHIFT_DST)
                    >> S123_SHIFT_DST) as i32)
                    .abs()
                    - thr;
                if x < 0 {
                    x = 0;
                }
                let x_sq = (((x as i64 * x as i64) + add_shift_sq) >> shift_sq) as i32;
                inner_d += ((x_sq as i64 * x as i64) + add_shift_cub) >> shift_cub;
            }
        }

        accum_h += (inner_h + add_shift_inner_accum) >> shift_inner_accum;
        accum_v += (inner_v + add_shift_inner_accum) >> shift_inner_accum;
        accum_d += (inner_d + add_shift_inner_accum) >> shift_inner_accum;
    }

    let area = (bottom - top) * (right - left);
    let f_accum_h = accum_h as f32 / final_shift;
    let f_accum_v = accum_v as f32 / final_shift;
    let f_accum_d = accum_d as f32 / final_shift;

    let powf_add = (area as f32 / 32.0).powf(1.0 / 3.0);
    f_accum_h.powf(1.0 / 3.0)
        + powf_add
        + f_accum_v.powf(1.0 / 3.0)
        + powf_add
        + f_accum_d.powf(1.0 / 3.0)
        + powf_add
}

#[cfg(test)]
pub(crate) fn score_scale0_reference(
    ref_h: &[i16],
    ref_v: &[i16],
    ref_d: &[i16],
    dis_h: &[i16],
    dis_v: &[i16],
    dis_d: &[i16],
    adm_enhn_gain_limit: f64,
    width: usize,
    height: usize,
) -> (f32, f32) {
    let n = width * height;
    debug_assert_eq!(ref_h.len(), n);
    debug_assert_eq!(ref_v.len(), n);
    debug_assert_eq!(ref_d.len(), n);
    debug_assert_eq!(dis_h.len(), n);
    debug_assert_eq!(dis_v.len(), n);
    debug_assert_eq!(dis_d.len(), n);

    let mut rst_h = Vec::with_capacity(n);
    let mut rst_v = Vec::with_capacity(n);
    let mut rst_d = Vec::with_capacity(n);
    let mut art_h = Vec::with_capacity(n);
    let mut art_v = Vec::with_capacity(n);
    let mut art_d = Vec::with_capacity(n);

    for k in 0..n {
        let (rh, rv, rd, ah, av, ad) = decouple_scale0(
            ref_h[k],
            ref_v[k],
            ref_d[k],
            dis_h[k],
            dis_v[k],
            dis_d[k],
            adm_enhn_gain_limit,
        );
        rst_h.push(rh);
        rst_v.push(rv);
        rst_d.push(rd);
        art_h.push(ah);
        art_v.push(av);
        art_d.push(ad);
    }

    let rfactor = noise_floor::rfactor(0);
    let (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d) =
        csf_scale0_reference(&art_h, &art_v, &art_d);
    let num = adm_cm_scale0_reference(
        &rst_h, &rst_v, &rst_d, &csf_a_h, &csf_a_v, &csf_a_d, &csf_f_h, &csf_f_v, &csf_f_d, width,
        height,
    );
    let den = adm_csf_den_scale0(ref_h, ref_v, ref_d, width, height, rfactor);
    (num, den)
}

#[cfg(test)]
pub(crate) fn score_scale_s123_reference(
    ref_h: &[i32],
    ref_v: &[i32],
    ref_d: &[i32],
    dis_h: &[i32],
    dis_v: &[i32],
    dis_d: &[i32],
    adm_enhn_gain_limit: f64,
    scale: usize,
    width: usize,
    height: usize,
) -> (f32, f32) {
    let n = width * height;
    debug_assert_eq!(ref_h.len(), n);
    debug_assert_eq!(ref_v.len(), n);
    debug_assert_eq!(ref_d.len(), n);
    debug_assert_eq!(dis_h.len(), n);
    debug_assert_eq!(dis_v.len(), n);
    debug_assert_eq!(dis_d.len(), n);

    let mut rst_h = Vec::with_capacity(n);
    let mut rst_v = Vec::with_capacity(n);
    let mut rst_d = Vec::with_capacity(n);
    let mut art_h = Vec::with_capacity(n);
    let mut art_v = Vec::with_capacity(n);
    let mut art_d = Vec::with_capacity(n);

    for k in 0..n {
        let (rh, rv, rd, ah, av, ad) = decouple_s123(
            ref_h[k],
            ref_v[k],
            ref_d[k],
            dis_h[k],
            dis_v[k],
            dis_d[k],
            adm_enhn_gain_limit,
        );
        rst_h.push(rh);
        rst_v.push(rv);
        rst_d.push(rd);
        art_h.push(ah);
        art_v.push(av);
        art_d.push(ad);
    }

    let rfactor = noise_floor::rfactor(scale);
    let i_rfactor = scale123_rfactor_fp(rfactor);
    let (csf_a_h, csf_a_v, csf_a_d, csf_f_h, csf_f_v, csf_f_d) =
        csf_s123_reference(&art_h, &art_v, &art_d, i_rfactor);
    let num = adm_cm_s123_reference(
        &rst_h, &rst_v, &rst_d, &csf_a_h, &csf_a_v, &csf_a_d, &csf_f_h, &csf_f_v, &csf_f_d, scale,
        width, height, i_rfactor,
    );
    let den = adm_csf_den_s123(ref_h, ref_v, ref_d, scale, width, height, rfactor);
    (num, den)
}

/// Score scale 0: returns `(num_scale, den_scale)`.
pub(crate) fn score_scale0(
    ref_h: &[i16],
    ref_v: &[i16],
    ref_d: &[i16],
    dis_h: &[i16],
    dis_v: &[i16],
    dis_d: &[i16],
    adm_enhn_gain_limit: f64,
    width: usize,
    height: usize,
) -> (f32, f32) {
    let rfactor = noise_floor::rfactor(0);
    let num = adm_cm_scale0(
        ref_h,
        ref_v,
        ref_d,
        dis_h,
        dis_v,
        dis_d,
        width,
        height,
        adm_enhn_gain_limit,
    );
    let den = adm_csf_den_scale0(ref_h, ref_v, ref_d, width, height, rfactor);
    (num, den)
}

/// Score scales 1–3: returns `(num_scale, den_scale)`.
pub(crate) fn score_scale_s123(
    ref_h: &[i32],
    ref_v: &[i32],
    ref_d: &[i32],
    dis_h: &[i32],
    dis_v: &[i32],
    dis_d: &[i32],
    adm_enhn_gain_limit: f64,
    scale: usize,
    width: usize,
    height: usize,
) -> (f32, f32) {
    let rfactor = noise_floor::rfactor(scale);
    let i_rfactor = scale123_rfactor_fp(rfactor);
    let num = adm_cm_s123(
        ref_h,
        ref_v,
        ref_d,
        dis_h,
        dis_v,
        dis_d,
        scale,
        width,
        height,
        i_rfactor,
        adm_enhn_gain_limit,
    );
    let den = adm_csf_den_s123(ref_h, ref_v, ref_d, scale, width, height, rfactor);
    (num, den)
}
