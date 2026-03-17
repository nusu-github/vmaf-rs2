//! DWT filter constants and 2D computation — spec §4.3.1–4.3.3

use std::mem::MaybeUninit;

use crate::math::reflect_index;
use vmaf_cpu::{Align32, AlignedScratch};

pub(crate) const FILTER_LO: [i32; 4] = [15826, 27411, 7345, -4240];
pub(crate) const FILTER_HI: [i32; 4] = [-4240, -7345, 27411, -15826];
pub(crate) const DWT_LO_SUM: i32 = 46342;

/// Normalize `abs_x` (≥ 32768) to a 15-bit mantissa — spec §4.3.8.
///
/// Returns `(mantissa, shift)` where `shift = 17 − clz32(abs_x)`.
pub(crate) fn get_best15_from32(abs_x: u32) -> (u16, i32) {
    let shift = 17 - abs_x.leading_zeros() as i32;
    let mantissa = ((abs_x + (1u32 << (shift - 1))) >> shift) as u16;
    (mantissa, shift)
}

/// Scale-0 DWT output (int16 subbands).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Bands16 {
    pub a: Vec<i16>,
    pub v: Vec<i16>,
    pub h: Vec<i16>,
    pub d: Vec<i16>,
    pub width: usize,
    pub height: usize,
}

/// Scales 1–3 DWT output (int32 subbands).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Bands32 {
    pub a: Vec<i32>,
    pub v: Vec<i32>,
    pub h: Vec<i32>,
    pub d: Vec<i32>,
    pub width: usize,
    pub height: usize,
}

/// 2D DWT on a uint16 luma plane (scale 0, int16 output) — spec §4.3.3.
pub(crate) fn dwt_scale0(src: &[u16], width: usize, height: usize, bpc: u8) -> Bands16 {
    let h_half = height.div_ceil(2);
    let mut tmplo = AlignedScratch::<MaybeUninit<i16>, Align32>::uninit(h_half * width);
    let mut tmphi = AlignedScratch::<MaybeUninit<i16>, Align32>::uninit(h_half * width);

    dwt_scale0_vertical_scalar(
        src,
        width,
        height,
        bpc,
        tmplo.as_mut_slice(),
        tmphi.as_mut_slice(),
    );
    // SAFETY: `dwt_scale0_vertical_scalar` writes every temporary coefficient.
    let tmplo = unsafe { tmplo.assume_init() };
    // SAFETY: `dwt_scale0_vertical_scalar` writes every temporary coefficient.
    let tmphi = unsafe { tmphi.assume_init() };
    dwt_scale0_horizontal_scalar(tmplo.as_slice(), tmphi.as_slice(), width, h_half)
}

pub(crate) fn dwt_scale0_vertical_scalar(
    src: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    tmplo: &mut [MaybeUninit<i16>],
    tmphi: &mut [MaybeUninit<i16>],
) {
    let shift_vp = if bpc == 8 { 8u32 } else { bpc as u32 };
    let round_vp = 1i32 << (shift_vp - 1);
    let h_half = height.div_ceil(2);
    let lo_bias = round_vp.wrapping_sub(DWT_LO_SUM.wrapping_mul(round_vp));

    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let base = 2 * i as i32;
        let r0 = reflect_index(base - 1, height as i32);
        let r1 = i * 2;
        let r2 = reflect_index(base + 1, height as i32);
        let r3 = reflect_index(base + 2, height as i32);
        let row_offset = i * width;

        for j in 0..width {
            let s0 = src[r0 * width + j] as i32;
            let s1 = src[r1 * width + j] as i32;
            let s2 = src[r2 * width + j] as i32;
            let s3 = src[r3 * width + j] as i32;

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

            tmplo[row_offset + j].write((al.wrapping_add(lo_bias) >> shift_vp) as i16);
            tmphi[row_offset + j].write((ah.wrapping_add(round_vp) >> shift_vp) as i16);
        }
    }
}

#[inline]
pub(crate) fn dwt_scale0_horizontal_scalar_at(
    tmplo_row: &[i16],
    tmphi_row: &[i16],
    width: usize,
    j: usize,
) -> (i16, i16, i16, i16) {
    let base = 2 * j as i32;
    let c0 = reflect_index(base - 1, width as i32);
    let c1 = j * 2;
    let c2 = reflect_index(base + 1, width as i32);
    let c3 = reflect_index(base + 2, width as i32);
    let round_hp = 32768i64;

    let hp = |buf: &[i16], filt: &[i32; 4]| -> i16 {
        let s0 = buf[c0] as i64;
        let s1 = buf[c1] as i64;
        let s2 = buf[c2] as i64;
        let s3 = buf[c3] as i64;
        ((filt[0] as i64 * s0
            + filt[1] as i64 * s1
            + filt[2] as i64 * s2
            + filt[3] as i64 * s3
            + round_hp)
            >> 16) as i16
    };

    (
        hp(tmplo_row, &FILTER_LO),
        hp(tmplo_row, &FILTER_HI),
        hp(tmphi_row, &FILTER_LO),
        hp(tmphi_row, &FILTER_HI),
    )
}

pub(crate) fn dwt_scale0_horizontal_scalar(
    tmplo: &[i16],
    tmphi: &[i16],
    width: usize,
    h_half: usize,
) -> Bands16 {
    let w_half = width.div_ceil(2);
    let n = h_half * w_half;
    let mut band_a = Vec::with_capacity(n);
    let mut band_v = Vec::with_capacity(n);
    let mut band_h = Vec::with_capacity(n);
    let mut band_d = Vec::with_capacity(n);

    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let tmplo_row = &tmplo[src_row_start..src_row_end];
        let tmphi_row = &tmphi[src_row_start..src_row_end];

        for j in 0..w_half {
            let (a, v, h, d) = dwt_scale0_horizontal_scalar_at(tmplo_row, tmphi_row, width, j);
            band_a.push(a);
            band_v.push(v);
            band_h.push(h);
            band_d.push(d);
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

/// Scale-specific constants: `(round_VP, shift_VP, round_HP, shift_HP)`.
pub(crate) const SCALE_PARAMS: [(i64, u32, i64, u32); 4] = [
    (0, 0, 0, 0),           // scale 0 — not used here
    (0, 0, 16384, 15),      // scale 1
    (32768, 16, 32768, 16), // scale 2
    (32768, 16, 16384, 15), // scale 3
];

/// 2D DWT on an int32 LL band (scales 1–3, int32 output) — spec §4.3.3.
pub(crate) fn dwt_s123(ll: &[i32], width: usize, height: usize, scale: usize) -> Bands32 {
    let h_half = height.div_ceil(2);
    let mut tmplo = AlignedScratch::<MaybeUninit<i32>, Align32>::uninit(h_half * width);
    let mut tmphi = AlignedScratch::<MaybeUninit<i32>, Align32>::uninit(h_half * width);

    dwt_s123_vertical_scalar(
        ll,
        width,
        height,
        scale,
        tmplo.as_mut_slice(),
        tmphi.as_mut_slice(),
    );
    // SAFETY: `dwt_s123_vertical_scalar` writes every temporary coefficient.
    let tmplo = unsafe { tmplo.assume_init() };
    // SAFETY: `dwt_s123_vertical_scalar` writes every temporary coefficient.
    let tmphi = unsafe { tmphi.assume_init() };
    dwt_s123_horizontal_scalar(tmplo.as_slice(), tmphi.as_slice(), width, h_half, scale)
}

pub(crate) fn dwt_s123_vertical_scalar(
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
    tmplo: &mut [MaybeUninit<i32>],
    tmphi: &mut [MaybeUninit<i32>],
) {
    let (round_vp, shift_vp, _, _) = SCALE_PARAMS[scale];
    let h_half = height.div_ceil(2);

    debug_assert_eq!(ll.len(), width * height);
    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let base = 2 * i as i32;
        let r0 = reflect_index(base - 1, height as i32);
        let r1 = i * 2;
        let r2 = reflect_index(base + 1, height as i32);
        let r3 = reflect_index(base + 2, height as i32);
        let row_offset = i * width;

        for j in 0..width {
            let s0 = ll[r0 * width + j] as i64;
            let s1 = ll[r1 * width + j] as i64;
            let s2 = ll[r2 * width + j] as i64;
            let s3 = ll[r3 * width + j] as i64;

            let al = FILTER_LO[0] as i64 * s0
                + FILTER_LO[1] as i64 * s1
                + FILTER_LO[2] as i64 * s2
                + FILTER_LO[3] as i64 * s3;
            let ah = FILTER_HI[0] as i64 * s0
                + FILTER_HI[1] as i64 * s1
                + FILTER_HI[2] as i64 * s2
                + FILTER_HI[3] as i64 * s3;

            tmplo[row_offset + j].write(((al + round_vp) >> shift_vp) as i32);
            tmphi[row_offset + j].write(((ah + round_vp) >> shift_vp) as i32);
        }
    }
}

#[inline]
pub(crate) fn dwt_s123_horizontal_scalar_at(
    tmplo_row: &[i32],
    tmphi_row: &[i32],
    width: usize,
    j: usize,
    round_hp: i64,
    shift_hp: u32,
) -> (i32, i32, i32, i32) {
    let base = 2 * j as i32;
    let c0 = reflect_index(base - 1, width as i32);
    let c1 = j * 2;
    let c2 = reflect_index(base + 1, width as i32);
    let c3 = reflect_index(base + 2, width as i32);

    let hp = |buf: &[i32], filt: &[i32; 4]| -> i32 {
        let s0 = buf[c0] as i64;
        let s1 = buf[c1] as i64;
        let s2 = buf[c2] as i64;
        let s3 = buf[c3] as i64;
        ((filt[0] as i64 * s0
            + filt[1] as i64 * s1
            + filt[2] as i64 * s2
            + filt[3] as i64 * s3
            + round_hp)
            >> shift_hp) as i32
    };

    (
        hp(tmplo_row, &FILTER_LO),
        hp(tmplo_row, &FILTER_HI),
        hp(tmphi_row, &FILTER_LO),
        hp(tmphi_row, &FILTER_HI),
    )
}

pub(crate) fn dwt_s123_horizontal_scalar(
    tmplo: &[i32],
    tmphi: &[i32],
    width: usize,
    h_half: usize,
    scale: usize,
) -> Bands32 {
    let (_, _, round_hp, shift_hp) = SCALE_PARAMS[scale];
    let w_half = width.div_ceil(2);
    let n = h_half * w_half;
    let mut band_a = Vec::with_capacity(n);
    let mut band_v = Vec::with_capacity(n);
    let mut band_h = Vec::with_capacity(n);
    let mut band_d = Vec::with_capacity(n);

    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let tmplo_row = &tmplo[src_row_start..src_row_end];
        let tmphi_row = &tmphi[src_row_start..src_row_end];

        for j in 0..w_half {
            let (a, v, h, d) =
                dwt_s123_horizontal_scalar_at(tmplo_row, tmphi_row, width, j, round_hp, shift_hp);
            band_a.push(a);
            band_v.push(v);
            band_h.push(h);
            band_d.push(d);
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
