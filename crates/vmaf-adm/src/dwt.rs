//! DWT filter constants and 2D computation — spec §4.3.1–4.3.3

use crate::math::reflect_index;

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
pub(crate) struct Bands16 {
    pub a: Vec<i16>,
    pub v: Vec<i16>,
    pub h: Vec<i16>,
    pub d: Vec<i16>,
    pub width: usize,
    pub height: usize,
}

/// Scales 1–3 DWT output (int32 subbands).
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
    let shift_vp = if bpc == 8 { 8u32 } else { bpc as u32 };
    let round_vp = 1i32 << (shift_vp - 1);
    let h_half = height.div_ceil(2);
    let w_half = width.div_ceil(2);

    // Vertical pass → tmplo / tmphi each h_half × width
    let mut tmplo = vec![0i16; h_half * width];
    let mut tmphi = vec![0i16; h_half * width];

    for i in 0..h_half {
        let base = 2 * i as i32;
        let r0 = reflect_index(base - 1, height as i32);
        let r1 = i * 2; // always in [0, height-1]
        let r2 = reflect_index(base + 1, height as i32);
        let r3 = reflect_index(base + 2, height as i32);

        for j in 0..width {
            let s0 = src[r0 * width + j] as i32;
            let s1 = src[r1 * width + j] as i32;
            let s2 = src[r2 * width + j] as i32;
            let s3 = src[r3 * width + j] as i32;

            let al = FILTER_LO[0] * s0 + FILTER_LO[1] * s1 + FILTER_LO[2] * s2 + FILTER_LO[3] * s3;
            let ah = FILTER_HI[0] * s0 + FILTER_HI[1] * s1 + FILTER_HI[2] * s2 + FILTER_HI[3] * s3;

            tmplo[i * width + j] = ((al - DWT_LO_SUM * round_vp + round_vp) >> shift_vp) as i16;
            tmphi[i * width + j] = ((ah + round_vp) >> shift_vp) as i16;
        }
    }

    // Horizontal pass (i64 accumulator to avoid overflow) → h_half × w_half
    let round_hp = 32768i64;
    let mut band_a = vec![0i16; h_half * w_half];
    let mut band_v = vec![0i16; h_half * w_half];
    let mut band_h = vec![0i16; h_half * w_half];
    let mut band_d = vec![0i16; h_half * w_half];

    for i in 0..h_half {
        for j in 0..w_half {
            let base = 2 * j as i32;
            let c0 = reflect_index(base - 1, width as i32);
            let c1 = j * 2; // always in [0, width-1]
            let c2 = reflect_index(base + 1, width as i32);
            let c3 = reflect_index(base + 2, width as i32);

            macro_rules! hp {
                ($buf:expr, $filt:expr) => {{
                    let s0 = $buf[i * width + c0] as i64;
                    let s1 = $buf[i * width + c1] as i64;
                    let s2 = $buf[i * width + c2] as i64;
                    let s3 = $buf[i * width + c3] as i64;
                    (($filt[0] as i64 * s0
                        + $filt[1] as i64 * s1
                        + $filt[2] as i64 * s2
                        + $filt[3] as i64 * s3
                        + round_hp)
                        >> 16) as i16
                }};
            }

            band_a[i * w_half + j] = hp!(tmplo, FILTER_LO);
            band_v[i * w_half + j] = hp!(tmplo, FILTER_HI);
            band_h[i * w_half + j] = hp!(tmphi, FILTER_LO);
            band_d[i * w_half + j] = hp!(tmphi, FILTER_HI);
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
const SCALE_PARAMS: [(i64, u32, i64, u32); 4] = [
    (0, 0, 0, 0),           // scale 0 — not used here
    (0, 0, 16384, 15),      // scale 1
    (32768, 16, 32768, 16), // scale 2
    (32768, 16, 16384, 15), // scale 3
];

/// 2D DWT on an int32 LL band (scales 1–3, int32 output) — spec §4.3.3.
pub(crate) fn dwt_s123(ll: &[i32], width: usize, height: usize, scale: usize) -> Bands32 {
    let (round_vp, shift_vp, round_hp, shift_hp) = SCALE_PARAMS[scale];
    let h_half = height.div_ceil(2);
    let w_half = width.div_ceil(2);

    let mut tmplo = vec![0i32; h_half * width];
    let mut tmphi = vec![0i32; h_half * width];

    // Vertical pass (i64 accumulator, truncating cast to i32 per spec §4.3.3)
    for i in 0..h_half {
        let base = 2 * i as i32;
        let r0 = reflect_index(base - 1, height as i32);
        let r1 = i * 2;
        let r2 = reflect_index(base + 1, height as i32);
        let r3 = reflect_index(base + 2, height as i32);

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

            // No saturating clamp — truncating cast to i32 per spec
            tmplo[i * width + j] = ((al + round_vp) >> shift_vp) as i32;
            tmphi[i * width + j] = ((ah + round_vp) >> shift_vp) as i32;
        }
    }

    // Horizontal pass (i64 accumulator, truncating cast to i32)
    let mut band_a = vec![0i32; h_half * w_half];
    let mut band_v = vec![0i32; h_half * w_half];
    let mut band_h = vec![0i32; h_half * w_half];
    let mut band_d = vec![0i32; h_half * w_half];

    for i in 0..h_half {
        for j in 0..w_half {
            let base = 2 * j as i32;
            let c0 = reflect_index(base - 1, width as i32);
            let c1 = j * 2;
            let c2 = reflect_index(base + 1, width as i32);
            let c3 = reflect_index(base + 2, width as i32);

            macro_rules! hp32 {
                ($buf:expr, $filt:expr) => {{
                    let s0 = $buf[i * width + c0] as i64;
                    let s1 = $buf[i * width + c1] as i64;
                    let s2 = $buf[i * width + c2] as i64;
                    let s3 = $buf[i * width + c3] as i64;
                    (($filt[0] as i64 * s0
                        + $filt[1] as i64 * s1
                        + $filt[2] as i64 * s2
                        + $filt[3] as i64 * s3
                        + round_hp)
                        >> shift_hp) as i32
                }};
            }

            band_a[i * w_half + j] = hp32!(tmplo, FILTER_LO);
            band_v[i * w_half + j] = hp32!(tmplo, FILTER_HI);
            band_h[i * w_half + j] = hp32!(tmphi, FILTER_LO);
            band_d[i * w_half + j] = hp32!(tmphi, FILTER_HI);
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
