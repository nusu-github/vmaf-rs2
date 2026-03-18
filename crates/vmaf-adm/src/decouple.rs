//! Integer angle-based masking (decouple) — spec §4.3.8

use crate::{dwt::get_best15_from32, tables::DIV_LOOKUP};

// ── Scale 0 (int16) ──────────────────────────────────────────────────────────

/// Decouple one pixel triplet at scale 0 — spec §4.3.8.
///
/// Returns `(rst_h, rst_v, rst_d, art_h, art_v, art_d)`.
pub(crate) fn decouple_scale0(
    ref_h: i16,
    ref_v: i16,
    ref_d: i16,
    dis_h: i16,
    dis_v: i16,
    dis_d: i16,
    adm_enhn_gain_limit: f64,
) -> (i16, i16, i16, i16, i16, i16) {
    // Angle coherence test (libvmaf does this in f32 after converting fixed-point back to float).
    let ot_dp: i64 = ref_h as i64 * dis_h as i64 + ref_v as i64 * dis_v as i64;
    let o_mag_sq: i64 = ref_h as i64 * ref_h as i64 + ref_v as i64 * ref_v as i64;
    let t_mag_sq: i64 = dis_h as i64 * dis_h as i64 + dis_v as i64 * dis_v as i64;
    const COS_SQ: f32 = f32::from_bits(0x3f7fec0a); // cos(1°)^2 as binary32
    let dp_f = ot_dp as f32 / 4096.0;
    let o_f = o_mag_sq as f32 / 4096.0;
    let t_f = t_mag_sq as f32 / 4096.0;
    let angle_flag = dp_f >= 0.0 && dp_f * dp_f >= COS_SQ * o_f * t_f;

    let process = |ref_c: i16, dis_c: i16| -> (i16, i16) {
        let tmp_k: i32 = if ref_c == 0 {
            32768
        } else {
            let idx = (ref_c as i32 + 32768) as usize;
            let raw = ((DIV_LOOKUP[idx] as i64 * dis_c as i64 + 16384) >> 15) as i32;
            raw.clamp(0, 32768)
        };
        let mut rst = ((tmp_k * ref_c as i32 + 16384) >> 15) as i16;
        let mut art = dis_c - rst;
        let rst_f = (tmp_k as f32 / 32768.0) * (ref_c as f32 / 64.0);
        if angle_flag {
            if rst_f > 0.0 {
                let scaled = (rst as f64 * adm_enhn_gain_limit) as i32;
                rst = scaled.min(dis_c as i32) as i16;
            } else if rst_f < 0.0 {
                let scaled = (rst as f64 * adm_enhn_gain_limit) as i32;
                rst = scaled.max(dis_c as i32) as i16;
            }
            art = dis_c - rst;
        }
        (rst, art)
    };

    let (rst_h, art_h) = process(ref_h, dis_h);
    let (rst_v, art_v) = process(ref_v, dis_v);
    let (rst_d, art_d) = process(ref_d, dis_d);
    (rst_h, rst_v, rst_d, art_h, art_v, art_d)
}

// ── Scales 1–3 (int32) ───────────────────────────────────────────────────────

/// Decouple one pixel triplet at scales 1–3 — spec §4.3.8.
///
/// Returns `(rst_h, rst_v, rst_d, art_h, art_v, art_d)`.
pub(crate) fn decouple_s123(
    ref_h: i32,
    ref_v: i32,
    ref_d: i32,
    dis_h: i32,
    dis_v: i32,
    dis_d: i32,
    adm_enhn_gain_limit: f64,
) -> (i32, i32, i32, i32, i32, i32) {
    // Angle coherence test (f32 with /4096 scaling)
    let ot_dp = (ref_h as i64) * dis_h as i64 + (ref_v as i64) * dis_v as i64;
    let o_mag_sq = (ref_h as i64) * ref_h as i64 + (ref_v as i64) * ref_v as i64;
    let t_mag_sq = (dis_h as i64) * dis_h as i64 + (dis_v as i64) * dis_v as i64;
    // libvmaf uses COS_SQ = cos(1°)^2 as a binary32 constant (0x3f7fec0a).
    const COS_SQ: f32 = f32::from_bits(0x3f7fec0a);
    let dp_f = ot_dp as f32 / 4096.0;
    let o_f = o_mag_sq as f32 / 4096.0;
    let t_f = t_mag_sq as f32 / 4096.0;
    let angle_flag = dp_f >= 0.0 && dp_f * dp_f >= COS_SQ * o_f * t_f;

    let process = |o_c: i32, t_c: i32| -> (i32, i32) {
        let abs_o = o_c.unsigned_abs();
        let sign_c: i64 = if o_c < 0 { -1 } else { 1 };

        let (mantissa, shift) = if abs_o < 32768 {
            (abs_o as u16, 0i32)
        } else {
            get_best15_from32(abs_o)
        };

        let tmp_k: i64 = if o_c == 0 {
            32768
        } else {
            let dl = DIV_LOOKUP[(mantissa as i32 + 32768) as usize] as i64;
            (dl * t_c as i64 * sign_c + (1i64 << (14 + shift))) >> (15 + shift)
        };
        let k_c = tmp_k.clamp(0, 32768) as i32;

        let mut rst = ((k_c as i64 * o_c as i64 + 16384) >> 15) as i32;

        // Enhancement sign test uses f32 on original-domain coefficient
        let rst_f = (k_c as f32 / 32768.0) * (o_c as f32 / 64.0);
        if angle_flag {
            if rst_f > 0.0 {
                let scaled = (rst as f64 * adm_enhn_gain_limit) as i32;
                rst = scaled.min(t_c);
            } else if rst_f < 0.0 {
                let scaled = (rst as f64 * adm_enhn_gain_limit) as i32;
                rst = scaled.max(t_c);
            }
        }
        let art = t_c - rst;
        (rst, art)
    };

    let (rst_h, art_h) = process(ref_h, dis_h);
    let (rst_v, art_v) = process(ref_v, dis_v);
    let (rst_d, art_d) = process(ref_d, dis_d);
    (rst_h, rst_v, rst_d, art_h, art_v, art_d)
}
