//! ADM feature extractor — spec §4.3

use crate::decouple::{decouple_s123, decouple_scale0};
use crate::dwt::{dwt_s123, dwt_scale0};
use crate::score::{score_scale0, score_scale_s123};

/// Stateless ADM extractor: computes the `adm2` score for one reference/distorted frame pair.
pub struct AdmExtractor {
    width: usize,
    height: usize,
    bpc: u8,
    adm_enhn_gain_limit: f64,
}

impl AdmExtractor {
    pub fn new(width: usize, height: usize, bpc: u8, adm_enhn_gain_limit: f64) -> Self {
        Self {
            width,
            height,
            bpc,
            adm_enhn_gain_limit,
        }
    }

    /// Compute the `adm2` score for one frame pair — spec §4.3.
    ///
    /// `ref_plane` and `dis_plane` are row-major luma planes with `width × height` samples.
    pub fn compute_frame(&self, ref_plane: &[u16], dis_plane: &[u16]) -> f64 {
        let (w, h, bpc) = (self.width, self.height, self.bpc);
        let limit = self.adm_enhn_gain_limit;

        // Scale 0 DWT
        let ref0 = dwt_scale0(ref_plane, w, h, bpc);
        let dis0 = dwt_scale0(dis_plane, w, h, bpc);
        let (w0, h0) = (ref0.width, ref0.height);
        let n0 = w0 * h0;

        // Decouple scale 0 (all pixels)
        let mut rst_h0 = vec![0i16; n0];
        let mut rst_v0 = vec![0i16; n0];
        let mut rst_d0 = vec![0i16; n0];
        let mut art_h0 = vec![0i16; n0];
        let mut art_v0 = vec![0i16; n0];
        let mut art_d0 = vec![0i16; n0];
        for k in 0..n0 {
            let (rh, rv, rd, ah, av, ad) = decouple_scale0(
                ref0.h[k], ref0.v[k], ref0.d[k], dis0.h[k], dis0.v[k], dis0.d[k], limit,
            );
            rst_h0[k] = rh;
            rst_v0[k] = rv;
            rst_d0[k] = rd;
            art_h0[k] = ah;
            art_v0[k] = av;
            art_d0[k] = ad;
        }

        // Score scale 0 (integer_adm fixed-point path)
        let (num0, den0) = score_scale0(
            &ref0.h, &ref0.v, &ref0.d, &rst_h0, &rst_v0, &rst_d0, &art_h0, &art_v0, &art_d0, w0, h0,
        );

        // Widen scale-0 LL to i32 (plain sign-extension, no shift)
        let mut cur_ref_ll: Vec<i32> = ref0.a.iter().map(|&x| x as i32).collect();
        let mut cur_dis_ll: Vec<i32> = dis0.a.iter().map(|&x| x as i32).collect();
        let mut cur_w = w0;
        let mut cur_h = h0;

        let mut num_total = num0;
        let mut den_total = den0;

        // Scales 1–3
        for scale in 1..=3usize {
            let ref_s = dwt_s123(&cur_ref_ll, cur_w, cur_h, scale);
            let dis_s = dwt_s123(&cur_dis_ll, cur_w, cur_h, scale);
            let (ws, hs) = (ref_s.width, ref_s.height);
            let ns = ws * hs;

            let mut rst_h = vec![0i32; ns];
            let mut rst_v = vec![0i32; ns];
            let mut rst_d = vec![0i32; ns];
            let mut art_h = vec![0i32; ns];
            let mut art_v = vec![0i32; ns];
            let mut art_d = vec![0i32; ns];

            for k in 0..ns {
                let (rh, rv, rd, ah, av, ad) = decouple_s123(
                    ref_s.h[k], ref_s.v[k], ref_s.d[k], dis_s.h[k], dis_s.v[k], dis_s.d[k], limit,
                );
                rst_h[k] = rh;
                rst_v[k] = rv;
                rst_d[k] = rd;
                art_h[k] = ah;
                art_v[k] = av;
                art_d[k] = ad;
            }

            let (num_s, den_s) = score_scale_s123(
                &ref_s.h, &ref_s.v, &ref_s.d, &rst_h, &rst_v, &rst_d, &art_h, &art_v, &art_d,
                scale, ws, hs,
            );
            num_total += num_s;
            den_total += den_s;

            cur_ref_ll = ref_s.a;
            cur_dis_ll = dis_s.a;
            cur_w = ws;
            cur_h = hs;
        }

        // Final adm2 score — spec §4.3.9
        let numden_limit = 1e-10 * (w * h) as f64 / (1920.0 * 1080.0);
        let num_d = if (num_total as f64) < numden_limit {
            0.0
        } else {
            num_total as f64
        };
        let den_d = if (den_total as f64) < numden_limit {
            0.0
        } else {
            den_total as f64
        };
        if den_d == 0.0 {
            1.0
        } else {
            num_d / den_d
        }
    }
}
