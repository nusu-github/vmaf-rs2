//! ADM feature extractor — spec §4.3

use crate::score::{score_scale0, score_scale_s123};
use crate::simd;
use vmaf_cpu::SimdBackend;

/// Stateless ADM extractor: computes the `adm2` score for one reference/distorted frame pair.
pub struct AdmExtractor {
    width: usize,
    height: usize,
    bpc: u8,
    adm_enhn_gain_limit: f64,
    backend: SimdBackend,
}

impl AdmExtractor {
    pub fn new(width: usize, height: usize, bpc: u8, adm_enhn_gain_limit: f64) -> Self {
        Self::with_backend(
            width,
            height,
            bpc,
            adm_enhn_gain_limit,
            simd::select_backend(),
        )
    }

    #[cfg(test)]
    pub(crate) fn with_backend_for_tests(
        width: usize,
        height: usize,
        bpc: u8,
        adm_enhn_gain_limit: f64,
        backend: SimdBackend,
    ) -> Self {
        Self::with_backend(width, height, bpc, adm_enhn_gain_limit, backend)
    }

    fn with_backend(
        width: usize,
        height: usize,
        bpc: u8,
        adm_enhn_gain_limit: f64,
        backend: SimdBackend,
    ) -> Self {
        Self {
            width,
            height,
            bpc,
            adm_enhn_gain_limit,
            backend: simd::effective_backend(backend),
        }
    }

    /// Compute the `adm2` score for one frame pair — spec §4.3.
    ///
    /// `ref_plane` and `dis_plane` are row-major luma planes with `width × height` samples.
    pub fn compute_frame(&self, ref_plane: &[u16], dis_plane: &[u16]) -> f64 {
        let (w, h, bpc) = (self.width, self.height, self.bpc);
        let limit = self.adm_enhn_gain_limit;
        let backend = self.backend;

        // Scale 0 DWT
        let ref0 = simd::dwt_scale0(backend, ref_plane, w, h, bpc);
        let dis0 = simd::dwt_scale0(backend, dis_plane, w, h, bpc);
        let (w0, h0) = (ref0.width, ref0.height);
        // Score scale 0 (integrated decouple + integer_adm fixed-point path)
        let (num0, den0) = score_scale0(
            &ref0.h, &ref0.v, &ref0.d, &dis0.h, &dis0.v, &dis0.d, limit, w0, h0,
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
            let ref_s = simd::dwt_s123(backend, &cur_ref_ll, cur_w, cur_h, scale);
            let dis_s = simd::dwt_s123(backend, &cur_dis_ll, cur_w, cur_h, scale);
            let (ws, hs) = (ref_s.width, ref_s.height);
            let (num_s, den_s) = score_scale_s123(
                &ref_s.h, &ref_s.v, &ref_s.d, &dis_s.h, &dis_s.v, &dis_s.d, limit, scale, ws, hs,
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
