//! ADM feature extractor — spec §4.3

use vmaf_cpu::{FrameGeometry, GainLimit, SimdBackend};

use crate::{
    dwt::{Bands16Buffer, Bands32Buffer, Scale0DwtWorkspace, Scale123DwtWorkspace},
    score::{score_scale_s123, score_scale0},
    simd,
};

/// Reusable internal buffers for per-frame ADM extraction.
#[doc(hidden)]
pub struct AdmWorkspace {
    cur_ref_ll: Vec<i32>,
    cur_dis_ll: Vec<i32>,
    scale0: Scale0DwtWorkspace,
    scale123: Scale123DwtWorkspace,
    ref_scale0: Bands16Buffer,
    dis_scale0: Bands16Buffer,
    ref_scale: Bands32Buffer,
    dis_scale: Bands32Buffer,
}

impl AdmWorkspace {
    fn new(width: usize, height: usize) -> Self {
        let scale0_area = width.div_ceil(2) * height.div_ceil(2);
        Self {
            cur_ref_ll: Vec::with_capacity(scale0_area),
            cur_dis_ll: Vec::with_capacity(scale0_area),
            scale0: Scale0DwtWorkspace::new(width, height),
            scale123: Scale123DwtWorkspace::new(width.div_ceil(2), height.div_ceil(2)),
            ref_scale0: Bands16Buffer::with_capacity(scale0_area),
            dis_scale0: Bands16Buffer::with_capacity(scale0_area),
            ref_scale: Bands32Buffer::with_capacity(scale0_area),
            dis_scale: Bands32Buffer::with_capacity(scale0_area),
        }
    }
}

#[inline]
fn copy_i16_slice_as_i32(dst: &mut Vec<i32>, src: &[i16]) {
    dst.clear();
    dst.reserve(src.len().saturating_sub(dst.capacity()));
    let spare = &mut dst.spare_capacity_mut()[..src.len()];
    for (slot, value) in spare.iter_mut().zip(src.iter().copied()) {
        slot.write(i32::from(value));
    }
    // SAFETY: each slot in the written prefix was initialized exactly once.
    unsafe {
        dst.set_len(src.len());
    }
}

/// Stateless ADM extractor: computes the `adm2` score for one reference/distorted frame pair.
pub struct AdmExtractor {
    geometry: FrameGeometry,
    adm_enhn_gain_limit: GainLimit,
    backend: SimdBackend,
}

impl AdmExtractor {
    /// Create an ADM extractor for one frame geometry.
    pub fn new(geometry: FrameGeometry, adm_enhn_gain_limit: GainLimit) -> Self {
        Self::with_backend(
            geometry,
            adm_enhn_gain_limit,
            SimdBackend::detect_effective(),
        )
    }

    #[cfg(test)]
    pub(crate) fn with_backend_for_tests(
        geometry: FrameGeometry,
        adm_enhn_gain_limit: GainLimit,
        backend: SimdBackend,
    ) -> Self {
        Self::with_backend(geometry, adm_enhn_gain_limit, backend)
    }

    fn with_backend(
        geometry: FrameGeometry,
        adm_enhn_gain_limit: GainLimit,
        backend: SimdBackend,
    ) -> Self {
        Self {
            geometry,
            adm_enhn_gain_limit,
            backend: backend.effective(),
        }
    }

    #[doc(hidden)]
    pub fn make_workspace(&self) -> AdmWorkspace {
        AdmWorkspace::new(self.geometry.width(), self.geometry.height())
    }

    #[doc(hidden)]
    pub fn compute_frame_with_workspace(
        &self,
        workspace: &mut AdmWorkspace,
        ref_plane: &[u16],
        dis_plane: &[u16],
    ) -> f64 {
        let geometry = self.geometry;
        let (w, h, bpc) = (geometry.width(), geometry.height(), geometry.bpc());
        let limit = self.adm_enhn_gain_limit.value();
        let backend = self.backend;

        simd::dwt_scale0_into(
            backend,
            ref_plane,
            w,
            h,
            bpc,
            &mut workspace.scale0,
            &mut workspace.ref_scale0,
        );
        simd::dwt_scale0_into(
            backend,
            dis_plane,
            w,
            h,
            bpc,
            &mut workspace.scale0,
            &mut workspace.dis_scale0,
        );
        let (w0, h0) = (workspace.ref_scale0.width, workspace.ref_scale0.height);
        let (num0, den0) = score_scale0(
            &workspace.ref_scale0.h,
            &workspace.ref_scale0.v,
            &workspace.ref_scale0.d,
            &workspace.dis_scale0.h,
            &workspace.dis_scale0.v,
            &workspace.dis_scale0.d,
            limit,
            w0,
            h0,
        );

        copy_i16_slice_as_i32(&mut workspace.cur_ref_ll, &workspace.ref_scale0.a);
        copy_i16_slice_as_i32(&mut workspace.cur_dis_ll, &workspace.dis_scale0.a);
        let mut cur_w = w0;
        let mut cur_h = h0;

        let mut num_total = num0;
        let mut den_total = den0;

        for scale in 1..=3usize {
            simd::dwt_s123_into(
                backend,
                &workspace.cur_ref_ll,
                cur_w,
                cur_h,
                scale,
                &mut workspace.scale123,
                &mut workspace.ref_scale,
            );
            simd::dwt_s123_into(
                backend,
                &workspace.cur_dis_ll,
                cur_w,
                cur_h,
                scale,
                &mut workspace.scale123,
                &mut workspace.dis_scale,
            );
            let (ws, hs) = (workspace.ref_scale.width, workspace.ref_scale.height);
            let (num_s, den_s) = score_scale_s123(
                &workspace.ref_scale.h,
                &workspace.ref_scale.v,
                &workspace.ref_scale.d,
                &workspace.dis_scale.h,
                &workspace.dis_scale.v,
                &workspace.dis_scale.d,
                limit,
                scale,
                ws,
                hs,
            );
            num_total += num_s;
            den_total += den_s;

            std::mem::swap(&mut workspace.cur_ref_ll, &mut workspace.ref_scale.a);
            std::mem::swap(&mut workspace.cur_dis_ll, &mut workspace.dis_scale.a);
            cur_w = ws;
            cur_h = hs;
        }

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
        if den_d == 0.0 { 1.0 } else { num_d / den_d }
    }

    /// Compute the `adm2` score for one frame pair — spec §4.3.
    ///
    /// `ref_plane` and `dis_plane` are row-major luma planes with `width × height` samples.
    pub fn compute_frame(&self, ref_plane: &[u16], dis_plane: &[u16]) -> f64 {
        let mut workspace = self.make_workspace();
        self.compute_frame_with_workspace(&mut workspace, ref_plane, dis_plane)
    }
}

#[cfg(test)]
mod tests {
    use vmaf_cpu::{FrameGeometry, FrameValidationError, GainLimit, SimdBackend};

    use super::*;

    fn geometry(width: usize, height: usize, bpc: u8) -> FrameGeometry {
        FrameGeometry::new(width, height, bpc).unwrap()
    }

    fn gain_limit(value: f64) -> GainLimit {
        GainLimit::new(value).unwrap()
    }

    fn patterned_plane(width: usize, height: usize, modulus: u16, bias: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    ((x * 17 + y * 19 + (x ^ y) * 5 + x * y * 3 + bias) % modulus as usize) as u16
                })
            })
            .collect()
    }

    #[test]
    fn workspace_path_matches_plain_compute() {
        let extractor = AdmExtractor::with_backend_for_tests(
            geometry(64, 64, 8),
            gain_limit(100.0),
            SimdBackend::Scalar,
        );
        let reference = patterned_plane(64, 64, 255, 7);
        let distorted = patterned_plane(64, 64, 255, 23);
        let mut workspace = extractor.make_workspace();

        let plain = extractor.compute_frame(&reference, &distorted);
        let reused1 =
            extractor.compute_frame_with_workspace(&mut workspace, &reference, &distorted);
        let reused2 =
            extractor.compute_frame_with_workspace(&mut workspace, &reference, &distorted);

        assert_eq!(plain.to_bits(), reused1.to_bits());
        assert_eq!(plain.to_bits(), reused2.to_bits());
    }

    #[test]
    fn constructor_rejects_invalid_dimensions_and_bpc() {
        assert!(matches!(
            FrameGeometry::new(15, 16, 8),
            Err(FrameValidationError::InvalidDimensions {
                width: 15,
                height: 16
            })
        ));
        assert!(matches!(
            FrameGeometry::new(16, 15, 8),
            Err(FrameValidationError::InvalidDimensions {
                width: 16,
                height: 15
            })
        ));
        assert!(matches!(
            FrameGeometry::new(16, 16, 9),
            Err(FrameValidationError::InvalidBitDepth { bpc: 9 })
        ));
    }
}
