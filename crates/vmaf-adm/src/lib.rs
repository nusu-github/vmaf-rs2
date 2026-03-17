//! ADM (Additive Distortion Measure) — spec §4.3
//!
//! Pipeline: integer DWT → integer decouple → fixed-point CSF/CM → shifted cube accumulators.
//! Scale 0 subbands are i16; scales 1–3 are i32 (§4.3.3).
//! Scoring follows libvmaf `integer_adm.c` (uses `powf(x, 1/3)` and fixed-point shifts).

mod decouple;
mod dwt;
mod extractor;
mod math;
mod noise_floor;
mod score;
mod simd;
mod tables;

pub use extractor::{AdmExtractor, AdmWorkspace};

#[cfg(test)]
mod tests {
    use super::decouple::decouple_scale0;
    use super::dwt::{dwt_s123, dwt_scale0, get_best15_from32};
    use super::extractor::AdmExtractor;
    use super::math::reflect_index;
    use super::score::{
        score_scale0, score_scale0_reference, score_scale_s123, score_scale_s123_reference,
    };
    use super::simd;
    use super::tables::DIV_LOOKUP;
    use vmaf_cpu::SimdBackend;

    // ── div_lookup §8 conformance ─────────────────────────────────────────────

    #[test]
    fn div_lookup_reference_vectors() {
        assert_eq!(DIV_LOOKUP[32768], 0); // x =  0
        assert_eq!(DIV_LOOKUP[32769], 1073741824); // x =  1
        assert_eq!(DIV_LOOKUP[32770], 536870912); // x =  2
        assert_eq!(DIV_LOOKUP[32771], 357913941); // x =  3
        assert_eq!(DIV_LOOKUP[32775], 153391689); // x =  7
        assert_eq!(DIV_LOOKUP[32868], 10737418); // x =  100
        assert_eq!(DIV_LOOKUP[33024], 4194304); // x =  256
        assert_eq!(DIV_LOOKUP[33768], 1073741); // x =  1000
        assert_eq!(DIV_LOOKUP[49152], 65536); // x =  16384
        assert_eq!(DIV_LOOKUP[65536], 32768); // x =  32768
        assert_eq!(DIV_LOOKUP[32767], -1073741824); // x = -1
        assert_eq!(DIV_LOOKUP[0], -32768); // x = -32768
    }

    // ── reflect_index ─────────────────────────────────────────────────────────

    #[test]
    fn reflect_in_range() {
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(3, 5), 3);
        assert_eq!(reflect_index(4, 5), 4);
    }

    #[test]
    fn reflect_negative() {
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
    }

    #[test]
    fn reflect_past_end() {
        assert_eq!(reflect_index(5, 5), 3); // 2*5 - 2 - 5 = 3
        assert_eq!(reflect_index(6, 5), 2); // 2*5 - 2 - 6 = 2
    }

    // ── get_best15_from32 §8 conformance ──────────────────────────────────────

    #[test]
    fn get_best15_from32_reference_vectors() {
        assert_eq!(get_best15_from32(32768), (16384, 1));
        assert_eq!(get_best15_from32(32769), (16385, 1));
        assert_eq!(get_best15_from32(65535), (32768, 1));
        assert_eq!(get_best15_from32(65536), (16384, 2));
        assert_eq!(get_best15_from32(1000000), (31250, 5));
    }

    // ── decouple_scale0 §8 reference case ────────────────────────────────────

    /// ref_h=100, ref_v=100, ref_d=50, dis_h=80, dis_v=90, dis_d=60
    /// → angle_flag=false; k_d clamped from 39322 → 32768 → rst_d=50, art_d=10
    #[test]
    fn decouple_scale0_reference_case() {
        let (rst_h, rst_v, rst_d, art_h, art_v, art_d) =
            decouple_scale0(100, 100, 50, 80, 90, 60, 100.0);
        assert_eq!(rst_h, 80);
        assert_eq!(art_h, 0);
        assert_eq!(rst_v, 90);
        assert_eq!(art_v, 0);
        assert_eq!(rst_d, 50);
        assert_eq!(art_d, 10);
    }

    fn plane_pattern(width: usize, height: usize, max_value: u16) -> Vec<u16> {
        let modulus = max_value as usize + 1;
        let mut plane = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let value = (x * 37 + y * 53 + (x ^ y) * 11 + (x * y) * 3) % modulus;
                plane.push(value as u16);
            }
        }
        plane
    }

    fn ll_pattern(width: usize, height: usize) -> Vec<i32> {
        let mut ll = Vec::with_capacity(width * height);
        for idx in 0..(width * height) {
            let base = ((idx as i32 * 97) % 4096) - 2048;
            let value = match idx % 5 {
                0 => base * 31,
                1 => -base * 17,
                2 => base * 9 - 777,
                3 => -base * 5 + 1234,
                _ => base * 13,
            };
            ll.push(value);
        }
        ll
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn dwt_scale0_sse2_matches_scalar_on_edges() {
        if !SimdBackend::X86Sse2.is_available() {
            return;
        }

        let width = 11;
        let height = 7;
        let bpc = 10;
        let plane = plane_pattern(width, height, (1u16 << bpc) - 1);
        let scalar = dwt_scale0(&plane, width, height, bpc);
        let simd = simd::dwt_scale0(SimdBackend::X86Sse2, &plane, width, height, bpc);

        assert_eq!(simd, scalar);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn dwt_scale0_avx2_matches_scalar_on_odd_frames() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let width = 33;
        let height = 19;
        let bpc = 12;
        let plane = plane_pattern(width, height, (1u16 << bpc) - 1);
        let scalar = dwt_scale0(&plane, width, height, bpc);
        let simd = simd::dwt_scale0(SimdBackend::X86Avx2Fma, &plane, width, height, bpc);

        assert_eq!(simd, scalar);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn dwt_s123_avx2_matches_scalar_across_scales() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let width = 21;
        let height = 15;
        let ll = ll_pattern(width, height);

        for scale in 1..=3 {
            let scalar = dwt_s123(&ll, width, height, scale);
            let simd = simd::dwt_s123(SimdBackend::X86Avx2Fma, &ll, width, height, scale);
            assert_eq!(simd, scalar, "scale {scale} mismatch");
        }
    }

    #[test]
    fn score_scale0_matches_reference_pipeline() {
        let width = 27;
        let height = 19;
        let bpc = 10;
        let limit = 100.0;
        let max_value = (1u16 << bpc) - 1;

        let ref_plane = plane_pattern(width, height, max_value);
        let mut dis_plane = plane_pattern(width, height, max_value);
        for (idx, value) in dis_plane.iter_mut().enumerate() {
            let delta = ((idx * 11 + 5) % 23) as u16;
            *value = value.saturating_sub(delta / 2).min(max_value);
        }

        let ref0 = dwt_scale0(&ref_plane, width, height, bpc);
        let dis0 = dwt_scale0(&dis_plane, width, height, bpc);

        let actual = score_scale0(
            &ref0.h,
            &ref0.v,
            &ref0.d,
            &dis0.h,
            &dis0.v,
            &dis0.d,
            limit,
            ref0.width,
            ref0.height,
        );
        let expected = score_scale0_reference(
            &ref0.h,
            &ref0.v,
            &ref0.d,
            &dis0.h,
            &dis0.v,
            &dis0.d,
            limit,
            ref0.width,
            ref0.height,
        );

        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    #[test]
    fn score_scale_s123_matches_reference_pipeline() {
        let width = 27;
        let height = 19;
        let bpc = 10;
        let limit = 100.0;
        let max_value = (1u16 << bpc) - 1;

        let ref_plane = plane_pattern(width, height, max_value);
        let mut dis_plane = plane_pattern(width, height, max_value);
        for (idx, value) in dis_plane.iter_mut().enumerate() {
            let delta = ((idx * 13 + 9) % 29) as u16;
            *value = value.saturating_sub(delta / 2).min(max_value);
        }

        let ref0 = dwt_scale0(&ref_plane, width, height, bpc);
        let dis0 = dwt_scale0(&dis_plane, width, height, bpc);

        let mut cur_ref_ll: Vec<i32> = ref0.a.iter().map(|&x| x as i32).collect();
        let mut cur_dis_ll: Vec<i32> = dis0.a.iter().map(|&x| x as i32).collect();
        let mut cur_w = ref0.width;
        let mut cur_h = ref0.height;

        for scale in 1..=3usize {
            let ref_s = dwt_s123(&cur_ref_ll, cur_w, cur_h, scale);
            let dis_s = dwt_s123(&cur_dis_ll, cur_w, cur_h, scale);

            let actual = score_scale_s123(
                &ref_s.h,
                &ref_s.v,
                &ref_s.d,
                &dis_s.h,
                &dis_s.v,
                &dis_s.d,
                limit,
                scale,
                ref_s.width,
                ref_s.height,
            );
            let expected = score_scale_s123_reference(
                &ref_s.h,
                &ref_s.v,
                &ref_s.d,
                &dis_s.h,
                &dis_s.v,
                &dis_s.d,
                limit,
                scale,
                ref_s.width,
                ref_s.height,
            );

            assert_eq!(
                actual.0.to_bits(),
                expected.0.to_bits(),
                "num mismatch at scale {scale}"
            );
            assert_eq!(
                actual.1.to_bits(),
                expected.1.to_bits(),
                "den mismatch at scale {scale}"
            );

            cur_ref_ll = ref_s.a;
            cur_dis_ll = dis_s.a;
            cur_w = ref_s.width;
            cur_h = ref_s.height;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn adm_extractor_backend_matches_scalar() {
        let requested = if SimdBackend::X86Avx2Fma.is_available() {
            SimdBackend::X86Avx2Fma
        } else if SimdBackend::X86Sse2.is_available() {
            SimdBackend::X86Sse2
        } else {
            return;
        };

        let width = 27;
        let height = 19;
        let bpc = 10;
        let max_value = (1u16 << bpc) - 1;
        let ref_plane = plane_pattern(width, height, max_value);
        let mut dis_plane = plane_pattern(width, height, max_value);
        for (idx, value) in dis_plane.iter_mut().enumerate() {
            let delta = ((idx * 7 + 13) % 19) as u16;
            *value = value.saturating_sub(delta / 2).min(max_value);
        }

        let scalar =
            AdmExtractor::with_backend_for_tests(width, height, bpc, 100.0, SimdBackend::Scalar);
        let simd_backend =
            AdmExtractor::with_backend_for_tests(width, height, bpc, 100.0, requested);

        assert_eq!(
            simd_backend.compute_frame(&ref_plane, &dis_plane).to_bits(),
            scalar.compute_frame(&ref_plane, &dis_plane).to_bits(),
        );
    }
}
