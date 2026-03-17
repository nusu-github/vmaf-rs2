//! Integer VIF (Visual Information Fidelity) — spec §4.2
//!
//! Outputs 4 per-scale scores (`vif_scale{0..3}`) and one combined score.
//! All transcendental math is handled via a precomputed Q11 log2 LUT (§4.2.1).

#![allow(clippy::needless_range_loop)]

mod extractor;
mod filter;
mod math;
mod stat;
mod tables;

pub use extractor::{VifExtractor, VifScores, VifWorkspace};

#[cfg(test)]
mod tests {
    use super::extractor::VifExtractor;
    use super::filter::subsample;
    use super::math::{log2_32, log2_64, reflect_index};
    use super::stat::vif_statistic;
    use super::tables::LOG2_TABLE;
    use vmaf_cpu::SimdBackend;

    fn patterned_plane(width: usize, height: usize, modulus: u16, bias: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    ((x * 19 + y * 23 + ((x ^ y) * 7) + (x * y * 3) + bias) % modulus as usize)
                        as u16
                })
            })
            .collect()
    }

    fn assert_scores_match(
        expected: &super::extractor::VifScores,
        actual: &super::extractor::VifScores,
    ) {
        for scale in 0..4 {
            assert_eq!(
                expected.scale[scale].to_bits(),
                actual.scale[scale].to_bits()
            );
        }
        assert_eq!(expected.combined.to_bits(), actual.combined.to_bits());
    }

    fn assert_scale_stat_match(expected: &super::stat::ScaleStat, actual: &super::stat::ScaleStat) {
        assert_eq!(expected.num.to_bits(), actual.num.to_bits());
        assert_eq!(expected.den.to_bits(), actual.den.to_bits());
    }

    // ── log2_table §8 conformance ─────────────────────────────────────────────

    #[test]
    fn log2_table_reference_vectors() {
        assert_eq!(LOG2_TABLE[32768], 30720);
        assert_eq!(LOG2_TABLE[49152], 31918);
        assert_eq!(LOG2_TABLE[65535], 32768);
    }

    // ── reflect_index ─────────────────────────────────────────────────────────

    #[test]
    fn reflect_index_in_range() {
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(3, 5), 3);
        assert_eq!(reflect_index(4, 5), 4);
    }

    #[test]
    fn reflect_index_negative() {
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
    }

    #[test]
    fn reflect_index_overflow() {
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
    }

    // ── log2_32 ───────────────────────────────────────────────────────────────

    #[test]
    fn log2_32_known_power_of_two() {
        assert_eq!(log2_32(&LOG2_TABLE, 131072), 34816);
        assert_eq!(log2_32(&LOG2_TABLE, 262144), 36864);
    }

    // ── log2_64 ───────────────────────────────────────────────────────────────

    #[test]
    fn log2_64_known_power_of_two() {
        assert_eq!(log2_64(&LOG2_TABLE, 131072), 34816);
        assert_eq!(log2_64(&LOG2_TABLE, 1 << 20), 40960);
    }

    // ── subsample ─────────────────────────────────────────────────────────────

    #[test]
    fn subsample_halves_dimensions() {
        let w = 32;
        let h = 32;
        let frame = vec![128u16; w * h];
        let (_, _, ow, oh) = subsample(&frame, &frame, w, h, 8, 0, SimdBackend::Scalar);
        assert_eq!(ow, 16);
        assert_eq!(oh, 16);
    }

    #[test]
    fn subsample_flat_produces_flat() {
        let w = 32;
        let h = 32;
        let ref_f = vec![128u16; w * h];
        let dis_f = vec![200u16; w * h];
        let (out_r, out_d, _, _) = subsample(&ref_f, &dis_f, w, h, 8, 0, SimdBackend::Scalar);
        let v_r = out_r[0];
        let v_d = out_d[0];
        assert!(
            out_r.iter().all(|&x| x == v_r),
            "ref subsampled not uniform"
        );
        assert!(
            out_d.iter().all(|&x| x == v_d),
            "dis subsampled not uniform"
        );
    }

    // ── VifExtractor ──────────────────────────────────────────────────────────

    #[test]
    fn vif_identical_flat_frames_score_one() {
        let w = 32;
        let h = 32;
        let frame = vec![128u16; w * h];
        let ext = VifExtractor::with_backend(w, h, 8, 100.0, SimdBackend::Scalar);
        let s = ext.compute_frame(&frame, &frame);
        for sc in 0..4 {
            assert!(
                (s.scale[sc] - 1.0).abs() < 1e-9,
                "scale {sc}: {}",
                s.scale[sc]
            );
        }
        assert!((s.combined - 1.0).abs() < 1e-9, "combined: {}", s.combined);
    }

    #[test]
    fn vif_identical_gradient_frames_score_near_one() {
        let w = 32;
        let h = 32;
        let frame: Vec<u16> = (0..h).flat_map(|_| 0u16..w as u16).collect();
        let ext = VifExtractor::with_backend(w, h, 8, 100.0, SimdBackend::Scalar);
        let s = ext.compute_frame(&frame, &frame);
        for sc in 0..4 {
            assert!(s.scale[sc] >= 0.99, "scale {sc}: {}", s.scale[sc]);
            assert!(s.scale[sc] <= 1.01, "scale {sc}: {}", s.scale[sc]);
        }
        assert!(
            s.combined >= 0.99 && s.combined <= 1.01,
            "combined: {}",
            s.combined
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn subsample_detected_backend_matches_scalar() {
        let backend = SimdBackend::detect();
        if matches!(backend, SimdBackend::Scalar) {
            return;
        }

        let w = 23;
        let h = 19;
        let ref_plane = patterned_plane(w, h, 1024, 11);
        let dis_plane = patterned_plane(w, h, 1024, 97);
        let scalar = subsample(&ref_plane, &dis_plane, w, h, 10, 0, SimdBackend::Scalar);
        let simd = subsample(&ref_plane, &dis_plane, w, h, 10, 0, backend);
        assert_eq!(scalar, simd);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn vif_statistic_detected_backend_matches_scalar_non_wrapping_scale0() {
        let backend = SimdBackend::detect();
        if matches!(backend, SimdBackend::Scalar) {
            return;
        }

        let w = 21;
        let h = 17;
        let ref_plane = patterned_plane(w, h, 1024, 19);
        let dis_plane = patterned_plane(w, h, 1024, 41);
        let scalar = vif_statistic(
            &ref_plane,
            &dis_plane,
            w,
            h,
            10,
            0,
            100.0,
            SimdBackend::Scalar,
        );
        let simd = vif_statistic(&ref_plane, &dis_plane, w, h, 10, 0, 100.0, backend);
        assert_scale_stat_match(&scalar, &simd);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn vif_statistic_detected_backend_matches_scalar_wrapping_scale0() {
        let backend = SimdBackend::detect();
        if matches!(backend, SimdBackend::Scalar) {
            return;
        }

        let w = 27;
        let h = 17;
        let ref_plane = patterned_plane(w, h, 256, 5);
        let dis_plane = patterned_plane(w, h, 256, 133);
        let scalar = vif_statistic(
            &ref_plane,
            &dis_plane,
            w,
            h,
            8,
            0,
            100.0,
            SimdBackend::Scalar,
        );
        let simd = vif_statistic(&ref_plane, &dis_plane, w, h, 8, 0, 100.0, backend);
        assert_scale_stat_match(&scalar, &simd);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn vif_statistic_explicit_avx2_matches_scalar_on_misaligned_odd_non_wrapping_scale0() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let w = 29usize;
        let h = 13usize;
        let bpc = 12u8;
        let modulus = 1usize << bpc;
        let len = w * h;
        let mut ref_storage = vec![0u16; len + 5];
        let mut dis_storage = vec![0u16; len + 7];
        let ref_plane = &mut ref_storage[1..1 + len];
        let dis_plane = &mut dis_storage[3..3 + len];
        ref_plane.copy_from_slice(&patterned_plane(w, h, modulus as u16, 19));
        dis_plane.copy_from_slice(&patterned_plane(w, h, modulus as u16, 71));

        let scalar = vif_statistic(
            ref_plane,
            dis_plane,
            w,
            h,
            bpc,
            0,
            100.0,
            SimdBackend::Scalar,
        );
        let avx2 = vif_statistic(
            ref_plane,
            dis_plane,
            w,
            h,
            bpc,
            0,
            100.0,
            SimdBackend::X86Avx2Fma,
        );
        assert_scale_stat_match(&scalar, &avx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn vif_statistic_explicit_avx2_matches_scalar_on_misaligned_odd_wrapping_scale0() {
        if !SimdBackend::X86Avx2Fma.is_available() {
            return;
        }

        let w = 27usize;
        let h = 17usize;
        let len = w * h;
        let mut ref_storage = vec![0u16; len + 7];
        let mut dis_storage = vec![0u16; len + 9];
        let ref_plane = &mut ref_storage[3..3 + len];
        let dis_plane = &mut dis_storage[1..1 + len];
        ref_plane.copy_from_slice(&patterned_plane(w, h, 256, 5));
        dis_plane.copy_from_slice(&patterned_plane(w, h, 256, 133));

        let scalar = vif_statistic(ref_plane, dis_plane, w, h, 8, 0, 100.0, SimdBackend::Scalar);
        let avx2 = vif_statistic(
            ref_plane,
            dis_plane,
            w,
            h,
            8,
            0,
            100.0,
            SimdBackend::X86Avx2Fma,
        );
        assert_scale_stat_match(&scalar, &avx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn extractor_detected_backend_matches_scalar() {
        let backend = SimdBackend::detect();
        if matches!(backend, SimdBackend::Scalar) {
            return;
        }

        let w = 46;
        let h = 38;
        let ref_plane = patterned_plane(w, h, 1024, 3);
        let dis_plane = patterned_plane(w, h, 1024, 67);
        let scalar = VifExtractor::with_backend(w, h, 10, 100.0, SimdBackend::Scalar);
        let simd = VifExtractor::with_backend(w, h, 10, 100.0, backend);
        let scalar_scores = scalar.compute_frame(&ref_plane, &dis_plane);
        let simd_scores = simd.compute_frame(&ref_plane, &dis_plane);
        assert_scores_match(&scalar_scores, &simd_scores);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn extractor_explicit_backends_match_scalar_on_misaligned_odd_12bit_pyramid() {
        let w = 19usize;
        let h = 17usize;
        let bpc = 12u8;
        let modulus = 1usize << bpc;
        let len = w * h;
        let mut ref_storage = vec![0u16; len + 3];
        let mut dis_storage = vec![0u16; len + 5];
        let ref_plane = &mut ref_storage[1..1 + len];
        let dis_plane = &mut dis_storage[3..3 + len];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let reference =
                    (x * 37 + y * 61 + ((x ^ y) * 17) + (x * y * 5) + idx * 3) % modulus;
                let delta = ((x * 11 + y * 7 + idx * 13) % 43) as u16;
                ref_plane[idx] = reference as u16;
                dis_plane[idx] = (reference as u16).saturating_sub(delta);
            }
        }

        let scalar = VifExtractor::with_backend(w, h, bpc, 100.0, SimdBackend::Scalar);
        let expected = scalar.compute_frame(ref_plane, dis_plane);

        for backend in [SimdBackend::X86Sse2, SimdBackend::X86Avx2Fma] {
            if !backend.is_available() {
                continue;
            }

            let simd = VifExtractor::with_backend(w, h, bpc, 100.0, backend);
            let actual = simd.compute_frame(ref_plane, dis_plane);
            assert_scores_match(&expected, &actual);
        }
    }
}
