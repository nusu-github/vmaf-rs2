//! Integer VIF (Visual Information Fidelity) — spec §4.2
//!
//! Outputs 4 per-scale scores (`vif_scale{0..3}`) and one combined score.
//! All transcendental math is handled via a precomputed Q11 log2 LUT (§4.2.1).

mod extractor;
mod filter;
mod math;
mod stat;
mod tables;

pub use extractor::{VifExtractor, VifScores};

#[cfg(test)]
mod tests {
    use super::extractor::VifExtractor;
    use super::filter::subsample;
    use super::math::{log2_32, log2_64, reflect_index};
    use super::tables::LOG2_TABLE;

    // ── log2_table §8 conformance ─────────────────────────────────────────────

    /// §8 conformance vectors for log2_table.
    /// CRITICAL: must use f32 (not f64) log2 — spec §4.2.1.
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
        assert_eq!(reflect_index(5, 5), 3); // 2*5 - 2 - 5 = 3
        assert_eq!(reflect_index(6, 5), 2); // 2*5 - 2 - 6 = 2
    }

    // ── log2_32 ───────────────────────────────────────────────────────────────

    /// log2_32 — spec §4.2.2
    #[test]
    fn log2_32_known_power_of_two() {
        assert_eq!(log2_32(&LOG2_TABLE, 131072), 34816); // log2(2^17) * 2048 = 34816
        assert_eq!(log2_32(&LOG2_TABLE, 262144), 36864); // log2(2^18) * 2048 = 36864
    }

    // ── log2_64 ───────────────────────────────────────────────────────────────

    /// log2_64 — spec §4.2.3
    #[test]
    fn log2_64_known_power_of_two() {
        assert_eq!(log2_64(&LOG2_TABLE, 131072), 34816);
        assert_eq!(log2_64(&LOG2_TABLE, 1 << 20), 40960); // log2(2^20) * 2048 = 40960
    }

    // ── subsample ─────────────────────────────────────────────────────────────

    /// Subsampled output must be half the input dimensions (integer division).
    #[test]
    fn subsample_halves_dimensions() {
        let w = 32;
        let h = 32;
        let frame = vec![128u16; w * h];
        let (_, _, ow, oh) = subsample(&frame, &frame, w, h, 8, 0);
        assert_eq!(ow, 16);
        assert_eq!(oh, 16);
    }

    /// Subsampling a uniform frame produces a uniform output (same value across all pixels).
    #[test]
    fn subsample_flat_produces_flat() {
        let w = 32;
        let h = 32;
        let ref_f = vec![128u16; w * h];
        let dis_f = vec![200u16; w * h];
        let (out_r, out_d, _, _) = subsample(&ref_f, &dis_f, w, h, 8, 0);
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

    /// Identical flat frames → every scale score == 1.0.
    ///
    /// For constant frames: sigma1_sq = sigma2_sq = sigma12 = 0 for all pixels.
    /// All pixels take the non-log path (sigma1_sq < SIGMA_NSQ).
    /// non_log penalty = 0 / 16384 / 65025 = 0.
    /// num = den = H*W → score = 1.0.
    #[test]
    fn vif_identical_flat_frames_score_one() {
        let w = 32;
        let h = 32;
        let frame = vec![128u16; w * h];
        let ext = VifExtractor::new(w, h, 8, 100.0);
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

    /// Identical non-flat frames → scores close to 1.0.
    ///
    /// For identical ref/dis: sigma12 = sigma1_sq and sigma2_sq = sigma1_sq.
    /// The log-path gain g ≈ 1, sv_sq ≈ 0, numer1_tmp ≈ numer1 → num ≈ den.
    #[test]
    fn vif_identical_gradient_frames_score_near_one() {
        let w = 32;
        let h = 32;
        // Horizontal ramp 0..=31, tiled across rows
        let frame: Vec<u16> = (0..h).flat_map(|_| 0u16..w as u16).collect();
        let ext = VifExtractor::new(w, h, 8, 100.0);
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
}
