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
mod tables;

pub use extractor::AdmExtractor;

#[cfg(test)]
mod tests {
    use super::decouple::decouple_scale0;
    use super::dwt::get_best15_from32;
    use super::math::reflect_index;
    use super::tables::DIV_LOOKUP;

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
}
