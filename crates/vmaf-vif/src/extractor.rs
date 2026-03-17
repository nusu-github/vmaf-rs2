//! VIF feature extractor — spec §4.2

use crate::filter::subsample;
use crate::stat::vif_statistic;
use vmaf_cpu::SimdBackend;

/// Per-frame VIF scores — 4 scales + combined.
pub struct VifScores {
    pub scale: [f64; 4],
    pub combined: f64,
}

/// Stateless VIF extractor: computes all 4 scale scores for one frame pair.
///
/// The extractor auto-selects the best available SIMD backend at construction
/// time while preserving the existing public API.
pub struct VifExtractor {
    width: usize,
    height: usize,
    bpc: u8,
    vif_enhn_gain_limit: f64,
    backend: SimdBackend,
}

impl VifExtractor {
    pub fn new(width: usize, height: usize, bpc: u8, vif_enhn_gain_limit: f64) -> Self {
        Self::with_backend(
            width,
            height,
            bpc,
            vif_enhn_gain_limit,
            SimdBackend::detect(),
        )
    }

    pub(crate) fn with_backend(
        width: usize,
        height: usize,
        bpc: u8,
        vif_enhn_gain_limit: f64,
        backend: SimdBackend,
    ) -> Self {
        Self {
            width,
            height,
            bpc,
            vif_enhn_gain_limit,
            backend: effective_backend(backend),
        }
    }

    /// Compute VIF scores for one ref/dis frame pair — spec §4.2.
    ///
    /// `ref_plane` and `dis_plane` are row-major luma planes (`width × height`).
    pub fn compute_frame(&self, ref_plane: &[u16], dis_plane: &[u16]) -> VifScores {
        let (w, h, bpc) = (self.width, self.height, self.bpc);
        let limit = self.vif_enhn_gain_limit;
        let backend = self.backend;

        let mut nums = [0.0f64; 4];
        let mut dens = [0.0f64; 4];

        let s0 = vif_statistic(ref_plane, dis_plane, w, h, bpc, 0, limit, backend);
        nums[0] = s0.num;
        dens[0] = s0.den;

        let mut cur_ref = ref_plane.to_vec();
        let mut cur_dis = dis_plane.to_vec();
        let mut cur_w = w;
        let mut cur_h = h;

        for scale in 0..3usize {
            let (next_ref, next_dis, next_w, next_h) =
                subsample(&cur_ref, &cur_dis, cur_w, cur_h, bpc, scale, backend);

            let ss = vif_statistic(
                &next_ref,
                &next_dis,
                next_w,
                next_h,
                bpc,
                scale + 1,
                limit,
                backend,
            );
            nums[scale + 1] = ss.num;
            dens[scale + 1] = ss.den;

            cur_ref = next_ref;
            cur_dis = next_dis;
            cur_w = next_w;
            cur_h = next_h;
        }

        let scale_scores = std::array::from_fn(|s| {
            if dens[s] > 0.0 {
                nums[s] / dens[s]
            } else {
                1.0
            }
        });

        let total_num: f64 = nums.iter().sum();
        let total_den: f64 = dens.iter().sum();
        let combined = if total_den > 0.0 {
            total_num / total_den
        } else {
            1.0
        };

        VifScores {
            scale: scale_scores,
            combined,
        }
    }
}

fn effective_backend(backend: SimdBackend) -> SimdBackend {
    if !backend.is_available() {
        return SimdBackend::Scalar;
    }

    match backend {
        SimdBackend::X86Avx512 => {
            if SimdBackend::X86Avx2Fma.is_available() {
                SimdBackend::X86Avx2Fma
            } else if SimdBackend::X86Sse2.is_available() {
                SimdBackend::X86Sse2
            } else {
                SimdBackend::Scalar
            }
        }
        other => other,
    }
}
