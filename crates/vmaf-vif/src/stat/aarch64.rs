use vmaf_cpu::SimdBackend;

use super::ScaleStat;

pub(super) fn vif_statistic(
    ref_plane: &[u16],
    dis_plane: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    vif_enhn_gain_limit: f64,
    backend: SimdBackend,
) -> ScaleStat {
    let _ = backend;
    // Future NEON kernels can reuse this dispatch entry point without reshaping
    // the extractor or scalar fallbacks.
    super::vif_statistic_scalar(
        ref_plane,
        dis_plane,
        width,
        height,
        bpc,
        scale,
        vif_enhn_gain_limit,
    )
}
