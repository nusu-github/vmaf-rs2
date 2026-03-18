use vmaf_cpu::SimdBackend;

use super::{ScaleStat, VifGainLimitMode, VifStatWorkspace};

pub(super) fn vif_statistic(
    ref_plane: &[u16],
    dis_plane: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    vif_gain_limit_mode: VifGainLimitMode,
    workspace: &mut VifStatWorkspace,
    backend: SimdBackend,
) -> ScaleStat {
    let _ = backend;
    // Future NEON kernels can reuse this dispatch entry point without reshaping
    // the extractor or scalar fallbacks.
    super::vif_statistic_scalar_with_workspace(
        ref_plane,
        dis_plane,
        width,
        height,
        bpc,
        scale,
        vif_gain_limit_mode,
        workspace,
    )
}
