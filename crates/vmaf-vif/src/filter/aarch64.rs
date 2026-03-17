use vmaf_cpu::SimdBackend;

use super::SubsampleWorkspace;

pub(super) fn subsample_into(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    backend: SimdBackend,
    workspace: &mut SubsampleWorkspace,
    out_ref: &mut Vec<u16>,
    out_dis: &mut Vec<u16>,
) -> (usize, usize) {
    let _ = backend;
    // NEON hooks will plug into the same dispatch shape; for now preserve exact
    // scalar behavior while reusing the caller-owned workspace.
    super::subsample_scalar_into(
        ref_in, dis_in, width, height, bpc, scale, workspace, out_ref, out_dis,
    )
}
