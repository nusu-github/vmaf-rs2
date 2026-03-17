use vmaf_cpu::SimdBackend;

pub(super) fn subsample(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    backend: SimdBackend,
) -> (Vec<u16>, Vec<u16>, usize, usize) {
    let _ = backend;
    // NEON hooks will plug into the same dispatch shape; for now preserve exact
    // scalar behavior.
    super::subsample_scalar(ref_in, dis_in, width, height, bpc, scale)
}
