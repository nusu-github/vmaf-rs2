//! AArch64 NEON hook.
//!
//! The current ADM SIMD implementation is x86/x86_64-first. This module keeps
//! the dispatch shape ready for future NEON kernels while preserving the same
//! caller-owned workspaces today.

use crate::dwt::{
    dwt_s123_into as scalar_dwt_s123_into, dwt_scale0_into as scalar_dwt_scale0_into,
    Bands16Buffer, Bands32Buffer, Scale0DwtWorkspace, Scale123DwtWorkspace,
};

pub(crate) fn dwt_scale0_into(
    src: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    workspace: &mut Scale0DwtWorkspace,
    bands: &mut Bands16Buffer,
) {
    scalar_dwt_scale0_into(src, width, height, bpc, workspace, bands)
}

pub(crate) fn dwt_s123_into(
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
    workspace: &mut Scale123DwtWorkspace,
    bands: &mut Bands32Buffer,
) {
    scalar_dwt_s123_into(ll, width, height, scale, workspace, bands)
}
