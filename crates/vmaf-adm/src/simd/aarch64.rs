//! AArch64 NEON hook.
//!
//! The current ADM SIMD implementation is x86/x86_64-first. This module keeps
//! the dispatch shape ready for future NEON kernels while preserving the same
//! call sites today.

use crate::dwt::{dwt_s123 as scalar_dwt_s123, dwt_scale0 as scalar_dwt_scale0, Bands16, Bands32};

pub(crate) fn dwt_scale0(src: &[u16], width: usize, height: usize, bpc: u8) -> Bands16 {
    scalar_dwt_scale0(src, width, height, bpc)
}

pub(crate) fn dwt_s123(ll: &[i32], width: usize, height: usize, scale: usize) -> Bands32 {
    scalar_dwt_s123(ll, width, height, scale)
}
