//! Shared SIMD backend selection, alignment helpers, and checked casting
//! patterns for future VMAF kernels.
//!
//! The initial first-class target is `x86`/`x86_64`, using `cpufeatures` for
//! runtime backend selection. `aarch64` is modeled as an extension hook so
//! future NEON kernels can plug into the same backend and buffer APIs without
//! reshaping downstream code.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod alignment;
mod backend;
mod cast;
mod probe;

pub use alignment::{
    Align16, Align32, Align64, AlignedAllocError, AlignedBlock, AlignedScratch, Alignment,
    ConstAlign16, ConstAlign32, ConstAlign64, assume_init_slice, avec_assume_init, avec_uninit,
    avec_uninit_32, avec_uninit_64, avec_zeroed, avec_zeroed_32, avec_zeroed_64, try_avec_uninit,
    try_avec_zeroed,
};
pub use backend::SimdBackend;
pub use bytemuck::{AnyBitPattern, NoUninit, Pod, PodCastError, Zeroable};
pub use cast::{
    align_to_slice, align_to_slice_mut, try_cast_mut, try_cast_ref, try_cast_slice,
    try_cast_slice_mut,
};
