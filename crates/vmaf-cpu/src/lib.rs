//! Shared SIMD backend selection, alignment helpers, and reusable validation
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
mod math;
mod probe;
mod validation;

pub use alignment::{
    Align16, Align32, Align64, AlignedAllocError, AlignedBlock, AlignedScratch, Alignment,
    ConstAlign16, ConstAlign32, ConstAlign64, assume_init_slice, avec_assume_init, avec_uninit,
    avec_uninit_32, avec_uninit_64, avec_zeroed, avec_zeroed_32, avec_zeroed_64, try_avec_uninit,
    try_avec_zeroed,
};
pub use backend::SimdBackend;
/// Splits a borrowed slice into a misaligned prefix, aligned body, and suffix.
///
/// This is the preferred pattern when VMAF cannot control the original buffer
/// alignment but still wants to process the aligned middle region efficiently.
pub use bytemuck::pod_align_to as align_to_slice;
/// Mutable version of [`align_to_slice`].
pub use bytemuck::pod_align_to_mut as align_to_slice_mut;
/// Attempts to reinterpret an aligned mutable reference as another plain-data type.
pub use bytemuck::try_cast_mut;
/// Attempts to reinterpret an aligned reference as another plain-data type.
pub use bytemuck::try_cast_ref;
/// Attempts to reinterpret a slice as another plain-data slice when length and
/// alignment are compatible.
pub use bytemuck::try_cast_slice;
/// Attempts to reinterpret a mutable slice as another plain-data slice when
/// length and alignment are compatible.
pub use bytemuck::try_cast_slice_mut;
pub use bytemuck::{AnyBitPattern, NoUninit, Pod, PodCastError, Zeroable};
pub use math::reflect_index;
pub use validation::{
    FrameGeometry, FrameValidationError, GainLimit, GainLimitError, MIN_FRAME_DIMENSION,
    checked_sample_count, validate_frame_geometry,
};

#[cfg(test)]
mod tests {
    use super::{Align16, AlignedBlock, align_to_slice, try_cast_slice, try_cast_slice_mut};

    #[test]
    fn checked_cast_accepts_aligned_storage() {
        let block = AlignedBlock::<[u8; 16], Align16>::new([0; 16]);
        let words = try_cast_slice::<u8, u32>(&block.as_ref()[..]).unwrap();

        assert_eq!(words.len(), 4);
    }

    #[test]
    fn checked_cast_rejects_misaligned_storage() {
        let block = AlignedBlock::<[u8; 9], Align16>::new([0; 9]);

        assert!(try_cast_slice::<u8, u32>(&block.as_ref()[1..]).is_err());
    }

    #[test]
    fn checked_cast_mut_updates_underlying_bytes() {
        let mut block = AlignedBlock::<[u8; 8], Align16>::new([0; 8]);
        let words = try_cast_slice_mut::<u8, u32>(&mut block.as_mut()[..]).unwrap();

        words[1] = u32::from_ne_bytes([1, 2, 3, 4]);
        assert_eq!(&block.as_ref()[4..8], &[1, 2, 3, 4]);
    }

    #[test]
    fn align_to_slice_returns_aligned_middle() {
        let block = AlignedBlock::<[u8; 17], Align16>::new([0; 17]);
        let (prefix, words, suffix) = align_to_slice::<u8, u32>(&block.as_ref()[1..]);

        assert_eq!(prefix.len(), 3);
        assert_eq!(words.len(), 3);
        assert_eq!(suffix.len(), 1);
    }
}
