use bytemuck::{Pod, PodCastError};

/// Attempts to reinterpret an aligned reference as another plain-data type.
pub fn try_cast_ref<Src: Pod, Dst: Pod>(value: &Src) -> Result<&Dst, PodCastError> {
    bytemuck::try_cast_ref(value)
}

/// Attempts to reinterpret an aligned mutable reference as another plain-data
/// type.
pub fn try_cast_mut<Src: Pod, Dst: Pod>(value: &mut Src) -> Result<&mut Dst, PodCastError> {
    bytemuck::try_cast_mut(value)
}

/// Attempts to reinterpret a slice as another plain-data slice when length and
/// alignment are compatible.
pub fn try_cast_slice<Src: Pod, Dst: Pod>(slice: &[Src]) -> Result<&[Dst], PodCastError> {
    bytemuck::try_cast_slice(slice)
}

/// Attempts to reinterpret a mutable slice as another plain-data slice when
/// length and alignment are compatible.
pub fn try_cast_slice_mut<Src: Pod, Dst: Pod>(
    slice: &mut [Src],
) -> Result<&mut [Dst], PodCastError> {
    bytemuck::try_cast_slice_mut(slice)
}

/// Splits a borrowed slice into a misaligned prefix, aligned body, and suffix.
///
/// This is the preferred pattern when VMAF cannot control the original buffer
/// alignment but still wants to process the aligned middle region efficiently.
pub fn align_to_slice<Src: Pod, Dst: Pod>(slice: &[Src]) -> (&[Src], &[Dst], &[Src]) {
    bytemuck::pod_align_to(slice)
}

/// Mutable version of [`align_to_slice`].
pub fn align_to_slice_mut<Src: Pod, Dst: Pod>(
    slice: &mut [Src],
) -> (&mut [Src], &mut [Dst], &mut [Src]) {
    bytemuck::pod_align_to_mut(slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Align16, AlignedBlock};

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
