use std::{
    alloc::handle_alloc_error,
    fmt,
    marker::PhantomData,
    mem::{MaybeUninit, align_of},
    ops::{Deref, DerefMut},
};

use aligned_vec::{AVec, RuntimeAlign, TryReserveError};
use bytemuck::Zeroable;
use thiserror::Error;

/// Compile-time `16`-byte alignment for [`AVec`].
pub type ConstAlign16 = aligned_vec::ConstAlign<16>;
/// Compile-time `32`-byte alignment for [`AVec`].
pub type ConstAlign32 = aligned_vec::ConstAlign<32>;
/// Compile-time `64`-byte alignment for [`AVec`].
pub type ConstAlign64 = aligned_vec::ConstAlign<64>;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for explicit storage alignments owned by VMAF.
///
/// Use [`AlignedBlock`] for fixed-size storage and [`AlignedScratch`] for
/// callers that still want a stable, alignment-parametrized scratch wrapper.
pub trait Alignment: Copy + Default + fmt::Debug + 'static + sealed::Sealed {
    /// Requested alignment in bytes.
    const BYTES: usize;
}

macro_rules! define_alignment {
    ($name:ident, $bytes:literal) => {
        #[doc = concat!("Alignment marker for ", stringify!($bytes), "-byte storage.")]
        #[repr(align($bytes))]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl Alignment for $name {
            const BYTES: usize = $bytes;
        }
    };
}

define_alignment!(Align16, 16);
define_alignment!(Align32, 32);
define_alignment!(Align64, 64);

/// Wraps a value in an explicitly aligned block.
///
/// This is best suited to fixed-size storage such as coefficient tables,
/// staging blocks, or compile-time scratch regions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AlignedBlock<T, A: Alignment> {
    _align: [A; 0],
    value: T,
}

impl<T, A: Alignment> AlignedBlock<T, A> {
    /// Creates a new aligned block.
    pub fn new(value: T) -> Self {
        Self { _align: [], value }
    }

    /// Returns the effective alignment in bytes.
    pub fn alignment() -> usize {
        align_of::<Self>()
    }

    /// Returns a raw pointer to the wrapped value.
    pub fn as_ptr(&self) -> *const T {
        &self.value
    }

    /// Returns a mutable raw pointer to the wrapped value.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        &mut self.value
    }

    /// Consumes the block and returns the wrapped value.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T, A: Alignment> From<T> for AlignedBlock<T, A> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T, A: Alignment> AsRef<T> for AlignedBlock<T, A> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T, A: Alignment> AsMut<T> for AlignedBlock<T, A> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T, A: Alignment> Deref for AlignedBlock<T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T, A: Alignment> DerefMut for AlignedBlock<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

/// Errors that can occur while creating aligned buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum AlignedAllocError {
    /// The requested allocation would overflow buffer layout bounds.
    #[error("aligned allocation layout overflow")]
    LayoutOverflow,
    /// The allocator failed for the requested buffer layout.
    #[error("aligned allocation failed for {size} bytes with {align}-byte alignment")]
    AllocationFailed {
        /// Requested allocation size in bytes.
        size: usize,
        /// Requested allocation alignment in bytes.
        align: usize,
    },
}

fn map_try_reserve_error(error: TryReserveError) -> AlignedAllocError {
    match error {
        TryReserveError::CapacityOverflow => AlignedAllocError::LayoutOverflow,
        TryReserveError::AllocError { layout } => AlignedAllocError::AllocationFailed {
            size: layout.size(),
            align: layout.align(),
        },
    }
}

fn allocation_layout_error(len: usize) -> ! {
    panic!("aligned scratch layout overflow for {len} elements")
}

fn abort_allocation(error: AlignedAllocError, len: usize) -> ! {
    match error {
        AlignedAllocError::LayoutOverflow => allocation_layout_error(len),
        AlignedAllocError::AllocationFailed { size, align } => {
            let layout = std::alloc::Layout::from_size_align(size, align)
                .expect("saved allocation layout is valid");
            handle_alloc_error(layout);
        }
    }
}

fn try_runtime_uninit_vec<T>(
    len: usize,
    alignment: usize,
) -> Result<AVec<MaybeUninit<T>, RuntimeAlign>, AlignedAllocError> {
    let mut values = AVec::<MaybeUninit<T>, RuntimeAlign>::new(alignment);
    values
        .try_reserve_exact(len)
        .map_err(map_try_reserve_error)?;
    // SAFETY: capacity was reserved for `len` elements.
    unsafe {
        values.set_len(len);
    }
    Ok(values)
}

/// Allocates a zeroed aligned vector.
pub fn try_avec_zeroed<T: Zeroable, const ALIGN: usize>(
    len: usize,
) -> Result<AVec<T, aligned_vec::ConstAlign<ALIGN>>, AlignedAllocError> {
    let mut values = AVec::<MaybeUninit<T>, aligned_vec::ConstAlign<ALIGN>>::new(ALIGN);
    values
        .try_reserve_exact(len)
        .map_err(map_try_reserve_error)?;
    // SAFETY: capacity was reserved for `len` elements and all bytes are zeroed
    // before the vector is converted into initialized storage.
    unsafe {
        values.set_len(len);
        std::ptr::write_bytes(values.as_mut_ptr(), 0, len);
        Ok(avec_assume_init(values))
    }
}

/// Allocates a zeroed aligned vector or aborts on OOM.
pub fn avec_zeroed<T: Zeroable, const ALIGN: usize>(
    len: usize,
) -> AVec<T, aligned_vec::ConstAlign<ALIGN>> {
    match try_avec_zeroed::<T, ALIGN>(len) {
        Ok(values) => values,
        Err(error) => abort_allocation(error, len),
    }
}

/// Allocates a zeroed `32`-byte aligned vector or aborts on OOM.
pub fn avec_zeroed_32<T: Zeroable>(len: usize) -> AVec<T, ConstAlign32> {
    avec_zeroed::<T, 32>(len)
}

/// Allocates a zeroed `64`-byte aligned vector or aborts on OOM.
pub fn avec_zeroed_64<T: Zeroable>(len: usize) -> AVec<T, ConstAlign64> {
    avec_zeroed::<T, 64>(len)
}

/// Allocates an uninitialized aligned vector.
pub fn try_avec_uninit<T, const ALIGN: usize>(
    len: usize,
) -> Result<AVec<MaybeUninit<T>, aligned_vec::ConstAlign<ALIGN>>, AlignedAllocError> {
    let mut values = AVec::<MaybeUninit<T>, aligned_vec::ConstAlign<ALIGN>>::new(ALIGN);
    values
        .try_reserve_exact(len)
        .map_err(map_try_reserve_error)?;
    // SAFETY: capacity was reserved for `len` elements, so setting the logical
    // length is valid and exposes `MaybeUninit<T>` slots to the caller.
    unsafe {
        values.set_len(len);
    }
    Ok(values)
}

/// Allocates an uninitialized aligned vector or aborts on OOM.
pub fn avec_uninit<T, const ALIGN: usize>(
    len: usize,
) -> AVec<MaybeUninit<T>, aligned_vec::ConstAlign<ALIGN>> {
    match try_avec_uninit::<T, ALIGN>(len) {
        Ok(values) => values,
        Err(error) => abort_allocation(error, len),
    }
}

/// Allocates an uninitialized `32`-byte aligned vector or aborts on OOM.
pub fn avec_uninit_32<T>(len: usize) -> AVec<MaybeUninit<T>, ConstAlign32> {
    avec_uninit::<T, 32>(len)
}

/// Allocates an uninitialized `64`-byte aligned vector or aborts on OOM.
pub fn avec_uninit_64<T>(len: usize) -> AVec<MaybeUninit<T>, ConstAlign64> {
    avec_uninit::<T, 64>(len)
}

/// Converts an uninitialized aligned vector into initialized storage.
///
/// # Safety
///
/// Every element in `values` must have been written with a valid `T`.
pub unsafe fn avec_assume_init<T, A: aligned_vec::Alignment>(
    values: AVec<MaybeUninit<T>, A>,
) -> AVec<T, A> {
    let (ptr, align, len, capacity) = values.into_raw_parts();
    // SAFETY: the caller guarantees that each slot contains a valid `T`, and
    // the raw parts originate from the `AVec<MaybeUninit<T>, A>` allocation.
    unsafe { AVec::from_raw_parts(ptr.cast::<T>(), align, len, capacity) }
}

/// Reinterprets an initialized prefix of `MaybeUninit<T>` as `&[T]`.
///
/// # Safety
///
/// Every element in `slice` must already be initialized.
pub unsafe fn assume_init_slice<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: the caller guarantees every element has been initialized.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<T>(), slice.len()) }
}

/// Heap-backed scratch storage with an explicit alignment guarantee.
///
/// This remains as a compatibility wrapper, but its storage is now backed by
/// [`AVec`] instead of a hand-rolled allocator.
pub struct AlignedScratch<T, A: Alignment> {
    values: AVec<T, RuntimeAlign>,
    _align: PhantomData<A>,
}

impl<T, A: Alignment> AlignedScratch<T, A> {
    /// Returns the effective alignment in bytes.
    pub fn alignment() -> usize {
        align_of::<T>().max(A::BYTES)
    }

    fn new(values: AVec<T, RuntimeAlign>) -> Self {
        Self {
            values,
            _align: PhantomData,
        }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when the buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns a raw pointer to the first element.
    pub fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }

    /// Returns a mutable raw pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.values.as_mut_ptr()
    }

    /// Borrows the buffer as a slice.
    pub fn as_slice(&self) -> &[T] {
        self.values.as_slice()
    }

    /// Borrows the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }
}

impl<T: Zeroable, A: Alignment> AlignedScratch<T, A> {
    /// Allocates a zero-initialized aligned scratch buffer.
    pub fn try_zeroed(len: usize) -> Result<Self, AlignedAllocError> {
        let alignment = Self::alignment();
        let mut values = try_runtime_uninit_vec::<T>(len, alignment)?;
        // SAFETY: the vector is fully allocated for `len` elements, and zeroed
        // bytes are a valid representation because `T: Zeroable`.
        unsafe {
            std::ptr::write_bytes(values.as_mut_ptr(), 0, len);
            Ok(Self::new(avec_assume_init(values)))
        }
    }

    /// Allocates a zero-initialized aligned scratch buffer or aborts on OOM.
    pub fn zeroed(len: usize) -> Self {
        match Self::try_zeroed(len) {
            Ok(scratch) => scratch,
            Err(error) => abort_allocation(error, len),
        }
    }
}

impl<T, A: Alignment> AlignedScratch<MaybeUninit<T>, A> {
    /// Allocates an uninitialized aligned scratch buffer.
    ///
    /// The returned buffer is safe to write through as a
    /// `&mut [MaybeUninit<T>]`. Call [`Self::assume_init`] only after every
    /// element has been fully initialized with a valid `T`.
    pub fn try_uninit(len: usize) -> Result<Self, AlignedAllocError> {
        Ok(Self::new(try_runtime_uninit_vec::<T>(
            len,
            Self::alignment(),
        )?))
    }

    /// Allocates an uninitialized aligned scratch buffer or aborts on OOM.
    pub fn uninit(len: usize) -> Self {
        match Self::try_uninit(len) {
            Ok(scratch) => scratch,
            Err(error) => abort_allocation(error, len),
        }
    }

    /// Converts an uninitialized scratch buffer into initialized storage.
    ///
    /// # Safety
    ///
    /// Every element in the buffer must have been written with a valid `T`
    /// before calling this function.
    pub unsafe fn assume_init(self) -> AlignedScratch<T, A> {
        // SAFETY: the caller guarantees that every element is initialized.
        let values = unsafe { avec_assume_init(self.values) };
        AlignedScratch::new(values)
    }
}

impl<T: Zeroable, A: Alignment> Default for AlignedScratch<T, A> {
    fn default() -> Self {
        Self::zeroed(0)
    }
}

impl<T, A: Alignment> Deref for AlignedScratch<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, A: Alignment> DerefMut for AlignedScratch<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: fmt::Debug, A: Alignment> fmt::Debug for AlignedScratch<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedScratch")
            .field("len", &self.len())
            .field("alignment", &Self::alignment())
            .field("values", &self.as_slice())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_block_uses_requested_alignment() {
        let block = AlignedBlock::<[u8; 8], Align32>::new([0; 8]);

        assert_eq!(AlignedBlock::<[u8; 8], Align32>::alignment(), 32);
        assert_eq!(block.as_ptr() as usize % 32, 0);
        assert_eq!(block.into_inner(), [0; 8]);
    }

    #[test]
    fn avec_zeroed_honors_alignment() {
        let mut values = avec_zeroed::<u16, 32>(19);

        assert_eq!(values.alignment(), 32);
        assert_eq!(values.len(), 19);
        assert_eq!(values.as_ptr() as usize % 32, 0);
        assert!(values.iter().all(|&value| value == 0));

        values[3] = 7;
        assert_eq!(values.as_slice()[3], 7);
    }

    #[test]
    fn aligned_scratch_is_zeroed_and_aligned() {
        let mut scratch = AlignedScratch::<u16, Align64>::zeroed(19);

        assert_eq!(AlignedScratch::<u16, Align64>::alignment(), 64);
        assert_eq!(scratch.len(), 19);
        assert_eq!(scratch.as_ptr() as usize % 64, 0);
        assert!(scratch.iter().all(|&value| value == 0));

        scratch[3] = 7;
        assert_eq!(scratch.as_slice()[3], 7);
    }

    #[test]
    fn aligned_scratch_reports_layout_overflow() {
        let err = AlignedScratch::<u64, Align64>::try_zeroed(usize::MAX).unwrap_err();
        assert_eq!(err, AlignedAllocError::LayoutOverflow);
    }

    #[test]
    fn aligned_uninit_scratch_can_be_initialized_in_place() {
        let mut scratch = AlignedScratch::<MaybeUninit<u16>, Align32>::uninit(8);

        assert_eq!(scratch.as_ptr() as usize % 32, 0);
        for (idx, slot) in scratch.iter_mut().enumerate() {
            slot.write((idx as u16) * 3);
        }

        // SAFETY: every slot was written in the loop above.
        let scratch = unsafe { scratch.assume_init() };
        assert_eq!(scratch.as_slice(), &[0, 3, 6, 9, 12, 15, 18, 21]);
    }

    #[test]
    fn aligned_scratch_is_send_and_sync_when_element_type_is() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<AlignedScratch<u16, Align32>>();
        assert_sync::<AlignedScratch<u16, Align32>>();
        assert_send::<AlignedScratch<MaybeUninit<u16>, Align32>>();
        assert_sync::<AlignedScratch<MaybeUninit<u16>, Align32>>();
    }
}
