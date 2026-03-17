use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

use bytemuck::Zeroable;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for explicit storage alignments owned by VMAF.
///
/// Use [`AlignedBlock`] and [`AlignedScratch`] when the project controls the
/// allocation and wants to guarantee a known alignment boundary for future SIMD
/// kernels.
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

    /// Returns an immutable reference to the wrapped value.
    pub fn as_ref(&self) -> &T {
        &self.value
    }

    /// Returns a mutable reference to the wrapped value.
    pub fn as_mut(&mut self) -> &mut T {
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

/// Errors that can occur while creating an [`AlignedScratch`] buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlignedAllocError {
    /// The requested allocation would overflow `Layout` bounds.
    LayoutOverflow,
    /// The allocator returned a null pointer for the requested layout.
    AllocationFailed {
        /// Requested allocation size in bytes.
        size: usize,
        /// Requested allocation alignment in bytes.
        align: usize,
    },
}

impl fmt::Display for AlignedAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutOverflow => f.write_str("aligned allocation layout overflow"),
            Self::AllocationFailed { size, align } => {
                write!(
                    f,
                    "aligned allocation failed for {size} bytes with {align}-byte alignment"
                )
            }
        }
    }
}

impl std::error::Error for AlignedAllocError {}

/// Heap-backed scratch storage with an explicit alignment guarantee.
///
/// This is intended for temporary buffers where VMAF owns the allocation and
/// wants a stable alignment contract for later SIMD kernels.
pub struct AlignedScratch<T, A: Alignment> {
    ptr: NonNull<T>,
    len: usize,
    _align: PhantomData<A>,
}

impl<T, A: Alignment> AlignedScratch<T, A> {
    /// Returns the effective alignment in bytes.
    pub fn alignment() -> usize {
        align_of::<T>().max(A::BYTES)
    }

    fn layout(len: usize) -> Result<Layout, AlignedAllocError> {
        let size = size_of::<T>()
            .checked_mul(len)
            .ok_or(AlignedAllocError::LayoutOverflow)?;
        Layout::from_size_align(size, Self::alignment())
            .map_err(|_| AlignedAllocError::LayoutOverflow)
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the buffer has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the first element.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the first element.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Borrows the buffer as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` either comes from an allocation for `len` elements of `T`
        // or is a dangling pointer used only for zero-sized/empty slices.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Borrows the buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same reasoning as `as_slice`, plus `&mut self` guarantees
        // unique access to the allocation.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable, A: Alignment> AlignedScratch<T, A> {
    /// Allocates a zero-initialized aligned scratch buffer.
    pub fn try_zeroed(len: usize) -> Result<Self, AlignedAllocError> {
        let layout = Self::layout(len)?;
        if layout.size() == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                _align: PhantomData,
            });
        }

        // SAFETY: `layout` was validated above. `alloc_zeroed` returns a pointer
        // suitable for deallocation with the same layout, and zeroed bytes are a
        // valid representation because `T: Zeroable`.
        let ptr = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(ptr.cast::<T>()) else {
            return Err(AlignedAllocError::AllocationFailed {
                size: layout.size(),
                align: layout.align(),
            });
        };

        Ok(Self {
            ptr,
            len,
            _align: PhantomData,
        })
    }

    /// Allocates a zero-initialized aligned scratch buffer or aborts on OOM.
    pub fn zeroed(len: usize) -> Self {
        match Self::try_zeroed(len) {
            Ok(scratch) => scratch,
            Err(AlignedAllocError::LayoutOverflow) => {
                panic!("aligned scratch layout overflow for {len} elements")
            }
            Err(AlignedAllocError::AllocationFailed { size, align }) => {
                let layout =
                    Layout::from_size_align(size, align).expect("saved allocation layout is valid");
                handle_alloc_error(layout);
            }
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
        let layout = Self::layout(len)?;
        if layout.size() == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                _align: PhantomData,
            });
        }

        // SAFETY: `layout` was validated above. `alloc` returns a pointer
        // suitable for deallocation with the same layout, and
        // `MaybeUninit<T>` permits uninitialized bytes.
        let ptr = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(ptr.cast::<MaybeUninit<T>>()) else {
            return Err(AlignedAllocError::AllocationFailed {
                size: layout.size(),
                align: layout.align(),
            });
        };

        Ok(Self {
            ptr,
            len,
            _align: PhantomData,
        })
    }

    /// Allocates an uninitialized aligned scratch buffer or aborts on OOM.
    pub fn uninit(len: usize) -> Self {
        match Self::try_uninit(len) {
            Ok(scratch) => scratch,
            Err(AlignedAllocError::LayoutOverflow) => {
                panic!("aligned scratch layout overflow for {len} elements")
            }
            Err(AlignedAllocError::AllocationFailed { size, align }) => {
                let layout =
                    Layout::from_size_align(size, align).expect("saved allocation layout is valid");
                handle_alloc_error(layout);
            }
        }
    }

    /// Converts an uninitialized scratch buffer into initialized storage.
    ///
    /// # Safety
    ///
    /// Every element in the buffer must have been written with a valid `T`
    /// before calling this function.
    pub unsafe fn assume_init(self) -> AlignedScratch<T, A> {
        let scratch = ManuallyDrop::new(self);
        AlignedScratch {
            ptr: scratch.ptr.cast::<T>(),
            len: scratch.len,
            _align: PhantomData,
        }
    }
}

impl<T: Zeroable, A: Alignment> Default for AlignedScratch<T, A> {
    fn default() -> Self {
        Self::zeroed(0)
    }
}

impl<T, A: Alignment> Drop for AlignedScratch<T, A> {
    fn drop(&mut self) {
        // SAFETY: the allocation stores `len` initialized `T` elements.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ))
        };

        let layout = Self::layout(self.len).expect("existing scratch buffer has a valid layout");
        if layout.size() == 0 {
            return;
        }

        // SAFETY: `ptr` was allocated with `alloc`/`alloc_zeroed` using this
        // exact layout and has not been deallocated yet.
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
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
            .field("len", &self.len)
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
}
