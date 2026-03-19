//! Math primitives for Integer VIF (spec §4.1.1, §4.2.2, §4.2.3)

pub(crate) use vmaf_cpu::reflect_index;

use crate::tables::{LOG2_TABLE_LEN, LOG2_TABLE_MIN_INDEX};

/// Q11 log2 from a 32-bit value using the precomputed LUT — spec §4.2.2.
///
/// Precondition: `x >= SIGMA_NSQ (131072 = 2^17)`.
/// Returns `log2(x)` in Q11 fixed-point (scale factor 2048).
#[inline]
pub(crate) fn log2_32(lut: &[u16; LOG2_TABLE_LEN], x: u32) -> i32 {
    let k = 16 - x.leading_zeros() as i32; // k >= 2 given precondition
    let idx = (x >> k) as usize;
    debug_assert!(idx >= LOG2_TABLE_MIN_INDEX);
    let table_idx = idx - LOG2_TABLE_MIN_INDEX;
    debug_assert!(table_idx < LOG2_TABLE_LEN);
    // SAFETY: valid VIF callers only produce indices in 32768..=65535.
    unsafe { *lut.get_unchecked(table_idx) as i32 + 2048 * k }
}

/// Q11 log2 from a 64-bit value using the precomputed LUT — spec §4.2.3.
///
/// Precondition: `x >= 131072 (= 2^17)`.
/// Returns `log2(x)` in Q11 fixed-point (scale factor 2048).
#[inline]
pub(crate) fn log2_64(lut: &[u16; LOG2_TABLE_LEN], x: u64) -> i32 {
    let k = 48 - x.leading_zeros() as i32; // 48 = 64 - 16
    let idx = (x >> k) as usize;
    debug_assert!(idx >= LOG2_TABLE_MIN_INDEX);
    let table_idx = idx - LOG2_TABLE_MIN_INDEX;
    debug_assert!(table_idx < LOG2_TABLE_LEN);
    // SAFETY: valid VIF callers only produce indices in 32768..=65535.
    unsafe { *lut.get_unchecked(table_idx) as i32 + 2048 * k }
}
