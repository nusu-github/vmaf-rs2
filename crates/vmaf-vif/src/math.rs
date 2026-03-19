//! Math primitives for Integer VIF (spec §4.1.1, §4.2.2, §4.2.3)

pub(crate) use vmaf_cpu::reflect_index;

/// Q11 log2 from a 32-bit value using the precomputed LUT — spec §4.2.2.
///
/// Precondition: `x >= SIGMA_NSQ (131072 = 2^17)`.
/// Returns `log2(x)` in Q11 fixed-point (scale factor 2048).
#[inline]
pub(crate) fn log2_32(lut: &[u16; 65536], x: u32) -> i32 {
    let k = 16 - x.leading_zeros() as i32; // k >= 2 given precondition
    let idx = (x >> k) as usize;
    lut[idx] as i32 + 2048 * k
}

/// Q11 log2 from a 64-bit value using the precomputed LUT — spec §4.2.3.
///
/// Precondition: `x >= 131072 (= 2^17)`.
/// Returns `log2(x)` in Q11 fixed-point (scale factor 2048).
#[inline]
pub(crate) fn log2_64(lut: &[u16; 65536], x: u64) -> i32 {
    let k = 48 - x.leading_zeros() as i32; // 48 = 64 - 16
    let idx = (x >> k) as usize;
    lut[idx] as i32 + 2048 * k
}
