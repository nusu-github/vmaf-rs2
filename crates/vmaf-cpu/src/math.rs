//! Shared math helpers reused across VMAF feature crates.

/// Mirror-reflect index `i` into range `[0, w)`.
///
/// Precondition: `w >= 2`.
#[inline]
pub const fn reflect_index(i: i32, w: i32) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= w {
        (2 * w - 2 - i) as usize
    } else {
        i as usize
    }
}
