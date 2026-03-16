//! Reflective boundary indexing — spec §4.1.1

/// Mirror-reflect index `i` into range `[0, w)`.
///
/// Precondition: `w >= 2`.
#[inline]
pub(crate) const fn reflect_index(i: i32, w: i32) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= w {
        (2 * w - 2 - i) as usize
    } else {
        i as usize
    }
}
