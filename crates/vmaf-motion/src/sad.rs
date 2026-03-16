//! Sum of Absolute Differences for motion score — spec §4.4.2

/// Compute the normalised SAD between two blurred frames.
///
/// Both buffers are flat row-major `[u16]` with layout `[row * width + col]`.
///
/// Formula (spec §4.4.2):
/// ```text
/// sad = Σ |buf_a[i][j] - buf_b[i][j]|   (u64 accumulator)
/// return f32(sad) / 256.0_f32 / f32(width * height)
/// ```
/// CRITICAL: cast to f32 **before** dividing by `width * height` — spec note.
pub(crate) fn compute_sad(buf_a: &[u16], buf_b: &[u16], width: usize, height: usize) -> f32 {
    let mut sad = 0u64;
    for (&a, &b) in buf_a.iter().zip(buf_b.iter()) {
        sad += (a as i32 - b as i32).unsigned_abs() as u64;
    }
    (sad as f32 / 256.0_f32) / (width * height) as f32
}
