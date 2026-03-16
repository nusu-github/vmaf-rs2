//! 5-tap Gaussian blur for motion estimation — spec §4.4.1

/// Coefficients of the 5-tap Gaussian filter (sums to 65536 = 2^16).
const MOTION_FILTER: [u32; 5] = [3571, 16004, 26386, 16004, 3571];

/// Reflective (mirror) padding — spec §4.1.1.
#[inline]
fn reflect(i: i32, len: i32) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= len {
        (2 * len - 2 - i) as usize
    } else {
        i as usize
    }
}

/// Gaussian-blur one luma plane and return the blurred frame as a flat `Vec<u16>`.
///
/// - `src`: flat luma samples in row-major order, `src[row * stride + col]`
/// - `stride`: row stride **in samples** (not bytes)
/// - `bpc`: bits per component (8, 10, or 12)
///
/// Output layout: `blurred[row * width + col]` (stride = width).
pub(crate) fn blur_frame(
    src: &[u16],
    stride: usize,
    width: usize,
    height: usize,
    bpc: u8,
) -> Vec<u16> {
    let n = width * height;
    let mut tmp = vec![0u16; n];
    let mut out = vec![0u16; n];

    let round_v = 1u32 << (bpc - 1);
    let shift_v = bpc;

    // --- Vertical pass: src → tmp ---
    for i in 0..height {
        for j in 0..width {
            let mut accum = 0u32;
            for k in 0..5usize {
                let ii = reflect(i as i32 - 2 + k as i32, height as i32);
                accum = accum.wrapping_add(MOTION_FILTER[k] * src[ii * stride + j] as u32);
            }
            tmp[i * width + j] = ((accum + round_v) >> shift_v) as u16;
        }
    }

    // --- Horizontal pass: tmp → out ---
    for i in 0..height {
        for j in 0..width {
            let mut accum = 0u32;
            for k in 0..5usize {
                let jj = reflect(j as i32 - 2 + k as i32, width as i32);
                accum = accum.wrapping_add(MOTION_FILTER[k] * tmp[i * width + jj] as u32);
            }
            out[i * width + j] = ((accum + 32768) >> 16) as u16;
        }
    }

    out
}
