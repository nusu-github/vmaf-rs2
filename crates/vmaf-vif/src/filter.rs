//! Gaussian subsampling: low-pass filter + 2:1 decimate — spec §4.2.5

use crate::math::reflect_index;
use crate::tables::{FILTER, FILTER_WIDTH};

/// Low-pass filter a pair of planes and decimate 2:1 (scale s → scale s+1).
///
/// Uses `FILTER[scale + 1]` (the subsampling filter for the next scale).
/// Returns `(out_ref, out_dis, out_width, out_height)`.
pub(crate) fn subsample(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
) -> (Vec<u16>, Vec<u16>, usize, usize) {
    let filt = &FILTER[scale + 1];
    let fw = FILTER_WIDTH[scale + 1];
    let half = fw / 2;

    // Phase 1: vertical filter → tmp (H × W)
    let (shift_v, round_v) = if scale == 0 {
        (bpc as u32, 1u32 << (bpc - 1))
    } else {
        (16u32, 32768u32)
    };

    let mut tmp_ref = vec![0u16; height * width];
    let mut tmp_dis = vec![0u16; height * width];

    for i in 0..height {
        for j in 0..width {
            let mut ar = 0u32;
            let mut ad = 0u32;
            for k in 0..fw {
                let ii = reflect_index(i as i32 - half as i32 + k as i32, height as i32);
                let c = filt[k] as u32;
                ar += c * ref_in[ii * width + j] as u32;
                ad += c * dis_in[ii * width + j] as u32;
            }
            tmp_ref[i * width + j] = ((ar + round_v) >> shift_v) as u16;
            tmp_dis[i * width + j] = ((ad + round_v) >> shift_v) as u16;
        }
    }

    // Phase 2: horizontal filter → filt (H × W), always shift=16, round=32768
    let mut filt_ref = vec![0u16; height * width];
    let mut filt_dis = vec![0u16; height * width];

    for i in 0..height {
        for j in 0..width {
            let mut ar = 0u32;
            let mut ad = 0u32;
            for k in 0..fw {
                let jj = reflect_index(j as i32 - half as i32 + k as i32, width as i32);
                let c = filt[k] as u32;
                ar += c * tmp_ref[i * width + jj] as u32;
                ad += c * tmp_dis[i * width + jj] as u32;
            }
            filt_ref[i * width + j] = ((ar + 32768) >> 16) as u16;
            filt_dis[i * width + j] = ((ad + 32768) >> 16) as u16;
        }
    }

    // Phase 3: decimate 2:1 in both dimensions
    let out_w = width / 2;
    let out_h = height / 2;
    let mut out_ref = vec![0u16; out_h * out_w];
    let mut out_dis = vec![0u16; out_h * out_w];

    for i in 0..out_h {
        for j in 0..out_w {
            out_ref[i * out_w + j] = filt_ref[(2 * i) * width + (2 * j)];
            out_dis[i * out_w + j] = filt_dis[(2 * i) * width + (2 * j)];
        }
    }

    (out_ref, out_dis, out_w, out_h)
}
