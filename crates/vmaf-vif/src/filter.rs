//! Gaussian subsampling: low-pass filter + 2:1 decimate — spec §4.2.5

use crate::math::reflect_index;
use crate::tables::{FILTER, FILTER_WIDTH};
use vmaf_cpu::{Align32, AlignedScratch, SimdBackend};

mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

const FILTER_TAP_CAP: usize = 18;
const HORIZONTAL_SHIFT: u32 = 16;
const HORIZONTAL_ROUND: u32 = 32768;

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
    backend: SimdBackend,
) -> (Vec<u16>, Vec<u16>, usize, usize) {
    match backend {
        SimdBackend::Scalar => subsample_scalar(ref_in, dis_in, width, height, bpc, scale),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Sse2 | SimdBackend::X86Avx2Fma | SimdBackend::X86Avx512 => {
            x86::subsample(ref_in, dis_in, width, height, bpc, scale, backend)
        }
        SimdBackend::Aarch64Neon => {
            aarch64::subsample(ref_in, dis_in, width, height, bpc, scale, backend)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        _ => subsample_scalar(ref_in, dis_in, width, height, bpc, scale),
    }
}

fn subsample_scalar(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
) -> (Vec<u16>, Vec<u16>, usize, usize) {
    let filt = &FILTER[scale + 1][..FILTER_WIDTH[scale + 1]];
    let half = filt.len() / 2;

    let (shift_v, round_v) = if scale == 0 {
        (bpc as u32, 1u32 << (bpc - 1))
    } else {
        (16u32, 32768u32)
    };

    let mut tmp_ref_buf = AlignedScratch::<u16, Align32>::zeroed(height * width);
    let mut tmp_dis_buf = AlignedScratch::<u16, Align32>::zeroed(height * width);
    let mut filt_ref_buf = AlignedScratch::<u16, Align32>::zeroed(height * width);
    let mut filt_dis_buf = AlignedScratch::<u16, Align32>::zeroed(height * width);
    let tmp_ref = tmp_ref_buf.as_mut_slice();
    let tmp_dis = tmp_dis_buf.as_mut_slice();
    let filt_ref = filt_ref_buf.as_mut_slice();
    let filt_dis = filt_dis_buf.as_mut_slice();

    for i in 0..height {
        let row_offsets = reflected_row_offsets(i, height, width, half, filt.len());
        let row = i * width;
        vertical_scalar_range(
            ref_in,
            dis_in,
            &row_offsets[..filt.len()],
            filt,
            shift_v,
            round_v,
            0,
            width,
            &mut tmp_ref[row..row + width],
            &mut tmp_dis[row..row + width],
        );
    }

    for i in 0..height {
        let row = i * width;
        horizontal_scalar_range(
            &tmp_ref[row..row + width],
            &tmp_dis[row..row + width],
            filt,
            half,
            0,
            width,
            &mut filt_ref[row..row + width],
            &mut filt_dis[row..row + width],
        );
    }

    decimate_filtered(filt_ref, filt_dis, width, height)
}

#[inline]
fn reflected_row_offsets(
    i: usize,
    height: usize,
    width: usize,
    half: usize,
    taps: usize,
) -> [usize; FILTER_TAP_CAP] {
    let mut row_offsets = [0usize; FILTER_TAP_CAP];

    for k in 0..taps {
        let ii = reflect_index(i as i32 - half as i32 + k as i32, height as i32);
        row_offsets[k] = ii * width;
    }

    row_offsets
}

#[inline]
fn vertical_scalar_range(
    ref_in: &[u16],
    dis_in: &[u16],
    row_offsets: &[usize],
    coeffs: &[u16],
    shift: u32,
    round: u32,
    start: usize,
    end: usize,
    tmp_ref_row: &mut [u16],
    tmp_dis_row: &mut [u16],
) {
    for j in start..end {
        let mut acc_ref = 0u32;
        let mut acc_dis = 0u32;

        for (tap, &coeff) in coeffs.iter().enumerate() {
            let idx = row_offsets[tap] + j;
            let c = coeff as u32;
            acc_ref += c * ref_in[idx] as u32;
            acc_dis += c * dis_in[idx] as u32;
        }

        tmp_ref_row[j] = ((acc_ref + round) >> shift) as u16;
        tmp_dis_row[j] = ((acc_dis + round) >> shift) as u16;
    }
}

#[inline]
fn horizontal_scalar_range(
    tmp_ref_row: &[u16],
    tmp_dis_row: &[u16],
    coeffs: &[u16],
    half: usize,
    start: usize,
    end: usize,
    filt_ref_row: &mut [u16],
    filt_dis_row: &mut [u16],
) {
    let width = tmp_ref_row.len();

    for j in start..end {
        let mut acc_ref = 0u32;
        let mut acc_dis = 0u32;

        for (tap, &coeff) in coeffs.iter().enumerate() {
            let jj = reflect_index(j as i32 - half as i32 + tap as i32, width as i32);
            let c = coeff as u32;
            acc_ref += c * tmp_ref_row[jj] as u32;
            acc_dis += c * tmp_dis_row[jj] as u32;
        }

        filt_ref_row[j] = ((acc_ref + HORIZONTAL_ROUND) >> HORIZONTAL_SHIFT) as u16;
        filt_dis_row[j] = ((acc_dis + HORIZONTAL_ROUND) >> HORIZONTAL_SHIFT) as u16;
    }
}

#[inline]
fn horizontal_simd_body_range(width: usize, half: usize, lanes: usize) -> (usize, usize) {
    let start = half.min(width);
    let interior_end = width.saturating_sub(half);

    if interior_end <= start {
        return (start, start);
    }

    let simd_end = start + ((interior_end - start) / lanes) * lanes;
    (start, simd_end)
}

#[inline]
fn decimate_filtered(
    filt_ref: &[u16],
    filt_dis: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, Vec<u16>, usize, usize) {
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
