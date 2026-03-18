//! Gaussian subsampling: low-pass filter + 2:1 decimate — spec §4.2.5

use aligned_vec::AVec;
use vmaf_cpu::{ConstAlign32, SimdBackend, avec_zeroed_32};

use crate::{
    math::reflect_index,
    tables::{FILTER, FILTER_WIDTH},
};

mod aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

const FILTER_TAP_CAP: usize = 18;
const HORIZONTAL_SHIFT: u32 = 16;
const HORIZONTAL_ROUND: u32 = 32768;

type AlignedVec32<T> = AVec<T, ConstAlign32>;

/// Reusable aligned row buffers for one subsampling step.
#[derive(Debug)]
pub(crate) struct SubsampleWorkspace {
    tmp_ref_row: AlignedVec32<u16>,
    tmp_dis_row: AlignedVec32<u16>,
    filt_ref_row: AlignedVec32<u16>,
    filt_dis_row: AlignedVec32<u16>,
}

impl Default for SubsampleWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}

impl SubsampleWorkspace {
    pub(crate) fn new(max_width: usize) -> Self {
        Self {
            tmp_ref_row: avec_zeroed_32(max_width),
            tmp_dis_row: avec_zeroed_32(max_width),
            filt_ref_row: avec_zeroed_32(max_width),
            filt_dis_row: avec_zeroed_32(max_width),
        }
    }

    fn prepare_rows(&mut self, width: usize) {
        if self.tmp_ref_row.len() < width {
            self.tmp_ref_row = avec_zeroed_32(width);
        }
        if self.tmp_dis_row.len() < width {
            self.tmp_dis_row = avec_zeroed_32(width);
        }
        if self.filt_ref_row.len() < width {
            self.filt_ref_row = avec_zeroed_32(width);
        }
        if self.filt_dis_row.len() < width {
            self.filt_dis_row = avec_zeroed_32(width);
        }
    }
}

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
    let mut workspace = SubsampleWorkspace::new(width);
    let mut out_ref = Vec::with_capacity((width / 2) * (height / 2));
    let mut out_dis = Vec::with_capacity((width / 2) * (height / 2));
    let (out_width, out_height) = subsample_into(
        ref_in,
        dis_in,
        width,
        height,
        bpc,
        scale,
        backend,
        &mut workspace,
        &mut out_ref,
        &mut out_dis,
    );
    (out_ref, out_dis, out_width, out_height)
}

pub(crate) fn subsample_into(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    backend: SimdBackend,
    workspace: &mut SubsampleWorkspace,
    out_ref: &mut Vec<u16>,
    out_dis: &mut Vec<u16>,
) -> (usize, usize) {
    match backend {
        SimdBackend::Scalar => subsample_scalar_into(
            ref_in, dis_in, width, height, bpc, scale, workspace, out_ref, out_dis,
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::X86Sse2 | SimdBackend::X86Avx2Fma | SimdBackend::X86Avx512 => {
            x86::subsample_into(
                ref_in, dis_in, width, height, bpc, scale, backend, workspace, out_ref, out_dis,
            )
        }
        SimdBackend::Aarch64Neon => aarch64::subsample_into(
            ref_in, dis_in, width, height, bpc, scale, backend, workspace, out_ref, out_dis,
        ),
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        _ => subsample_scalar_into(
            ref_in, dis_in, width, height, bpc, scale, workspace, out_ref, out_dis,
        ),
    }
}

fn subsample_scalar_into(
    ref_in: &[u16],
    dis_in: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    scale: usize,
    workspace: &mut SubsampleWorkspace,
    out_ref: &mut Vec<u16>,
    out_dis: &mut Vec<u16>,
) -> (usize, usize) {
    let filt = &FILTER[scale + 1][..FILTER_WIDTH[scale + 1]];
    let half = filt.len() / 2;
    let out_w = width / 2;
    let out_h = height / 2;
    let out_len = out_w * out_h;

    let (shift_v, round_v) = if scale == 0 {
        (bpc as u32, 1u32 << (bpc - 1))
    } else {
        (16u32, 32768u32)
    };

    out_ref.resize(out_len, 0);
    out_dis.resize(out_len, 0);

    workspace.prepare_rows(width);
    let tmp_ref_row = &mut workspace.tmp_ref_row.as_mut_slice()[..width];
    let tmp_dis_row = &mut workspace.tmp_dis_row.as_mut_slice()[..width];
    let filt_ref_row = &mut workspace.filt_ref_row.as_mut_slice()[..width];
    let filt_dis_row = &mut workspace.filt_dis_row.as_mut_slice()[..width];

    for out_i in 0..out_h {
        let src_i = out_i * 2;
        let row_offsets = reflected_row_offsets(src_i, height, width, half, filt.len());
        vertical_scalar_range(
            ref_in,
            dis_in,
            &row_offsets[..filt.len()],
            filt,
            shift_v,
            round_v,
            0,
            width,
            tmp_ref_row,
            tmp_dis_row,
        );
        horizontal_scalar_range(
            tmp_ref_row,
            tmp_dis_row,
            filt,
            half,
            0,
            width,
            filt_ref_row,
            filt_dis_row,
        );

        let dst_row = out_i * out_w;
        decimate_filtered_row_into(
            filt_ref_row,
            filt_dis_row,
            &mut out_ref[dst_row..dst_row + out_w],
            &mut out_dis[dst_row..dst_row + out_w],
        );
    }

    (out_w, out_h)
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
fn decimate_filtered_row_into(
    filt_ref_row: &[u16],
    filt_dis_row: &[u16],
    out_ref_row: &mut [u16],
    out_dis_row: &mut [u16],
) {
    for (j, (dst_ref, dst_dis)) in out_ref_row.iter_mut().zip(out_dis_row).enumerate() {
        let src = j * 2;
        *dst_ref = filt_ref_row[src];
        *dst_dis = filt_dis_row[src];
    }
}

#[cfg(test)]
mod tests {
    use vmaf_cpu::SimdBackend;

    use super::*;

    fn patterned_plane(width: usize, height: usize, modulus: u16, bias: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    ((x * 11 + y * 17 + (x ^ y) * 5 + x * y * 3 + bias) % modulus as usize) as u16
                })
            })
            .collect()
    }

    #[test]
    fn workspace_path_matches_owned_wrapper() {
        let width = 23;
        let height = 19;
        let reference = patterned_plane(width, height, 1024, 7);
        let distorted = patterned_plane(width, height, 1024, 29);
        let expected = subsample(
            &reference,
            &distorted,
            width,
            height,
            10,
            0,
            SimdBackend::Scalar,
        );

        let mut workspace = SubsampleWorkspace::new(width);
        let mut out_ref = Vec::new();
        let mut out_dis = Vec::new();
        let dims = subsample_into(
            &reference,
            &distorted,
            width,
            height,
            10,
            0,
            SimdBackend::Scalar,
            &mut workspace,
            &mut out_ref,
            &mut out_dis,
        );
        assert_eq!((expected.2, expected.3), dims);
        assert_eq!(expected.0, out_ref);
        assert_eq!(expected.1, out_dis);

        let dims2 = subsample_into(
            &reference,
            &distorted,
            width,
            height,
            10,
            0,
            SimdBackend::Scalar,
            &mut workspace,
            &mut out_ref,
            &mut out_dis,
        );
        assert_eq!((expected.2, expected.3), dims2);
        assert_eq!(expected.0, out_ref);
        assert_eq!(expected.1, out_dis);
    }
}
