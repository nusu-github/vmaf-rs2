//! DWT filter constants and 2D computation — spec §4.3.1–4.3.3

use std::mem::MaybeUninit;

use crate::math::reflect_index;
use vmaf_cpu::{Align32, AlignedScratch};

pub(crate) const FILTER_LO: [i32; 4] = [15826, 27411, 7345, -4240];
pub(crate) const FILTER_HI: [i32; 4] = [-4240, -7345, 27411, -15826];
pub(crate) const DWT_LO_SUM: i32 = 46342;

#[inline]
fn reserve_output<T>(buffer: &mut Vec<T>, len: usize) {
    buffer.clear();
    if buffer.capacity() < len {
        buffer.reserve(len - buffer.capacity());
    }
}

#[inline]
unsafe fn assume_init_slice<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    std::slice::from_raw_parts(slice.as_ptr().cast::<T>(), slice.len())
}

/// Normalize `abs_x` (≥ 32768) to a 15-bit mantissa — spec §4.3.8.
///
/// Returns `(mantissa, shift)` where `shift = 17 − clz32(abs_x)`.
pub(crate) fn get_best15_from32(abs_x: u32) -> (u16, i32) {
    let shift = 17 - abs_x.leading_zeros() as i32;
    let mantissa = ((abs_x + (1u32 << (shift - 1))) >> shift) as u16;
    (mantissa, shift)
}

/// Scale-0 DWT output (int16 subbands).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Bands16 {
    pub a: Vec<i16>,
    pub v: Vec<i16>,
    pub h: Vec<i16>,
    pub d: Vec<i16>,
    pub width: usize,
    pub height: usize,
}

/// Reusable scale-0 DWT bands.
#[derive(Debug, Default)]
pub(crate) struct Bands16Buffer {
    pub a: Vec<i16>,
    pub v: Vec<i16>,
    pub h: Vec<i16>,
    pub d: Vec<i16>,
    pub width: usize,
    pub height: usize,
}

impl Bands16Buffer {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            a: Vec::with_capacity(capacity),
            v: Vec::with_capacity(capacity),
            h: Vec::with_capacity(capacity),
            d: Vec::with_capacity(capacity),
            width: 0,
            height: 0,
        }
    }

    pub(crate) fn prepare(&mut self, width: usize, height: usize) {
        self.width = width.div_ceil(2);
        self.height = height.div_ceil(2);
        let len = self.width * self.height;
        reserve_output(&mut self.a, len);
        reserve_output(&mut self.v, len);
        reserve_output(&mut self.h, len);
        reserve_output(&mut self.d, len);
    }

    pub(crate) fn into_owned(self) -> Bands16 {
        Bands16 {
            a: self.a,
            v: self.v,
            h: self.h,
            d: self.d,
            width: self.width,
            height: self.height,
        }
    }
}

/// Scales 1–3 DWT output (int32 subbands).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Bands32 {
    pub a: Vec<i32>,
    pub v: Vec<i32>,
    pub h: Vec<i32>,
    pub d: Vec<i32>,
    pub width: usize,
    pub height: usize,
}

/// Reusable scale 1–3 DWT bands.
#[derive(Debug, Default)]
pub(crate) struct Bands32Buffer {
    pub a: Vec<i32>,
    pub v: Vec<i32>,
    pub h: Vec<i32>,
    pub d: Vec<i32>,
    pub width: usize,
    pub height: usize,
}

impl Bands32Buffer {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            a: Vec::with_capacity(capacity),
            v: Vec::with_capacity(capacity),
            h: Vec::with_capacity(capacity),
            d: Vec::with_capacity(capacity),
            width: 0,
            height: 0,
        }
    }

    pub(crate) fn prepare(&mut self, width: usize, height: usize) {
        self.width = width.div_ceil(2);
        self.height = height.div_ceil(2);
        let len = self.width * self.height;
        reserve_output(&mut self.a, len);
        reserve_output(&mut self.v, len);
        reserve_output(&mut self.h, len);
        reserve_output(&mut self.d, len);
    }

    pub(crate) fn into_owned(self) -> Bands32 {
        Bands32 {
            a: self.a,
            v: self.v,
            h: self.h,
            d: self.d,
            width: self.width,
            height: self.height,
        }
    }
}

/// Reusable aligned temporaries for scale-0 DWT vertical filtering.
#[derive(Debug)]
pub(crate) struct Scale0DwtWorkspace {
    tmplo: AlignedScratch<MaybeUninit<i16>, Align32>,
    tmphi: AlignedScratch<MaybeUninit<i16>, Align32>,
}

impl Default for Scale0DwtWorkspace {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Scale0DwtWorkspace {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let len = height.div_ceil(2) * width;
        Self {
            tmplo: AlignedScratch::uninit(len),
            tmphi: AlignedScratch::uninit(len),
        }
    }

    pub(crate) fn prepare_len(&mut self, len: usize) {
        if self.tmplo.len() < len {
            self.tmplo = AlignedScratch::uninit(len);
        }
        if self.tmphi.len() < len {
            self.tmphi = AlignedScratch::uninit(len);
        }
    }

    pub(crate) fn uninit_slices(
        &mut self,
        len: usize,
    ) -> (&mut [MaybeUninit<i16>], &mut [MaybeUninit<i16>]) {
        (
            &mut self.tmplo.as_mut_slice()[..len],
            &mut self.tmphi.as_mut_slice()[..len],
        )
    }

    pub(crate) fn init_slices(&self, len: usize) -> (&[i16], &[i16]) {
        // SAFETY: callers only request initialized views after fully writing the
        // prefix through `uninit_slices`.
        unsafe {
            (
                assume_init_slice(&self.tmplo.as_slice()[..len]),
                assume_init_slice(&self.tmphi.as_slice()[..len]),
            )
        }
    }
}

/// Reusable aligned temporaries for scale 1–3 DWT vertical filtering.
#[derive(Debug)]
pub(crate) struct Scale123DwtWorkspace {
    tmplo: AlignedScratch<MaybeUninit<i32>, Align32>,
    tmphi: AlignedScratch<MaybeUninit<i32>, Align32>,
}

impl Default for Scale123DwtWorkspace {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Scale123DwtWorkspace {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let len = height.div_ceil(2) * width;
        Self {
            tmplo: AlignedScratch::uninit(len),
            tmphi: AlignedScratch::uninit(len),
        }
    }

    pub(crate) fn prepare_len(&mut self, len: usize) {
        if self.tmplo.len() < len {
            self.tmplo = AlignedScratch::uninit(len);
        }
        if self.tmphi.len() < len {
            self.tmphi = AlignedScratch::uninit(len);
        }
    }

    pub(crate) fn uninit_slices(
        &mut self,
        len: usize,
    ) -> (&mut [MaybeUninit<i32>], &mut [MaybeUninit<i32>]) {
        (
            &mut self.tmplo.as_mut_slice()[..len],
            &mut self.tmphi.as_mut_slice()[..len],
        )
    }

    pub(crate) fn init_slices(&self, len: usize) -> (&[i32], &[i32]) {
        // SAFETY: callers only request initialized views after fully writing the
        // prefix through `uninit_slices`.
        unsafe {
            (
                assume_init_slice(&self.tmplo.as_slice()[..len]),
                assume_init_slice(&self.tmphi.as_slice()[..len]),
            )
        }
    }
}

/// 2D DWT on a uint16 luma plane (scale 0, int16 output) — spec §4.3.3.
pub(crate) fn dwt_scale0(src: &[u16], width: usize, height: usize, bpc: u8) -> Bands16 {
    let mut workspace = Scale0DwtWorkspace::new(width, height);
    let mut bands = Bands16Buffer::with_capacity(width.div_ceil(2) * height.div_ceil(2));
    dwt_scale0_into(src, width, height, bpc, &mut workspace, &mut bands);
    bands.into_owned()
}

pub(crate) fn dwt_scale0_into(
    src: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    workspace: &mut Scale0DwtWorkspace,
    bands: &mut Bands16Buffer,
) {
    let h_half = height.div_ceil(2);
    let tmp_len = h_half * width;
    workspace.prepare_len(tmp_len);

    {
        let (tmplo, tmphi) = workspace.uninit_slices(tmp_len);
        dwt_scale0_vertical_scalar(src, width, height, bpc, tmplo, tmphi);
    }

    let (tmplo, tmphi) = workspace.init_slices(tmp_len);
    dwt_scale0_horizontal_scalar_into(tmplo, tmphi, width, h_half, bands);
}

pub(crate) fn dwt_scale0_vertical_scalar(
    src: &[u16],
    width: usize,
    height: usize,
    bpc: u8,
    tmplo: &mut [MaybeUninit<i16>],
    tmphi: &mut [MaybeUninit<i16>],
) {
    let shift_vp = if bpc == 8 { 8u32 } else { bpc as u32 };
    let round_vp = 1i32 << (shift_vp - 1);
    let h_half = height.div_ceil(2);
    let lo_bias = round_vp.wrapping_sub(DWT_LO_SUM.wrapping_mul(round_vp));

    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let base = 2 * i as i32;
        let r0 = reflect_index(base - 1, height as i32);
        let r1 = i * 2;
        let r2 = reflect_index(base + 1, height as i32);
        let r3 = reflect_index(base + 2, height as i32);
        let row_offset = i * width;

        for j in 0..width {
            let s0 = src[r0 * width + j] as i32;
            let s1 = src[r1 * width + j] as i32;
            let s2 = src[r2 * width + j] as i32;
            let s3 = src[r3 * width + j] as i32;

            let al = FILTER_LO[0]
                .wrapping_mul(s0)
                .wrapping_add(FILTER_LO[1].wrapping_mul(s1))
                .wrapping_add(FILTER_LO[2].wrapping_mul(s2))
                .wrapping_add(FILTER_LO[3].wrapping_mul(s3));
            let ah = FILTER_HI[0]
                .wrapping_mul(s0)
                .wrapping_add(FILTER_HI[1].wrapping_mul(s1))
                .wrapping_add(FILTER_HI[2].wrapping_mul(s2))
                .wrapping_add(FILTER_HI[3].wrapping_mul(s3));

            tmplo[row_offset + j].write((al.wrapping_add(lo_bias) >> shift_vp) as i16);
            tmphi[row_offset + j].write((ah.wrapping_add(round_vp) >> shift_vp) as i16);
        }
    }
}

#[inline]
pub(crate) fn dwt_scale0_horizontal_scalar_at(
    tmplo_row: &[i16],
    tmphi_row: &[i16],
    width: usize,
    j: usize,
) -> (i16, i16, i16, i16) {
    let base = 2 * j as i32;
    let c0 = reflect_index(base - 1, width as i32);
    let c1 = j * 2;
    let c2 = reflect_index(base + 1, width as i32);
    let c3 = reflect_index(base + 2, width as i32);
    let round_hp = 32768i64;

    let hp = |buf: &[i16], filt: &[i32; 4]| -> i16 {
        let s0 = buf[c0] as i64;
        let s1 = buf[c1] as i64;
        let s2 = buf[c2] as i64;
        let s3 = buf[c3] as i64;
        ((filt[0] as i64 * s0
            + filt[1] as i64 * s1
            + filt[2] as i64 * s2
            + filt[3] as i64 * s3
            + round_hp)
            >> 16) as i16
    };

    (
        hp(tmplo_row, &FILTER_LO),
        hp(tmplo_row, &FILTER_HI),
        hp(tmphi_row, &FILTER_LO),
        hp(tmphi_row, &FILTER_HI),
    )
}

pub(crate) fn dwt_scale0_horizontal_scalar_into(
    tmplo: &[i16],
    tmphi: &[i16],
    width: usize,
    h_half: usize,
    bands: &mut Bands16Buffer,
) {
    bands.prepare(width, h_half * 2);

    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let tmplo_row = &tmplo[src_row_start..src_row_end];
        let tmphi_row = &tmphi[src_row_start..src_row_end];

        for j in 0..bands.width {
            let (a, v, h, d) = dwt_scale0_horizontal_scalar_at(tmplo_row, tmphi_row, width, j);
            bands.a.push(a);
            bands.v.push(v);
            bands.h.push(h);
            bands.d.push(d);
        }
    }
}

/// Scale-specific constants: `(round_VP, shift_VP, round_HP, shift_HP)`.
pub(crate) const SCALE_PARAMS: [(i64, u32, i64, u32); 4] = [
    (0, 0, 0, 0),           // scale 0 — not used here
    (0, 0, 16384, 15),      // scale 1
    (32768, 16, 32768, 16), // scale 2
    (32768, 16, 16384, 15), // scale 3
];

/// 2D DWT on an int32 LL band (scales 1–3, int32 output) — spec §4.3.3.
pub(crate) fn dwt_s123(ll: &[i32], width: usize, height: usize, scale: usize) -> Bands32 {
    let mut workspace = Scale123DwtWorkspace::new(width, height);
    let mut bands = Bands32Buffer::with_capacity(width.div_ceil(2) * height.div_ceil(2));
    dwt_s123_into(ll, width, height, scale, &mut workspace, &mut bands);
    bands.into_owned()
}

pub(crate) fn dwt_s123_into(
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
    workspace: &mut Scale123DwtWorkspace,
    bands: &mut Bands32Buffer,
) {
    let h_half = height.div_ceil(2);
    let tmp_len = h_half * width;
    workspace.prepare_len(tmp_len);

    {
        let (tmplo, tmphi) = workspace.uninit_slices(tmp_len);
        dwt_s123_vertical_scalar(ll, width, height, scale, tmplo, tmphi);
    }

    let (tmplo, tmphi) = workspace.init_slices(tmp_len);
    dwt_s123_horizontal_scalar_into(tmplo, tmphi, width, h_half, scale, bands);
}

pub(crate) fn dwt_s123_vertical_scalar(
    ll: &[i32],
    width: usize,
    height: usize,
    scale: usize,
    tmplo: &mut [MaybeUninit<i32>],
    tmphi: &mut [MaybeUninit<i32>],
) {
    let (round_vp, shift_vp, _, _) = SCALE_PARAMS[scale];
    let h_half = height.div_ceil(2);

    debug_assert_eq!(ll.len(), width * height);
    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let base = 2 * i as i32;
        let r0 = reflect_index(base - 1, height as i32);
        let r1 = i * 2;
        let r2 = reflect_index(base + 1, height as i32);
        let r3 = reflect_index(base + 2, height as i32);
        let row_offset = i * width;

        for j in 0..width {
            let s0 = ll[r0 * width + j] as i64;
            let s1 = ll[r1 * width + j] as i64;
            let s2 = ll[r2 * width + j] as i64;
            let s3 = ll[r3 * width + j] as i64;

            let al = FILTER_LO[0] as i64 * s0
                + FILTER_LO[1] as i64 * s1
                + FILTER_LO[2] as i64 * s2
                + FILTER_LO[3] as i64 * s3;
            let ah = FILTER_HI[0] as i64 * s0
                + FILTER_HI[1] as i64 * s1
                + FILTER_HI[2] as i64 * s2
                + FILTER_HI[3] as i64 * s3;

            tmplo[row_offset + j].write(((al + round_vp) >> shift_vp) as i32);
            tmphi[row_offset + j].write(((ah + round_vp) >> shift_vp) as i32);
        }
    }
}

#[inline]
pub(crate) fn dwt_s123_horizontal_scalar_at(
    tmplo_row: &[i32],
    tmphi_row: &[i32],
    width: usize,
    j: usize,
    round_hp: i64,
    shift_hp: u32,
) -> (i32, i32, i32, i32) {
    let base = 2 * j as i32;
    let c0 = reflect_index(base - 1, width as i32);
    let c1 = j * 2;
    let c2 = reflect_index(base + 1, width as i32);
    let c3 = reflect_index(base + 2, width as i32);

    let hp = |buf: &[i32], filt: &[i32; 4]| -> i32 {
        let s0 = buf[c0] as i64;
        let s1 = buf[c1] as i64;
        let s2 = buf[c2] as i64;
        let s3 = buf[c3] as i64;
        ((filt[0] as i64 * s0
            + filt[1] as i64 * s1
            + filt[2] as i64 * s2
            + filt[3] as i64 * s3
            + round_hp)
            >> shift_hp) as i32
    };

    (
        hp(tmplo_row, &FILTER_LO),
        hp(tmplo_row, &FILTER_HI),
        hp(tmphi_row, &FILTER_LO),
        hp(tmphi_row, &FILTER_HI),
    )
}

pub(crate) fn dwt_s123_horizontal_scalar_into(
    tmplo: &[i32],
    tmphi: &[i32],
    width: usize,
    h_half: usize,
    scale: usize,
    bands: &mut Bands32Buffer,
) {
    let (_, _, round_hp, shift_hp) = SCALE_PARAMS[scale];
    bands.prepare(width, h_half * 2);

    debug_assert_eq!(tmplo.len(), h_half * width);
    debug_assert_eq!(tmphi.len(), h_half * width);

    for i in 0..h_half {
        let src_row_start = i * width;
        let src_row_end = src_row_start + width;
        let tmplo_row = &tmplo[src_row_start..src_row_end];
        let tmphi_row = &tmphi[src_row_start..src_row_end];

        for j in 0..bands.width {
            let (a, v, h, d) =
                dwt_s123_horizontal_scalar_at(tmplo_row, tmphi_row, width, j, round_hp, shift_hp);
            bands.a.push(a);
            bands.v.push(v);
            bands.h.push(h);
            bands.d.push(d);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_pattern(width: usize, height: usize, max_value: u16) -> Vec<u16> {
        let modulus = max_value as usize + 1;
        let mut plane = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let value = (x * 23 + y * 31 + (x ^ y) * 7 + x * y * 3) % modulus;
                plane.push(value as u16);
            }
        }
        plane
    }

    fn ll_pattern(width: usize, height: usize) -> Vec<i32> {
        let mut ll = Vec::with_capacity(width * height);
        for idx in 0..(width * height) {
            let base = ((idx as i32 * 97) % 4096) - 2048;
            ll.push(match idx % 4 {
                0 => base * 17,
                1 => -base * 9,
                2 => base * 5 - 777,
                _ => -base * 3 + 1234,
            });
        }
        ll
    }

    fn assert_bands16_match(expected: &Bands16, actual: &Bands16Buffer) {
        assert_eq!(expected.width, actual.width);
        assert_eq!(expected.height, actual.height);
        assert_eq!(expected.a, actual.a);
        assert_eq!(expected.v, actual.v);
        assert_eq!(expected.h, actual.h);
        assert_eq!(expected.d, actual.d);
    }

    fn assert_bands32_match(expected: &Bands32, actual: &Bands32Buffer) {
        assert_eq!(expected.width, actual.width);
        assert_eq!(expected.height, actual.height);
        assert_eq!(expected.a, actual.a);
        assert_eq!(expected.v, actual.v);
        assert_eq!(expected.h, actual.h);
        assert_eq!(expected.d, actual.d);
    }

    #[test]
    fn scale0_into_matches_owned_wrapper() {
        let width = 33;
        let height = 19;
        let bpc = 10;
        let plane = plane_pattern(width, height, (1u16 << bpc) - 1);
        let expected = dwt_scale0(&plane, width, height, bpc);
        let mut workspace = Scale0DwtWorkspace::new(width, height);
        let mut bands = Bands16Buffer::with_capacity(width.div_ceil(2) * height.div_ceil(2));

        dwt_scale0_into(&plane, width, height, bpc, &mut workspace, &mut bands);
        assert_bands16_match(&expected, &bands);
        dwt_scale0_into(&plane, width, height, bpc, &mut workspace, &mut bands);
        assert_bands16_match(&expected, &bands);
    }

    #[test]
    fn scale123_into_matches_owned_wrapper() {
        let width = 21;
        let height = 15;
        let scale = 2;
        let ll = ll_pattern(width, height);
        let expected = dwt_s123(&ll, width, height, scale);
        let mut workspace = Scale123DwtWorkspace::new(width, height);
        let mut bands = Bands32Buffer::with_capacity(width.div_ceil(2) * height.div_ceil(2));

        dwt_s123_into(&ll, width, height, scale, &mut workspace, &mut bands);
        assert_bands32_match(&expected, &bands);
        dwt_s123_into(&ll, width, height, scale, &mut workspace, &mut bands);
        assert_bands32_match(&expected, &bands);
    }
}
