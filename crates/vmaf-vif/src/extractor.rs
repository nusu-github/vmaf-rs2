//! VIF feature extractor — spec §4.2

use crate::filter::{subsample_into, SubsampleWorkspace};
use crate::stat::{vif_statistic_with_workspace, VifStatWorkspace};
use std::{error::Error, fmt};
use vmaf_cpu::SimdBackend;

const MIN_FRAME_DIMENSION: usize = 16;

/// Errors produced by [`VifExtractor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VifError {
    /// Width or height is below the minimum required by the spec kernels.
    InvalidDimensions { width: usize, height: usize },
    /// Bit depth is not supported by the integer pipeline.
    InvalidBitDepth { bpc: u8 },
    /// Width × height overflowed `usize`.
    SampleCountOverflow { width: usize, height: usize },
}

impl fmt::Display for VifError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => write!(
                f,
                "invalid frame dimensions {width}x{height}: width and height must be at least {MIN_FRAME_DIMENSION}"
            ),
            Self::InvalidBitDepth { bpc } => {
                write!(f, "invalid bit depth {bpc}: expected one of 8, 10, or 12")
            }
            Self::SampleCountOverflow { width, height } => {
                write!(f, "sample count overflow for dimensions {width}x{height}")
            }
        }
    }
}

impl Error for VifError {}

/// Per-frame VIF scores — 4 scales + combined.
pub struct VifScores {
    pub scale: [f64; 4],
    pub combined: f64,
}

/// Reusable internal buffers for per-frame VIF extraction.
///
/// This is intentionally exposed to sibling crates so orchestration can keep a
/// worker-local workspace without forcing public API callers to manage scratch.
#[doc(hidden)]
pub struct VifWorkspace {
    ref_a: Vec<u16>,
    ref_b: Vec<u16>,
    dis_a: Vec<u16>,
    dis_b: Vec<u16>,
    stat: VifStatWorkspace,
    subsample: SubsampleWorkspace,
}

impl VifWorkspace {
    fn new(width: usize, height: usize) -> Self {
        let next_level_area = width.div_ceil(2) * height.div_ceil(2);
        Self {
            ref_a: Vec::with_capacity(next_level_area),
            ref_b: Vec::with_capacity(next_level_area),
            dis_a: Vec::with_capacity(next_level_area),
            dis_b: Vec::with_capacity(next_level_area),
            stat: VifStatWorkspace::new(width),
            subsample: SubsampleWorkspace::new(width * height),
        }
    }
}

#[derive(Clone, Copy)]
enum CurrentScaleLevel {
    Input,
    A,
    B,
}

/// Stateless VIF extractor: computes all 4 scale scores for one frame pair.
///
/// The extractor auto-selects the best available SIMD backend at construction
/// time while preserving the existing public API.
pub struct VifExtractor {
    width: usize,
    height: usize,
    bpc: u8,
    vif_enhn_gain_limit: f64,
    backend: SimdBackend,
}

impl VifExtractor {
    /// Create a VIF extractor for one frame geometry.
    ///
    /// # Errors
    ///
    /// Returns [`VifError`] when dimensions are below the spec minimum, the
    /// bit depth is unsupported, or the frame area overflows `usize`.
    pub fn new(
        width: usize,
        height: usize,
        bpc: u8,
        vif_enhn_gain_limit: f64,
    ) -> Result<Self, VifError> {
        Self::with_backend(
            width,
            height,
            bpc,
            vif_enhn_gain_limit,
            SimdBackend::detect(),
        )
    }

    pub(crate) fn with_backend(
        width: usize,
        height: usize,
        bpc: u8,
        vif_enhn_gain_limit: f64,
        backend: SimdBackend,
    ) -> Result<Self, VifError> {
        validate_frame_geometry(width, height, bpc)?;
        Ok(Self {
            width,
            height,
            bpc,
            vif_enhn_gain_limit,
            backend: effective_backend(backend),
        })
    }

    #[doc(hidden)]
    pub fn make_workspace(&self) -> VifWorkspace {
        VifWorkspace::new(self.width, self.height)
    }

    #[doc(hidden)]
    pub fn compute_frame_with_workspace(
        &self,
        workspace: &mut VifWorkspace,
        ref_plane: &[u16],
        dis_plane: &[u16],
    ) -> VifScores {
        let (w, h, bpc) = (self.width, self.height, self.bpc);
        let limit = self.vif_enhn_gain_limit;
        let backend = self.backend;

        let mut nums = [0.0f64; 4];
        let mut dens = [0.0f64; 4];

        let s0 = vif_statistic_with_workspace(
            ref_plane,
            dis_plane,
            w,
            h,
            bpc,
            0,
            limit,
            &mut workspace.stat,
            backend,
        );
        nums[0] = s0.num;
        dens[0] = s0.den;

        let mut cur_level = CurrentScaleLevel::Input;
        let mut cur_w = w;
        let mut cur_h = h;

        for scale in 0..3usize {
            let (ss, next_level, next_w, next_h) = match cur_level {
                CurrentScaleLevel::Input => {
                    let (next_w, next_h) = subsample_into(
                        ref_plane,
                        dis_plane,
                        cur_w,
                        cur_h,
                        bpc,
                        scale,
                        backend,
                        &mut workspace.subsample,
                        &mut workspace.ref_a,
                        &mut workspace.dis_a,
                    );
                    let next_len = next_w * next_h;
                    let ss = vif_statistic_with_workspace(
                        &workspace.ref_a[..next_len],
                        &workspace.dis_a[..next_len],
                        next_w,
                        next_h,
                        bpc,
                        scale + 1,
                        limit,
                        &mut workspace.stat,
                        backend,
                    );
                    (ss, CurrentScaleLevel::A, next_w, next_h)
                }
                CurrentScaleLevel::A => {
                    let cur_len = cur_w * cur_h;
                    let (next_w, next_h) = subsample_into(
                        &workspace.ref_a[..cur_len],
                        &workspace.dis_a[..cur_len],
                        cur_w,
                        cur_h,
                        bpc,
                        scale,
                        backend,
                        &mut workspace.subsample,
                        &mut workspace.ref_b,
                        &mut workspace.dis_b,
                    );
                    let next_len = next_w * next_h;
                    let ss = vif_statistic_with_workspace(
                        &workspace.ref_b[..next_len],
                        &workspace.dis_b[..next_len],
                        next_w,
                        next_h,
                        bpc,
                        scale + 1,
                        limit,
                        &mut workspace.stat,
                        backend,
                    );
                    (ss, CurrentScaleLevel::B, next_w, next_h)
                }
                CurrentScaleLevel::B => {
                    let cur_len = cur_w * cur_h;
                    let (next_w, next_h) = subsample_into(
                        &workspace.ref_b[..cur_len],
                        &workspace.dis_b[..cur_len],
                        cur_w,
                        cur_h,
                        bpc,
                        scale,
                        backend,
                        &mut workspace.subsample,
                        &mut workspace.ref_a,
                        &mut workspace.dis_a,
                    );
                    let next_len = next_w * next_h;
                    let ss = vif_statistic_with_workspace(
                        &workspace.ref_a[..next_len],
                        &workspace.dis_a[..next_len],
                        next_w,
                        next_h,
                        bpc,
                        scale + 1,
                        limit,
                        &mut workspace.stat,
                        backend,
                    );
                    (ss, CurrentScaleLevel::A, next_w, next_h)
                }
            };
            nums[scale + 1] = ss.num;
            dens[scale + 1] = ss.den;
            cur_level = next_level;
            cur_w = next_w;
            cur_h = next_h;
        }

        let scale_scores = std::array::from_fn(|s| {
            if dens[s] > 0.0 {
                nums[s] / dens[s]
            } else {
                1.0
            }
        });

        let total_num: f64 = nums.iter().sum();
        let total_den: f64 = dens.iter().sum();
        let combined = if total_den > 0.0 {
            total_num / total_den
        } else {
            1.0
        };

        VifScores {
            scale: scale_scores,
            combined,
        }
    }

    /// Compute VIF scores for one ref/dis frame pair — spec §4.2.
    ///
    /// `ref_plane` and `dis_plane` are row-major luma planes (`width × height`).
    pub fn compute_frame(&self, ref_plane: &[u16], dis_plane: &[u16]) -> VifScores {
        let mut workspace = self.make_workspace();
        self.compute_frame_with_workspace(&mut workspace, ref_plane, dis_plane)
    }
}

fn validate_frame_geometry(width: usize, height: usize, bpc: u8) -> Result<(), VifError> {
    if width < MIN_FRAME_DIMENSION || height < MIN_FRAME_DIMENSION {
        return Err(VifError::InvalidDimensions { width, height });
    }
    if !matches!(bpc, 8 | 10 | 12) {
        return Err(VifError::InvalidBitDepth { bpc });
    }
    width
        .checked_mul(height)
        .ok_or(VifError::SampleCountOverflow { width, height })?;
    Ok(())
}

fn effective_backend(backend: SimdBackend) -> SimdBackend {
    if !backend.is_available() {
        return SimdBackend::Scalar;
    }

    match backend {
        SimdBackend::X86Avx512 => {
            if SimdBackend::X86Avx2Fma.is_available() {
                SimdBackend::X86Avx2Fma
            } else if SimdBackend::X86Sse2.is_available() {
                SimdBackend::X86Sse2
            } else {
                SimdBackend::Scalar
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vmaf_cpu::SimdBackend;

    fn patterned_plane(width: usize, height: usize, modulus: u16, bias: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    ((x * 11 + y * 13 + (x ^ y) * 7 + x * y * 3 + bias) % modulus as usize) as u16
                })
            })
            .collect()
    }

    #[test]
    fn workspace_path_matches_plain_compute() {
        let extractor = VifExtractor::with_backend(48, 48, 8, 100.0, SimdBackend::Scalar).unwrap();
        let reference = patterned_plane(48, 48, 255, 3);
        let distorted = patterned_plane(48, 48, 255, 17);
        let mut workspace = extractor.make_workspace();

        let plain = extractor.compute_frame(&reference, &distorted);
        let reused1 =
            extractor.compute_frame_with_workspace(&mut workspace, &reference, &distorted);
        let reused2 =
            extractor.compute_frame_with_workspace(&mut workspace, &reference, &distorted);

        for idx in 0..4 {
            assert_eq!(plain.scale[idx].to_bits(), reused1.scale[idx].to_bits());
            assert_eq!(plain.scale[idx].to_bits(), reused2.scale[idx].to_bits());
        }
        assert_eq!(plain.combined.to_bits(), reused1.combined.to_bits());
        assert_eq!(plain.combined.to_bits(), reused2.combined.to_bits());
    }

    #[test]
    fn constructor_rejects_invalid_dimensions_and_bpc() {
        assert!(matches!(
            VifExtractor::new(15, 16, 8, 100.0),
            Err(VifError::InvalidDimensions {
                width: 15,
                height: 16
            })
        ));
        assert!(matches!(
            VifExtractor::new(16, 15, 8, 100.0),
            Err(VifError::InvalidDimensions {
                width: 16,
                height: 15
            })
        ));
        assert!(matches!(
            VifExtractor::new(16, 16, 9, 100.0),
            Err(VifError::InvalidBitDepth { bpc: 9 })
        ));
    }
}
