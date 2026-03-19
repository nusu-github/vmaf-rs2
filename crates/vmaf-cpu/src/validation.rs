//! Shared validated geometry and numeric boundary helpers.

use std::fmt;

use thiserror::Error;

/// Minimum width or height accepted by the integer VMAF kernels.
pub const MIN_FRAME_DIMENSION: usize = 16;

/// Validated frame geometry shared by VIF, ADM, Motion, and orchestration code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameGeometry {
    width: usize,
    height: usize,
    bpc: u8,
    sample_count: usize,
}

impl FrameGeometry {
    /// Construct validated frame geometry for the integer VMAF pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`FrameValidationError`] when dimensions are below the spec minimum,
    /// the bit depth is unsupported, or the frame area overflows `usize`.
    pub fn new(width: usize, height: usize, bpc: u8) -> Result<Self, FrameValidationError> {
        if width < MIN_FRAME_DIMENSION || height < MIN_FRAME_DIMENSION {
            return Err(FrameValidationError::InvalidDimensions { width, height });
        }
        if !matches!(bpc, 8 | 10 | 12) {
            return Err(FrameValidationError::InvalidBitDepth { bpc });
        }

        let sample_count = checked_sample_count(width, height)?;
        Ok(Self {
            width,
            height,
            bpc,
            sample_count,
        })
    }

    /// Frame width in samples.
    #[inline]
    pub fn width(self) -> usize {
        self.width
    }

    /// Frame height in samples.
    #[inline]
    pub fn height(self) -> usize {
        self.height
    }

    /// Bit depth in bits per component.
    #[inline]
    pub fn bpc(self) -> u8 {
        self.bpc
    }

    /// Total sample count (`width * height`).
    #[inline]
    pub fn sample_count(self) -> usize {
        self.sample_count
    }
}

impl fmt::Display for FrameGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}@{}bpc", self.width, self.height, self.bpc)
    }
}

/// Validated enhancement gain limit shared by ADM and VIF model settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainLimit(f64);

impl GainLimit {
    /// Construct a validated enhancement gain limit.
    ///
    /// # Errors
    ///
    /// Returns [`GainLimitError`] when the value is non-finite or smaller than 1.0.
    pub fn new(value: f64) -> Result<Self, GainLimitError> {
        if !value.is_finite() {
            return Err(GainLimitError::NonFinite { value });
        }
        if value < 1.0 {
            return Err(GainLimitError::TooSmall { value });
        }
        Ok(Self(value))
    }

    /// Access the validated floating-point value.
    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl fmt::Display for GainLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Validation failures for enhancement gain limits.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GainLimitError {
    /// Gain limits must remain finite for the model pipeline.
    #[error("gain limit must be finite, got {value}")]
    NonFinite {
        /// The rejected raw gain-limit value.
        value: f64,
    },
    /// Gain limits smaller than one violate the model schema.
    #[error("gain limit must be >= 1.0, got {value}")]
    TooSmall {
        /// The rejected raw gain-limit value.
        value: f64,
    },
}

/// Common validation errors for frame geometry and sample counts.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrameValidationError {
    /// Width or height is below the minimum required by the spec kernels.
    #[error(
        "invalid frame dimensions {width}x{height}: width and height must be at least {MIN_FRAME_DIMENSION}"
    )]
    InvalidDimensions {
        /// The rejected frame width.
        width: usize,
        /// The rejected frame height.
        height: usize,
    },
    /// Bit depth is not supported by the integer pipeline.
    #[error("invalid bit depth {bpc}: expected one of 8, 10, or 12")]
    InvalidBitDepth {
        /// The rejected bits-per-component value.
        bpc: u8,
    },
    /// Width × height overflowed `usize`.
    #[error("sample count overflow for dimensions {width}x{height}")]
    SampleCountOverflow {
        /// The frame width involved in the overflow.
        width: usize,
        /// The frame height involved in the overflow.
        height: usize,
    },
}

/// Validate frame geometry shared by VIF, ADM, Motion, and orchestration code.
///
/// # Errors
///
/// Returns [`FrameValidationError`] when dimensions are below the spec minimum,
/// the bit depth is unsupported, or the frame area overflows `usize`.
pub fn validate_frame_geometry(
    width: usize,
    height: usize,
    bpc: u8,
) -> Result<(), FrameValidationError> {
    FrameGeometry::new(width, height, bpc)?;
    Ok(())
}

/// Multiply `width * height` while preserving the shared overflow error shape.
///
/// # Errors
///
/// Returns [`FrameValidationError::SampleCountOverflow`] when the product does
/// not fit in `usize`.
pub fn checked_sample_count(width: usize, height: usize) -> Result<usize, FrameValidationError> {
    width
        .checked_mul(height)
        .ok_or(FrameValidationError::SampleCountOverflow { width, height })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_frame_geometry_rejects_small_dimensions() {
        assert_eq!(
            validate_frame_geometry(15, 16, 8),
            Err(FrameValidationError::InvalidDimensions {
                width: 15,
                height: 16,
            })
        );
    }

    #[test]
    fn validate_frame_geometry_rejects_invalid_bit_depth() {
        assert_eq!(
            validate_frame_geometry(16, 16, 9),
            Err(FrameValidationError::InvalidBitDepth { bpc: 9 })
        );
    }

    #[test]
    fn checked_sample_count_reports_overflow() {
        assert_eq!(
            checked_sample_count(usize::MAX, 2),
            Err(FrameValidationError::SampleCountOverflow {
                width: usize::MAX,
                height: 2,
            })
        );
    }

    #[test]
    fn frame_geometry_exposes_validated_components() {
        let geometry = FrameGeometry::new(32, 24, 10).unwrap();

        assert_eq!(geometry.width(), 32);
        assert_eq!(geometry.height(), 24);
        assert_eq!(geometry.bpc(), 10);
        assert_eq!(geometry.sample_count(), 32 * 24);
    }

    #[test]
    fn gain_limit_rejects_non_finite_values() {
        assert_eq!(
            GainLimit::new(f64::INFINITY),
            Err(GainLimitError::NonFinite {
                value: f64::INFINITY,
            })
        );
    }

    #[test]
    fn gain_limit_rejects_small_values() {
        assert_eq!(
            GainLimit::new(0.5),
            Err(GainLimitError::TooSmall { value: 0.5 })
        );
    }

    #[test]
    fn gain_limit_accepts_model_default() {
        let limit = GainLimit::new(100.0).unwrap();

        assert_eq!(limit.value(), 100.0);
    }
}
