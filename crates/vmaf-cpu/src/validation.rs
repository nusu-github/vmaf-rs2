//! Shared frame-geometry validation helpers.

use thiserror::Error;

/// Minimum width or height accepted by the integer VMAF kernels.
pub const MIN_FRAME_DIMENSION: usize = 16;

/// Common validation errors for frame geometry and sample counts.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrameValidationError {
    /// Width or height is below the minimum required by the spec kernels.
    #[error(
        "invalid frame dimensions {width}x{height}: width and height must be at least {MIN_FRAME_DIMENSION}"
    )]
    InvalidDimensions { width: usize, height: usize },
    /// Bit depth is not supported by the integer pipeline.
    #[error("invalid bit depth {bpc}: expected one of 8, 10, or 12")]
    InvalidBitDepth { bpc: u8 },
    /// Width × height overflowed `usize`.
    #[error("sample count overflow for dimensions {width}x{height}")]
    SampleCountOverflow { width: usize, height: usize },
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
    if width < MIN_FRAME_DIMENSION || height < MIN_FRAME_DIMENSION {
        return Err(FrameValidationError::InvalidDimensions { width, height });
    }
    if !matches!(bpc, 8 | 10 | 12) {
        return Err(FrameValidationError::InvalidBitDepth { bpc });
    }
    checked_sample_count(width, height)?;
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
}
