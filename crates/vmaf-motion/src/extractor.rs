//! MotionExtractor: ring-buffer state machine — spec §4.4.2

use thiserror::Error;
use vmaf_cpu::SimdBackend;

use crate::{blur::blur_frame_with_backend, sad::compute_sad_with_backend, simd};

const MIN_FRAME_DIMENSION: usize = 16;

/// Errors produced by [`MotionExtractor`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MotionError {
    /// Width or height is below the minimum required by the spec kernels.
    #[error(
        "invalid frame dimensions {width}x{height}: width and height must be at least {MIN_FRAME_DIMENSION}"
    )]
    InvalidDimensions { width: usize, height: usize },
    /// Bit depth is not supported by the integer pipeline.
    #[error("invalid bit depth {bpc}: expected one of 8, 10, or 12")]
    InvalidBitDepth { bpc: u8 },
    /// Width × height or stride × height overflowed `usize`.
    #[error("sample count overflow for dimensions {width}x{height}")]
    SampleCountOverflow { width: usize, height: usize },
    /// The extractor has already been flushed and is now terminal.
    #[error("motion extractor has already been flushed")]
    AlreadyFlushed,
    /// The provided stride is smaller than the configured frame width.
    #[error("invalid stride {stride}: expected at least width {width}")]
    InvalidStride { stride: usize, width: usize },
    /// The provided input plane is too short for the configured stride and height.
    #[error("invalid plane length {actual}: expected at least {required} samples")]
    InvalidPlaneLength { actual: usize, required: usize },
    /// A pre-blurred plane did not match the configured frame area.
    #[error("invalid blurred plane length {actual}: expected exactly {required} samples")]
    InvalidBlurredPlaneLength { actual: usize, required: usize },
}

/// Stateful motion extractor for a single video sequence.
///
/// Call [`push_frame`] for every reference frame in order, then [`flush`] once.
///
/// Emission schedule:
/// | Frame n  | push returns           | flush returns              |
/// |----------|------------------------|----------------------------|
/// | 0        | `Some((0, 0.0))`       | —                          |
/// | 1        | `None`                 | —                          |
/// | n ≥ 2    | `Some((n-1, m2[n-1]))` | —                          |
/// | (end)    | —                      | `Some((n_last, m1[n_last]))`|
pub struct MotionExtractor {
    width: usize,
    height: usize,
    bpc: u8,
    backend: SimdBackend,
    /// Three blurred-frame slots; slot `n % 3` holds frame `n`.
    slots: [Vec<u16>; 3],
    /// Number of frames pushed so far.
    frame_count: usize,
    /// motion1 of the last pushed frame (needed for motion2 computation and flush).
    motion1_prev: f32,
    /// Whether [`flush`] has already made the extractor terminal.
    flushed: bool,
}

impl MotionExtractor {
    /// Create a motion extractor for one sequence.
    ///
    /// # Errors
    ///
    /// Returns [`MotionError`] when dimensions are below the spec minimum, the
    /// bit depth is unsupported, or the frame area overflows `usize`.
    pub fn new(width: usize, height: usize, bpc: u8) -> Result<Self, MotionError> {
        Self::with_backend(width, height, bpc, simd::select_backend())
    }

    #[cfg(test)]
    pub(crate) fn with_backend_for_tests(
        width: usize,
        height: usize,
        bpc: u8,
        backend: SimdBackend,
    ) -> Result<Self, MotionError> {
        Self::with_backend(width, height, bpc, backend)
    }

    fn with_backend(
        width: usize,
        height: usize,
        bpc: u8,
        backend: SimdBackend,
    ) -> Result<Self, MotionError> {
        validate_frame_geometry(width, height, bpc)?;
        Ok(Self {
            width,
            height,
            bpc,
            backend: simd::effective_backend(backend),
            slots: [Vec::new(), Vec::new(), Vec::new()],
            frame_count: 0,
            motion1_prev: 0.0,
            flushed: false,
        })
    }

    /// Push one reference frame.
    ///
    /// `luma`: flat row-major u16 samples, `luma[row * stride + col]`.
    /// `stride`: row stride **in samples**.
    ///
    /// Returns `Some((frame_index, motion2_score))` when a score becomes
    /// available, or `None` if more frames are needed.
    ///
    /// # Errors
    ///
    /// Returns [`MotionError`] if the extractor has been flushed or the input
    /// plane shape does not match the configured frame geometry.
    pub fn push_frame(
        &mut self,
        luma: &[u16],
        stride: usize,
    ) -> Result<Option<(usize, f32)>, MotionError> {
        self.ensure_active()?;
        self.validate_input_plane(luma, stride)?;
        let blurred = self.prepare_blurred_frame(luma, stride);
        self.push_blurred_frame(blurred)
    }

    /// Blur one reference frame using the extractor's cached backend selection.
    ///
    /// This is intended for batch workflows that precompute blur in parallel and
    /// later feed the result into [`push_blurred_frame`].
    pub fn prepare_blurred_frame(&self, luma: &[u16], stride: usize) -> Vec<u16> {
        blur_frame_with_backend(
            self.backend,
            luma,
            stride,
            self.width,
            self.height,
            self.bpc,
        )
    }

    /// Push a pre-computed blurred reference frame.
    ///
    /// Use [`prepare_blurred_frame`] to compute the blurred frame with this
    /// extractor's cached backend, allowing blur to be computed in parallel
    /// across multiple frames before sequential state update.
    ///
    /// # Errors
    ///
    /// Returns [`MotionError`] if the extractor has been flushed or the blurred
    /// frame does not match the configured frame area.
    pub fn push_blurred_frame(
        &mut self,
        blurred_luma: Vec<u16>,
    ) -> Result<Option<(usize, f32)>, MotionError> {
        self.ensure_active()?;
        self.validate_blurred_plane(&blurred_luma)?;

        let n = self.frame_count;
        let w = self.width;
        let h = self.height;

        // Store this pre-blurred frame into slot n % 3.
        self.slots[n % 3] = blurred_luma;

        let motion1_n = if n == 0 {
            0.0_f32
        } else {
            // Compare current frame (slot n%3) with one-back (slot (n+2)%3).
            compute_sad_with_backend(
                self.backend,
                &self.slots[(n + 2) % 3],
                &self.slots[n % 3],
                w,
                h,
            )
        };

        let result = if n == 0 {
            Some((0, 0.0_f32))
        } else if n >= 2 {
            // motion2[n-1] = min(motion1[n-1], motion1[n])
            Some((n - 1, self.motion1_prev.min(motion1_n)))
        } else {
            None
        };

        self.motion1_prev = motion1_n;
        self.frame_count += 1;
        Ok(result)
    }

    /// Emit the pending motion2 score for the final frame.
    ///
    /// After the first call, the extractor becomes terminal and subsequent
    /// calls return `None`.
    ///
    /// Returns `Some((n_last, motion2[n_last]))` when `n_last >= 1`,
    /// `None` for a single-frame sequence (motion2[0] was already emitted).
    pub fn flush(&mut self) -> Option<(usize, f32)> {
        if self.flushed {
            return None;
        }
        self.flushed = true;

        let n_last = self.frame_count.checked_sub(1)?;
        if n_last >= 1 {
            // motion2[n_last] = motion1[n_last] (no subsequent frame to compare)
            Some((n_last, self.motion1_prev))
        } else {
            None // motion2[0] = 0.0 already emitted by push_frame(0)
        }
    }

    fn frame_len(&self) -> usize {
        self.width * self.height
    }

    fn ensure_active(&self) -> Result<(), MotionError> {
        if self.flushed {
            return Err(MotionError::AlreadyFlushed);
        }
        Ok(())
    }

    fn validate_input_plane(&self, luma: &[u16], stride: usize) -> Result<(), MotionError> {
        if stride < self.width {
            return Err(MotionError::InvalidStride {
                stride,
                width: self.width,
            });
        }

        let required = checked_sample_count(stride, self.height).map_err(|_| {
            MotionError::SampleCountOverflow {
                width: stride,
                height: self.height,
            }
        })?;
        if luma.len() < required {
            return Err(MotionError::InvalidPlaneLength {
                actual: luma.len(),
                required,
            });
        }
        Ok(())
    }

    fn validate_blurred_plane(&self, blurred_luma: &[u16]) -> Result<(), MotionError> {
        let required = self.frame_len();
        if blurred_luma.len() != required {
            return Err(MotionError::InvalidBlurredPlaneLength {
                actual: blurred_luma.len(),
                required,
            });
        }
        Ok(())
    }
}

fn validate_frame_geometry(width: usize, height: usize, bpc: u8) -> Result<(), MotionError> {
    if width < MIN_FRAME_DIMENSION || height < MIN_FRAME_DIMENSION {
        return Err(MotionError::InvalidDimensions { width, height });
    }
    if !matches!(bpc, 8 | 10 | 12) {
        return Err(MotionError::InvalidBitDepth { bpc });
    }
    checked_sample_count(width, height)?;
    Ok(())
}

fn checked_sample_count(width: usize, height: usize) -> Result<usize, MotionError> {
    width
        .checked_mul(height)
        .ok_or(MotionError::SampleCountOverflow { width, height })
}
