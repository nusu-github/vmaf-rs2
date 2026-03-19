//! MotionExtractor: ring-buffer state machine — spec §4.4.2

use thiserror::Error;
use vmaf_cpu::{FrameGeometry, FrameValidationError, SimdBackend, checked_sample_count};

use crate::{blur::blur_frame_with_backend, sad::compute_sad_with_backend, simd};

/// Errors produced by [`MotionExtractor`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MotionError {
    /// Shared sample-count validation errors surfaced while checking strides.
    #[error(transparent)]
    FrameValidation(#[from] FrameValidationError),
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

#[doc(hidden)]
pub struct CollectingData {
    /// Three blurred-frame slots; slot `n % 3` holds frame `n`.
    slots: [Vec<u16>; 3],
    /// Number of frames pushed so far.
    frame_count: usize,
    /// motion1 of the last pushed frame (needed for motion2 computation and flush).
    motion1_prev: f32,
}

#[doc(hidden)]
pub struct FlushedData;

#[doc(hidden)]
pub trait MotionState: private::Sealed {
    #[doc(hidden)]
    type Data;
}

/// Motion extractor state that still accepts input frames.
pub struct Collecting;

/// Motion extractor state after the final pending score has been emitted.
pub struct Flushed;

impl MotionState for Collecting {
    type Data = CollectingData;
}

impl MotionState for Flushed {
    type Data = FlushedData;
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::Collecting {}
    impl Sealed for super::Flushed {}
}

/// Stateful motion extractor for a single video sequence.
///
/// Call [`push_frame`](MotionExtractor::push_frame) for every reference frame in
/// order, then [`flush`](MotionExtractor::flush) once.
///
/// Emission schedule:
/// | Frame n  | push returns           | flush returns              |
/// |----------|------------------------|----------------------------|
/// | 0        | `Some((0, 0.0))`       | —                          |
/// | 1        | `None`                 | —                          |
/// | n ≥ 2    | `Some((n-1, m2[n-1]))` | —                          |
/// | (end)    | —                      | `Some((n_last, m1[n_last]))`|
pub struct MotionExtractor<S: MotionState = Collecting> {
    geometry: FrameGeometry,
    backend: SimdBackend,
    state: S::Data,
}

impl<S: MotionState> MotionExtractor<S> {
    /// Blur one reference frame using the extractor's cached backend selection.
    ///
    /// This is intended for batch workflows that precompute blur in parallel and
    /// later feed the result into [`push_blurred_frame`](Self::push_blurred_frame).
    pub fn prepare_blurred_frame(&self, luma: &[u16], stride: usize) -> Vec<u16> {
        let geometry = self.geometry;
        blur_frame_with_backend(
            self.backend,
            luma,
            stride,
            geometry.width(),
            geometry.height(),
            geometry.bpc(),
        )
    }

    fn frame_len(&self) -> usize {
        self.geometry.sample_count()
    }

    fn validate_input_plane(&self, luma: &[u16], stride: usize) -> Result<(), MotionError> {
        let width = self.geometry.width();
        if stride < width {
            return Err(MotionError::InvalidStride { stride, width });
        }

        let required =
            checked_sample_count(stride, self.geometry.height()).map_err(MotionError::from)?;
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

impl MotionExtractor<Collecting> {
    /// Create a motion extractor for one sequence.
    pub fn new(geometry: FrameGeometry) -> Self {
        Self::with_backend(geometry, simd::default_backend())
    }

    #[cfg(test)]
    pub(crate) fn with_backend_for_tests(geometry: FrameGeometry, backend: SimdBackend) -> Self {
        Self::with_backend(geometry, backend)
    }

    fn with_backend(geometry: FrameGeometry, backend: SimdBackend) -> Self {
        Self {
            geometry,
            backend: backend.effective(),
            state: CollectingData {
                slots: [Vec::new(), Vec::new(), Vec::new()],
                frame_count: 0,
                motion1_prev: 0.0,
            },
        }
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
    /// Returns [`MotionError`] when the input plane shape does not match the
    /// configured frame geometry.
    pub fn push_frame(
        &mut self,
        luma: &[u16],
        stride: usize,
    ) -> Result<Option<(usize, f32)>, MotionError> {
        self.validate_input_plane(luma, stride)?;
        let blurred = self.prepare_blurred_frame(luma, stride);
        self.push_blurred_frame(blurred)
    }

    /// Push a pre-computed blurred reference frame.
    ///
    /// Use [`prepare_blurred_frame`](Self::prepare_blurred_frame) to compute the
    /// blurred frame with this extractor's cached backend, allowing blur to be
    /// computed in parallel across multiple frames before sequential state update.
    ///
    /// # Errors
    ///
    /// Returns [`MotionError`] if the blurred frame does not match the configured
    /// frame area.
    pub fn push_blurred_frame(
        &mut self,
        blurred_luma: Vec<u16>,
    ) -> Result<Option<(usize, f32)>, MotionError> {
        self.validate_blurred_plane(&blurred_luma)?;

        let state = &mut self.state;
        let n = state.frame_count;
        let w = self.geometry.width();
        let h = self.geometry.height();

        // Store this pre-blurred frame into slot n % 3.
        state.slots[n % 3] = blurred_luma;

        let motion1_n = if n == 0 {
            0.0_f32
        } else {
            // Compare current frame (slot n%3) with one-back (slot (n+2)%3).
            compute_sad_with_backend(
                self.backend,
                &state.slots[(n + 2) % 3],
                &state.slots[n % 3],
                w,
                h,
            )
        };

        let result = if n == 0 {
            Some((0, 0.0_f32))
        } else if n >= 2 {
            // motion2[n-1] = min(motion1[n-1], motion1[n])
            Some((n - 1, state.motion1_prev.min(motion1_n)))
        } else {
            None
        };

        state.motion1_prev = motion1_n;
        state.frame_count += 1;
        Ok(result)
    }

    /// Emit the pending motion2 score for the final frame and transition into
    /// the terminal flushed state.
    pub fn flush(self) -> (MotionExtractor<Flushed>, Option<(usize, f32)>) {
        let pending = self.pending_score();
        let flushed = MotionExtractor {
            geometry: self.geometry,
            backend: self.backend,
            state: FlushedData,
        };
        (flushed, pending)
    }

    fn pending_score(&self) -> Option<(usize, f32)> {
        let n_last = self.state.frame_count.checked_sub(1)?;
        if n_last >= 1 {
            // motion2[n_last] = motion1[n_last] (no subsequent frame to compare)
            Some((n_last, self.state.motion1_prev))
        } else {
            None // motion2[0] = 0.0 already emitted by push_frame(0)
        }
    }
}
