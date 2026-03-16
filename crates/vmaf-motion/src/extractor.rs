//! MotionExtractor: ring-buffer state machine — spec §4.4.2

use crate::blur::blur_frame;
use crate::sad::compute_sad;

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
    /// Three blurred-frame slots; slot `n % 3` holds frame `n`.
    slots: [Vec<u16>; 3],
    /// Number of frames pushed so far.
    frame_count: usize,
    /// motion1 of the last pushed frame (needed for motion2 computation and flush).
    motion1_prev: f32,
}

impl MotionExtractor {
    pub fn new(width: usize, height: usize, bpc: u8) -> Self {
        let empty = vec![0u16; width * height];
        Self {
            width,
            height,
            bpc,
            slots: [empty.clone(), empty.clone(), empty],
            frame_count: 0,
            motion1_prev: 0.0,
        }
    }

    /// Push one reference frame.
    ///
    /// `luma`: flat row-major u16 samples, `luma[row * stride + col]`.
    /// `stride`: row stride **in samples**.
    ///
    /// Returns `Some((frame_index, motion2_score))` when a score becomes
    /// available, or `None` if more frames are needed.
    pub fn push_frame(&mut self, luma: &[u16], stride: usize) -> Option<(usize, f32)> {
        let n = self.frame_count;
        let w = self.width;
        let h = self.height;

        // Blur this frame into slot n % 3.
        self.slots[n % 3] = blur_frame(luma, stride, w, h, self.bpc);

        let motion1_n = if n == 0 {
            0.0_f32
        } else {
            // Compare current frame (slot n%3) with one-back (slot (n+2)%3).
            compute_sad(&self.slots[(n + 2) % 3], &self.slots[n % 3], w, h)
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
        result
    }

    /// Emit the pending motion2 score for the final frame.
    ///
    /// Must be called exactly once after all frames have been pushed.
    ///
    /// Returns `Some((n_last, motion2[n_last]))` when `n_last >= 1`,
    /// `None` for a single-frame sequence (motion2[0] was already emitted).
    pub fn flush(&mut self) -> Option<(usize, f32)> {
        let n_last = self.frame_count.checked_sub(1)?;
        if n_last >= 1 {
            // motion2[n_last] = motion1[n_last] (no subsequent frame to compare)
            Some((n_last, self.motion1_prev))
        } else {
            None // motion2[0] = 0.0 already emitted by push_frame(0)
        }
    }
}
