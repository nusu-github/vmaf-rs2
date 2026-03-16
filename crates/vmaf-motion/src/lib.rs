//! Integer Motion — spec §4.4
//!
//! Computes Motion2 from reference frames only (distorted frame is ignored).
//! Maintains a 3-slot ring buffer of Gaussian-blurred frames.
//! Requires `flush()` after the final frame to emit the pending motion2 score (§4.4.2).

mod blur;
mod sad;

pub use extractor::MotionExtractor;
mod extractor;

#[cfg(test)]
mod tests {
    use super::blur::blur_frame;
    use super::sad::compute_sad;
    use super::MotionExtractor;

    // --- blur_frame ---

    /// Flat 8-bit frame: every pixel must survive blur unchanged (value-wise).
    ///
    /// For pixel value V=100, bpc=8:
    ///   vertical:   (65536*100 + 128) >> 8 = 25600
    ///   horizontal: (65536*25600 + 32768) >> 16 = 25600
    #[test]
    fn blur_frame_flat_8bit() {
        let w = 4;
        let h = 4;
        let src = vec![100u16; w * h];
        let blurred = blur_frame(&src, w, w, h, 8);
        assert!(
            blurred.iter().all(|&v| v == 25600),
            "expected all 25600, got {:?}",
            &blurred[..4]
        );
    }

    /// Flat 10-bit frame: pixel=512, bpc=10.
    ///   vertical:   (65536*512 + 512) >> 10 = 32768
    ///   horizontal: (65536*32768 + 32768) >> 16 = 32768
    #[test]
    fn blur_frame_flat_10bit() {
        let w = 4;
        let h = 4;
        let src = vec![512u16; w * h];
        let blurred = blur_frame(&src, w, w, h, 10);
        assert!(
            blurred.iter().all(|&v| v == 32768),
            "expected all 32768, got {:?}",
            &blurred[..4]
        );
    }

    // --- compute_sad ---

    /// Identical frames → SAD = 0.0
    #[test]
    fn compute_sad_identical() {
        let buf = vec![25600u16; 4 * 4];
        assert_eq!(compute_sad(&buf, &buf, 4, 4), 0.0_f32);
    }

    /// Two flat frames differing by Δ=50 in original pixel space (8-bit).
    ///   blurred A = 25600 (from V=100), blurred B = 12800 (from V=50)
    ///   sad = 16 * 12800 = 204800
    ///   f32(204800) / 256.0 / 16.0 = 50.0
    #[test]
    fn compute_sad_flat_delta() {
        let buf_a = vec![25600u16; 4 * 4];
        let buf_b = vec![12800u16; 4 * 4];
        assert_eq!(compute_sad(&buf_a, &buf_b, 4, 4), 50.0_f32);
    }

    // --- MotionExtractor ---

    /// Single frame: push returns motion2[0] = 0.0; flush returns None.
    #[test]
    fn motion_single_frame() {
        let mut m = MotionExtractor::new(4, 4, 8);
        let frame = vec![100u16; 4 * 4];
        assert_eq!(m.push_frame(&frame, 4), Some((0, 0.0_f32)));
        assert_eq!(m.flush(), None);
    }

    /// Two identical frames: motion2[0]=0, motion2[1]=0 via flush.
    #[test]
    fn motion_identical_frames() {
        let mut m = MotionExtractor::new(4, 4, 8);
        let frame = vec![100u16; 4 * 4];
        assert_eq!(m.push_frame(&frame, 4), Some((0, 0.0_f32)));
        assert_eq!(m.push_frame(&frame, 4), None);
        assert_eq!(m.flush(), Some((1, 0.0_f32)));
    }

    /// Three frames: f0=100, f1=200, f2=100.
    ///   blurred: f0→25600, f1→51200, f2→25600
    ///   motion1[1] = (16 * 25600) / 256.0 / 16 = 100.0
    ///   motion1[2] = same = 100.0
    ///   motion2[1] = min(100.0, 100.0) = 100.0   ← emitted at push(f2)
    ///   flush → motion2[2] = motion1[2] = 100.0
    #[test]
    fn motion_three_frames() {
        let w = 4;
        let h = 4;
        let mut m = MotionExtractor::new(w, h, 8);
        let f0 = vec![100u16; w * h];
        let f1 = vec![200u16; w * h];
        let f2 = vec![100u16; w * h];
        assert_eq!(m.push_frame(&f0, w), Some((0, 0.0_f32)));
        assert_eq!(m.push_frame(&f1, w), None);
        let r2 = m.push_frame(&f2, w);
        assert!(r2.is_some());
        let (idx, score) = r2.unwrap();
        assert_eq!(idx, 1);
        assert!(
            (score - 100.0_f32).abs() < 0.5,
            "expected ~100.0, got {score}"
        );
        let flush = m.flush();
        assert!(flush.is_some());
        let (fidx, fscore) = flush.unwrap();
        assert_eq!(fidx, 2);
        assert!(
            (fscore - 100.0_f32).abs() < 0.5,
            "expected ~100.0, got {fscore}"
        );
    }

    /// §8: flush must emit the pending motion2[n_last] score — spec §4.4.2
    #[test]
    fn motion_flush_emits_pending_score() {
        let mut m = MotionExtractor::new(4, 4, 8);
        let fa = vec![100u16; 4 * 4];
        let fb = vec![200u16; 4 * 4]; // different frame
        m.push_frame(&fa, 4);
        m.push_frame(&fb, 4);
        // flush must return the pending motion2[1] = motion1[1]
        let result = m.flush();
        assert!(result.is_some(), "flush must return Some for n_last >= 1");
        assert_eq!(result.unwrap().0, 1, "frame index must be 1");
    }
}
