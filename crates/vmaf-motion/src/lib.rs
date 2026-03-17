//! Integer Motion — spec §4.4
//!
//! Computes Motion2 from reference frames only (distorted frame is ignored).
//! Maintains a 3-slot ring buffer of Gaussian-blurred frames.
//! Requires `flush()` after the final frame to emit the pending motion2 score (§4.4.2).

pub use blur::blur_frame;
mod blur;
mod sad;
mod simd;

pub use extractor::MotionExtractor;
mod extractor;

#[cfg(test)]
mod tests {
    use super::blur::{blur_frame, blur_frame_scalar_reference, blur_frame_with_backend};
    use super::sad::{compute_sad, compute_sad_scalar_reference, compute_sad_with_backend};
    use super::MotionExtractor;
    use vmaf_cpu::SimdBackend;

    fn patterned_frame(width: usize, height: usize, stride: usize, modulo: usize) -> Vec<u16> {
        let mut frame = vec![u16::MAX; stride * height];
        for row in 0..height {
            for col in 0..width {
                let value = (row * 131 + col * 29 + (row ^ col) * 7) % modulo;
                frame[row * stride + col] = value as u16;
            }
        }
        frame
    }

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

    #[test]
    fn blur_frame_matches_scalar_reference_with_stride_and_tails() {
        let w = 21;
        let h = 5;
        let stride = 25;
        let src = patterned_frame(w, h, stride, 1 << 10);

        let expected = blur_frame_scalar_reference(&src, stride, w, h, 10);
        let actual = blur_frame(&src, stride, w, h, 10);

        assert_eq!(actual, expected);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn blur_frame_explicit_backends_match_scalar_on_misaligned_stride_and_bpc_edges() {
        let cases = [
            (7usize, 5usize, 13usize, 8u8),
            (17usize, 9usize, 25usize, 10u8),
            (19usize, 7usize, 29usize, 12u8),
        ];

        for &(w, h, stride, bpc) in &cases {
            let modulus = 1usize << bpc;
            let mut storage = vec![0u16; stride * h + 3];
            let src = &mut storage[3..];

            for row in 0..h {
                for col in 0..w {
                    let value = (row * 173 + col * 43 + (row ^ col) * 19 + row * col * 5) % modulus;
                    src[row * stride + col] = value as u16;
                }
            }

            let expected = blur_frame_scalar_reference(src, stride, w, h, bpc);

            for backend in [SimdBackend::X86Sse2, SimdBackend::X86Avx2Fma] {
                if !backend.is_available() {
                    continue;
                }

                let actual = blur_frame_with_backend(backend, src, stride, w, h, bpc);
                assert_eq!(
                    actual, expected,
                    "mismatch for {w}x{h} stride {stride} bpc {bpc}"
                );
            }
        }
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

    #[test]
    fn compute_sad_matches_scalar_reference_for_unsigned_tail() {
        let w = 17;
        let h = 3;
        let len = w * h;
        let modulus = usize::from(u16::MAX) + 1;
        let mut buf_a = Vec::with_capacity(len);
        let mut buf_b = Vec::with_capacity(len);

        for idx in 0..len {
            buf_a.push(((idx * 977) % modulus) as u16);
            buf_b.push((u16::MAX as usize - ((idx * 1237) % modulus)) as u16);
        }
        buf_a[0] = 0;
        buf_b[0] = u16::MAX;
        buf_a[1] = u16::MAX;
        buf_b[1] = 0;

        let expected = compute_sad_scalar_reference(&buf_a, &buf_b, w, h);
        let actual = compute_sad(&buf_a, &buf_b, w, h);

        assert_eq!(actual, expected);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn compute_sad_explicit_backends_match_scalar_on_misaligned_tails() {
        let w = 31;
        let h = 5;
        let len = w * h;
        let modulus = usize::from(u16::MAX) + 1;
        let mut storage_a = vec![0u16; len + 3];
        let mut storage_b = vec![0u16; len + 5];
        let buf_a = &mut storage_a[1..1 + len];
        let buf_b = &mut storage_b[3..3 + len];

        for idx in 0..len {
            buf_a[idx] = ((idx * 977 + idx / w * 71) % modulus) as u16;
            buf_b[idx] = (u16::MAX as usize - ((idx * 1237 + idx / h * 29) % modulus)) as u16;
        }
        buf_a[0] = 0;
        buf_b[0] = u16::MAX;
        buf_a[1] = u16::MAX;
        buf_b[1] = 0;

        let expected = compute_sad_scalar_reference(buf_a, buf_b, w, h);

        for backend in [SimdBackend::X86Sse2, SimdBackend::X86Avx2Fma] {
            if !backend.is_available() {
                continue;
            }

            let actual = compute_sad_with_backend(backend, buf_a, buf_b, w, h);
            assert_eq!(actual.to_bits(), expected.to_bits(), "mismatch for {w}x{h}");
        }
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn extractor_detected_backend_matches_scalar() {
        let requested = SimdBackend::detect();
        if matches!(requested, SimdBackend::Scalar) {
            return;
        }

        let w = 19;
        let h = 7;
        let stride = 23;
        let bpc = 10;
        let modulus = 1usize << bpc;
        let frames: Vec<Vec<u16>> = (0..4usize)
            .map(|seed| {
                let mut frame = patterned_frame(w, h, stride, modulus);
                for (idx, value) in frame.iter_mut().enumerate() {
                    *value = ((*value as usize + idx * 7 + seed * 29) % modulus) as u16;
                }
                frame
            })
            .collect();

        let mut scalar = MotionExtractor::with_backend_for_tests(w, h, bpc, SimdBackend::Scalar);
        let mut simd = MotionExtractor::with_backend_for_tests(w, h, bpc, requested);

        for frame in &frames {
            let scalar_blur = scalar.prepare_blurred_frame(frame, stride);
            let simd_blur = simd.prepare_blurred_frame(frame, stride);
            assert_eq!(scalar_blur, simd_blur);
            assert_eq!(
                scalar.push_blurred_frame(scalar_blur),
                simd.push_blurred_frame(simd_blur),
            );
        }

        assert_eq!(scalar.flush(), simd.flush());
    }
}
