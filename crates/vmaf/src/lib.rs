//! vmaf — public API and pipeline orchestration — spec §2
#![deny(unsafe_code)]

use thiserror::Error;
use vmaf_adm::{AdmError, AdmExtractor, AdmWorkspace};
pub use vmaf_model::{
    LibsvmParseError, LoadModelError, ModelValidationError, PoolMethod, VmafModel, load_model,
};
use vmaf_model::{
    collect_scores, denormalize, normalize_features, pool, score_transform, svm_predict,
};
use vmaf_motion::{MotionError, MotionExtractor};
use vmaf_vif::{VifError, VifExtractor, VifWorkspace};

const MIN_FRAME_DIMENSION: usize = 16;

/// Errors produced by [`VmafContext`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VmafError {
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
    /// Reference plane length does not match the configured frame size.
    #[error("invalid reference plane length {actual}: expected exactly {required} samples")]
    InvalidReferenceLength { actual: usize, required: usize },
    /// Distorted plane length does not match the configured frame size.
    #[error("invalid distorted plane length {actual}: expected exactly {required} samples")]
    InvalidDistortedLength { actual: usize, required: usize },
    /// The context has already been flushed and is now terminal.
    #[error("vmaf context has already been flushed")]
    AlreadyFlushed,
    /// A lower-level motion error that should be surfaced to callers.
    #[error(transparent)]
    Motion(MotionError),
}

impl From<VifError> for VmafError {
    fn from(err: VifError) -> Self {
        match err {
            VifError::InvalidDimensions { width, height } => {
                Self::InvalidDimensions { width, height }
            }
            VifError::InvalidBitDepth { bpc } => Self::InvalidBitDepth { bpc },
            VifError::SampleCountOverflow { width, height } => {
                Self::SampleCountOverflow { width, height }
            }
        }
    }
}

impl From<AdmError> for VmafError {
    fn from(err: AdmError) -> Self {
        match err {
            AdmError::InvalidDimensions { width, height } => {
                Self::InvalidDimensions { width, height }
            }
            AdmError::InvalidBitDepth { bpc } => Self::InvalidBitDepth { bpc },
            AdmError::SampleCountOverflow { width, height } => {
                Self::SampleCountOverflow { width, height }
            }
        }
    }
}

impl From<MotionError> for VmafError {
    fn from(err: MotionError) -> Self {
        match err {
            MotionError::InvalidDimensions { width, height } => {
                Self::InvalidDimensions { width, height }
            }
            MotionError::InvalidBitDepth { bpc } => Self::InvalidBitDepth { bpc },
            MotionError::SampleCountOverflow { width, height } => {
                Self::SampleCountOverflow { width, height }
            }
            MotionError::AlreadyFlushed => Self::AlreadyFlushed,
            MotionError::InvalidPlaneLength { actual, required }
            | MotionError::InvalidBlurredPlaneLength { actual, required } => {
                Self::InvalidReferenceLength { actual, required }
            }
            err @ MotionError::InvalidStride { .. } => Self::Motion(err),
        }
    }
}

/// Per-frame VMAF scores and features.
#[derive(Clone, Debug)]
pub struct FrameScore {
    pub frame_index: usize,
    pub score: f64,
    pub adm2: f64,
    pub motion2: f64,
    pub vif_scale0: f64,
    pub vif_scale1: f64,
    pub vif_scale2: f64,
    pub vif_scale3: f64,
}

struct PendingFrame {
    frame_idx: usize,
    adm: f64,
    vif: [f64; 4],
}

/// Options controlling VMAF score post-processing.
#[derive(Clone, Copy, Debug, Default)]
pub struct VmafOptions {
    /// Apply the model's `score_transform` block (polynomial/knots/rectify).
    ///
    /// Note: the reference `./vmaf` binary in this repo (libvmaf-style) does not
    /// apply `score_transform` for the bundled v0.6.x models; to match it, keep
    /// this disabled.
    pub apply_score_transform: bool,
}

/// VMAF scoring context for a single video sequence.
pub struct VmafContext {
    model: VmafModel,
    vif: VifExtractor,
    vif_workspace: VifWorkspace,
    adm: AdmExtractor,
    adm_workspace: AdmWorkspace,
    motion: MotionExtractor,
    width: usize,
    height: usize,
    pending: Vec<PendingFrame>,
    per_frame_scores: Vec<FrameScore>,
    frame_count: usize,
    opts: VmafOptions,
    flushed: bool,
}

impl VmafContext {
    /// Create a VMAF scoring context for one sequence.
    ///
    /// # Errors
    ///
    /// Returns [`VmafError`] when dimensions are below the spec minimum, the
    /// bit depth is unsupported, or the frame area overflows `usize`.
    pub fn new(model: VmafModel, width: usize, height: usize, bpc: u8) -> Result<Self, VmafError> {
        Self::new_with_options(model, width, height, bpc, VmafOptions::default())
    }

    /// Create a VMAF scoring context with custom options.
    ///
    /// # Errors
    ///
    /// Returns [`VmafError`] when dimensions are below the spec minimum, the
    /// bit depth is unsupported, or the frame area overflows `usize`.
    pub fn new_with_options(
        model: VmafModel,
        width: usize,
        height: usize,
        bpc: u8,
        opts: VmafOptions,
    ) -> Result<Self, VmafError> {
        validate_frame_geometry(width, height, bpc)?;

        let vif = VifExtractor::new(width, height, bpc, model.vif_enhn_gain_limit)?;
        let vif_workspace = vif.make_workspace();
        let adm = AdmExtractor::new(width, height, bpc, model.adm_enhn_gain_limit)?;
        let adm_workspace = adm.make_workspace();
        Ok(Self {
            vif,
            vif_workspace,
            adm,
            adm_workspace,
            motion: MotionExtractor::new(width, height, bpc)?,
            model,
            width,
            height,
            pending: Vec::new(),
            per_frame_scores: Vec::new(),
            frame_count: 0,
            opts,
            flushed: false,
        })
    }

    /// Push a reference/distorted frame pair. Returns a `FrameScore` when one
    /// becomes available (motion has a 1-frame lag).
    ///
    /// # Errors
    ///
    /// Returns [`VmafError`] if the context has already been flushed or either
    /// plane does not match the configured frame size.
    pub fn push_frame(
        &mut self,
        reference: &[u16],
        distorted: &[u16],
    ) -> Result<Option<FrameScore>, VmafError> {
        self.ensure_active()?;
        self.validate_frame_inputs(reference, distorted)?;

        let vif_scores =
            self.vif
                .compute_frame_with_workspace(&mut self.vif_workspace, reference, distorted);
        let adm_score =
            self.adm
                .compute_frame_with_workspace(&mut self.adm_workspace, reference, distorted);
        let motion_result = self.motion.push_frame(reference, self.width)?;

        self.pending.push(PendingFrame {
            frame_idx: self.frame_count,
            adm: adm_score,
            vif: vif_scores.scale,
        });
        self.frame_count += 1;

        Ok(motion_result.and_then(|(idx, m2)| self.finalize_frame(idx, m2 as f64)))
    }

    /// Push a batch of reference/distorted frame pairs in parallel.
    /// Returns a list of `FrameScore`s that have become available.
    ///
    /// # Errors
    ///
    /// Returns [`VmafError`] if the context has already been flushed or any
    /// frame pair does not match the configured frame size.
    pub fn push_frame_batch(
        &mut self,
        frames: &[(&[u16], &[u16])],
    ) -> Result<Vec<FrameScore>, VmafError> {
        self.ensure_active()?;
        for &(reference, distorted) in frames {
            self.validate_frame_inputs(reference, distorted)?;
        }

        if frames.len() <= 1 || rayon::current_num_threads() <= 1 {
            let mut out = Vec::with_capacity(frames.len());
            for &(reference, distorted) in frames {
                if let Some(score) = self.push_frame(reference, distorted)? {
                    out.push(score);
                }
            }
            return Ok(out);
        }

        use rayon::prelude::*;

        let vif = &self.vif;
        let adm = &self.adm;
        let motion = &self.motion;
        let stride = self.width;
        let extracted: Vec<_> = frames
            .par_iter()
            .map_init(
                || (vif.make_workspace(), adm.make_workspace()),
                |(vif_workspace, adm_workspace), (r, d)| {
                    let vif_scores = vif.compute_frame_with_workspace(vif_workspace, r, d);
                    let adm_score = adm.compute_frame_with_workspace(adm_workspace, r, d);
                    let blur = motion.prepare_blurred_frame(r, stride);
                    (vif_scores, adm_score, blur)
                },
            )
            .collect();

        let mut out = Vec::new();
        for (vif_scores, adm_score, blur) in extracted {
            self.pending.push(PendingFrame {
                frame_idx: self.frame_count,
                adm: adm_score,
                vif: vif_scores.scale,
            });
            self.frame_count += 1;

            if let Some((idx, m2)) = self.motion.push_blurred_frame(blur)? {
                if let Some(fs) = self.finalize_frame(idx, m2 as f64) {
                    out.push(fs);
                }
            }
        }
        Ok(out)
    }

    /// Flush the final pending frame. Call once after all frames are pushed.
    ///
    /// The first call makes the context terminal; subsequent calls return
    /// `Ok(None)`.
    pub fn flush(&mut self) -> Result<Option<FrameScore>, VmafError> {
        if self.flushed {
            return Ok(None);
        }
        self.flushed = true;

        Ok(self
            .motion
            .flush()
            .and_then(|(idx, m2)| self.finalize_frame(idx, m2 as f64)))
    }

    /// All finalized per-frame scores in frame-index order.
    pub fn per_frame_scores(&self) -> &[FrameScore] {
        &self.per_frame_scores
    }

    /// Pool all per-frame scores using the given method and subsampling factor.
    pub fn pool_score(&self, method: PoolMethod, n_subsample: usize) -> f64 {
        let scores: Vec<f64> = self.per_frame_scores.iter().map(|fs| fs.score).collect();
        let n = scores.len();
        if n == 0 {
            return 0.0;
        }
        let selected = collect_scores(&scores, 0, n - 1, n_subsample);
        pool(&selected, method)
    }

    fn finalize_frame(&mut self, idx: usize, m2: f64) -> Option<FrameScore> {
        let pos = self.pending.iter().position(|p| p.frame_idx == idx)?;
        let pf = self.pending.remove(pos);

        // Feature order: [adm2, motion2, vif_scale0, vif_scale1, vif_scale2, vif_scale3] — spec §6
        let raw: [f64; 6] = [pf.adm, m2, pf.vif[0], pf.vif[1], pf.vif[2], pf.vif[3]];
        let normed = normalize_features(
            &raw,
            &self.model.feature_slopes,
            &self.model.feature_intercepts,
        );
        let raw_svm = svm_predict(&self.model.svm, &normed);
        let denorm = denormalize(raw_svm, self.model.score_slope, self.model.score_intercept);
        let st = if self.opts.apply_score_transform {
            self.model.score_transform.as_ref()
        } else {
            None
        };
        let score = score_transform(denorm, st, self.model.score_clip);

        let fs = FrameScore {
            frame_index: idx,
            score,
            adm2: pf.adm,
            motion2: m2,
            vif_scale0: pf.vif[0],
            vif_scale1: pf.vif[1],
            vif_scale2: pf.vif[2],
            vif_scale3: pf.vif[3],
        };
        self.per_frame_scores.push(fs.clone());
        Some(fs)
    }

    fn frame_len(&self) -> usize {
        self.width * self.height
    }

    fn ensure_active(&self) -> Result<(), VmafError> {
        if self.flushed {
            return Err(VmafError::AlreadyFlushed);
        }
        Ok(())
    }

    fn validate_frame_inputs(&self, reference: &[u16], distorted: &[u16]) -> Result<(), VmafError> {
        let required = self.frame_len();
        if reference.len() != required {
            return Err(VmafError::InvalidReferenceLength {
                actual: reference.len(),
                required,
            });
        }
        if distorted.len() != required {
            return Err(VmafError::InvalidDistortedLength {
                actual: distorted.len(),
                required,
            });
        }
        Ok(())
    }
}

fn validate_frame_geometry(width: usize, height: usize, bpc: u8) -> Result<(), VmafError> {
    if width < MIN_FRAME_DIMENSION || height < MIN_FRAME_DIMENSION {
        return Err(VmafError::InvalidDimensions { width, height });
    }
    if !matches!(bpc, 8 | 10 | 12) {
        return Err(VmafError::InvalidBitDepth { bpc });
    }
    width
        .checked_mul(height)
        .ok_or(VmafError::SampleCountOverflow { width, height })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg_model_feature_opts_change_scores() {
        let pos = load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap();
        let neg = load_model(include_str!("../../../models/vmaf_v0.6.1neg.json")).unwrap();

        assert_eq!(pos.adm_enhn_gain_limit, 100.0);
        assert_eq!(pos.vif_enhn_gain_limit, 100.0);
        assert_eq!(neg.adm_enhn_gain_limit, 1.0);
        assert_eq!(neg.vif_enhn_gain_limit, 1.0);

        let (w, h, bpc) = (64usize, 64usize, 8u8);

        let reference: Vec<u16> = (0..h)
            .flat_map(|y| (0..w).map(move |x| ((x + y) % 128) as u16))
            .collect();
        let distorted: Vec<u16> = reference.iter().map(|&v| (v * 2).min(255)).collect();

        let mut ctx_pos = VmafContext::new(pos, w, h, bpc).unwrap();
        let mut ctx_neg = VmafContext::new(neg, w, h, bpc).unwrap();

        // Push 2 frames to satisfy motion's 1-frame lag.
        for _ in 0..2 {
            ctx_pos.push_frame(&reference, &distorted).unwrap();
            ctx_neg.push_frame(&reference, &distorted).unwrap();
        }
        ctx_pos.flush().unwrap();
        ctx_neg.flush().unwrap();

        let a = ctx_pos.per_frame_scores();
        let b = ctx_neg.per_frame_scores();
        assert_eq!(a.len(), b.len());
        assert!(!a.is_empty());

        // Under enhancement (distorted is effectively a gain>1 transform of reference),
        // neg models clamp enhancement gain to 1.0, so VIF (and thus overall score)
        // should differ from the default model.
        let last_a = &a[a.len() - 1];
        let last_b = &b[b.len() - 1];

        let dv0 = (last_a.vif_scale0 - last_b.vif_scale0).abs();
        let dv3 = (last_a.vif_scale3 - last_b.vif_scale3).abs();
        let ds = (last_a.score - last_b.score).abs();

        assert!(dv0 > 1e-9 || dv3 > 1e-9 || ds > 1e-9);
    }

    #[test]
    fn batch_and_sequential_paths_match() {
        let model_json = include_str!("../../../models/vmaf_v0.6.1.json");
        let (w, h, bpc) = (64usize, 48usize, 8u8);

        let frames: Vec<(Vec<u16>, Vec<u16>)> = (0..5usize)
            .map(|frame_idx| {
                let reference: Vec<u16> = (0..h)
                    .flat_map(|y| {
                        (0..w)
                            .map(move |x| ((x * 3 + y * 5 + frame_idx * 11 + (x ^ y)) % 256) as u16)
                    })
                    .collect();
                let distorted = reference
                    .iter()
                    .enumerate()
                    .map(|(idx, &value)| {
                        value.saturating_sub(((frame_idx * 7 + idx * 3) % 19) as u16)
                    })
                    .collect();
                (reference, distorted)
            })
            .collect();

        let mut sequential = VmafContext::new(load_model(model_json).unwrap(), w, h, bpc).unwrap();
        let mut batch = VmafContext::new(load_model(model_json).unwrap(), w, h, bpc).unwrap();

        let mut sequential_scores = Vec::new();
        for (reference, distorted) in &frames {
            if let Some(score) = sequential.push_frame(reference, distorted).unwrap() {
                sequential_scores.push(score);
            }
        }
        if let Some(score) = sequential.flush().unwrap() {
            sequential_scores.push(score);
        }

        let batch_inputs_a: Vec<_> = frames[..3]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();
        let batch_inputs_b: Vec<_> = frames[3..]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();

        let mut batch_scores = batch.push_frame_batch(&batch_inputs_a).unwrap();
        batch_scores.extend(batch.push_frame_batch(&batch_inputs_b).unwrap());
        if let Some(score) = batch.flush().unwrap() {
            batch_scores.push(score);
        }

        assert_eq!(sequential_scores.len(), batch_scores.len());

        for (sequential_score, batch_score) in sequential_scores.iter().zip(&batch_scores) {
            assert_eq!(sequential_score.frame_index, batch_score.frame_index);
            assert_eq!(
                sequential_score.score.to_bits(),
                batch_score.score.to_bits()
            );
            assert_eq!(sequential_score.adm2.to_bits(), batch_score.adm2.to_bits());
            assert_eq!(
                sequential_score.motion2.to_bits(),
                batch_score.motion2.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale0.to_bits(),
                batch_score.vif_scale0.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale1.to_bits(),
                batch_score.vif_scale1.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale2.to_bits(),
                batch_score.vif_scale2.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale3.to_bits(),
                batch_score.vif_scale3.to_bits()
            );
        }
    }

    #[test]
    fn batch_and_sequential_paths_match_on_odd_10bit_frames() {
        let model_json = include_str!("../../../models/vmaf_v0.6.1.json");
        let (w, h, bpc) = (53usize, 37usize, 10u8);
        let modulus = 1usize << bpc;

        let frames: Vec<(Vec<u16>, Vec<u16>)> = (0..6usize)
            .map(|frame_idx| {
                let reference: Vec<u16> = (0..h)
                    .flat_map(|y| {
                        (0..w).map(move |x| {
                            ((x * 17 + y * 29 + frame_idx * 31 + (x ^ (y * 3)) + x * y * 5)
                                % modulus) as u16
                        })
                    })
                    .collect();
                let distorted = reference
                    .iter()
                    .enumerate()
                    .map(|(idx, &value)| {
                        value.saturating_sub(((frame_idx * 13 + idx * 5 + idx / w) % 37) as u16)
                    })
                    .collect();
                (reference, distorted)
            })
            .collect();

        let mut sequential = VmafContext::new(load_model(model_json).unwrap(), w, h, bpc).unwrap();
        let mut batch = VmafContext::new(load_model(model_json).unwrap(), w, h, bpc).unwrap();

        let mut sequential_scores = Vec::new();
        for (reference, distorted) in &frames {
            if let Some(score) = sequential.push_frame(reference, distorted).unwrap() {
                sequential_scores.push(score);
            }
        }
        if let Some(score) = sequential.flush().unwrap() {
            sequential_scores.push(score);
        }

        let batch_inputs_a: Vec<_> = frames[..1]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();
        let batch_inputs_b: Vec<_> = frames[1..4]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();
        let batch_inputs_c: Vec<_> = frames[4..]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();

        let mut batch_scores = batch.push_frame_batch(&batch_inputs_a).unwrap();
        batch_scores.extend(batch.push_frame_batch(&batch_inputs_b).unwrap());
        batch_scores.extend(batch.push_frame_batch(&batch_inputs_c).unwrap());
        if let Some(score) = batch.flush().unwrap() {
            batch_scores.push(score);
        }

        assert_eq!(sequential_scores.len(), batch_scores.len());

        for (sequential_score, batch_score) in sequential_scores.iter().zip(&batch_scores) {
            assert_eq!(sequential_score.frame_index, batch_score.frame_index);
            assert_eq!(
                sequential_score.score.to_bits(),
                batch_score.score.to_bits()
            );
            assert_eq!(sequential_score.adm2.to_bits(), batch_score.adm2.to_bits());
            assert_eq!(
                sequential_score.motion2.to_bits(),
                batch_score.motion2.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale0.to_bits(),
                batch_score.vif_scale0.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale1.to_bits(),
                batch_score.vif_scale1.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale2.to_bits(),
                batch_score.vif_scale2.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale3.to_bits(),
                batch_score.vif_scale3.to_bits()
            );
        }
    }

    #[test]
    fn batch_path_matches_sequential_inside_single_thread_pool() {
        let model_json = include_str!("../../../models/vmaf_v0.6.1.json");
        let (w, h, bpc) = (48usize, 32usize, 8u8);

        let frames: Vec<(Vec<u16>, Vec<u16>)> = (0..4usize)
            .map(|frame_idx| {
                let reference: Vec<u16> = (0..h)
                    .flat_map(|y| {
                        (0..w)
                            .map(move |x| ((x * 9 + y * 7 + frame_idx * 13 + (x ^ y)) % 256) as u16)
                    })
                    .collect();
                let distorted = reference
                    .iter()
                    .enumerate()
                    .map(|(idx, &value)| {
                        value.saturating_sub(((frame_idx * 5 + idx * 11 + idx / w) % 23) as u16)
                    })
                    .collect();
                (reference, distorted)
            })
            .collect();

        let mut sequential = VmafContext::new(load_model(model_json).unwrap(), w, h, bpc).unwrap();
        let mut single_thread_batch =
            VmafContext::new(load_model(model_json).unwrap(), w, h, bpc).unwrap();

        let mut sequential_scores = Vec::new();
        for (reference, distorted) in &frames {
            if let Some(score) = sequential.push_frame(reference, distorted).unwrap() {
                sequential_scores.push(score);
            }
        }
        if let Some(score) = sequential.flush().unwrap() {
            sequential_scores.push(score);
        }

        let batch_inputs: Vec<_> = frames
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut batch_scores =
            pool.install(|| single_thread_batch.push_frame_batch(&batch_inputs).unwrap());
        if let Some(score) = single_thread_batch.flush().unwrap() {
            batch_scores.push(score);
        }

        assert_eq!(sequential_scores.len(), batch_scores.len());

        for (sequential_score, batch_score) in sequential_scores.iter().zip(&batch_scores) {
            assert_eq!(sequential_score.frame_index, batch_score.frame_index);
            assert_eq!(
                sequential_score.score.to_bits(),
                batch_score.score.to_bits()
            );
            assert_eq!(sequential_score.adm2.to_bits(), batch_score.adm2.to_bits());
            assert_eq!(
                sequential_score.motion2.to_bits(),
                batch_score.motion2.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale0.to_bits(),
                batch_score.vif_scale0.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale1.to_bits(),
                batch_score.vif_scale1.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale2.to_bits(),
                batch_score.vif_scale2.to_bits()
            );
            assert_eq!(
                sequential_score.vif_scale3.to_bits(),
                batch_score.vif_scale3.to_bits()
            );
        }
    }

    #[test]
    fn context_rejects_invalid_dimensions_and_bpc() {
        assert!(matches!(
            VmafContext::new(
                load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap(),
                15,
                16,
                8
            ),
            Err(VmafError::InvalidDimensions {
                width: 15,
                height: 16
            })
        ));
        assert!(matches!(
            VmafContext::new(
                load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap(),
                16,
                15,
                8
            ),
            Err(VmafError::InvalidDimensions {
                width: 16,
                height: 15
            })
        ));
        assert!(matches!(
            VmafContext::new(
                load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap(),
                16,
                16,
                9
            ),
            Err(VmafError::InvalidBitDepth { bpc: 9 })
        ));
    }

    #[test]
    fn flush_is_terminal_and_idempotent() {
        let model = load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap();
        let (w, h, bpc) = (64usize, 64usize, 8u8);
        let reference: Vec<u16> = (0..h)
            .flat_map(|y| (0..w).map(move |x| ((x * 7 + y * 11) % 256) as u16))
            .collect();
        let distorted = reference
            .iter()
            .enumerate()
            .map(|(idx, &value)| value.saturating_sub((idx % 17) as u16))
            .collect::<Vec<_>>();

        let mut ctx = VmafContext::new(model, w, h, bpc).unwrap();
        assert!(ctx.push_frame(&reference, &distorted).unwrap().is_some());
        assert!(ctx.push_frame(&reference, &distorted).unwrap().is_none());

        let flushed = ctx.flush().unwrap();
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().frame_index, 1);
        assert!(ctx.flush().unwrap().is_none());
        assert_eq!(
            ctx.push_frame(&reference, &distorted).unwrap_err(),
            VmafError::AlreadyFlushed
        );
        assert_eq!(
            ctx.push_frame_batch(&[(&reference, &distorted)])
                .unwrap_err(),
            VmafError::AlreadyFlushed
        );
    }
}
