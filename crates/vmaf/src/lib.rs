//! vmaf — public API and pipeline orchestration — spec §2
#![deny(unsafe_code)]

use std::{
    cell::RefCell,
    collections::VecDeque,
    ops::AddAssign,
    time::{Duration, Instant},
};

use thiserror::Error;
use vmaf_adm::{AdmExtractor, AdmWorkspace};
use vmaf_cpu::MIN_FRAME_DIMENSION;
pub use vmaf_cpu::{FrameGeometry, FrameValidationError, GainLimit, GainLimitError};
pub use vmaf_model::{
    LibsvmParseError, LoadModelError, ModelValidationError, PoolMethod, VmafModel, load_model,
};
use vmaf_model::{
    collect_scores, denormalize, normalize_features, pool, score_transform, svm_predict,
};
use vmaf_motion::{Collecting as MotionCollecting, MotionError, MotionExtractor};
use vmaf_vif::{VifExtractor, VifWorkspace};
const MIN_PARALLEL_JOB_LEN: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GeometryKey {
    width: usize,
    height: usize,
    bpc: u8,
}

impl GeometryKey {
    fn new(geometry: &FrameGeometry) -> Self {
        Self {
            width: geometry.width(),
            height: geometry.height(),
            bpc: geometry.bpc(),
        }
    }
}

struct BatchWorkerScratch {
    key: GeometryKey,
    vif: VifWorkspace,
    adm: AdmWorkspace,
}

impl BatchWorkerScratch {
    fn new(vif: &VifExtractor, adm: &AdmExtractor, key: GeometryKey) -> Self {
        Self {
            key,
            vif: vif.make_workspace(),
            adm: adm.make_workspace(),
        }
    }
}

std::thread_local! {
    static BATCH_WORKER_SCRATCH: RefCell<Option<BatchWorkerScratch>> = const { RefCell::new(None) };
}

struct BatchWorkerScratchLease {
    scratch: Option<BatchWorkerScratch>,
}

impl BatchWorkerScratchLease {
    fn acquire(vif: &VifExtractor, adm: &AdmExtractor, key: GeometryKey) -> Self {
        let scratch = BATCH_WORKER_SCRATCH.with(|cell| {
            let maybe_scratch = cell.borrow_mut().take();
            match maybe_scratch {
                Some(scratch) if scratch.key == key => scratch,
                _ => BatchWorkerScratch::new(vif, adm, key),
            }
        });
        Self {
            scratch: Some(scratch),
        }
    }

    fn scratch_mut(&mut self) -> &mut BatchWorkerScratch {
        self.scratch
            .as_mut()
            .expect("worker scratch should exist while the lease is alive")
    }
}

impl Drop for BatchWorkerScratchLease {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            BATCH_WORKER_SCRATCH.with(|cell| {
                *cell.borrow_mut() = Some(scratch);
            });
        }
    }
}

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
    /// A lower-level motion error that should be surfaced to callers.
    #[error(transparent)]
    Motion(MotionError),
}

impl From<FrameValidationError> for VmafError {
    fn from(err: FrameValidationError) -> Self {
        match err {
            FrameValidationError::InvalidDimensions { width, height } => {
                Self::InvalidDimensions { width, height }
            }
            FrameValidationError::InvalidBitDepth { bpc } => Self::InvalidBitDepth { bpc },
            FrameValidationError::SampleCountOverflow { width, height } => {
                Self::SampleCountOverflow { width, height }
            }
        }
    }
}

impl From<MotionError> for VmafError {
    fn from(err: MotionError) -> Self {
        match err {
            MotionError::FrameValidation(err) => err.into(),
            MotionError::InvalidPlaneLength { actual, required }
            | MotionError::InvalidBlurredPlaneLength { actual, required } => {
                Self::InvalidReferenceLength { actual, required }
            }
            err @ MotionError::InvalidStride { .. } => Self::Motion(err),
        }
    }
}

/// Per-frame VMAF scores and features.
#[derive(Clone, Copy, Debug)]
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

/// Profiling timings for one or more frame-processing calls.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessingTimings {
    pub validation: Duration,
    pub feature_extraction: Duration,
    pub motion: Duration,
    pub finalize: Duration,
    pub total: Duration,
}

impl AddAssign for ProcessingTimings {
    fn add_assign(&mut self, rhs: Self) {
        self.validation += rhs.validation;
        self.feature_extraction += rhs.feature_extraction;
        self.motion += rhs.motion;
        self.finalize += rhs.finalize;
        self.total += rhs.total;
    }
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

#[doc(hidden)]
pub struct CollectingData {
    vif_workspace: VifWorkspace,
    adm_workspace: AdmWorkspace,
    motion: MotionExtractor<MotionCollecting>,
    pending: VecDeque<PendingFrame>,
    frame_count: usize,
}

#[doc(hidden)]
pub struct FinalizedData;

#[doc(hidden)]
pub trait ContextState: private::Sealed {
    #[doc(hidden)]
    type Data;
}

/// VMAF context state that still accepts input frames.
pub struct Collecting;

/// VMAF context state after the final pending score has been finalized.
pub struct Finalized;

impl ContextState for Collecting {
    type Data = CollectingData;
}

impl ContextState for Finalized {
    type Data = FinalizedData;
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::Collecting {}
    impl Sealed for super::Finalized {}
}

/// VMAF scoring context for a single video sequence.
pub struct VmafContext<S: ContextState = Collecting> {
    model: VmafModel,
    vif: VifExtractor,
    adm: AdmExtractor,
    geometry: FrameGeometry,
    per_frame_scores: Vec<FrameScore>,
    opts: VmafOptions,
    state: S::Data,
}

impl VmafContext<Collecting> {
    /// Create a VMAF scoring context for one sequence.
    pub fn new(model: VmafModel, geometry: FrameGeometry) -> Self {
        Self::new_with_options(model, geometry, VmafOptions::default())
    }

    /// Create a VMAF scoring context with custom options.
    pub fn new_with_options(model: VmafModel, geometry: FrameGeometry, opts: VmafOptions) -> Self {
        let vif = VifExtractor::new(geometry, model.vif_enhn_gain_limit);
        let vif_workspace = vif.make_workspace();
        let adm = AdmExtractor::new(geometry, model.adm_enhn_gain_limit);
        let adm_workspace = adm.make_workspace();
        Self {
            vif,
            adm,
            model,
            geometry,
            per_frame_scores: Vec::new(),
            opts,
            state: CollectingData {
                vif_workspace,
                adm_workspace,
                motion: MotionExtractor::new(geometry),
                pending: VecDeque::with_capacity(2),
                frame_count: 0,
            },
        }
    }

    /// Push a reference/distorted frame pair. Returns a `FrameScore` when one
    /// becomes available (motion has a 1-frame lag).
    ///
    /// # Errors
    ///
    /// Returns [`VmafError`] if either plane does not match the configured frame
    /// size.
    pub fn push_frame(
        &mut self,
        reference: &[u16],
        distorted: &[u16],
    ) -> Result<Option<FrameScore>, VmafError> {
        self.push_frame_impl(reference, distorted, true, None)
    }

    /// Push a batch of reference/distorted frame pairs in parallel.
    /// Returns a list of `FrameScore`s that have become available.
    ///
    /// # Errors
    ///
    /// Returns [`VmafError`] if any frame pair does not match the configured
    /// frame size.
    pub fn push_frame_batch(
        &mut self,
        frames: &[(&[u16], &[u16])],
    ) -> Result<Vec<FrameScore>, VmafError> {
        self.push_frame_batch_impl(frames, None)
    }

    /// Push a batch of frames and return profiling timings for the work performed.
    #[doc(hidden)]
    pub fn push_frame_batch_with_timings(
        &mut self,
        frames: &[(&[u16], &[u16])],
    ) -> Result<(Vec<FrameScore>, ProcessingTimings), VmafError> {
        let mut timings = ProcessingTimings::default();
        let scores = self.push_frame_batch_impl(frames, Some(&mut timings))?;
        Ok((scores, timings))
    }

    fn push_frame_batch_impl(
        &mut self,
        frames: &[(&[u16], &[u16])],
        mut timings: Option<&mut ProcessingTimings>,
    ) -> Result<Vec<FrameScore>, VmafError> {
        let validation_start = Instant::now();
        for &(reference, distorted) in frames {
            self.validate_frame_inputs(reference, distorted)?;
        }
        if let Some(timings) = timings.as_mut() {
            let elapsed = validation_start.elapsed();
            timings.validation += elapsed;
            timings.total += elapsed;
        }

        if frames.len() <= 1 || rayon::current_num_threads() <= 1 {
            let mut out = Vec::with_capacity(frames.len());
            for &(reference, distorted) in frames {
                if let Some(score) =
                    self.push_frame_impl(reference, distorted, false, timings.as_deref_mut())?
                {
                    out.push(score);
                }
            }
            return Ok(out);
        }

        use rayon::prelude::*;

        let vif = &self.vif;
        let adm = &self.adm;
        let motion = &self.state.motion;
        let stride = self.geometry.width();
        let geometry_key = GeometryKey::new(&self.geometry);
        let processing_start = Instant::now();
        let feature_start = Instant::now();
        let min_job_len = frames
            .len()
            .div_ceil(rayon::current_num_threads())
            .max(MIN_PARALLEL_JOB_LEN);
        let extracted: Vec<_> = frames
            .par_iter()
            .with_min_len(min_job_len)
            .map_init(
                || BatchWorkerScratchLease::acquire(vif, adm, geometry_key),
                |lease, (r, d)| {
                    let scratch = lease.scratch_mut();
                    let vif_scores = vif.compute_frame_with_workspace(&mut scratch.vif, r, d);
                    let adm_score = adm.compute_frame_with_workspace(&mut scratch.adm, r, d);
                    let blur = motion.prepare_blurred_frame(r, stride);
                    (vif_scores, adm_score, blur)
                },
            )
            .collect();
        if let Some(timings) = timings.as_mut() {
            timings.feature_extraction += feature_start.elapsed();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (vif_scores, adm_score, blur) in extracted {
            let enqueue_start = Instant::now();
            self.state.pending.push_back(PendingFrame {
                frame_idx: self.state.frame_count,
                adm: adm_score,
                vif: vif_scores.scale,
            });
            self.state.frame_count += 1;
            if let Some(timings) = timings.as_mut() {
                timings.finalize += enqueue_start.elapsed();
            }

            let motion_start = Instant::now();
            let motion_result = self.state.motion.push_blurred_frame(blur)?;
            if let Some(timings) = timings.as_mut() {
                timings.motion += motion_start.elapsed();
            }

            if let Some((idx, m2)) = motion_result {
                let finalize_start = Instant::now();
                if let Some(fs) = self.finalize_frame(idx, m2 as f64) {
                    out.push(fs);
                }
                if let Some(timings) = timings.as_mut() {
                    timings.finalize += finalize_start.elapsed();
                }
            }
        }
        if let Some(timings) = timings.as_mut() {
            timings.total += processing_start.elapsed();
        }
        Ok(out)
    }

    /// Flush the final pending frame and transition into the finalized state.
    pub fn flush(mut self) -> VmafContext<Finalized> {
        let CollectingData {
            motion,
            mut pending,
            ..
        } = self.state;
        let (_motion, pending_score) = motion.flush();
        if let Some((idx, m2)) = pending_score {
            let _ = finalize_frame_components(
                &self.model,
                self.opts,
                &mut pending,
                &mut self.per_frame_scores,
                idx,
                m2 as f64,
            );
        }
        debug_assert!(pending.is_empty());

        VmafContext {
            model: self.model,
            vif: self.vif,
            adm: self.adm,
            geometry: self.geometry,
            per_frame_scores: self.per_frame_scores,
            opts: self.opts,
            state: FinalizedData,
        }
    }

    fn finalize_frame(&mut self, idx: usize, m2: f64) -> Option<FrameScore> {
        finalize_frame_components(
            &self.model,
            self.opts,
            &mut self.state.pending,
            &mut self.per_frame_scores,
            idx,
            m2,
        )
    }

    fn push_frame_impl(
        &mut self,
        reference: &[u16],
        distorted: &[u16],
        validate_inputs: bool,
        mut timings: Option<&mut ProcessingTimings>,
    ) -> Result<Option<FrameScore>, VmafError> {
        let total_start = Instant::now();

        if validate_inputs {
            let validation_start = Instant::now();
            self.validate_frame_inputs(reference, distorted)?;
            if let Some(timings) = timings.as_mut() {
                timings.validation += validation_start.elapsed();
            }
        }

        let feature_start = Instant::now();
        let vif_scores = self.vif.compute_frame_with_workspace(
            &mut self.state.vif_workspace,
            reference,
            distorted,
        );
        let adm_score = self.adm.compute_frame_with_workspace(
            &mut self.state.adm_workspace,
            reference,
            distorted,
        );
        if let Some(timings) = timings.as_mut() {
            timings.feature_extraction += feature_start.elapsed();
        }

        let motion_start = Instant::now();
        let motion_result = self
            .state
            .motion
            .push_frame(reference, self.geometry.width())?;
        if let Some(timings) = timings.as_mut() {
            timings.motion += motion_start.elapsed();
        }

        let finalize_start = Instant::now();
        self.state.pending.push_back(PendingFrame {
            frame_idx: self.state.frame_count,
            adm: adm_score,
            vif: vif_scores.scale,
        });
        self.state.frame_count += 1;

        let result = motion_result.and_then(|(idx, m2)| self.finalize_frame(idx, m2 as f64));
        if let Some(timings) = timings.as_mut() {
            timings.finalize += finalize_start.elapsed();
            timings.total += total_start.elapsed();
        }
        Ok(result)
    }

    fn validate_frame_inputs(&self, reference: &[u16], distorted: &[u16]) -> Result<(), VmafError> {
        let required = self.geometry.sample_count();
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

impl VmafContext<Finalized> {
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
}

fn finalize_frame_components(
    model: &VmafModel,
    opts: VmafOptions,
    pending: &mut VecDeque<PendingFrame>,
    per_frame_scores: &mut Vec<FrameScore>,
    idx: usize,
    m2: f64,
) -> Option<FrameScore> {
    let pf = match pending.front() {
        Some(front) if front.frame_idx == idx => pending.pop_front(),
        _ => {
            let pos = pending.iter().position(|p| p.frame_idx == idx)?;
            pending.remove(pos)
        }
    }?;

    // Feature order: [adm2, motion2, vif_scale0, vif_scale1, vif_scale2, vif_scale3] — spec §6
    let raw: [f64; 6] = [pf.adm, m2, pf.vif[0], pf.vif[1], pf.vif[2], pf.vif[3]];
    let normed = normalize_features(&raw, &model.feature_slopes, &model.feature_intercepts);
    let raw_svm = svm_predict(&model.svm, &normed);
    let denorm = denormalize(raw_svm, model.score_slope, model.score_intercept);
    let st = if opts.apply_score_transform {
        model.score_transform.as_ref()
    } else {
        None
    };
    let score = score_transform(denorm, st, model.score_clip);

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
    per_frame_scores.push(fs);
    Some(fs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: usize, height: usize, bpc: u8) -> FrameGeometry {
        FrameGeometry::new(width, height, bpc).unwrap()
    }

    #[test]
    fn parallel_job_len_has_two_frame_floor() {
        let len = 8usize.div_ceil(8).max(MIN_PARALLEL_JOB_LEN);
        assert_eq!(len, 2);
    }

    #[test]
    fn neg_model_feature_opts_change_scores() {
        let pos = load_model(include_str!("../../../models/vmaf_v0.6.1.json")).unwrap();
        let neg = load_model(include_str!("../../../models/vmaf_v0.6.1neg.json")).unwrap();

        assert_eq!(pos.adm_enhn_gain_limit.value(), 100.0);
        assert_eq!(pos.vif_enhn_gain_limit.value(), 100.0);
        assert_eq!(neg.adm_enhn_gain_limit.value(), 1.0);
        assert_eq!(neg.vif_enhn_gain_limit.value(), 1.0);

        let (w, h, bpc) = (64usize, 64usize, 8u8);

        let reference: Vec<u16> = (0..h)
            .flat_map(|y| (0..w).map(move |x| ((x + y) % 128) as u16))
            .collect();
        let distorted: Vec<u16> = reference.iter().map(|&v| (v * 2).min(255)).collect();

        let mut ctx_pos = VmafContext::new(pos, geometry(w, h, bpc));
        let mut ctx_neg = VmafContext::new(neg, geometry(w, h, bpc));

        // Push 2 frames to satisfy motion's 1-frame lag.
        for _ in 0..2 {
            ctx_pos.push_frame(&reference, &distorted).unwrap();
            ctx_neg.push_frame(&reference, &distorted).unwrap();
        }
        let ctx_pos: VmafContext<Finalized> = ctx_pos.flush();
        let ctx_neg: VmafContext<Finalized> = ctx_neg.flush();

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

        let mut sequential = VmafContext::new(load_model(model_json).unwrap(), geometry(w, h, bpc));
        let mut batch = VmafContext::new(load_model(model_json).unwrap(), geometry(w, h, bpc));

        for (reference, distorted) in &frames {
            sequential.push_frame(reference, distorted).unwrap();
        }
        let sequential: VmafContext<Finalized> = sequential.flush();

        let batch_inputs_a: Vec<_> = frames[..3]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();
        let batch_inputs_b: Vec<_> = frames[3..]
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();

        batch.push_frame_batch(&batch_inputs_a).unwrap();
        batch.push_frame_batch(&batch_inputs_b).unwrap();
        let batch: VmafContext<Finalized> = batch.flush();
        let sequential_scores = sequential.per_frame_scores();
        let batch_scores = batch.per_frame_scores();

        assert_eq!(sequential_scores.len(), batch_scores.len());

        for (sequential_score, batch_score) in sequential_scores.iter().zip(batch_scores.iter()) {
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

        let mut sequential = VmafContext::new(load_model(model_json).unwrap(), geometry(w, h, bpc));
        let mut batch = VmafContext::new(load_model(model_json).unwrap(), geometry(w, h, bpc));

        for (reference, distorted) in &frames {
            sequential.push_frame(reference, distorted).unwrap();
        }
        let sequential: VmafContext<Finalized> = sequential.flush();

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

        batch.push_frame_batch(&batch_inputs_a).unwrap();
        batch.push_frame_batch(&batch_inputs_b).unwrap();
        batch.push_frame_batch(&batch_inputs_c).unwrap();
        let batch: VmafContext<Finalized> = batch.flush();
        let sequential_scores = sequential.per_frame_scores();
        let batch_scores = batch.per_frame_scores();

        assert_eq!(sequential_scores.len(), batch_scores.len());

        for (sequential_score, batch_score) in sequential_scores.iter().zip(batch_scores.iter()) {
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

        let mut sequential = VmafContext::new(load_model(model_json).unwrap(), geometry(w, h, bpc));
        let mut single_thread_batch =
            VmafContext::new(load_model(model_json).unwrap(), geometry(w, h, bpc));

        for (reference, distorted) in &frames {
            sequential.push_frame(reference, distorted).unwrap();
        }
        let sequential: VmafContext<Finalized> = sequential.flush();

        let batch_inputs: Vec<_> = frames
            .iter()
            .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| single_thread_batch.push_frame_batch(&batch_inputs).unwrap());
        let single_thread_batch: VmafContext<Finalized> = single_thread_batch.flush();
        let sequential_scores = sequential.per_frame_scores();
        let batch_scores = single_thread_batch.per_frame_scores();

        assert_eq!(sequential_scores.len(), batch_scores.len());

        for (sequential_score, batch_score) in sequential_scores.iter().zip(batch_scores.iter()) {
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
            FrameGeometry::new(15, 16, 8),
            Err(FrameValidationError::InvalidDimensions {
                width: 15,
                height: 16
            })
        ));
        assert!(matches!(
            FrameGeometry::new(16, 15, 8),
            Err(FrameValidationError::InvalidDimensions {
                width: 16,
                height: 15
            })
        ));
        assert!(matches!(
            FrameGeometry::new(16, 16, 9),
            Err(FrameValidationError::InvalidBitDepth { bpc: 9 })
        ));
    }

    #[test]
    fn flush_transitions_to_finalized_state() {
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

        let mut ctx = VmafContext::new(model, geometry(w, h, bpc));
        assert!(ctx.push_frame(&reference, &distorted).unwrap().is_some());
        assert!(ctx.push_frame(&reference, &distorted).unwrap().is_none());

        let ctx: VmafContext<Finalized> = ctx.flush();
        assert_eq!(ctx.per_frame_scores().len(), 2);
        assert_eq!(ctx.per_frame_scores()[1].frame_index, 1);
        assert!(ctx.pool_score(PoolMethod::Mean, 1).is_finite());
    }
}
