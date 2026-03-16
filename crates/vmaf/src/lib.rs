//! vmaf — public API and pipeline orchestration — spec §2
#![deny(unsafe_code)]

use vmaf_adm::AdmExtractor;
use vmaf_model::{
    collect_scores, denormalize, normalize_features, pool, score_transform, svm_predict,
};
use vmaf_motion::MotionExtractor;
use vmaf_vif::VifExtractor;

pub use vmaf_model::{load_model, PoolMethod, VmafModel};

/// Per-frame VMAF scores and features.
#[derive(Clone)]
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
#[derive(Clone, Copy, Debug)]
#[derive(Default)]
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
    adm: AdmExtractor,
    motion: MotionExtractor,
    width: usize,
    pending: Vec<PendingFrame>,
    per_frame_scores: Vec<FrameScore>,
    frame_count: usize,
    opts: VmafOptions,
}

impl VmafContext {
    pub fn new(model: VmafModel, width: usize, height: usize, bpc: u8) -> Self {
        Self::new_with_options(model, width, height, bpc, VmafOptions::default())
    }

    pub fn new_with_options(
        model: VmafModel,
        width: usize,
        height: usize,
        bpc: u8,
        opts: VmafOptions,
    ) -> Self {
        Self {
            vif: VifExtractor::new(width, height, bpc, model.vif_enhn_gain_limit),
            adm: AdmExtractor::new(width, height, bpc, model.adm_enhn_gain_limit),
            motion: MotionExtractor::new(width, height, bpc),
            model,
            width,
            pending: Vec::new(),
            per_frame_scores: Vec::new(),
            frame_count: 0,
            opts,
        }
    }

    /// Push a reference/distorted frame pair. Returns a `FrameScore` when one
    /// becomes available (motion has a 1-frame lag).
    pub fn push_frame(&mut self, reference: &[u16], distorted: &[u16]) -> Option<FrameScore> {
        let vif_scores = self.vif.compute_frame(reference, distorted);
        let adm_score = self.adm.compute_frame(reference, distorted);
        let motion_result = self.motion.push_frame(reference, self.width);

        self.pending.push(PendingFrame {
            frame_idx: self.frame_count,
            adm: adm_score,
            vif: vif_scores.scale,
        });
        self.frame_count += 1;

        motion_result.and_then(|(idx, m2)| self.finalize_frame(idx, m2 as f64))
    }

    /// Flush the final pending frame. Call once after all frames are pushed.
    pub fn flush(&mut self) -> Option<FrameScore> {
        self.motion
            .flush()
            .and_then(|(idx, m2)| self.finalize_frame(idx, m2 as f64))
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

        let mut ctx_pos = VmafContext::new(pos, w, h, bpc);
        let mut ctx_neg = VmafContext::new(neg, w, h, bpc);

        // Push 2 frames to satisfy motion's 1-frame lag.
        for _ in 0..2 {
            ctx_pos.push_frame(&reference, &distorted);
            ctx_neg.push_frame(&reference, &distorted);
        }
        ctx_pos.flush();
        ctx_neg.flush();

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
}
