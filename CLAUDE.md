# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vmaf-rs2 is a Rust re-implementation of Netflix's VMAF (Video Multi-method Assessment Fusion) integer pipeline. It implements the specification in `docs/VMAF_DESIGN_SPEC.md` (IEEE 1016-2009 format SDD). The goal is bit-exact or functionally equivalent output to libvmaf's integer path.

## Build & Test Commands

```bash
cargo build                          # Debug build
cargo build --release                # Release build (LTO, single codegen unit)
cargo test                           # All workspace tests
cargo test -p vmaf-vif               # Single crate tests
cargo test test_name                 # Single test by name
cargo clippy --workspace             # Lint all crates
cargo run --release -p vmaf-cli -- --reference ref.y4m --distorted dist.y4m --model models/vmaf_v0.6.1.json
```

## Workspace Crates

Seven crates in `crates/`, ordered by dependency:

- **vmaf-cpu** — SIMD backend detection (`SimdBackend` enum: Scalar, SSE2, AVX2/FMA, AVX-512, Neon). Runtime CPU feature detection, alignment helpers, safe casting.
- **vmaf-vif** — Visual Information Fidelity (spec §4.2). 4-scale Gaussian pyramid, filter-based subsampling. Has `build.rs` generating Q11 log2 lookup table (must use f32, not f64).
- **vmaf-adm** — Additive Distortion Measure (spec §4.3). Integer DWT, decouple, CSF/CM weighting, scoring. Scale 0 uses i16 subbands, scales 1-3 use i32. DIV_LOOKUP reciprocal table.
- **vmaf-motion** — Motion2 feature (spec §4.4). Stateful 3-slot ring buffer for frame blur states. 1-frame lag: `push_frame` returns motion2[n-1], `flush` returns motion2[last].
- **vmaf-model** — SVM model loading and scoring (spec §3.3, §5). JSON model parser, feature normalization, RBF kernel SVM-Nu SVR, score transform, pooling.
- **vmaf** — Orchestration library. `VmafContext` with `push_frame()`, `push_frame_batch()` (Rayon parallel), `flush()`, `pool_score()`. Feature order: [adm2, motion2, vif_scale0-3].
- **vmaf-cli** — CLI tool. Y4M input, clap args, progress bar, JSON output, configurable thread count.

## Architecture

**Pipeline flow:** Frame pixels → VIF/ADM/Motion extractors → 6 features → normalize → SVM predict → denormalize → score_transform → clip [0,100] → pool across frames.

**Key patterns:**

- VIF and ADM extractors are stateless (compute per-frame independently).
- Motion extractor is stateful (ring buffer, 1-frame lag).
- All feature extraction uses fixed-point integer arithmetic throughout, matching libvmaf's integer path.
- SIMD backends are selected at runtime via `SimdBackend::detect()` with fallback chain (AVX-512 → AVX2/FMA → SSE2 → Scalar on x86).
- Each feature crate has platform-specific kernel modules under `backend/` (x86, aarch64).

**Batch processing:** `push_frame_batch()` uses Rayon to parallelize feature extraction across frames but preserves motion2's sequential dependency internally.

## Models

JSON model files in `models/`. Default: `vmaf_v0.6.1.json`. The `neg` variant adds enhancement gain limiting for ADM/VIF.

## Spec Conformance

Tests verify against §8 conformance vectors from the design spec. When implementing new functionality, refer to the relevant spec section in `docs/VMAF_DESIGN_SPEC.md`.
