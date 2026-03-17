# vmaf-rs2

vmaf-rs2 is a project that implements the integer arithmetic pipeline of [VMAF](https://github.com/Netflix/vmaf), a video quality evaluation metric, in Rust. It provides both a CLI tool and a library.

It implements the integer arithmetic path in compliance with the [Design Specification](docs/VMAF_DESIGN_SPEC.md) and supports runtime SIMD dynamic dispatch as well as parallel batch processing.

## Quick Start

```bash
cargo build --release

./target/release/vmaf \
  -r reference.y4m \
  -d distorted.y4m \
  -m models/vmaf_v0.6.1.json
```

## CLI Usage

```bash
vmaf -r <reference.y4m> -d <distorted.y4m> -m <model.json> [OPTIONS]
```

| Flag                      | Default    | Description                                    |
| ------------------------- | ---------- | ---------------------------------------------- |
| `-r, --reference`         | *required* | Reference Y4M file                             |
| `-d, --distorted`         | *required* | Distorted Y4M file                             |
| `-m, --model`             | *required* | VMAF JSON model file                           |
| `--pool-method`           | `mean`     | Pooling: `mean`, `harmonic_mean`, `min`, `max` |
| `--n-subsample`           | `1`        | Frame subsampling factor (1 = all frames)      |
| `--threads`               | `0`        | Thread count (0 = auto)                        |
| `-q, --quiet`             |            | Suppress progress on stderr                    |
| `--json`                  |            | Output per-frame JSON report                   |
| `--apply-score-transform` |            | Apply model's score transform block            |

### Examples

```bash
# Basic scoring
vmaf -r ref.y4m -d dist.y4m -m models/vmaf_v0.6.1.json

# Per-frame JSON, no progress bar
vmaf -r ref.y4m -d dist.y4m -m models/vmaf_v0.6.1.json --quiet --json

# 4 threads, harmonic mean pooling
vmaf -r ref.y4m -d dist.y4m -m models/vmaf_v0.6.1.json --threads 4 --pool-method harmonic_mean

# Score every 5th frame with the negative-enhancement model
vmaf -r ref.y4m -d dist.y4m -m models/vmaf_v0.6.1neg.json --n-subsample 5
```

## Library Usage

```rust
use vmaf::{load_model, PoolMethod, VmafContext};

let model = load_model(&std::fs::read_to_string("models/vmaf_v0.6.1.json")?)?;
let mut ctx = VmafContext::new(model, width, height, 8 /* bpc */);

for (ref_frame, dis_frame) in frames {
    if let Some(score) = ctx.push_frame(ref_frame, dis_frame) {
        println!("Frame {}: {:.4}", score.frame_index, score.score);
    }
}
ctx.flush(); // emit final frame (motion has 1-frame lag)

let pooled = ctx.pool_score(PoolMethod::Mean, 1);
```

For parallel processing, use `push_frame_batch()` which distributes VIF/ADM extraction across a Rayon thread pool while preserving motion's sequential dependency.

## Performance

- Release profile enables LTO and single codegen unit
- Runtime SIMD dispatch: AVX-512 > AVX2/FMA > SSE2 > scalar (x86); Neon (aarch64)
- Batch processing parallelizes VIF and ADM extraction with Rayon
- Only luma plane is consumed; 8/10/12-bit input supported

## Specification

The implementation follows `docs/VMAF_DESIGN_SPEC.md`, an IEEE 1016-2009 format software design description covering:

- Feature extraction: VIF (§4.2), ADM (§4.3), Motion (§4.4)
- SVM model scoring (§3.3, §5)
- Score transform and pooling (§6)
- Conformance vectors (§8)
