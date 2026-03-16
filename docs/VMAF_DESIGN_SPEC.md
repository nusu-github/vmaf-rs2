# VMAF (libvmaf) Software Design Description (SDD)

**Standard:** IEEE 1016-2009 Optimized
**Target:** Clean-room re-implementation in any programming language
**Version:** 2.3

---

## 1. Introduction

### 1.1 Purpose

This document provides a complete technical specification of the VMAF (Video Multi-method Assessment Fusion) integer-pipeline implementation (`libvmaf`). It is designed to allow a software engineer to implement a **bit-exact or functionally equivalent** version of VMAF in **any programming language** without access to any existing implementation source code.

All algorithms are expressed in language-neutral pseudocode. No language-specific libraries or idioms are assumed.

### 1.2 Scope

This specification covers the **integer-based pipeline**, which is the production standard for VMAF. It includes:

- Frame management and pixel format handling
- Feature extraction (VIF, ADM, Motion)
- SVM-based score prediction
- Score transformation and pooling

### 1.3 Notation

- `>>` right bit shift (logical unless noted)
- `<<` left bit shift
- `^` exponentiation (not XOR)
- `|x|` absolute value of x
- `floor(x)` largest integer ≤ x
- `clz32(x)` count of leading zero bits in a 32-bit unsigned integer
- `clz64(x)` count of leading zero bits in a 64-bit unsigned integer
- `uint8`, `uint16`, `uint32`, `uint64` unsigned integers
- `int16`, `int32`, `int64` signed integers
- `f32` 32-bit floating point (single precision)
- `f64` 64-bit floating point (double precision)

**Portable clz implementations:**

```
function clz32(x: uint32) -> int:
    if x == 0: return 32
    n = 0
    if (x & 0xFFFF0000) == 0: n += 16; x <<= 16
    if (x & 0xFF000000) == 0: n +=  8; x <<=  8
    if (x & 0xF0000000) == 0: n +=  4; x <<=  4
    if (x & 0xC0000000) == 0: n +=  2; x <<=  2
    if (x & 0x80000000) == 0: n +=  1
    return n
```

The 64-bit variant `clz64` follows the same pattern over 64 bits.

### 1.4 Portability and Conformance Requirements

This specification is intentionally **language-neutral** but not **numerically permissive**. A conforming implementation in another programming language must reproduce the arithmetic behavior defined here, even if the host language has different default numeric semantics.

**Required numeric model:**

- Unsigned integer arithmetic on `uintN` values is modulo `2^N`.
- Signed integer arithmetic must use two's-complement ranges for `int16`, `int32`, and `int64`.
- Any right shift explicitly described as **arithmetic** must sign-extend negative values. If the host language does not define arithmetic right shift on signed integers, implement it manually.
- Any right shift not explicitly marked arithmetic operates on unsigned values and is therefore logical.
- `f32` and `f64` mean IEEE 754 binary32 and binary64 respectively.
- Floating-point expressions must not be algebraically reordered if reordering can change the final rounded result.
- `round(x)` means round to nearest integer with ties **away from zero**. If the host language's default rounding differs, provide a helper that matches this rule.
- Unless a different rule is stated explicitly, integer division truncates toward zero.
- A cast from `f32`/`f64` to a signed or unsigned integer type truncates toward zero.
- A float-to-integer cast is valid only when the source value is finite and the truncated result is representable in the destination type. Out-of-range or NaN/Inf casts are invalid under this specification.
- `clamp(x, lo, hi)` means `min(max(x, lo), hi)`.

**Math library requirements:**

- `log2`, `exp`, `log10`, and `cbrt` are normative mathematical functions in this specification.
- For bit-exact conformance, each such call must equal the **correctly rounded** result of the real-valued function in the destination precision (`f32` or `f64` as specified at each call site), using IEEE 754 round-to-nearest, ties-to-even.
- If the host runtime cannot guarantee that result, the implementation must replace the runtime call with a custom correctly rounded routine, a precomputed lookup table, or precomputed constants.
- A result that differs by 1 ulp or more from the correctly rounded destination-precision value is non-conforming.
- When this document requires a computation in `f32`, intermediate widening to `f64` and narrowing back to `f32` is not conforming.
- NaN and Inf are not valid intermediate or final values for well-formed inputs under this specification. If an implementation produces them, that implementation is non-conforming.

**Input validity requirements:**

- Frame width and height must each be at least 16. This guarantees that all four VIF scales retain dimensions of at least `2 × 2`.
- `bpc` must be exactly one of `8`, `10`, or `12`.
- Only the luma plane is consumed; chroma planes are ignored.
- All frame pairs presented to the context must have identical width, height, and `bpc`.
- The model JSON must contain the fields described in §3.3 with compatible types and array lengths.
- If `score_transform.knots` is present, it must contain at least 2 points, the `x` coordinates must be strictly increasing, and the `y` coordinates must be nondecreasing.

**Concurrency requirements:**

- Threading is an implementation choice, not part of the functional contract.
- A single-threaded implementation is conforming if it produces the same per-frame feature values, predictions, and pooled scores.
- When a concurrent implementation is used, observable results must be identical to a valid single-threaded execution.

---

## 2. System Architecture

### 2.1 Processing Pipeline

VMAF operates as a stateful engine (`VmafContext`):

```
1. Configuration  → thread count, subsampling, model path
2. Model Loading  → parse JSON (SVM params, feature normalization)
3. Frame Input    → accept pairs of VmafPicture (Reference, Distorted)
4. Feature Extraction → parallel execution per frame:
       ├── Integer VIF  (4 scale scores)
       ├── Integer ADM  (1 combined score)
       └── Integer Motion2 (1 score, from reference frames only)
5. Score Collection → buffer (feature_name, frame_index) → score
6. Prediction  → SVM per frame → raw VMAF score
7. Transform   → polynomial / piecewise-linear / rectify / clip
8. Pooling     → mean / harmonic-mean / min / max over frame range
```

**Subsampling semantics:** `subsampling` (also written `n_subsample` in §5.5) affects only the final frame-selection step used for pooled output aggregation. It does **not** permit skipping feature extraction, Motion state updates, or per-frame prediction on intermediate frames. A conforming implementation computes features and per-frame scores for every input frame in presentation order, then applies subsampling only when collecting the frame scores that participate in pooling.

### 2.2 Threading Model

- A thread pool processes frames in parallel.
- `FeatureCollector`: a thread-safe map from `(feature_name, frame_index)` to `f64`.
  - Multiple extractor threads write concurrently; use mutex or lock-free map.
  - Prediction for frame N **blocks** until all required features for frame N are written.

---

## 3. Data Structures

### 3.1 VmafPicture

| Field    | Type   | Description                                 |
| -------- | ------ | ------------------------------------------- |
| `data`   | byte[] | Pixel planes. Only the **Y-plane** is used. |
| `stride` | uint32 | Row byte length (may include padding)       |
| `width`  | uint32 | Frame width in pixels                       |
| `height` | uint32 | Frame height in pixels                      |
| `bpc`    | uint8  | Bits per component: 8, 10, or 12            |

**Normative Y-plane representation:**

- The Y plane is planar, not interleaved with chroma.
- `stride` is measured in **bytes per row**, not samples per row.
- For `bpc = 8`, each luma sample is stored as one `uint8`.
- For `bpc = 10` or `12`, each luma sample is stored as one `uint16`.
- For `bpc = 10` or `12`, the sample value is **right-justified** in the 16-bit element. The valid numeric range is `0 .. (2^bpc - 1)`. Bits above bit `bpc-1` must be zero for well-formed input.
- The algorithms in this document operate on sample values, not serialized file bytes. When reading from an external file format, any file-format-specific packing or endianness must be resolved before populating `VmafPicture`.
- For pseudocode that indexes `src[i][j]`, interpret the storage as:
  - 8-bit path: `sample = ((uint8*)data)[i * stride + j]`
  - 10/12-bit path: `sample = ((uint16*)data)[i * (stride / 2) + j]`

The contiguous Y/U/V allocation strategy used by some implementations is not part of the functional contract. Only the per-plane sample values and byte strides are normative.

### 3.2 Bit-Depth Normalization

For higher bit depths, pixel values are larger; the algorithms use bit-shift adjustments at the convolution stage rather than rescaling sigma_nsq.

### 3.3 VmafModel (JSON Schema)

```json
{
  "model_dict": {
    "model_type": "LIBSVMNUSVR",
    "norm_type": "linear_rescale",
    "model": "<embedded LIBSVM text format>",
    "feature_names": [
      "VMAF_integer_feature_adm2_score",
      "VMAF_integer_feature_motion2_score",
      "VMAF_integer_feature_vif_scale0_score",
      "VMAF_integer_feature_vif_scale1_score",
      "VMAF_integer_feature_vif_scale2_score",
      "VMAF_integer_feature_vif_scale3_score"
    ],
    "slopes":     [0.012020766, 2.8098077503, 0.0626440747, 1.2227634563, 1.5360318811, 1.7620864996, 2.0865646829],
    "intercepts": [-0.3092982,  -1.7993969,   -0.0030172,   -0.1728125,   -0.5294309,   -0.7577186,   -1.0834286],
    "slope":      0.012020766,
    "intercept":  -0.3092981928,
    "score_clip": [0.0, 100.0],
    "score_transform": {
      "p0": 1.70674692,
      "p1": 1.72643844,
      "p2": -0.00705305,
      "knots": [[0.0, 0.0], [100.0, 100.0]],
      "out_gte_in": true
    }
  }
}
```

Note: `slopes[0]` and `intercepts[0]` equal the top-level `slope`/`intercept` fields. Feature normalization uses `slopes[1..6]` and `intercepts[1..6]` (1-based indexing in the JSON array, matching SVM feature indices 1–6).

If `score_transform.knots` is present, its JSON representation is an array of 2-element numeric arrays:

```json
"knots": [
  [x0, y0],
  [x1, y1],
  ...
]
```

Each inner array must contain exactly two numbers, interpreted as `(x, y)`.

#### feature_opts_dicts (Per-feature option dictionaries)

Some models (notably the `*neg*` models) carry **per-feature parameters** inside the model JSON. These are encoded as an optional `feature_opts_dicts` field under `model_dict`.

**Schema:**

- `model_dict.feature_opts_dicts` (optional) is an array of JSON objects.
- Its length **must equal** `len(model_dict.feature_names)`.
- Entry `feature_opts_dicts[i]` applies to the feature named `feature_names[i]`.
- Each entry is a flat key/value object. Values must be one of:
  - JSON number (interpreted as IEEE-754 `f64` / binary64)
  - JSON boolean
  - JSON string
- Nested objects/arrays are non-conforming.

**Application rule (normative):** When extracting feature `feature_names[i]`, the implementation must apply the option dictionary `feature_opts_dicts[i]` as that feature's parameter set.

For this SDD, the following keys affect the numeric algorithm and are required for correct `vmaf_float_v0.6.1neg.json` behavior:

- `adm_enhn_gain_limit` (ADM): see §4.3.5 and §4.3.8.
- `vif_enhn_gain_limit` (VIF): see §4.2.6 and §4.2.7.

**Consistency requirement:** `vif_enhn_gain_limit` is conceptually a single parameter shared by the VIF computation across all 4 scales. Therefore, if a model specifies `vif_enhn_gain_limit` for any of the VIF scale features (`*_vif_scale0_score` .. `*_vif_scale3_score`), it must specify the **same numeric value** for all of them. Models shipped with libvmaf satisfy this.

**Example (excerpt, `vmaf_float_v0.6.1neg.json`):**

```json
"feature_opts_dicts": [
  { "adm_enhn_gain_limit": 1.0 },
  {},
  { "vif_enhn_gain_limit": 1.0 },
  { "vif_enhn_gain_limit": 1.0 },
  { "vif_enhn_gain_limit": 1.0 },
  { "vif_enhn_gain_limit": 1.0 }
]
```

#### Embedded LIBSVM Format

The `"model"` string contains a LIBSVM text-format model:

```
svm_type nu_svr
kernel_type rbf
gamma 0.04
nr_class 2
total_sv <N>
rho <rho_value>
SV
<alpha_1> 1:<v1> 2:<v2> 3:<v3> 4:<v4> 5:<v5> 6:<v6>
<alpha_2> 1:<v1> 2:<v2> ...
```

- Each SV line: coefficient (alpha×y), then sparse `index:value` pairs (1-based indices).
- Missing indices have value 0.

---

## 4. Feature Extraction Algorithms

### 4.1 Common Operations

#### 4.1.1 Reflective Padding (Mirror)

For a signal of length `W` and index `i`:

**Precondition:** `W >= 2`.

```
function reflect_index(i, W):
    if i < 0:    return -i
    if i >= W:   return 2*W - 2 - i
    return i
```

---

### 4.2 Integer VIF (Visual Information Fidelity)

VIF measures information fidelity across a Gaussian image pyramid, outputting **4 independent scale scores** and one combined score.

#### 4.2.1 Log2 Lookup Table (LUT)

**Purpose:** Fixed-point approximation of log₂, stored in Q11 format (scale factor = 2048).

```
log2_table: uint16[65537]

procedure log_generate(log2_table):
    for i = 32767 to 65535 inclusive:
        // IMPORTANT: use single-precision (f32) log2.
        // Double-precision log2 produces different rounded values for many entries,
        // which causes subtly different VIF scores across frames.
        log2_table[i] = round(f32_log2(f32(i)) * 2048.0f)
    // Index 32767 is populated but never accessed for valid inputs.
    // Valid access range: indices 32768..65535.
    // Accessing outside this range is a programming error (precondition violation).
```

Embedding the precomputed `log2_table[32768..65535]` directly is conforming and is the recommended portability strategy for bit-exact ports.

#### 4.2.2 Log2_32 Function (32-bit input → Q11 result)

**Precondition:** `x >= SIGMA_NSQ = 131072`. This ensures `clz32(x) <= 14` so `k >= 2` and the normalized index falls in `[32768, 65535]`.

```
function log2_32(log2_table, x: uint32) -> int32:
    k = 16 - clz32(x)    // k >= 2 given the precondition
    x = x >> k
    return log2_table[x] + 2048 * k
```

**Mathematical basis:** `log2(x) = log2(m) + k` where `m = x >> k ∈ [32768, 65535]`.

#### 4.2.3 Log2_64 Function (64-bit input → Q11 result)

**Precondition:** `x >= 131072` (= 2^17). Always satisfied because `numer1 >= SIGMA_NSQ`.

```
function log2_64(log2_table, x: uint64) -> int32:
    k = 48 - clz64(x)    // 48 = 64 - 16
    x = x >> k
    return log2_table[x] + 2048 * k
```

**The constant `2048 × 17`:** The expression

```
log2_32(log2_table, SIGMA_NSQ + sigma1_sq) - 2048 * 17
```

computes `log2(SIGMA_NSQ + sigma1_sq) - log2(2^17)` in Q11 = `log2(1 + sigma1_sq / SIGMA_NSQ)` in Q11.

#### 4.2.4 Gaussian Filter Kernels

Two distinct roles:

- **Statistics filter** `filter[scale]`: used by `vif_statistic` at scale `scale` for local variance computation.
- **Subsampling filter** `filter[scale+1]`: used by `subsample` to decimate from scale `scale` to `scale+1`.

These are two separate filters drawn from the same table. **Do not use the same filter for both purposes.**

Each filter's coefficients sum to 65536 (= 2^16).

| Index | Width | Coefficients (uint16)                                                                              |
| ----- | ----- | -------------------------------------------------------------------------------------------------- |
| 0     | 17    | `489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640, 935, 489` |
| 1     | 9     | `1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244`                                          |
| 2     | 5     | `3571, 16004, 26386, 16004, 3571`                                                                  |
| 3     | 3     | `10904, 43728, 10904`                                                                              |

(In memory the arrays are padded with a trailing `0` to width 18; the width values above are the functional tap counts.)

| Scale | Statistics filter    | Subsampling filter (→ next scale) |
| ----- | -------------------- | --------------------------------- |
| 0     | `filter[0]` (17-tap) | `filter[1]` (9-tap)               |
| 1     | `filter[1]` (9-tap)  | `filter[2]` (5-tap)               |
| 2     | `filter[2]` (5-tap)  | `filter[3]` (3-tap)               |
| 3     | `filter[3]` (3-tap)  | n/a                               |

#### 4.2.5 Multi-Scale Pyramid Processing

**4 scales** are processed. Scale 0 is the full image; each subsequent scale halves both dimensions.

**Subsampling (scale s → s+1) using `filter[s+1]`:**

The implementation first applies the 1D filter to **all rows** (full height), then decimates 2:1 in a separate step (take every other row and column).

```
procedure subsample(src_ref, src_dis, filt, bpc, scale, W, H):
    // --- Phase 1: Vertical filtering (all H rows) ---
    for i = 0 to H-1:
        for j = 0 to W-1:
            accum_ref = accum_dis = 0
            for k = 0 to filt_width-1:
                ii = reflect_index(i - filt_width/2 + k, H)
                accum_ref += filt[k] * src_ref[ii][j]
                accum_dis += filt[k] * src_dis[ii][j]

            if scale == 0:
                shift = bpc;  round = 1 << (bpc - 1)
            else:
                shift = 16;   round = 32768

            tmp_ref[i][j] = (accum_ref + round) >> shift   // uint16
            tmp_dis[i][j] = (accum_dis + round) >> shift   // uint16

    // --- Phase 2: Horizontal filtering (all W columns) ---
    for i = 0 to H-1:
        for j = 0 to W-1:
            accum_ref = accum_dis = 0
            for k = 0 to filt_width-1:
                jj = reflect_index(j - filt_width/2 + k, W)
                accum_ref += filt[k] * tmp_ref[i][jj]
                accum_dis += filt[k] * tmp_dis[i][jj]
            filt_ref[i][j] = (accum_ref + 32768) >> 16   // uint16
            filt_dis[i][j] = (accum_dis + 32768) >> 16   // uint16

    // --- Phase 3: Decimate 2:1 in both dimensions ---
    for i = 0 to H/2-1:
        for j = 0 to W/2-1:
            out_ref[i][j] = filt_ref[2*i][2*j]
            out_dis[i][j] = filt_dis[2*i][2*j]
```

Output dimensions: `(W/2) × (H/2)` (integer division).

#### 4.2.6 VIF Statistic Core Computation

This is the core of VIF. For each scale, it computes `num` and `den` accumulators using **`filter[scale]`** (statistics filter).

**Constants and model parameters:**

```
SIGMA_NSQ            = 131072         // 2^17; noise power in Q16 representing 2.0
vif_enhn_gain_limit  = 100.0          // default; must be >= 1.0; 1.0 disables enhancement gain
EPSILON              = 65536 * 1e-10  // ≈ 6.5536e-6; numerically negligible guard
```

`vif_enhn_gain_limit` may be overridden by the model JSON `feature_opts_dicts` entry corresponding to each VIF feature (`*_vif_scale{0..3}_score`). For conforming implementations, all VIF-scale features in a model must use the same `vif_enhn_gain_limit` value.

**CRITICAL: Accumulator types differ between 8-bit and 16-bit paths.**

For 8-bit input (`bpc = 8`) the vertical-pass squared accumulators are **uint32**. This is a normative part of the algorithm because the arithmetic is modulo `2^32` in that path. For 10/12-bit input and for all higher scales, they are **uint64**. An implementation that uses uint64 for the 8-bit vertical pass will produce different results.

**Step 1: Vertical convolution** (for each pixel (i, j) at scale):

```
fwidth = filter_width[scale]   // from table in §4.2.4

for i = 0 to H-1:
    for j = 0 to W-1:
        accum_mu1 = accum_mu2 = 0   // uint32

        if bpc == 8 and scale == 0:
            // 8-bit path: squared accumulators are uint32 (may wrap — intentional)
            accum_ref_sq = accum_dis_sq = accum_ref_dis = 0   // uint32
        else:
            // 10/12-bit or higher scales: squared accumulators are uint64
            accum_ref_sq = accum_dis_sq = accum_ref_dis = 0   // uint64

        for fi = 0 to fwidth-1:
            ii = reflect_index(i - fwidth/2 + fi, H)
            c = uint32(filter[scale][fi])
            r = uint32(ref[ii][j])
            d = uint32(dis[ii][j])

            accum_mu1   += c * r
            accum_mu2   += c * d
            accum_ref_sq  += (c * r) * r   // type matches accumulator declaration above
            accum_dis_sq  += (c * d) * d
            accum_ref_dis += (c * r) * d

        // Rounding and shift for mean accumulators
        if scale == 0:
            shift_mu = bpc;  round_mu = 1 << (bpc - 1)
        else:
            shift_mu = 16;   round_mu = 32768

        // Rounding and shift for squared accumulators (scale 0 only is special)
        if scale == 0 and bpc == 8:
            sq_shift = 0;  sq_round = 0    // no shift for 8-bit scale 0
        elif scale == 0:
            sq_shift = (bpc - 8) * 2
            sq_round = 1 << (sq_shift - 1)
        else:
            sq_shift = 16;  sq_round = 32768

        tmp_mu1[j]    = (accum_mu1   + round_mu) >> shift_mu   // uint16
        tmp_mu2[j]    = (accum_mu2   + round_mu) >> shift_mu   // uint16
        tmp_ref_sq[j] = (accum_ref_sq  + sq_round) >> sq_shift  // uint32
        tmp_dis_sq[j] = (accum_dis_sq  + sq_round) >> sq_shift  // uint32
        tmp_ref_dis[j]= (accum_ref_dis + sq_round) >> sq_shift  // uint32

    // Mirror-pad tmp arrays: reflect fwidth/2 elements beyond each edge
    // tmp[-(f)] = tmp[f],  tmp[W-1+f] = tmp[W-1-f]  for f = 1..fwidth/2
```

**Step 2: Horizontal convolution** (for each pixel (i, j)):

```
for i = 0 to H-1:
    for j = 0 to W-1:
        accum_mu1 = accum_mu2 = 0   // uint32
        accum_ref = accum_dis = accum_ref_dis = 0   // uint64

        for fj = 0 to fwidth-1:
            jj = j - fwidth/2 + fj    // negative jj accesses padded region
            c = uint32(filter[scale][fj])

            accum_mu1    += c * uint32(tmp_mu1[jj])
            accum_mu2    += c * uint32(tmp_mu2[jj])
            accum_ref    += uint64(c) * uint64(tmp_ref_sq[jj])
            accum_dis    += uint64(c) * uint64(tmp_dis_sq[jj])
            accum_ref_dis+= uint64(c) * uint64(tmp_ref_dis[jj])

        mu1_val = accum_mu1   // uint32
        mu2_val = accum_mu2   // uint32

        mu1_sq  = uint32((uint64(mu1_val) * mu1_val + 2147483648) >> 32)
        mu2_sq  = uint32((uint64(mu2_val) * mu2_val + 2147483648) >> 32)
        mu1_mu2 = uint32((uint64(mu1_val) * mu2_val + 2147483648) >> 32)

        ref_filt = uint32((accum_ref     + 32768) >> 16)
        dis_filt = uint32((accum_dis     + 32768) >> 16)
        rdi_filt = uint32((accum_ref_dis + 32768) >> 16)

        sigma1_sq = int32(ref_filt - mu1_sq)
        sigma2_sq = int32(dis_filt - mu2_sq)
        sigma12   = int32(rdi_filt - mu1_mu2)
        sigma2_sq = max(sigma2_sq, 0)

        // [accumulate VIF terms — see §4.2.7]
```

#### 4.2.7 VIF Accumulator Logic

For each pixel, after computing `sigma1_sq`, `sigma2_sq`, `sigma12`:

```
// Four int64 accumulators (initialised to 0 per scale per frame):
//   accum_num_log, accum_den_log         (Q11 log sums)
//   accum_num_non_log, accum_den_non_log (for low-variance pixels)

if sigma1_sq >= SIGMA_NSQ:
    accum_den_log += log2_32(log2_table, uint32(SIGMA_NSQ + sigma1_sq)) - 2048 * 17

    if sigma12 > 0 and sigma2_sq > 0:
        g_f64 = f64(sigma12) / (f64(sigma1_sq) + EPSILON)

        // sv_sq: residual variance; computed and clamped in integer domain
        sv_sq_int32 = int32(sigma2_sq) - int32(g_f64 * f64(sigma12))
        sv_sq_int32 = max(sv_sq_int32, 0)   // clamp in integer, not float

        g_f64 = min(g_f64, vif_enhn_gain_limit)

        numer1     = uint32(sv_sq_int32) + uint32(SIGMA_NSQ)
        // numer1_tmp is kept as int64 until the final log2 call to preserve the
        // signed intermediate expression. Under the branch conditions here
        // (sigma12 > 0, sigma2_sq > 0, g >= 0), numer1_tmp is guaranteed >= numer1.
        numer1_tmp = int64(g_f64 * g_f64 * f64(sigma1_sq)) + int64(numer1)

        accum_num_log += log2_64(log2_table, uint64(numer1_tmp)) - log2_64(log2_table, uint64(numer1))
else:
    accum_num_non_log += int64(sigma2_sq)
    accum_den_non_log += 1
```

**Note on `sv_sq` type:** The clamp `max(sv_sq_int32, 0)` is in the **integer** domain. Clamping after a float conversion would give different bit-exact results.

#### 4.2.8 Per-Scale Score Extraction

```
// Evaluate left-to-right; do not reorder the divisions:
non_log_penalty = f64(accum_num_non_log) / 16384.0 / 65025.0

num[scale] = f64(accum_num_log) / 2048.0
           + (f64(accum_den_non_log) - non_log_penalty)

den[scale] = f64(accum_den_log) / 2048.0 + f64(accum_den_non_log)
```

**Magic constant explanations:**

| Constant  | Value | Meaning                                                |
| --------- | ----- | ------------------------------------------------------ |
| `2048`    | 2^11  | Q11 de-scale factor (log2_table precision)             |
| `2048×17` | 34816 | Subtracts log₂(SIGMA_NSQ = 2^17) in Q11                |
| `16384`   | 2^14  | First divisor in non-log penalty                       |
| `65025`   | 255²  | Second divisor; normalizes by 8-bit full-scale squared |

#### 4.2.9 Final Combined VIF Score

```
total_num = sum(num[s] for s in 0..3)
total_den = sum(den[s] for s in 0..3)

vif_combined = (total_den > 0.0) ? total_num / total_den : 1.0
vif_scale{s}_score = (den[s] > 0.0) ? num[s] / den[s] : 1.0
```

The zero-denominator guards above are normative for this specification.

---

### 4.3 ADM (Additive Distortion Measure)

ADM measures perceptual distortion using a 4-level DWT pyramid and CSF model.

**Pipeline architecture:** integer DWT → integer decouple → f32 CSF weighting → f32 score accumulation.

**Two-tier integer pipeline:**

- **Scale 0:** DWT subbands are stored as `int16`.
- **Scales 1–3:** DWT subbands are stored as `int32`. Before computing the scale-1 DWT, the scale-0 LL band (`band_a`) is widened from `int16` to `int32` by plain sign-extension only: `int32_band_a[i][j] = int32(int16_band_a[i][j])`.

All scoring uses f32 (single precision) throughout.

#### 4.3.1 DWT Filter Coefficients

VMAF uses the **Daubechies DB2 Biorthogonal 7-9** wavelet as a 4-tap separable 2D convolution.

```
filter_lo = [15826, 27411, 7345, -4240]   // int32
filter_hi = [-4240, -7345, 27411, -15826]  // int32
dwt_lo_sum = 46342   // sum(filter_lo); used in vertical-pass rounding
dwt_hi_sum = 0       // sum(filter_hi); no rounding correction needed for hi band
```

#### 4.3.2 DWT Boundary Index Table

Pre-compute index tables using `reflect_index`. The first row (`i=0`) is specified explicitly here because the generic loop below starts at `i=1`:

```
H_half = (H + 1) / 2
W_half = (W + 1) / 2

// Explicit i=0 row (base = 2*0 = 0):
ind_y[0][0] = reflect_index(-1, H)
ind_y[1][0] = reflect_index( 0, H)
ind_y[2][0] = reflect_index( 1, H)
ind_y[3][0] = reflect_index( 2, H)

// Generic loop for i = 1 to H_half-1:
for i = 1 to H_half-1:
    base = 2 * i
    ind_y[0][i] = reflect_index(base - 1, H)
    ind_y[1][i] = base
    ind_y[2][i] = reflect_index(base + 1, H)
    ind_y[3][i] = reflect_index(base + 2, H)

// Same for ind_x using width W
```

#### 4.3.3 2D DWT Computation

**Vertical pass** (for each output row i = 0..H_half-1, all input columns j):

```
shift_VP = (input is 8-bit) ? 8 : bpc
round_VP = 1 << (shift_VP - 1)

for i = 0 to H_half-1:
    for j = 0 to W-1:
        s0 = src[ind_y[0][i]][j]
        s1 = src[ind_y[1][i]][j]
        s2 = src[ind_y[2][i]][j]
        s3 = src[ind_y[3][i]][j]

        accum_lo = filter_lo[0]*s0 + filter_lo[1]*s1 + filter_lo[2]*s2 + filter_lo[3]*s3
        accum_hi = filter_hi[0]*s0 + filter_hi[1]*s1 + filter_hi[2]*s2 + filter_hi[3]*s3

        tmplo[j] = int16((accum_lo - dwt_lo_sum * round_VP + round_VP) >> shift_VP)
        tmphi[j] = int16((accum_hi + round_VP) >> shift_VP)
        // Note: dwt_hi_sum = 0 so no correction term for tmphi
```

**Horizontal pass** (on tmplo → LL, LH; on tmphi → HL, HH):

```
shift_HP = 16
round_HP = 32768

for j = 0 to W_half-1:
    // tmplo → band_a (LL) and band_v (LH)
    s0..s3 from tmplo[ind_x[0..3][j]]
    band_a[i][j] = int16((filter_lo[0]*s0 + filter_lo[1]*s1 + filter_lo[2]*s2 + filter_lo[3]*s3 + round_HP) >> shift_HP)
    band_v[i][j] = int16((filter_hi[0]*s0 + filter_hi[1]*s1 + filter_hi[2]*s2 + filter_hi[3]*s3 + round_HP) >> shift_HP)

    // tmphi → band_h (HL) and band_d (HH)
    s0..s3 from tmphi[ind_x[0..3][j]]
    band_h[i][j] = int16((filter_lo[0]*s0 + filter_lo[1]*s1 + filter_lo[2]*s2 + filter_lo[3]*s3 + round_HP) >> shift_HP)
    band_d[i][j] = int16((filter_hi[0]*s0 + filter_hi[1]*s1 + filter_hi[2]*s2 + filter_hi[3]*s3 + round_HP) >> shift_HP)
```

Output subbands at scale 0 are `int16`. Before processing scale 1, widen the LL band to `int32` by **plain sign-extension only — no bit shift**:

```
// Scale 0 → Scale 1 widening (sign-extension, no shift):
for each pixel (i,j):
    int32_band_a[i][j] = int32(int16_band_a[i][j])   // plain widening, value unchanged
```

Scales 1–3 use `int32` subbands throughout (same DWT formulas, wider types). The DWT shift constants for scales 1–3 are: `shift_VP = 0` (scale 1), `shift_VP = 16` (scales 2–3); the scale-1 vertical pass uses no shift because the input is already in the correct range after sign-extension.

**Exact numeric types for scales 1–3:**

- Input LL coefficients are `int32`.
- Vertical scratch buffers `tmplo_ref`, `tmphi_ref`, `tmplo_dis`, `tmphi_dis` are `int32`.
- Every filter product and running sum in both vertical and horizontal passes is accumulated in `int64`.
- Output subbands `band_a`, `band_v`, `band_h`, `band_d` are written as `int32` after rounding and right shift.
- There is no saturating clamp between the `int64` accumulator and the final `int32` store; the narrowed stored value is the shifted result cast to `int32`.

**Scale-specific constants for scales 1–3:**

| Scale | `round_VP` | `shift_VP` | `round_HP` | `shift_HP` |
| ----- | ---------- | ---------- | ---------- | ---------- |
| 1     | `0`        | `0`        | `16384`    | `15`       |
| 2     | `32768`    | `16`       | `32768`    | `16`       |
| 3     | `32768`    | `16`       | `16384`    | `15`       |

Pseudocode for the scales 1–3 DWT path:

```
for each output row i:
    for each input column j:
        // ref path
        s0..s3 = int32 source samples from the current LL plane
        accum_ref: int64 = filter[0]*s0 + filter[1]*s1 + filter[2]*s2 + filter[3]*s3
        tmplo_ref[j] = int32((accum_ref + round_VP[scale]) >> shift_VP[scale])

        accum_ref: int64 = filter_hi[0]*s0 + filter_hi[1]*s1 + filter_hi[2]*s2 + filter_hi[3]*s3
        tmphi_ref[j] = int32((accum_ref + round_VP[scale]) >> shift_VP[scale])

        // distorted path uses the same rules and types

    for each output column j:
        s0..s3 = int32 values loaded from the scratch buffers
        accum_ref: int64 = ...
        band_a[i][j] = int32((accum_ref + round_HP[scale]) >> shift_HP[scale])
        ...
```

**Subband naming:**

| Band     | Filter pair | Orientation      |
| -------- | ----------- | ---------------- |
| `band_a` | lo×lo (LL)  | Approximation    |
| `band_v` | lo×hi (LH)  | Vertical edges   |
| `band_h` | hi×lo (HL)  | Horizontal edges |
| `band_d` | hi×hi (HH)  | Diagonal details |

#### 4.3.4 Multi-Scale ADM Structure

- 4 scales (0–3). Scale 0 processes the full image.
- Scale s+1 processes `band_a` from scale s.
- At each scale, the detail subbands (band_v, band_h, band_d) feed into CSF and scoring.

#### 4.3.5 CSF (Contrast Sensitivity Function) Weights

**Model parameters** (Y channel only):

```
a     = 0.495
k     = 0.466
f0    = 0.401
g_csf = [1.501, 1.0, 0.534, 1.0]   // indexed by theta (0..3)

adm_norm_view_dist     = 3.0
adm_ref_display_height = 1080.0     // pixels
adm_enhn_gain_limit    = 100.0      // default; must be >= 1.0; 1.0 disables enhancement gain
ADM_BORDER_FACTOR      = 0.1
```

**`adm_enhn_gain_limit` meaning and application (normative):**

`adm_enhn_gain_limit` controls how much *enhancement* (distorted detail coefficients aligned in direction with the reference) is credited as reconstructed signal during ADM decouple.

- It is applied only under `angle_flag == true`.
- It is applied in both `decouple_scale0` and `decouple_s123` at the enhancement step shown in §4.3.8.
- It must be `>= 1.0`.
  - `1.0` disables enhancement gain (no boost beyond the initial ratio-based reconstruction).
  - `100.0` is the default and behaves as effectively "unlimited" for typical content, allowing the reconstructed coefficient to reach the distorted coefficient under the min/max clamp.
- The value may be overridden by `model_dict.feature_opts_dicts` for the ADM feature (`*_adm2_score`). See §3.3.

**Basis function amplitudes** `A[lambda][theta]`:

```
A[0] = [0.62171, 0.67234, 0.72709, 0.67234]
A[1] = [0.34537, 0.41317, 0.49428, 0.41317]
A[2] = [0.18004, 0.22727, 0.28688, 0.22727]
A[3] = [0.091401, 0.11792, 0.15214, 0.11792]
A[4] = [0.045943, 0.059758, 0.077727, 0.059758]
A[5] = [0.023013, 0.030018, 0.039156, 0.030018]
```

**`dwt_quant_step(lambda, theta)`:**

```
function dwt_quant_step(lambda, theta):
    r    = adm_norm_view_dist * adm_ref_display_height * π / 180.0
    temp = log10(2^(lambda+1) * f0 * g_csf[theta] / r)
    Q    = 2.0 * a * 10^(k * temp^2) / A[lambda][theta]
    return Q
```

For bit-exact conformance, implementations do **not** evaluate `dwt_quant_step` at runtime. They must use the exact binary32 CSF weights specified below. The formula above is informational and is provided to explain where the constants come from.

**CSF weight per subband:** band_h and band_v both use `theta=1`; band_d uses `theta=2`. theta=0 and theta=3 are not used.

```
factor1 = f32(1.0 / dwt_quant_step(scale, 1))   // for band_h and band_v
factor2 = f32(1.0 / dwt_quant_step(scale, 2))   // for band_d

csf_weight[band_h] = factor1
csf_weight[band_v] = factor1
csf_weight[band_d] = factor2
```

For bit-exact ports, use the exact binary32 CSF weights below:

| Scale | `factor1` for `band_h`/`band_v` | Binary32     | `factor2` for `band_d` | Binary32     |
| ----- | ------------------------------- | ------------ | ---------------------- | ------------ |
| 0     | `0.017381533980369568`          | `0x3c8e63b8` | `0.00589068653061986`  | `0x3bc106a9` |
| 1     | `0.03198481351137161`           | `0x3d030282` | `0.014299066737294197` | `0x3c6a46a2` |
| 2     | `0.04337266460061073`           | `0x3d31a789` | `0.02439691312611103`  | `0x3cc7dc09` |
| 3     | `0.04567341133952141`           | `0x3d3b140b` | `0.0313127338886261`   | `0x3d0041c8` |

#### 4.3.6 div_lookup Table

ADM decouple uses a fixed-point reciprocal lookup table to avoid division:

```
div_lookup: int32[65537]   // index range -32768..32768 stored at index+32768

procedure build_div_lookup(div_lookup):
    div_lookup[32768] = 0   // entry for x=0: division by zero → 0 (MUST be set explicitly)
    for i = 1 to 32768:
        recip = int32(2^30 / i)          // Q30 reciprocal (truncated, not rounded)
        div_lookup[32768 + i] =  recip   // positive i
        div_lookup[32768 - i] = -recip   // negative i: negate the positive reciprocal
```

Access: `div_lookup[x + 32768]` gives approximately `2^30 / x` for `x ∈ [-32768, 32768]`.

Embedding the precomputed `div_lookup` directly is conforming and is the recommended portability strategy for bit-exact ports. See §8 for reference verification vectors.

**Critical:** The entry at `div_lookup[32768]` (x=0) must be **explicitly initialized to 0**. If the array is allocated on the stack without initialization, this entry will be garbage and will corrupt the decouple computation whenever `ref_c == 0`.

#### 4.3.7 Border Exclusion

Two distinct border regions are used:

**Decouple/CSF border** (expanded by filter radius):

```
decouple_left   = floor(W * ADM_BORDER_FACTOR - 0.5 - 1)
decouple_top    = floor(H * ADM_BORDER_FACTOR - 0.5 - 1)
decouple_right  = W - decouple_left + 2
decouple_bottom = H - decouple_top  + 2
(clamp to [0, W) and [0, H))
```

**Score accumulation border** (final num/den sums):

```
accum_left   = floor(W * ADM_BORDER_FACTOR - 0.5)
accum_top    = floor(H * ADM_BORDER_FACTOR - 0.5)
accum_right  = W - accum_left
accum_bottom = H - accum_top
(clamp to [0, W) and [0, H))
```

#### 4.3.8 Decouple (Angle-Based Masking)

Decouple separates each distorted subband into a reconstructed component (correlated with the reference) and an artifact component. The implementation uses integer fixed-point with `div_lookup`.

**Scale 0 (int16 subbands):**

```
function decouple_scale0(ref_h, ref_v, ref_d: int16,
                          dis_h, dis_v, dis_d: int16,
                          div_lookup) -> (rst_h, rst_v, rst_d, art_h, art_v, art_d: int16):

    // Angle coherence test (convert to f64 for comparison)
    dp     = f64(ref_h)*f64(dis_h) + f64(ref_v)*f64(dis_v)
    o_sq   = f64(ref_h)^2 + f64(ref_v)^2
    t_sq   = f64(dis_h)^2 + f64(dis_v)^2
    COS_SQ = cos(1°)^2 ≈ 0.9996954   // libvmaf uses binary32 0x3f7fec0a

    angle_flag = (dp >= 0.0) AND (dp*dp >= COS_SQ * o_sq * t_sq)

    for c in {h, v, d}:
        if ref_c == 0:
            tmp_k = 32768          // ratio = 1.0 in Q15
        else:
            // Fixed-point ratio: dis_c / ref_c in Q15, clamped to [0, 32768]
            raw_k = int32((int64(div_lookup[ref_c + 32768]) * int64(dis_c) + 16384) >> 15)
            tmp_k = clamp(raw_k, 0, 32768)

        rst_c = int16((int32(tmp_k) * int32(ref_c) + 16384) >> 15)
        art_c = int16(dis_c - rst_c)

        if angle_flag:
            if rst_c > 0: rst_c = int16(min(int32(rst_c) * adm_enhn_gain_limit, int32(dis_c)))
            if rst_c < 0: rst_c = int16(max(int32(rst_c) * adm_enhn_gain_limit, int32(dis_c)))
            art_c = int16(dis_c - rst_c)   // recompute after enhancement

    return (rst_h, rst_v, rst_d, art_h, art_v, art_d)
```

**Scales 1–3 (int32 subbands):** The same high-level decouple logic is used, but each reference component is normalized independently before the reciprocal lookup. The angle test still uses the original unshifted horizontal and vertical coefficients.

```
function get_best15_from32(abs_x: uint32) -> (mantissa: uint16, shift: int):
    // Precondition: 32768 <= abs_x <= 2147483647.
    shift = 17 - clz32(abs_x)
    mantissa = uint16((abs_x + (1 << (shift - 1))) >> shift)
    return (mantissa, shift)

function decouple_s123(ref_h, ref_v, ref_d: int32,
                       dis_h, dis_v, dis_d: int32,
                       div_lookup) -> (rst_h, rst_v, rst_d, art_h, art_v, art_d: int32):

    // Angle coherence test uses original H/V coefficients only.
    ot_dp   = int64(ref_h) * dis_h + int64(ref_v) * dis_v
    o_mag_sq = int64(ref_h) * ref_h + int64(ref_v) * ref_v
    t_mag_sq = int64(dis_h) * dis_h + int64(dis_v) * dis_v
    COS_SQ  = cos(1°)^2 ≈ 0.9996954   // libvmaf uses binary32 0x3f7fec0a

    angle_flag =
        (f32(ot_dp) / 4096.0 >= 0.0) AND
        ((f32(ot_dp) / 4096.0)^2 >= COS_SQ * (f32(o_mag_sq) / 4096.0) * (f32(t_mag_sq) / 4096.0))

    for c in {h, v, d}:
        o_c = ref_c
        t_c = dis_c
        shift_c = 0
        abs_o_c = uint32(|o_c|)
        sign_c  = (o_c < 0) ? -1 : 1

        if abs_o_c < 32768:
            mantissa_c = uint16(abs_o_c)
        else:
            (mantissa_c, shift_c) = get_best15_from32(abs_o_c)

        if o_c == 0:
            tmp_k = 32768
        else:
            tmp_k =
                (((int64(div_lookup[mantissa_c + 32768]) * int64(t_c)) * sign_c)
                  + (1 << (14 + shift_c))) >> (15 + shift_c)

        k_c = clamp(tmp_k, 0, 32768)

        // Reconstruction always uses the original unshifted reference coefficient.
        rst_c = int32((k_c * int64(o_c) + 16384) >> 15)

        // Enhancement sign test uses f32 on the original-domain coefficient.
        rst_c_f = (f32(k_c) / 32768.0) * (f32(o_c) / 64.0)
        if angle_flag and rst_c_f > 0.0: rst_c = min(int32(rst_c * adm_enhn_gain_limit), t_c)
        if angle_flag and rst_c_f < 0.0: rst_c = max(int32(rst_c * adm_enhn_gain_limit), t_c)

        art_c = t_c - rst_c

    return (rst_h, rst_v, rst_d, art_h, art_v, art_d)
```

**Critical details:**

- The normalization shift is computed **independently for h, v, d**. There is no shared shift across bands.
- The distorted coefficients `dis_h`, `dis_v`, `dis_d` remain in their original scale during ratio estimation.
- There is no explicit "shift back" step after ratio estimation; reconstruction uses the original unshifted reference coefficients directly.
- The rounding `+ (1 << (shift-1))` inside `get_best15_from32` and the rounding term `1 << (14 + shift_c)` in `tmp_k` are both required. Omitting either changes the decouple ratios.
- `k_c` is clamped to `[0, 32768]`. `rst_c` and `art_c` are not saturating-clamped after reconstruction/enhancement; they remain plain `int32` results.

#### 4.3.9 ADM Score Calculation

All scoring arithmetic is f32. Integer subband values are cast to f32 before use.

**CSF application:**

For each detail subband `b ∈ {h, v, d}` and each pixel (i,j) within the decouple border:

```
csf_a[b][i][j] = f32(art[b][i][j]) * csf_weight[b]    // CSF-weighted artifact
csf_f[b][i][j] = |csf_a[b][i][j]| / 30.0              // threshold pre-computation
```

**Numerator accumulation per scale:**

The main signal `x` comes from `decouple_r` (the **reconstructed** reference component), **not** from `decouple_a`. The masking threshold combines `csf_f` values from **all three orientation bands jointly**.

Neighbor access for the threshold uses mirror reflection at the image edges. For each neighbor offset `(di, dj)` with `di, dj ∈ {-1, 0, 1}` and `(di, dj) != (0, 0)`, access:

```
ii = reflect_index(i + di, H)
jj = reflect_index(j + dj, W)
```

This is equivalent to the explicit edge cases:

- top row uses reflected row indices `[1, 0, 1]`
- bottom row uses reflected row indices `[H-2, H-1, H-2]`
- left column uses reflected column indices `[1, 0, 1]`
- right column uses reflected column indices `[W-2, W-1, W-2]`

For each pixel (i,j) within the score accumulation border:

```
accum_num = 0.0   // f32

for each pixel (i,j) in accum_border:
    for b in {h, v, d}:
        x = f32(decouple_r[b][i][j]) * csf_weight[b]   // main signal from decouple_r

        // Cross-band threshold: sum csf_f over all 3 bands and 8 neighbors + center
        thr = 0.0
        for b2 in {h, v, d}:
            for (di, dj) in 8-connected neighbors of (i,j):
                ii = reflect_index(i + di, H)
                jj = reflect_index(j + dj, W)
                thr += csf_f[b2][ii][jj]
            thr += |f32(art[b2][i][j]) * csf_weight[b2]| / 15.0   // center: absolute post-CSF magnitude / 15

        val = max(|x| - thr, 0.0)
        accum_num += val^3

// W_border and H_border are the border-trimmed pixel dimensions at this scale
// (accum_right - accum_left) and (accum_bottom - accum_top) from §4.3.7
num_scale = cbrt(accum_num) + cbrt(f32(W_border * H_border) / 32.0)
```

**Important:** The threshold `thr` is computed once per pixel (i,j) and is shared across all three orientation terms for that pixel. It sums across all bands, not per-band separately.

**Denominator accumulation per scale:**

The denominator uses the **reference DWT subbands directly** (bypassing decouple entirely):

```
accum_den = 0.0   // f32

for each pixel (i,j) in accum_border:
    for b in {h, v, d}:
        x = |f32(ref_detail[b][i][j]) * csf_weight[b]|
        accum_den += x^3

den_scale = cbrt(accum_den) + cbrt(f32(W_border * H_border) / 32.0)
// W_border, H_border = (accum_right - accum_left), (accum_bottom - accum_top) at this scale
```

**Score combination:**

```
num_total = den_total = 0.0

for s = 0 to 3:
    // The per-scale accumulation combines the four scale-local num/den values directly.
    num_total += num_scale[s]
    den_total += den_scale[s]

// W_full, H_full = full frame dimensions (not border-trimmed)
numden_limit = 1e-10 * f64(W_full * H_full) / (1920.0 * 1080.0)

// Apply threshold independently to numerator and denominator before dividing:
if f64(num_total) < numden_limit: num_total = 0.0
if f64(den_total) < numden_limit: den_total = 0.0

// If den is zero, return 1.0 (perfect score on degenerate/flat content)
adm2_score = (den_total == 0.0) ? 1.0 : f64(num_total) / f64(den_total)
// Note: accumulation is in f32; widening to f64 occurs before the final division
```

---

### 4.4 Integer Motion

**Important:** Motion uses only **reference frames**. The distorted frame is not used.

Motion measures frame-to-frame change using 5-tap Gaussian-blurred reference frames.

#### 4.4.1 Blur Filter

```
MOTION_FILTER = [3571, 16004, 26386, 16004, 3571]   // uint16, sums to 65536
```

**Vertical pass** (src → tmp):

```
for each pixel (i, j):
    accum = 0  // uint32
    for k = 0 to 4:
        ii = reflect_index(i - 2 + k, H)
        accum += uint32(MOTION_FILTER[k]) * uint32(src[ii][j])
    tmp[i][j] = uint16((accum + (1 << (bpc-1))) >> bpc)
```

**Horizontal pass** (tmp → blurred):

```
for each pixel (i, j):
    accum = 0  // uint32
    for k = 0 to 4:
        jj = reflect_index(j - 2 + k, W)
        accum += uint32(MOTION_FILTER[k]) * uint32(tmp[i][jj])
    blurred[i][j] = uint16((accum + 32768) >> 16)
```

#### 4.4.2 Motion1 and Motion2

Only **Motion2** is the SVM feature.

**Ring buffer:** Three blur slots indexed by `slot = frame_index % 3`. After writing frame `n` to slot `n%3`, each slot contains the last frame that was written to it:

| After frame n | Slot 0  | Slot 1  | Slot 2  |
| ------------- | ------- | ------- | ------- |
| n=0           | frame 0 | (empty) | (empty) |
| n=1           | frame 0 | frame 1 | (empty) |
| n=2           | frame 0 | frame 1 | frame 2 |
| n=3           | frame 3 | frame 1 | frame 2 |
| n=4           | frame 3 | frame 4 | frame 2 |
| n=5           | frame 3 | frame 4 | frame 5 |

At frame `n ≥ 2`, translating to named slots:

- `blur[n%3]`      = frame `n`   (just written, current)
- `blur[(n+2)%3]`  = frame `n-1` (one frame back)
- `blur[(n+1)%3]`  = frame `n-2` (two frames back)

Verification for n=3: slot 0=F3 ✓, slot (3+2)%3=2=F2 ✓ (n-1=2), slot (3+1)%3=1=F1 ✓ (n-2=1).

**SAD computation:**

```
function compute_sad(buf_a, buf_b, W, H) -> f32:
    sad: uint64 = 0
    for i = 0 to H-1:
        for j = 0 to W-1:
            sad += uint64(|buf_a[i][j] - buf_b[i][j]|)
    return f32(sad / 256.0) / f32(W * H)
    // Cast to f32 before dividing by W*H; do not compute the entire expression in f64.
```

**Motion1** (consecutive-frame SAD):

```
after blurring frame n into blur[n%3]:

if n == 0:
    motion1[0] = 0.0
else:
    // Compare frame n (current) to frame n-1 (one back)
    motion1[n] = compute_sad(blur[(n+2)%3], blur[n%3], W, H)
```

**Motion2** (minimum of two consecutive motion1 values):

```
if n >= 2:
    // Also compute SAD between frame n-1 and frame n-2
    sad_prev = compute_sad(blur[(n+2)%3], blur[(n+1)%3], W, H)
    motion2[n-1] = min(sad_prev, motion1[n])

motion2[0] = 0.0
// motion2[1] is not available until frame 2 is processed

// --- Final frame flush ---
// When the video ends at frame n_last, there is one pending motion2 score
// that was never emitted (motion2[n_last] was never written because no frame n_last+1
// arrived to trigger the min). The flush step emits it:
//
//   motion2[n_last] = motion1[n_last]   // no next frame to compare against
//
// This is reported to the feature collector at index n_last.
// Implementations must call a flush() equivalent after processing all frames.
```

---

## 5. Prediction and Pooling

### 5.1 Feature Normalization

Apply per-feature linear rescaling using `slopes[1..6]` and `intercepts[1..6]` from the JSON:

```
for i = 0 to 5:
    f_normalized[i] = slopes[i+1] * f_raw[i] + intercepts[i+1]
```

(1-based indexing in the JSON array; `slopes[0]` / `intercepts[0]` are model-level and used in §5.3.)

### 5.2 SVM Nu-SVR Prediction

**SVM input vector (1-based indices, terminated with index=-1):**

```
node[0] = (index=1, value=f_normalized[0])   // adm2
node[1] = (index=2, value=f_normalized[1])   // motion2
node[2] = (index=3, value=f_normalized[2])   // vif_scale0
node[3] = (index=4, value=f_normalized[3])   // vif_scale1
node[4] = (index=5, value=f_normalized[4])   // vif_scale2
node[5] = (index=6, value=f_normalized[5])   // vif_scale3
node[6] = (index=-1, value=0.0)              // mandatory terminator
```

**RBF Kernel:**

```
function K(x_node, sv_node) -> f64:
    sq_diff = 0.0
    for k = 1 to 6:
        xk = value at index k in x_node  (0.0 if absent)
        sk = value at index k in sv_node (0.0 if absent)
        sq_diff += (xk - sk)^2
    return exp(-0.04 * sq_diff)   // gamma = 0.04
```

**Nu-SVR Decision:**

```
function svm_predict(model, x_node) -> f64:
    result = 0.0
    for i = 0 to model.total_sv - 1:
        result += model.sv_coef[i] * K(x_node, model.SV[i])
    return result - model.rho
```

### 5.3 Score Denormalization

```
raw = svm_predict(model, x_node)
denormalized = (raw - model.intercept) / model.slope
// model.slope = 0.012020766, model.intercept = -0.3092981928
```

### 5.4 Score Transformation Pipeline

`score_in` is the denormalized value and must remain fixed for the rectification step.

```
score_in = denormalized
score    = score_in

// Step 1: Polynomial (p0 + p1*x + p2*x^2)
if any of p0, p1, p2 is defined:
    poly = 0.0
    if p0 defined: poly += p0
    if p1 defined: poly += p1 * score
    if p2 defined: poly += p2 * score^2
    score = poly

// Step 2: Piecewise linear mapping via knots (if present)
if knots defined and len(knots) >= 2:
    score = piecewise_linear(score, knots)

// Step 3: Rectification — compare against score_in (not the post-step-1 value)
if out_lte_in: score = min(score, score_in)
if out_gte_in: score = max(score, score_in)

// Step 4: Clipping
score = clamp(score, score_clip[0], score_clip[1])   // default [0.0, 100.0]
```

**Piecewise linear with extrapolation:**

```
function piecewise_linear(x, knots[]) -> f64:
    // Precondition: len(knots) >= 2, x-coordinates strictly increasing,
    // y-coordinates nondecreasing.
    n_seg = len(knots) - 1
    for i = 0 to n_seg-1:
        x0, y0 = knots[i]
        x1, y1 = knots[i+1]

        if y0 == y1:   // horizontal segment
            val = y0   // constant
        else:
            slope  = (y1 - y0) / (x1 - x0)
            val    = y0 + slope * (x - x0)

        // In-range: return immediately
        if x0 <= x <= x1: return val

        // Extrapolation using first segment for x < x0
        if i == 0 and x < x0: return val

        // Extrapolation using last segment for x > x1
        if i == n_seg-1 and x > x1: return val

    return x   // fallback (should not be reached for well-formed knots)
```

### 5.5 Pooling Methods

```
function collect_scores(scores, index_low, index_high, n_subsample) -> f64[]:
    result = []
    for i = index_low to index_high:
        if n_subsample > 1 and i % n_subsample != 0: continue
        result.append(scores[i])
    return result
```

**Precondition:** `collect_scores(...)` must return at least one element. Requesting pooling over an empty frame set is an invalid call.

| Method        | Formula                                               |
| ------------- | ----------------------------------------------------- |
| Mean          | `sum(s) / N`                                          |
| Harmonic Mean | `(N / sum(1.0 / (s + 1.0))) - 1.0`  (explicit parens) |
| Minimum       | `min(s)`                                              |
| Maximum       | `max(s)`                                              |

**Default:** Mean. The `+1.0` offset in Harmonic Mean is a design choice, not an approximation.

---

## 6. Feature Dictionary

Standard feature names for `vmaf_v0.6.1`:

| SVM Index | Feature Name                            | Algorithm          |
| --------- | --------------------------------------- | ------------------ |
| 1         | `VMAF_integer_feature_adm2_score`       | ADM (§4.3)         |
| 2         | `VMAF_integer_feature_motion2_score`    | Motion2 (§4.4)     |
| 3         | `VMAF_integer_feature_vif_scale0_score` | VIF scale 0 (§4.2) |
| 4         | `VMAF_integer_feature_vif_scale1_score` | VIF scale 1 (§4.2) |
| 5         | `VMAF_integer_feature_vif_scale2_score` | VIF scale 2 (§4.2) |
| 6         | `VMAF_integer_feature_vif_scale3_score` | VIF scale 3 (§4.2) |

---

## 7. Key Constants Summary

| Constant                | Value             | Location   | Meaning                                                                               |
| ----------------------- | ----------------- | ---------- | ------------------------------------------------------------------------------------- |
| `SIGMA_NSQ`                    | 131072 (2^17)     | VIF        | Noise power in Q16 fixed-point                                                        |
| `vif_enhn_gain_limit`          | 100.0 (default)   | VIF        | Max gain cap (g); override via `feature_opts_dicts[].vif_enhn_gain_limit`             |
| `EPSILON`                      | 6.5536e-6         | VIF        | Division guard in gain; numerically negligible                                        |
| `LOG2_Q_SCALE`                 | 2048 (2^11)       | VIF        | Q11 precision for log2 LUT                                                            |
| `LOG2_LUT_ACCESS_RANGE`        | 32768–65535       | VIF        | Valid index range in log2_table                                                       |
| `VIF_NONLOG_DIVISOR`           | 16384.0 × 65025.0 | VIF        | Combined divisor for non-log penalty                                                  |
| `ADM_BORDER_FACTOR`            | 0.1               | ADM        | Fraction of image excluded at borders                                                 |
| `adm_enhn_gain_limit`          | 100.0 (default)   | ADM        | Enhancement gain in decouple; override via `feature_opts_dicts[].adm_enhn_gain_limit` |
| `ADM_NORM_VIEW_DIST`           | 3.0               | ADM        | Viewing distance / display height                                                     |
| `ADM_REF_DISP_HEIGHT`   | 1080.0 px         | ADM        | Reference display height                                                              |
| `ADM_DIV_Q_FACTOR`      | 2^30              | ADM        | Q30 scale for div_lookup table                                                        |
| `ADM_NUMDEN_LIMIT_BASE` | 1e-10             | ADM        | numden threshold base (× W×H / FHD)                                                   |
| `MOTION_SAD_NORM`       | 256.0             | Motion     | SAD pre-normalization divisor                                                         |
| `MOTION_SAD_PRECISION`  | f32               | Motion     | Result is cast to f32 before dividing by W×H                                          |
| `SVM_GAMMA`             | 0.04              | SVM        | RBF kernel parameter                                                                  |
| `SCORE_CLIP_MIN`        | 0.0               | Prediction | Minimum output score                                                                  |
| `SCORE_CLIP_MAX`        | 100.0             | Prediction | Maximum output score                                                                  |
| `HARMONIC_MEAN_OFFSET`  | 1.0               | Pooling    | Offset preventing division-by-zero in pooling                                         |

---

## 8. Clean-Room and Portability Compliance Checklist

An implementation is considered conforming to this specification only if all items below are true:

1. It is built from this document and the external model data alone, without consulting an existing implementation.
2. It uses only the behaviors defined in this document for integer widths, overflow, shifts, rounding, and floating-point precision.
3. For transcendental functions, it either:
   uses correctly rounded destination-precision `log2`/`exp`/`log10`/`cbrt`, or
   replaces them with the tables/constants allowed by this specification.
   The ADM CSF path must use the binary32 weights from §4.3.5 for bit-exact conformance.
4. It reproduces the required single-precision operations exactly where `f32` is specified, especially in VIF LUT generation, ADM scoring, and Motion SAD normalization.
5. It applies the exact boundary handling, padding, border exclusion, and final-frame Motion flush rules given in §§4.2-4.4.
6. It uses the scale-0 ADM LL widening rule from §4.3.3: plain sign-extension to `int32`, with no additional left shift.
7. It treats threading as optional infrastructure only; changing thread count must not change results.
8. It rejects or sanitizes invalid inputs in a way that does not change the outputs for valid inputs.
9. It explicitly initializes `div_lookup[32768]` (the x=0 entry) to 0 rather than relying on array default initialization.

Recommended acceptance tests for a new-language port:

- Verify that `log2_table[32768]`, `log2_table[65535]`, and several random interior entries match a known-good generator using `f32`.
- Verify that the 8-bit scale-0 VIF squared-accumulator path exhibits modulo-`2^32` behavior, not widened `uint64` behavior.
- Verify that ADM scale-0 LL widening to scale 1 is plain sign-extension.
- Verify that ADM scale-1/2/3 decouple uses per-component normalization shifts and reconstructs from the original unshifted reference coefficients.
- Verify that Motion emits the final pending `motion2[n_last]` value during flush.
- Verify that changing thread count does not change any per-frame score.
- Verify that `div_lookup[32768]` equals 0 and that representative positive, negative, and power-of-two entries match the reference vectors below.
- Verify that `get_best15_from32` returns the expected `(mantissa, shift)` pairs from the reference table below, including the boundary at `abs_x=32768` (min valid input) and `abs_x=65535` (shift-1 boundary).
- Verify that `decouple_scale0` with the reference inputs below produces the expected `rst` and `art` outputs, including the k-clamping case (`dis_d=60 > ref_d=50`).
- Verify that `csf_weight[scale=0, band_v/h]` equals the binary32 value `0x3c8e63b8` as specified in §4.3.5.

Reference numeric conformance vectors:

- `round(f32_log2(f32(32768.0)) * 2048.0) = 30720`
- `round(f32_log2(f32(49152.0)) * 2048.0) = 31918`
- `round(f32_log2(f32(65535.0)) * 2048.0) = 32768`
- `f32_cbrt(f32(0.5))` must equal the binary32 value `0x3f4b2ff5` (`0.7937005162239075`)
- `exp(-1.0)` in `f64` must equal `0.36787944117144233`
- `log10(0.401)` in `f64` must equal `-0.39685562737981767`

**`div_lookup` reference vectors** (`floor(2^30 / |x|)`, negated for negative `x`; `2^30 = 1073741824`):

| x      | index (`x + 32768`) | `div_lookup[index]` |
| ------ | ------------------- | ------------------- |
| 0      | 32768               | 0                   |
| 1      | 32769               | 1073741824          |
| 2      | 32770               | 536870912           |
| 3      | 32771               | 357913941           |
| 7      | 32775               | 153391689           |
| 100    | 32868               | 10737418            |
| 256    | 33024               | 4194304             |
| 1000   | 33768               | 1073741             |
| 16384  | 49152               | 65536               |
| 32768  | 65536               | 32768               |
| -1     | 32767               | -1073741824         |
| -32768 | 0                   | -32768              |

**`get_best15_from32` reference vectors** (`shift = 17 - clz32(abs_x)`, `mantissa = uint16((abs_x + (1 << (shift-1))) >> shift)`):

| abs_x   | clz32 | shift | mantissa |
| ------- | ----- | ----- | -------- |
| 32768   | 16    | 1     | 16384    |
| 32769   | 16    | 1     | 16385    |
| 65535   | 16    | 1     | 32768    |
| 65536   | 15    | 2     | 16384    |
| 1000000 | 12    | 5     | 31250    |

**`decouple_scale0` reference case** (scale-0 `int16` subbands; `angle_flag = false`):

```
Input:   ref_h=100, ref_v=100, ref_d=50
         dis_h=80,  dis_v=90,  dis_d=60

Angle test:
  dp    = 100×80 + 100×90 = 17000
  o_sq  = 100² + 100²     = 20000
  t_sq  =  80² +  90²     = 14500
  17000² = 289 000 000  <  0.9997 × 20 000 × 14 500 = 289 913 000  →  angle_flag = false

Intermediate k values (clamped to [0, 32768]):
  k_h = 26214  (raw_k = 26214;  not clamped)
  k_v = 29491  (raw_k = 29491;  not clamped)
  k_d = 32768  (raw_k = 39322 → clamped; dis_d=60 > ref_d=50)

Expected output:
  rst_h = 80,  art_h =  0
  rst_v = 90,  art_v =  0
  rst_d = 50,  art_d = 10  ← k_d clamping causes non-zero artifact
```
