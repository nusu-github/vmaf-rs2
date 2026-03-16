//! Noise-floor based DWT visibility model used by libvmaf integer_adm.
//!
//! This is the Watson/Yang/Solomon/Villasenor (1997) model for the 7-9
//! biorthogonal wavelet basis. libvmaf uses it to derive per-scale CSF weights
//! via `dwt_quant_step()`.

const M_PI: f64 = core::f64::consts::PI;

pub(crate) const DEFAULT_ADM_NORM_VIEW_DIST: f64 = 3.0;
pub(crate) const DEFAULT_ADM_REF_DISPLAY_HEIGHT: i32 = 1080;

#[derive(Clone, Copy)]
struct DwtModelParams {
    a: f32,
    k: f32,
    f0: f32,
    g: [f32; 4],
}

// 0 -> Y, 1 -> Cb, 2 -> Cr (we only use Y).
const DWT_7_9_Y_THRESHOLD: DwtModelParams = DwtModelParams {
    a: 0.495,
    k: 0.466,
    f0: 0.401,
    g: [1.501, 1.0, 0.534, 1.0],
};

// Transposed table: A[lambda][theta].
const DWT_7_9_BASIS_AMPLITUDES: [[f32; 4]; 6] = [
    [0.62171, 0.67234, 0.72709, 0.67234],
    [0.34537, 0.41317, 0.49428, 0.41317],
    [0.18004, 0.22727, 0.28688, 0.22727],
    [0.091401, 0.11792, 0.15214, 0.11792],
    [0.045943, 0.059758, 0.077727, 0.059758],
    [0.023013, 0.030018, 0.039156, 0.030018],
];

/// libvmaf `dwt_quant_step()` for Y channel.
///
/// - `lambda`: 0..=3 for ADM scales 0..=3.
/// - `theta`: wavelet orientation (0..=3). libvmaf calls this with 1 (HV) and 2 (D).
pub(crate) fn dwt_quant_step(
    lambda: usize,
    theta: usize,
    adm_norm_view_dist: f64,
    adm_ref_display_height: i32,
) -> f32 {
    let p = DWT_7_9_Y_THRESHOLD;

    // In libvmaf: `float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;`
    // The expression is evaluated in double and then truncated to binary32.
    let r: f32 = (adm_norm_view_dist * adm_ref_display_height as f64 * M_PI / 180.0) as f32;

    // `temp = log10(pow(2.0, lambda + 1) * f0 * g[theta] / r)`
    // pow/log10 are double; assigned to float.
    let temp: f32 = (((2.0f64).powi(lambda as i32 + 1) * p.f0 as f64 * p.g[theta] as f64)
        / (r as f64))
        .log10() as f32;

    // `Q = 2*a*pow(10.0, k*temp*temp) / basis[lambda][theta]`.
    // The exponent is computed in float precision then promoted to double for pow().
    let exp_f32: f32 = p.k * temp * temp;
    let pow_term: f64 = (10.0f64).powf(exp_f32 as f64);
    let basis = DWT_7_9_BASIS_AMPLITUDES[lambda][theta] as f64;

    (2.0f64 * p.a as f64 * pow_term / basis) as f32
}

/// Convenience: return the 3 orientation rfactor values used by integer_adm:
/// `[rfactor_h, rfactor_v, rfactor_d]`.
pub(crate) fn rfactor(scale: usize) -> [f32; 3] {
    let factor_hv = dwt_quant_step(
        scale,
        1,
        DEFAULT_ADM_NORM_VIEW_DIST,
        DEFAULT_ADM_REF_DISPLAY_HEIGHT,
    );
    let factor_d = dwt_quant_step(
        scale,
        2,
        DEFAULT_ADM_NORM_VIEW_DIST,
        DEFAULT_ADM_REF_DISPLAY_HEIGHT,
    );
    [1.0 / factor_hv, 1.0 / factor_hv, 1.0 / factor_d]
}
