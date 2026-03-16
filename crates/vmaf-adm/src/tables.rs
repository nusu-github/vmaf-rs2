//! Precomputed lookup tables for ADM (spec §4.3.6)

/// Fixed-point reciprocal table: `div_lookup[x + 32768] ≈ 2^30 / x` (Q30).
///
/// Index range: `x ∈ [-32768, 32768]`, stored at `index = x + 32768`.
///
/// Construction (§4.3.6):
/// - `div_lookup[32768] = 0`  (x = 0, explicit)
/// - `div_lookup[32768 + i] =  floor(2^30 / i)`   for i = 1..=32768
/// - `div_lookup[32768 - i] = -floor(2^30 / i)`   for i = 1..=32768
pub(crate) static DIV_LOOKUP: [i32; 65537] = build_div_lookup();

const fn build_div_lookup() -> [i32; 65537] {
    let mut table = [0i32; 65537];
    let mut i = 1usize;
    while i <= 32768 {
        let recip = (1i64 << 30) / (i as i64);
        table[32768 + i] = recip as i32;
        table[32768 - i] = -(recip as i32);
        i += 1;
    }
    // table[32768] = 0 already (array zero-init), but spec §4.3.6 mandates
    // explicit initialization — const fn zero-init satisfies this.
    table
}
