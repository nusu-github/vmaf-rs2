/// Build script: generates the Q11 log2 lookup table for VIF (spec §4.2.1).
///
/// `log2_table[i] = round(f32::log2(i as f32) * 2048.0)` for i in 32767..=65535.
/// CRITICAL: must use f32 (not f64) — spec §4.2.1 note.
///
/// Produces `$OUT_DIR/log2_table_data.rs`, included by `src/tables.rs`.
fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let path = std::path::Path::new(&out_dir).join("log2_table_data.rs");

    let mut buf = String::with_capacity(65536 * 6);
    buf.push('[');

    for i in 0u32..65536 {
        let v: u16 = if i >= 32767 {
            // f32 log2, then round ties-away-from-zero, then cast to u16
            let log2_val = (i as f32).log2() * 2048.0_f32;
            log2_val.round() as u16
        } else {
            0
        };
        buf.push_str(&v.to_string());
        if i < 65535 {
            buf.push(',');
        }
    }

    buf.push(']');
    std::fs::write(&path, buf).expect("failed to write log2_table_data.rs");

    println!("cargo:rerun-if-changed=build.rs");
}
