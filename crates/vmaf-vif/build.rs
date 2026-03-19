/// Build script: generates the Q11 log2 lookup table for VIF (spec §4.2.1).
///
/// `log2_table[i - 32768] = round(f32::log2(i as f32) * 2048.0)` for
/// `i in 32768..=65535`.
/// CRITICAL: must use f32 (not f64) — spec §4.2.1 note.
///
/// Produces `$OUT_DIR/log2_table_data.rs`, included by `src/tables.rs`.
fn main() {
    const LOG2_TABLE_MIN_INDEX: u32 = 32768;
    const LOG2_TABLE_MAX_EXCLUSIVE: u32 = 65536;

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let path = std::path::Path::new(&out_dir).join("log2_table_data.rs");

    let entry_count = (LOG2_TABLE_MAX_EXCLUSIVE - LOG2_TABLE_MIN_INDEX) as usize;
    let mut buf = String::with_capacity(entry_count * 6);
    buf.push('[');

    for i in LOG2_TABLE_MIN_INDEX..LOG2_TABLE_MAX_EXCLUSIVE {
        // f32 log2, then round ties-away-from-zero, then cast to u16
        let log2_val = (i as f32).log2() * 2048.0_f32;
        let v = log2_val.round() as u16;
        buf.push_str(&v.to_string());
        if i + 1 < LOG2_TABLE_MAX_EXCLUSIVE {
            buf.push(',');
        }
    }

    buf.push(']');
    std::fs::write(&path, buf).expect("failed to write log2_table_data.rs");

    println!("cargo:rerun-if-changed=build.rs");
}
