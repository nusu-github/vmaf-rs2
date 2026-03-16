//! vmaf-cli — command-line interface for VMAF scoring

use clap::Parser;
use std::path::PathBuf;
use vmaf::{load_model, PoolMethod, VmafContext, VmafOptions};

#[derive(Parser)]
struct Args {
    /// Reference (original) Y4M file
    #[arg(short, long)]
    reference: PathBuf,
    /// Distorted (encoded) Y4M file
    #[arg(short, long)]
    distorted: PathBuf,
    /// VMAF JSON model file
    #[arg(short, long)]
    model: PathBuf,
    /// Pooling method: mean, harmonic_mean, min, max
    #[arg(long, default_value = "mean")]
    pool_method: String,
    /// Frame subsampling factor (1 = all frames)
    #[arg(long, default_value_t = 1)]
    n_subsample: usize,
    /// Apply the model's score_transform (spec mode). Disabled by default to match
    /// the reference ./vmaf behavior for bundled v0.6.x models.
    #[arg(long, default_value_t = false)]
    apply_score_transform: bool,
}

fn luma_to_u16(y_plane: &[u8], bpc: u8) -> Vec<u16> {
    if bpc == 8 {
        y_plane.iter().map(|&b| b as u16).collect()
    } else {
        // 10/12-bit: little-endian u16 per sample
        y_plane
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

fn parse_pool_method(s: &str) -> Result<PoolMethod, Box<dyn std::error::Error>> {
    match s {
        "mean" => Ok(PoolMethod::Mean),
        "harmonic_mean" => Ok(PoolMethod::HarmonicMean),
        "min" => Ok(PoolMethod::Min),
        "max" => Ok(PoolMethod::Max),
        other => Err(format!("unknown pool method: {other}").into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let json = std::fs::read_to_string(&args.model)?;
    let model = load_model(&json).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let pool_method = parse_pool_method(&args.pool_method)?;

    let ref_file = std::fs::File::open(&args.reference)?;
    let dis_file = std::fs::File::open(&args.distorted)?;

    let mut ref_dec = y4m::Decoder::new(ref_file)?;
    let mut dis_dec = y4m::Decoder::new(dis_file)?;

    let width = ref_dec.get_width();
    let height = ref_dec.get_height();
    let bpc = ref_dec.get_colorspace().get_bit_depth() as u8;

    eprintln!("[vmaf] {}×{} bpc={} model loaded", width, height, bpc);

    let mut ctx = VmafContext::new_with_options(
        model,
        width,
        height,
        bpc,
        VmafOptions {
            apply_score_transform: args.apply_score_transform,
        },
    );
    let mut frame_count = 0usize;

    loop {
        let ref_frame = match ref_dec.read_frame() {
            Ok(f) => f,
            Err(y4m::Error::EOF) => break,
            Err(e) => return Err(Box::new(e)),
        };
        let dis_frame = match dis_dec.read_frame() {
            Ok(f) => f,
            Err(y4m::Error::EOF) => break,
            Err(e) => return Err(Box::new(e)),
        };

        let ref_luma = luma_to_u16(ref_frame.get_y_plane(), bpc);
        let dis_luma = luma_to_u16(dis_frame.get_y_plane(), bpc);

        if let Some(fs) = ctx.push_frame(&ref_luma, &dis_luma) {
            eprintln!(
                "[vmaf] frame {:3}  score={:.4}  adm2={:.4}  motion2={:.4}  \
                 vif=[{:.4},{:.4},{:.4},{:.4}]",
                fs.frame_index,
                fs.score,
                fs.adm2,
                fs.motion2,
                fs.vif_scale0,
                fs.vif_scale1,
                fs.vif_scale2,
                fs.vif_scale3,
            );
        }
        frame_count += 1;
        if frame_count.is_multiple_of(10) {
            eprintln!("[vmaf] pushed {} frames…", frame_count);
        }
    }

    eprintln!("[vmaf] pushed {} frames total, flushing…", frame_count);
    if let Some(fs) = ctx.flush() {
        eprintln!(
            "[vmaf] frame {:3}  score={:.4}  adm2={:.4}  motion2={:.4}  \
             vif=[{:.4},{:.4},{:.4},{:.4}]",
            fs.frame_index,
            fs.score,
            fs.adm2,
            fs.motion2,
            fs.vif_scale0,
            fs.vif_scale1,
            fs.vif_scale2,
            fs.vif_scale3,
        );
    }

    let pooled = ctx.pool_score(pool_method, args.n_subsample);

    // Manual JSON output (no serde_json dependency needed)
    print!("{{\n  \"frames\": [\n");
    for (i, fs) in ctx.per_frame_scores().iter().enumerate() {
        if i > 0 {
            println!(",");
        }
        print!(
            "    {{\"frameIndex\": {}, \"score\": {:.6}, \"adm2\": {:.6}, \
             \"motion2\": {:.6}, \"vifScale0\": {:.6}, \"vifScale1\": {:.6}, \
             \"vifScale2\": {:.6}, \"vifScale3\": {:.6}}}",
            fs.frame_index,
            fs.score,
            fs.adm2,
            fs.motion2,
            fs.vif_scale0,
            fs.vif_scale1,
            fs.vif_scale2,
            fs.vif_scale3,
        );
    }
    print!("\n  ],\n");
    println!("  \"pooledScore\": {:.6},", pooled);
    println!("  \"poolMethod\": \"{}\"", args.pool_method);
    println!("}}");

    Ok(())
}
