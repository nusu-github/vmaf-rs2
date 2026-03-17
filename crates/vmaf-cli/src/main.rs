//! vmaf-cli — command-line interface for VMAF scoring

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use vmaf::{load_model, FrameScore, PoolMethod, VmafContext, VmafOptions};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum OutputMode {
    /// Emit the full per-frame JSON report.
    Json,
    /// Emit only the pooled score.
    Pooled,
    /// Do not emit any stdout output.
    None,
}

#[derive(Debug, Parser)]
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
    /// Number of threads to use (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Suppress progress/status logging on stderr.
    ///
    /// Useful for benchmarking to avoid per-frame stderr overhead.
    #[arg(short, long, visible_alias = "no-progress")]
    quiet: bool,
    /// Stdout output mode.
    ///
    /// Keep the default full JSON for compatibility, or use `pooled`/`none`
    /// during benchmarks to avoid formatting the full report.
    #[arg(long, value_enum, default_value_t = OutputMode::Json)]
    output: OutputMode,
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

fn emit_frame_progress(quiet: bool, fs: &FrameScore) {
    if quiet {
        return;
    }

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

fn emit_full_json(ctx: &VmafContext, pooled: f64, pool_method: &str) {
    // Manual JSON output keeps the CLI lightweight without pulling in serde_json.
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
    println!("  \"poolMethod\": \"{}\"", pool_method);
    println!("}}");
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

    if !args.quiet {
        eprintln!("[vmaf] {}×{} bpc={} model loaded", width, height, bpc);
    }

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

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
    let batch_size = if args.threads > 0 {
        args.threads * 2
    } else {
        rayon::current_num_threads() * 2
    };
    let mut batch = Vec::with_capacity(batch_size);

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

        batch.push((ref_luma, dis_luma));

        if batch.len() >= batch_size {
            let refs: Vec<(&[u16], &[u16])> = batch
                .iter()
                .map(|(r, d)| (r.as_slice(), d.as_slice()))
                .collect();
            for fs in ctx.push_frame_batch(&refs) {
                emit_frame_progress(args.quiet, &fs);
            }
            frame_count += batch.len();
            if !args.quiet {
                eprintln!("[vmaf] pushed {} frames…", frame_count);
            }
            batch.clear();
        }
    }

    if !batch.is_empty() {
        let refs: Vec<(&[u16], &[u16])> = batch
            .iter()
            .map(|(r, d)| (r.as_slice(), d.as_slice()))
            .collect();
        for fs in ctx.push_frame_batch(&refs) {
            emit_frame_progress(args.quiet, &fs);
        }
        frame_count += batch.len();
    }

    if !args.quiet {
        eprintln!("[vmaf] pushed {} frames total, flushing…", frame_count);
    }
    if let Some(fs) = ctx.flush() {
        emit_frame_progress(args.quiet, &fs);
    }

    let pooled = ctx.pool_score(pool_method, args.n_subsample);

    match args.output {
        OutputMode::Json => emit_full_json(&ctx, pooled, &args.pool_method),
        OutputMode::Pooled => println!("{pooled:.6}"),
        OutputMode::None => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse_args(extra: &[&str]) -> Args {
        let mut argv = vec![
            "vmaf",
            "--reference",
            "reference.y4m",
            "--distorted",
            "distorted.y4m",
            "--model",
            "model.json",
        ];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("CLI args should parse")
    }

    #[test]
    fn cli_defaults_preserve_existing_behavior() {
        let args = parse_args(&[]);

        assert!(!args.quiet);
        assert_eq!(args.output, OutputMode::Json);
    }

    #[test]
    fn cli_supports_quiet_alias_and_lightweight_output_modes() {
        let args = parse_args(&["--no-progress", "--output", "none"]);

        assert!(args.quiet);
        assert_eq!(args.output, OutputMode::None);

        let args = parse_args(&["--quiet", "--output", "pooled"]);

        assert!(args.quiet);
        assert_eq!(args.output, OutputMode::Pooled);
    }

    #[test]
    fn help_text_mentions_benchmark_controls() {
        let mut cmd = Args::command();
        let help = cmd.render_long_help().to_string();

        assert!(help.contains("--quiet"));
        assert!(help.contains("no-progress"));
        assert!(help.contains("--output <OUTPUT>"));
        assert!(help.contains("pooled"));
        assert!(help.contains("none"));
    }
}
