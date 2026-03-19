//! vmaf-cli — command-line interface for VMAF scoring

use std::{
    io::Read,
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use thiserror::Error;
use vmaf::{
    Finalized, FrameGeometry, LoadModelError, PoolMethod, ProcessingTimings, VmafContext,
    VmafError, VmafOptions, load_model,
};

#[derive(Debug, Parser)]
#[command(version, about = "VMAF video quality metric")]
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
    #[arg(long, default_value = "mean", value_parser = parse_pool_method)]
    pool_method: PoolMethod,
    /// Frame subsampling factor (1 = all frames)
    #[arg(long, default_value_t = 1)]
    n_subsample: usize,
    /// Number of threads to use (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Suppress progress display on stderr.
    #[arg(short, long, visible_alias = "no-progress")]
    quiet: bool,
    /// Output full per-frame JSON report instead of just the pooled score.
    #[arg(long)]
    json: bool,
    /// Apply the model's score_transform (spec mode). Disabled by default to match
    /// the reference ./vmaf behavior for bundled v0.6.x models.
    #[arg(long, default_value_t = false)]
    apply_score_transform: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorspaceFamily {
    Mono,
    Cs420,
    Cs422,
    Cs444,
}

#[derive(Debug, Clone)]
struct StreamMetadata {
    geometry: FrameGeometry,
    colorspace: String,
    colorspace_family: ColorspaceFamily,
}

type LumaPair = (Vec<u16>, Vec<u16>);

#[derive(Debug)]
struct LumaBatch {
    pairs: Vec<LumaPair>,
    len: usize,
}

impl LumaBatch {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut pairs = Vec::with_capacity(capacity);
        pairs.resize_with(capacity, || (Vec::new(), Vec::new()));
        Self { pairs, len: 0 }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn is_full(&self) -> bool {
        self.len == self.pairs.len()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn active_pairs(&self) -> &[LumaPair] {
        &self.pairs[..self.len]
    }

    fn next_slot_mut(&mut self) -> (&mut Vec<u16>, &mut Vec<u16>) {
        debug_assert!(self.len < self.pairs.len());
        let (reference, distorted) = &mut self.pairs[self.len];
        (reference, distorted)
    }

    fn commit_slot(&mut self) {
        self.len += 1;
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error("{stream} stream dimensions must be at least 16x16, got {width}x{height}")]
    InvalidDimensions {
        stream: &'static str,
        width: usize,
        height: usize,
    },
    #[error("{stream} stream bit depth must be 8, 10, or 12, got {bit_depth} ({colorspace})")]
    InvalidBitDepth {
        stream: &'static str,
        bit_depth: u8,
        colorspace: String,
    },
    #[error("{stream} stream colorspace is unsupported: {colorspace}")]
    UnsupportedColorspace {
        stream: &'static str,
        colorspace: String,
    },
    #[error("reference and distorted {field} must match, got {reference} vs {distorted}")]
    MetadataMismatch {
        field: &'static str,
        reference: String,
        distorted: String,
    },
    #[error(
        "reference and distorted colorspaces must be compatible, got {reference} vs {distorted}"
    )]
    ColorspaceMismatch {
        reference: String,
        distorted: String,
    },
    #[error("{stream} stream ended before the other stream after {paired_frames} paired frames")]
    AsymmetricEof {
        stream: &'static str,
        paired_frames: usize,
    },
    #[error("reference and distorted inputs must contain at least one frame")]
    ZeroFrameInput,
    #[error("failed to read {stream} frame: {source}")]
    StreamRead {
        stream: &'static str,
        #[source]
        source: y4m::Error,
    },
}

#[derive(Debug, Error)]
enum FrameProcessingError<E> {
    #[error(transparent)]
    Cli(#[from] CliError),
    #[error(transparent)]
    Callback(E),
}

#[derive(Debug, Error)]
#[error("unknown pool method: {value}")]
struct PoolMethodParseError {
    value: String,
}

#[derive(Debug, Error)]
enum MainError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Decoder(#[from] y4m::Error),
    #[error(transparent)]
    Model(#[from] LoadModelError),
    #[error(transparent)]
    Cli(#[from] CliError),
    #[error(transparent)]
    FrameProcessing(#[from] FrameProcessingError<VmafError>),
    #[error(transparent)]
    Vmaf(#[from] VmafError),
    #[error(transparent)]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
}

#[derive(Debug, Clone, Copy, Default)]
struct IoTimings {
    frame_read: Duration,
    luma_convert: Duration,
}

const TARGET_BATCH_BYTES: usize = 128 * 1024 * 1024;
const TARGET_BATCH_FRAMES_PER_WORKER: usize = 2;

fn hotspot_profile_enabled() -> bool {
    std::env::var_os("VMAF_HOTSPOT_PROFILE").is_some()
}

fn compute_batch_size(width: usize, height: usize, worker_count: usize) -> usize {
    let worker_count = worker_count.max(1);
    if let Some(override_batch_size) = positive_usize_env("VMAF_BATCH_SIZE_OVERRIDE") {
        return override_batch_size;
    }
    let desired = worker_count.saturating_mul(TARGET_BATCH_FRAMES_PER_WORKER);
    let bytes_per_pair = width
        .checked_mul(height)
        .and_then(|area| area.checked_mul(std::mem::size_of::<u16>() * 2))
        .unwrap_or(usize::MAX);
    let memory_cap = if bytes_per_pair == 0 {
        desired
    } else {
        (TARGET_BATCH_BYTES / bytes_per_pair).max(1)
    };
    desired.min(memory_cap).max(1)
}

fn positive_usize_env(key: &str) -> Option<usize> {
    match std::env::var(key) {
        Ok(value) => match value.parse::<usize>() {
            Ok(parsed) if parsed > 0 => Some(parsed),
            _ => {
                eprintln!("ignoring {key}={value:?}: expected a positive integer");
                None
            }
        },
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!("ignoring {key}: value is not valid UTF-8");
            None
        }
    }
}

fn print_timing_line(label: &str, duration: Duration, total: Duration) {
    let total_secs = total.as_secs_f64();
    let percentage = if total_secs > 0.0 {
        duration.as_secs_f64() / total_secs * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  {label:<24} {:>8.3} ms {:>6.2}%",
        duration.as_secs_f64() * 1_000.0,
        percentage
    );
}

fn print_hotspot_summary(
    total: Duration,
    io: IoTimings,
    batch_total: Duration,
    processing: ProcessingTimings,
    progress_update: Duration,
    flush: Duration,
    pool: Duration,
) {
    eprintln!(
        "hotspot profile (processing total {:.3} ms)",
        total.as_secs_f64() * 1_000.0
    );
    print_timing_line("read_frame", io.frame_read, total);
    print_timing_line("luma_convert", io.luma_convert, total);
    print_timing_line("batch_callback", batch_total, total);
    print_timing_line("  batch.validate", processing.validation, total);
    print_timing_line("  batch.features", processing.feature_extraction, total);
    print_timing_line("  batch.motion", processing.motion, total);
    print_timing_line("  batch.finalize", processing.finalize, total);
    print_timing_line("  progress_update", progress_update, total);

    let batch_accounted = processing.validation
        + processing.feature_extraction
        + processing.motion
        + processing.finalize
        + progress_update;
    let batch_other = batch_total.saturating_sub(batch_accounted);
    print_timing_line("  batch.other", batch_other, total);

    print_timing_line("flush", flush, total);
    print_timing_line("pool", pool, total);

    let accounted = io.frame_read + io.luma_convert + batch_total + flush + pool;
    let other = total.saturating_sub(accounted);
    print_timing_line("other", other, total);
}

fn luma_to_u16_into(y_plane: &[u8], bpc: u8, out: &mut Vec<u16>) {
    let sample_count = if bpc == 8 {
        y_plane.len()
    } else {
        y_plane.len() / 2
    };
    out.clear();
    if out.capacity() < sample_count {
        out.reserve(sample_count - out.capacity());
    }

    let spare = &mut out.spare_capacity_mut()[..sample_count];
    if bpc == 8 {
        for (slot, value) in spare.iter_mut().zip(y_plane.iter().copied()) {
            slot.write(u16::from(value));
        }
    } else {
        let chunks = y_plane.chunks_exact(2);
        debug_assert!(chunks.remainder().is_empty());
        for (slot, chunk) in spare.iter_mut().zip(chunks) {
            slot.write(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }

    // SAFETY: the written prefix matches `sample_count`, and every slot in it
    // was initialized exactly once above.
    unsafe {
        out.set_len(sample_count);
    }
}

fn parse_pool_method(s: &str) -> Result<PoolMethod, PoolMethodParseError> {
    match s {
        "mean" => Ok(PoolMethod::Mean),
        "harmonic_mean" => Ok(PoolMethod::HarmonicMean),
        "min" => Ok(PoolMethod::Min),
        "max" => Ok(PoolMethod::Max),
        other => Err(PoolMethodParseError {
            value: other.to_string(),
        }),
    }
}

fn pool_method_label(method: PoolMethod) -> &'static str {
    match method {
        PoolMethod::Mean => "mean",
        PoolMethod::HarmonicMean => "harmonic_mean",
        PoolMethod::Min => "min",
        PoolMethod::Max => "max",
    }
}

fn colorspace_family(colorspace: y4m::Colorspace) -> Option<ColorspaceFamily> {
    match colorspace {
        y4m::Colorspace::Cmono | y4m::Colorspace::Cmono12 => Some(ColorspaceFamily::Mono),
        y4m::Colorspace::C420
        | y4m::Colorspace::C420p10
        | y4m::Colorspace::C420p12
        | y4m::Colorspace::C420jpeg
        | y4m::Colorspace::C420paldv
        | y4m::Colorspace::C420mpeg2 => Some(ColorspaceFamily::Cs420),
        y4m::Colorspace::C422 | y4m::Colorspace::C422p10 | y4m::Colorspace::C422p12 => {
            Some(ColorspaceFamily::Cs422)
        }
        y4m::Colorspace::C444 | y4m::Colorspace::C444p10 | y4m::Colorspace::C444p12 => {
            Some(ColorspaceFamily::Cs444)
        }
        _ => None,
    }
}

fn read_stream_metadata<R: Read>(
    stream: &'static str,
    decoder: &y4m::Decoder<R>,
) -> Result<StreamMetadata, CliError> {
    let colorspace = decoder.get_colorspace();
    let colorspace_label = format!("{colorspace:?}");
    let colorspace_family =
        colorspace_family(colorspace).ok_or_else(|| CliError::UnsupportedColorspace {
            stream,
            colorspace: colorspace_label.clone(),
        })?;
    let geometry = stream_geometry(
        stream,
        decoder.get_width(),
        decoder.get_height(),
        colorspace.get_bit_depth() as u8,
        &colorspace_label,
    )?;

    Ok(StreamMetadata {
        geometry,
        colorspace: colorspace_label,
        colorspace_family,
    })
}

fn stream_geometry(
    stream: &'static str,
    width: usize,
    height: usize,
    bit_depth: u8,
    colorspace: &str,
) -> Result<FrameGeometry, CliError> {
    FrameGeometry::new(width, height, bit_depth).map_err(|err| match err {
        vmaf::FrameValidationError::InvalidDimensions { width, height } => {
            CliError::InvalidDimensions {
                stream,
                width,
                height,
            }
        }
        vmaf::FrameValidationError::InvalidBitDepth { bpc } => CliError::InvalidBitDepth {
            stream,
            bit_depth: bpc,
            colorspace: colorspace.to_string(),
        },
        vmaf::FrameValidationError::SampleCountOverflow { width, height } => {
            CliError::InvalidDimensions {
                stream,
                width,
                height,
            }
        }
    })
}

fn validate_stream_metadata(
    reference: &StreamMetadata,
    distorted: &StreamMetadata,
) -> Result<FrameGeometry, CliError> {
    if reference.geometry.width() != distorted.geometry.width() {
        return Err(CliError::MetadataMismatch {
            field: "width",
            reference: reference.geometry.width().to_string(),
            distorted: distorted.geometry.width().to_string(),
        });
    }

    if reference.geometry.height() != distorted.geometry.height() {
        return Err(CliError::MetadataMismatch {
            field: "height",
            reference: reference.geometry.height().to_string(),
            distorted: distorted.geometry.height().to_string(),
        });
    }

    if reference.geometry.bpc() != distorted.geometry.bpc() {
        return Err(CliError::MetadataMismatch {
            field: "bit depth",
            reference: reference.geometry.bpc().to_string(),
            distorted: distorted.geometry.bpc().to_string(),
        });
    }

    if reference.colorspace_family != distorted.colorspace_family {
        return Err(CliError::ColorspaceMismatch {
            reference: reference.colorspace.clone(),
            distorted: distorted.colorspace.clone(),
        });
    }

    Ok(reference.geometry)
}

fn read_luma_pair_into_with_timings<R1: Read, R2: Read>(
    ref_dec: &mut y4m::Decoder<R1>,
    dis_dec: &mut y4m::Decoder<R2>,
    bpc: u8,
    paired_frames: usize,
    reference_out: &mut Vec<u16>,
    distorted_out: &mut Vec<u16>,
    mut timings: Option<&mut IoTimings>,
) -> Result<bool, CliError> {
    let read_start = Instant::now();
    let ref_frame = ref_dec.read_frame();
    let dis_frame = dis_dec.read_frame();
    if let Some(timings) = timings.as_mut() {
        timings.frame_read += read_start.elapsed();
    }

    match (ref_frame, dis_frame) {
        (Ok(ref_frame), Ok(dis_frame)) => {
            let convert_start = Instant::now();
            luma_to_u16_into(ref_frame.get_y_plane(), bpc, reference_out);
            luma_to_u16_into(dis_frame.get_y_plane(), bpc, distorted_out);
            if let Some(timings) = timings.as_mut() {
                timings.luma_convert += convert_start.elapsed();
            }
            Ok(true)
        }
        (Err(y4m::Error::EOF), Err(y4m::Error::EOF)) => Ok(false),
        (Err(y4m::Error::EOF), Ok(_)) => Err(CliError::AsymmetricEof {
            stream: "reference",
            paired_frames,
        }),
        (Ok(_), Err(y4m::Error::EOF)) => Err(CliError::AsymmetricEof {
            stream: "distorted",
            paired_frames,
        }),
        (Err(y4m::Error::EOF), Err(source)) => Err(CliError::StreamRead {
            stream: "distorted",
            source,
        }),
        (Err(source), Err(y4m::Error::EOF)) => Err(CliError::StreamRead {
            stream: "reference",
            source,
        }),
        (Err(source), _) => Err(CliError::StreamRead {
            stream: "reference",
            source,
        }),
        (_, Err(source)) => Err(CliError::StreamRead {
            stream: "distorted",
            source,
        }),
    }
}

fn flush_batch<F, E>(
    batch: &mut LumaBatch,
    frame_count: &mut usize,
    on_batch: &mut F,
) -> Result<(), FrameProcessingError<E>>
where
    F: FnMut(&[(&[u16], &[u16])], usize) -> Result<(), E>,
    E: std::error::Error + 'static,
{
    if batch.is_empty() {
        return Ok(());
    }

    let refs: Vec<(&[u16], &[u16])> = batch
        .active_pairs()
        .iter()
        .map(|(reference, distorted)| (reference.as_slice(), distorted.as_slice()))
        .collect();
    let new_frame_count = *frame_count + batch.len();
    on_batch(refs.as_slice(), new_frame_count).map_err(FrameProcessingError::Callback)?;
    *frame_count = new_frame_count;
    batch.clear();
    Ok(())
}

fn process_stream_frames<R1: Read, R2: Read, F, E>(
    ref_dec: &mut y4m::Decoder<R1>,
    dis_dec: &mut y4m::Decoder<R2>,
    bpc: u8,
    batch_size: usize,
    on_batch: F,
) -> Result<usize, FrameProcessingError<E>>
where
    F: FnMut(&[(&[u16], &[u16])], usize) -> Result<(), E>,
    E: std::error::Error + 'static,
{
    process_stream_frames_with_timings(ref_dec, dis_dec, bpc, batch_size, None, on_batch)
}

fn process_stream_frames_with_timings<R1: Read, R2: Read, F, E>(
    ref_dec: &mut y4m::Decoder<R1>,
    dis_dec: &mut y4m::Decoder<R2>,
    bpc: u8,
    batch_size: usize,
    mut timings: Option<&mut IoTimings>,
    mut on_batch: F,
) -> Result<usize, FrameProcessingError<E>>
where
    F: FnMut(&[(&[u16], &[u16])], usize) -> Result<(), E>,
    E: std::error::Error + 'static,
{
    let batch_size = batch_size.max(1);
    let mut batch = LumaBatch::new(batch_size);
    let mut frame_count = 0usize;

    loop {
        let paired_frames = frame_count + batch.len();
        let has_pair = {
            let (reference, distorted) = batch.next_slot_mut();
            read_luma_pair_into_with_timings(
                ref_dec,
                dis_dec,
                bpc,
                paired_frames,
                reference,
                distorted,
                timings.as_deref_mut(),
            )?
        };

        if !has_pair {
            break;
        }

        batch.commit_slot();
        if batch.is_full() {
            flush_batch(&mut batch, &mut frame_count, &mut on_batch)?;
        }
    }

    flush_batch(&mut batch, &mut frame_count, &mut on_batch)?;

    if frame_count == 0 {
        return Err(CliError::ZeroFrameInput.into());
    }

    Ok(frame_count)
}

fn emit_full_json(ctx: &VmafContext<Finalized>, pooled: f64, pool_method: &str) {
    // Manual JSON output keeps the CLI lightweight without pulling in serde_json.
    print!("{{\n  \"frames\": [\n");
    for (i, fs) in ctx.per_frame_scores().iter().enumerate() {
        if i > 0 {
            println!(",");
        }
        print!(
            "    {{\"frameIndex\": {}, \"score\": {:.6}, \"adm2\": {:.6}, \\
             \"motion2\": {:.6}, \"vifScale0\": {:.6}, \"vifScale1\": {:.6}, \\
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

fn make_progress_bar() -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{pos} frames {spinner} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn main() -> Result<(), MainError> {
    let args = Args::parse();
    let hotspot_profile = hotspot_profile_enabled();

    if !args.quiet {
        eprintln!("VMAF version {}", env!("CARGO_PKG_VERSION"));
    }

    let json = std::fs::read_to_string(&args.model)?;
    let model = load_model(&json)?;
    let ref_file = std::fs::File::open(&args.reference)?;
    let dis_file = std::fs::File::open(&args.distorted)?;

    let mut ref_dec = y4m::Decoder::new(ref_file)?;
    let mut dis_dec = y4m::Decoder::new(dis_file)?;

    let ref_meta = read_stream_metadata("reference", &ref_dec)?;
    let dis_meta = read_stream_metadata("distorted", &dis_dec)?;
    let geometry = validate_stream_metadata(&ref_meta, &dis_meta)?;

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }

    let mut ctx = VmafContext::new_with_options(
        model,
        geometry,
        VmafOptions {
            apply_score_transform: args.apply_score_transform,
        },
    );

    let pb = if args.quiet {
        ProgressBar::hidden()
    } else {
        make_progress_bar()
    };

    let worker_count = if args.threads > 0 {
        args.threads
    } else {
        rayon::current_num_threads()
    };
    let batch_size = compute_batch_size(geometry.width(), geometry.height(), worker_count);
    let start = Instant::now();
    let mut io_timings = IoTimings::default();
    let mut processing_timings = ProcessingTimings::default();
    let mut batch_total = Duration::ZERO;
    let mut progress_update = Duration::ZERO;

    let mut handle_batch = |refs: &[(&[u16], &[u16])], frame_count: usize| {
        let batch_start = Instant::now();
        if hotspot_profile {
            let (_scores, timings) = ctx.push_frame_batch_with_timings(refs)?;
            processing_timings += timings;
        } else {
            ctx.push_frame_batch(refs)?;
        }

        let progress_start = Instant::now();
        pb.set_position(frame_count as u64);
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            pb.set_message(format!("{:.2} FPS", frame_count as f64 / elapsed));
        }
        if hotspot_profile {
            progress_update += progress_start.elapsed();
            batch_total += batch_start.elapsed();
        }
        Ok::<_, vmaf::VmafError>(())
    };

    let process_result = if hotspot_profile {
        process_stream_frames_with_timings(
            &mut ref_dec,
            &mut dis_dec,
            geometry.bpc(),
            batch_size,
            Some(&mut io_timings),
            &mut handle_batch,
        )
    } else {
        process_stream_frames(
            &mut ref_dec,
            &mut dis_dec,
            geometry.bpc(),
            batch_size,
            &mut handle_batch,
        )
    };

    if let Err(err) = process_result {
        pb.finish_and_clear();
        return Err(err.into());
    }

    let flush_start = Instant::now();
    let ctx = ctx.flush();
    let flush_time = if hotspot_profile {
        flush_start.elapsed()
    } else {
        Duration::ZERO
    };
    pb.finish();

    let pool_start = Instant::now();
    let pooled = ctx.pool_score(args.pool_method, args.n_subsample);
    let pool_time = if hotspot_profile {
        pool_start.elapsed()
    } else {
        Duration::ZERO
    };

    if hotspot_profile {
        print_hotspot_summary(
            start.elapsed(),
            io_timings,
            batch_total,
            processing_timings,
            progress_update,
            flush_time,
            pool_time,
        );
    }

    if args.json {
        emit_full_json(&ctx, pooled, pool_method_label(args.pool_method));
    } else {
        println!("{}: {:.6}", args.model.display(), pooled);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use clap::CommandFactory;

    use super::*;

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

    fn plane_sizes(
        width: usize,
        height: usize,
        colorspace: y4m::Colorspace,
    ) -> (usize, usize, usize) {
        let bytes_per_sample = colorspace.get_bytes_per_sample();
        let y_len = width * height * bytes_per_sample;
        let c420_len = width.div_ceil(2) * height.div_ceil(2) * bytes_per_sample;
        let c422_len = width.div_ceil(2) * height * bytes_per_sample;

        match colorspace_family(colorspace) {
            Some(ColorspaceFamily::Mono) => (y_len, 0, 0),
            Some(ColorspaceFamily::Cs420) => (y_len, c420_len, c420_len),
            Some(ColorspaceFamily::Cs422) => (y_len, c422_len, c422_len),
            Some(ColorspaceFamily::Cs444) => (y_len, y_len, y_len),
            None => panic!("test colorspace should be supported"),
        }
    }

    fn make_y4m_stream(
        width: usize,
        height: usize,
        colorspace: y4m::Colorspace,
        frame_count: usize,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = y4m::encode(width, height, y4m::Ratio::new(24, 1))
            .with_colorspace(colorspace)
            .write_header(&mut bytes)
            .expect("Y4M header should be writable");
        let (y_len, u_len, v_len) = plane_sizes(width, height, colorspace);
        let y = vec![0u8; y_len];
        let u = vec![0u8; u_len];
        let v = vec![0u8; v_len];

        for _ in 0..frame_count {
            let frame = y4m::Frame::new([&y, &u, &v], None);
            encoder
                .write_frame(&frame)
                .expect("Y4M frame should be writable");
        }

        bytes
    }

    fn decoder_from_bytes(bytes: Vec<u8>) -> y4m::Decoder<Cursor<Vec<u8>>> {
        y4m::Decoder::new(Cursor::new(bytes)).expect("Y4M stream should decode")
    }

    #[test]
    fn cli_defaults() {
        let args = parse_args(&[]);

        assert!(!args.quiet);
        assert!(!args.json);
    }

    #[test]
    fn cli_quiet_and_json_flags() {
        let args = parse_args(&["--no-progress", "--json"]);

        assert!(args.quiet);
        assert!(args.json);

        let args = parse_args(&["--quiet"]);

        assert!(args.quiet);
        assert!(!args.json);
    }

    #[test]
    fn help_text_mentions_controls() {
        let mut cmd = Args::command();
        let help = cmd.render_long_help().to_string();

        assert!(help.contains("--quiet"));
        assert!(help.contains("no-progress"));
        assert!(help.contains("--json"));
    }

    #[test]
    fn compute_batch_size_scales_up_small_inputs() {
        assert_eq!(compute_batch_size(16, 16, 8), 16);
        assert_eq!(compute_batch_size(16, 16, 1), 2);
    }

    #[test]
    fn compute_batch_size_caps_large_inputs_by_memory() {
        assert_eq!(compute_batch_size(1920, 1080, 8), 16);
        assert_eq!(compute_batch_size(3840, 2160, 8), 4);
    }

    #[test]
    fn validate_stream_metadata_rejects_small_dimensions() {
        let err = stream_geometry("reference", 15, 16, 8, "C420").unwrap_err();
        assert_eq!(
            err.to_string(),
            "reference stream dimensions must be at least 16x16, got 15x16"
        );
    }

    #[test]
    fn validate_stream_metadata_rejects_invalid_bit_depth() {
        let err = stream_geometry("reference", 16, 16, 9, "C420p9").unwrap_err();
        assert_eq!(
            err.to_string(),
            "reference stream bit depth must be 8, 10, or 12, got 9 (C420p9)"
        );
    }

    #[test]
    fn validate_stream_metadata_rejects_width_mismatch() {
        let ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 1));
        let dis_dec = decoder_from_bytes(make_y4m_stream(32, 16, y4m::Colorspace::C420, 1));

        let ref_meta = read_stream_metadata("reference", &ref_dec).unwrap();
        let dis_meta = read_stream_metadata("distorted", &dis_dec).unwrap();
        let err = validate_stream_metadata(&ref_meta, &dis_meta).unwrap_err();

        assert_eq!(
            err.to_string(),
            "reference and distorted width must match, got 16 vs 32"
        );
    }

    #[test]
    fn validate_stream_metadata_rejects_bit_depth_mismatch() {
        let ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 1));
        let dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420p10, 1));

        let ref_meta = read_stream_metadata("reference", &ref_dec).unwrap();
        let dis_meta = read_stream_metadata("distorted", &dis_dec).unwrap();
        let err = validate_stream_metadata(&ref_meta, &dis_meta).unwrap_err();

        assert_eq!(
            err.to_string(),
            "reference and distorted bit depth must match, got 8 vs 10"
        );
    }

    #[test]
    fn validate_stream_metadata_rejects_incompatible_colorspaces() {
        let ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 1));
        let dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C444, 1));

        let ref_meta = read_stream_metadata("reference", &ref_dec).unwrap();
        let dis_meta = read_stream_metadata("distorted", &dis_dec).unwrap();
        let err = validate_stream_metadata(&ref_meta, &dis_meta).unwrap_err();

        assert_eq!(
            err.to_string(),
            "reference and distorted colorspaces must be compatible, got C420 vs C444"
        );
    }

    #[test]
    fn validate_stream_metadata_accepts_compatible_420_variants() {
        let ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420jpeg, 1));
        let dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420mpeg2, 1));

        let ref_meta = read_stream_metadata("reference", &ref_dec).unwrap();
        let dis_meta = read_stream_metadata("distorted", &dis_dec).unwrap();

        validate_stream_metadata(&ref_meta, &dis_meta)
            .expect("4:2:0 variants should be compatible");
    }

    #[test]
    fn process_stream_frames_rejects_distorted_eof_first() {
        let mut ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 2));
        let mut dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 1));

        let err = process_stream_frames(&mut ref_dec, &mut dis_dec, 8, 4, |_, _| {
            Ok::<_, CliError>(())
        })
        .expect_err("distorted EOF should be rejected");

        assert_eq!(
            err.to_string(),
            "distorted stream ended before the other stream after 1 paired frames"
        );
    }

    #[test]
    fn process_stream_frames_rejects_reference_eof_first() {
        let mut ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 1));
        let mut dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 2));

        let err = process_stream_frames(&mut ref_dec, &mut dis_dec, 8, 4, |_, _| {
            Ok::<_, CliError>(())
        })
        .expect_err("reference EOF should be rejected");

        assert_eq!(
            err.to_string(),
            "reference stream ended before the other stream after 1 paired frames"
        );
    }

    #[test]
    fn process_stream_frames_rejects_zero_frame_inputs() {
        let mut ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 0));
        let mut dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420, 0));

        let err = process_stream_frames(&mut ref_dec, &mut dis_dec, 8, 4, |_, _| {
            Ok::<_, CliError>(())
        })
        .expect_err("zero-frame inputs should be rejected");

        assert_eq!(
            err.to_string(),
            "reference and distorted inputs must contain at least one frame"
        );
    }

    #[test]
    fn process_stream_frames_counts_valid_pairs() {
        let mut ref_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420p10, 3));
        let mut dis_dec = decoder_from_bytes(make_y4m_stream(16, 16, y4m::Colorspace::C420p10, 3));
        let mut batches = Vec::new();

        let frame_count = process_stream_frames(&mut ref_dec, &mut dis_dec, 10, 2, |_, total| {
            batches.push(total);
            Ok::<_, CliError>(())
        })
        .expect("valid streams should process successfully");

        assert_eq!(frame_count, 3);
        assert_eq!(batches, vec![2, 3]);
    }
}
