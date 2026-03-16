//! LIBSVM embedded text format parser — spec §3.3

use crate::model::{SupportVector, SvmModel};

/// Parse the embedded LIBSVM text string from the model JSON — spec §3.3.
///
/// Expected format:
/// ```text
/// svm_type nu_svr
/// kernel_type rbf
/// gamma <f64>
/// nr_class 2
/// total_sv <usize>
/// rho <f64>
/// SV
/// <coef> 1:<v1> 2:<v2> 3:<v3> 4:<v4> 5:<v5> 6:<v6>
/// ...
/// ```
pub fn parse_libsvm(text: &str) -> Result<SvmModel, String> {
    let mut gamma = None::<f64>;
    let mut rho = None::<f64>;
    let mut support_vectors = Vec::new();
    let mut in_sv_section = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line == "SV" {
            in_sv_section = true;
            continue;
        }

        if in_sv_section {
            support_vectors.push(parse_sv_line(line)?);
            continue;
        }

        // Header key-value pairs
        if let Some(v) = line.strip_prefix("gamma ") {
            gamma = Some(v.trim().parse().map_err(|e| format!("gamma: {e}"))?);
        } else if let Some(v) = line.strip_prefix("rho ") {
            rho = Some(v.trim().parse().map_err(|e| format!("rho: {e}"))?);
        }
        // svm_type, kernel_type, nr_class, total_sv are informational; skip.
    }

    Ok(SvmModel {
        gamma: gamma.ok_or("missing gamma")?,
        rho: rho.ok_or("missing rho")?,
        support_vectors,
    })
}

/// Parse one SV line: `<coef> 1:<v1> 2:<v2> … 6:<v6>`
///
/// Missing feature indices default to 0.0 (spec §3.3).
fn parse_sv_line(line: &str) -> Result<SupportVector, String> {
    let mut parts = line.split_whitespace();
    let coef: f64 = parts
        .next()
        .ok_or("empty SV line")?
        .parse()
        .map_err(|e| format!("coef: {e}"))?;

    let mut values = [0.0_f64; 6];
    for tok in parts {
        let (idx_s, val_s) = tok
            .split_once(':')
            .ok_or_else(|| format!("bad token: {tok}"))?;
        let idx: usize = idx_s.parse().map_err(|e| format!("index: {e}"))?;
        let val: f64 = val_s.parse().map_err(|e| format!("value: {e}"))?;
        if !(1..=6).contains(&idx) {
            return Err(format!("feature index {idx} out of range 1–6"));
        }
        values[idx - 1] = val;
    }

    Ok(SupportVector { coef, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_MODEL: &str = "\
svm_type nu_svr
kernel_type rbf
gamma 0.04
nr_class 2
total_sv 2
rho 0.5
SV
1.0 1:0.5 2:0.3 3:0.1 4:0.2 5:0.4 6:0.6
-0.5 1:0.1 2:0.2 3:0.3 4:0.4 5:0.5 6:0.6
";

    #[test]
    fn parse_header() {
        let m = parse_libsvm(MINIMAL_MODEL).unwrap();
        assert_eq!(m.gamma, 0.04);
        assert_eq!(m.rho, 0.5);
        assert_eq!(m.support_vectors.len(), 2);
    }

    #[test]
    fn parse_sv_coefficients() {
        let m = parse_libsvm(MINIMAL_MODEL).unwrap();
        assert_eq!(m.support_vectors[0].coef, 1.0);
        assert_eq!(m.support_vectors[1].coef, -0.5);
    }

    #[test]
    fn parse_sv_values() {
        let m = parse_libsvm(MINIMAL_MODEL).unwrap();
        assert_eq!(m.support_vectors[0].values, [0.5, 0.3, 0.1, 0.2, 0.4, 0.6]);
        assert_eq!(m.support_vectors[1].values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    /// Missing feature indices default to 0.0.
    #[test]
    fn parse_sv_missing_features() {
        let text = "svm_type nu_svr\ngamma 0.04\nrho 0.0\nSV\n1.0 3:0.9\n";
        let m = parse_libsvm(text).unwrap();
        let v = m.support_vectors[0].values;
        assert_eq!(v[0], 0.0); // index 1 missing
        assert_eq!(v[1], 0.0); // index 2 missing
        assert_eq!(v[2], 0.9); // index 3 present
        assert_eq!(v[3], 0.0);
    }

    #[test]
    fn parse_error_missing_gamma() {
        let text = "rho 0.5\nSV\n1.0 1:0.5\n";
        assert!(parse_libsvm(text).is_err());
    }
}
