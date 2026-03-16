//! JSON deserialization for VmafModel — spec §3.3

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::libsvm::parse_libsvm;
use crate::model::{ScoreTransform, VmafModel};

// ---------- serde shapes ----------

#[derive(Deserialize)]
struct Root {
    model_dict: ModelDict,
}

#[derive(Deserialize)]
struct ModelDict {
    model: String,
    slopes: Vec<f64>,
    intercepts: Vec<f64>,
    feature_names: Vec<String>,
    #[serde(default)]
    feature_opts_dicts: Option<Vec<Map<String, Value>>>,

    /// Some models have explicit `slope`/`intercept`; others use `slopes[0]`/`intercepts[0]`.
    #[serde(default)]
    slope: Option<f64>,
    #[serde(default)]
    intercept: Option<f64>,
    score_clip: [f64; 2],
    #[serde(default)]
    score_transform: Option<ScoreTransformJson>,
}

/// Deserialize a bool that may appear as `true`/`false` or as `"true"`/`"false"`.
fn de_bool_or_str<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    struct V;
    impl<'de> serde::de::Visitor<'de> for V {
        type Value = bool;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("bool or string \"true\"/\"false\"")
        }
        fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<bool, E> {
            Ok(v)
        }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<bool, E> {
            match v {
                "true" => Ok(true),
                "false" => Ok(false),
                other => Err(E::invalid_value(serde::de::Unexpected::Str(other), &self)),
            }
        }
    }
    d.deserialize_any(V)
}

#[derive(Deserialize)]
struct ScoreTransformJson {
    p0: Option<f64>,
    p1: Option<f64>,
    p2: Option<f64>,
    /// `[[x0,y0],[x1,y1],…]`
    knots: Option<Vec<[f64; 2]>>,
    #[serde(default, deserialize_with = "de_bool_or_str")]
    out_gte_in: bool,
    #[serde(default, deserialize_with = "de_bool_or_str")]
    out_lte_in: bool,
}

// ---------- public API ----------

fn validate_feature_name(idx: usize, name: &str) -> bool {
    // Accept both naming families found in bundled models.
    match idx {
        0 => matches!(
            name,
            "VMAF_integer_feature_adm2_score" | "VMAF_feature_adm2_score"
        ),
        1 => matches!(
            name,
            "VMAF_integer_feature_motion2_score" | "VMAF_feature_motion2_score"
        ),
        2 => matches!(
            name,
            "VMAF_integer_feature_vif_scale0_score" | "VMAF_feature_vif_scale0_score"
        ),
        3 => matches!(
            name,
            "VMAF_integer_feature_vif_scale1_score" | "VMAF_feature_vif_scale1_score"
        ),
        4 => matches!(
            name,
            "VMAF_integer_feature_vif_scale2_score" | "VMAF_feature_vif_scale2_score"
        ),
        5 => matches!(
            name,
            "VMAF_integer_feature_vif_scale3_score" | "VMAF_feature_vif_scale3_score"
        ),
        _ => false,
    }
}

fn validate_feature_names(names: &[String; 6]) -> Result<(), String> {
    for (i, name) in names.iter().enumerate() {
        if !validate_feature_name(i, name) {
            return Err(format!("unsupported feature_names[{i}] = {name}"));
        }
    }
    Ok(())
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn parse_gain_limit(v: &Value, key: &str, idx: usize) -> Result<f64, String> {
    let n = match v {
        Value::Number(num) => num.as_f64().ok_or_else(|| {
            format!("feature_opts_dicts[{idx}].{key} is not representable as f64")
        })?,
        other => {
            return Err(format!(
                "feature_opts_dicts[{idx}].{key} must be a JSON number, got {}",
                json_type_name(other)
            ));
        }
    };

    if !n.is_finite() {
        return Err(format!("feature_opts_dicts[{idx}].{key} must be finite"));
    }
    if n < 1.0 {
        return Err(format!("feature_opts_dicts[{idx}].{key} must be >= 1.0"));
    }
    Ok(n)
}

/// Parse a VMAF model from its JSON string — spec §3.3.
pub fn load_model(json: &str) -> Result<VmafModel, String> {
    let root: Root = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let d = root.model_dict;

    if d.slopes.len() < 7 {
        return Err(format!(
            "slopes must have ≥ 7 entries, got {}",
            d.slopes.len()
        ));
    }
    if d.intercepts.len() < 7 {
        return Err(format!(
            "intercepts must have ≥ 7 entries, got {}",
            d.intercepts.len()
        ));
    }

    if d.feature_names.len() != 6 {
        return Err(format!(
            "feature_names must have length 6, got {}",
            d.feature_names.len()
        ));
    }
    let feature_names: [String; 6] = d.feature_names.clone().try_into().unwrap();
    validate_feature_names(&feature_names)?;

    let mut adm_enhn_gain_limit = 100.0f64;
    let mut vif_enhn_gain_limit = 100.0f64;

    if let Some(feature_opts_dicts) = d.feature_opts_dicts.as_ref() {
        if feature_opts_dicts.len() != 6 {
            return Err(format!(
                "feature_opts_dicts length must equal feature_names length (6), got {}",
                feature_opts_dicts.len()
            ));
        }

        // Validate option value types (flat primitives only).
        for (i, dict) in feature_opts_dicts.iter().enumerate() {
            for (k, v) in dict.iter() {
                match v {
                    Value::Number(_) | Value::Bool(_) | Value::String(_) => {}
                    other => {
                        return Err(format!(
                            "feature_opts_dicts[{i}].{k} must be number/bool/string, got {}",
                            json_type_name(other)
                        ));
                    }
                }
            }
        }

        // ADM (feature index 0)
        if let Some(v) = feature_opts_dicts[0].get("adm_enhn_gain_limit") {
            adm_enhn_gain_limit = parse_gain_limit(v, "adm_enhn_gain_limit", 0)?;
        }

        // VIF scales (feature indices 2..=5)
        let mut vif_vals: [Option<f64>; 4] = [None, None, None, None];
        for s in 0..4usize {
            if let Some(v) = feature_opts_dicts[2 + s].get("vif_enhn_gain_limit") {
                vif_vals[s] = Some(parse_gain_limit(v, "vif_enhn_gain_limit", 2 + s)?);
            }
        }

        if vif_vals.iter().any(|v| v.is_some()) {
            if vif_vals.iter().any(|v| v.is_none()) {
                return Err(
                    "vif_enhn_gain_limit must be specified for all 4 VIF scale features"
                        .to_string(),
                );
            }
            let first = vif_vals[0].unwrap();
            if !vif_vals.iter().all(|v| v.unwrap() == first) {
                return Err("vif_enhn_gain_limit must match across all VIF scales".to_string());
            }
            vif_enhn_gain_limit = first;
        }
    }

    let svm = parse_libsvm(&d.model)?;

    let feature_slopes = std::array::from_fn(|i| d.slopes[i + 1]);
    let feature_intercepts = std::array::from_fn(|i| d.intercepts[i + 1]);

    let score_transform = d.score_transform.map(|st| ScoreTransform {
        p0: st.p0,
        p1: st.p1,
        p2: st.p2,
        knots: st.knots,
        out_gte_in: st.out_gte_in,
        out_lte_in: st.out_lte_in,
    });

    // `slope`/`intercept` are optional; fall back to `slopes[0]`/`intercepts[0]`.
    let score_slope = d.slope.unwrap_or(d.slopes[0]);
    let score_intercept = d.intercept.unwrap_or(d.intercepts[0]);

    Ok(VmafModel {
        svm,
        feature_names,
        feature_slopes,
        feature_intercepts,
        score_slope,
        score_intercept,
        score_clip: d.score_clip,
        score_transform,
        adm_enhn_gain_limit,
        vif_enhn_gain_limit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_JSON: &str = r#"{
        "model_dict": {
            "model_type": "LIBSVMNUSVR",
            "norm_type": "linear_rescale",
            "model": "svm_type nu_svr\nkernel_type rbf\ngamma 0.04\nnr_class 2\ntotal_sv 1\nrho 0.5\nSV\n1.0 1:0.1 2:0.2 3:0.3 4:0.4 5:0.5 6:0.6\n",
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
    }"#;

    #[test]
    fn load_model_parses_svm() {
        let m = load_model(MINIMAL_JSON).unwrap();
        assert_eq!(m.svm.gamma, 0.04);
        assert_eq!(m.svm.rho, 0.5);
        assert_eq!(m.svm.support_vectors.len(), 1);
    }

    #[test]
    fn load_model_feature_slopes() {
        let m = load_model(MINIMAL_JSON).unwrap();
        // slopes[1] from JSON
        assert!((m.feature_slopes[0] - 2.8098077503).abs() < 1e-10);
    }

    #[test]
    fn load_model_score_pipeline() {
        let m = load_model(MINIMAL_JSON).unwrap();
        assert!((m.score_slope - 0.012020766).abs() < 1e-10);
        assert!((m.score_intercept - (-0.3092981928)).abs() < 1e-10);
        assert_eq!(m.score_clip, [0.0, 100.0]);
    }

    #[test]
    fn load_model_score_transform() {
        let m = load_model(MINIMAL_JSON).unwrap();
        let st = m.score_transform.as_ref().unwrap();
        assert!((st.p0.unwrap() - 1.70674692).abs() < 1e-8);
        assert!(st.out_gte_in);
        assert!(!st.out_lte_in);
        assert_eq!(
            m.svm.support_vectors[0].values,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        );
    }

    #[test]
    fn load_model_invalid_json() {
        assert!(load_model("{bad json}").is_err());
    }

    #[test]
    fn load_model_defaults_gain_limits() {
        let m = load_model(MINIMAL_JSON).unwrap();
        assert_eq!(m.adm_enhn_gain_limit, 100.0);
        assert_eq!(m.vif_enhn_gain_limit, 100.0);
    }

    #[test]
    fn load_model_parses_neg_model_feature_opts() {
        let json = include_str!("../../../models/vmaf_v0.6.1neg.json");
        let m = load_model(json).unwrap();
        assert_eq!(m.adm_enhn_gain_limit, 1.0);
        assert_eq!(m.vif_enhn_gain_limit, 1.0);
    }

    #[test]
    fn load_model_rejects_feature_opts_length_mismatch() {
        let bad = MINIMAL_JSON.replace(
            "\"feature_names\": [",
            "\"feature_names\": [", // keep anchor
        );
        // insert feature_opts_dicts with wrong length (5)
        let bad = bad.replace(
            "\"slopes\":",
            "\"feature_opts_dicts\": [{},{},{},{},{}],\n            \"slopes\":",
        );
        assert!(load_model(&bad).is_err());
    }

    #[test]
    fn load_model_rejects_nested_option_values() {
        let bad = MINIMAL_JSON.replace(
            "\"slopes\":",
            "\"feature_opts_dicts\": [{\"adm_enhn_gain_limit\": 1.0},{},{\"vif_enhn_gain_limit\": 1.0},{\"vif_enhn_gain_limit\": 1.0},{\"vif_enhn_gain_limit\": 1.0},{\"vif_enhn_gain_limit\": {\"nested\": 1}}],\n            \"slopes\":",
        );
        assert!(load_model(&bad).is_err());
    }

    #[test]
    fn load_model_requires_vif_gain_limit_all_scales() {
        let bad = MINIMAL_JSON.replace(
            "\"slopes\":",
            "\"feature_opts_dicts\": [{},{},{\"vif_enhn_gain_limit\": 2.0},{},{},{}],\n            \"slopes\":",
        );
        assert!(load_model(&bad).is_err());
    }

    #[test]
    fn load_model_requires_vif_gain_limit_same_across_scales() {
        let bad = MINIMAL_JSON.replace(
            "\"slopes\":",
            "\"feature_opts_dicts\": [{},{},{\"vif_enhn_gain_limit\": 2.0},{\"vif_enhn_gain_limit\": 2.0},{\"vif_enhn_gain_limit\": 3.0},{\"vif_enhn_gain_limit\": 2.0}],\n            \"slopes\":",
        );
        assert!(load_model(&bad).is_err());
    }
}
