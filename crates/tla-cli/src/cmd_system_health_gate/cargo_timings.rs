// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use serde_json::Value;

use super::{HealthCheck, CARGO_TIMINGS_SELF_TEST_CHECK_NAME};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CargoTimingsParseResult {
    pub(super) timings: Vec<(String, f64)>,
    pub(super) skipped_non_json_lines: usize,
}

pub(super) fn check_parse_cargo_timings_ndjson_self_test() -> HealthCheck {
    match parse_cargo_timings_ndjson_native_self_test() {
        Ok(detail) => HealthCheck::ok(CARGO_TIMINGS_SELF_TEST_CHECK_NAME, Some(detail.to_string())),
        Err(detail) => HealthCheck::err(CARGO_TIMINGS_SELF_TEST_CHECK_NAME, Some(detail)),
    }
}

fn parse_cargo_timings_ndjson_native_self_test() -> std::result::Result<&'static str, String> {
    let empty = parse_cargo_timings_ndjson_text("");
    if empty.timings != Vec::<(String, f64)>::new() {
        return Err(format!("empty timings={:?}, want []", empty.timings));
    }
    if empty.skipped_non_json_lines != 0 {
        return Err(format!(
            "empty skipped_non_json_lines={}, want 0",
            empty.skipped_non_json_lines
        ));
    }

    let mixed = parse_cargo_timings_ndjson_text(
        r#"
Compiling foo v0.1.0
{"reason":"compiler-artifact","duration":1.0}
{"reason":"timing-info","package_id":"p#0.1.0","duration":0.5}
{not json}
{"reason":"timing-info","package_id":"q#0.1.0","duration":"1.25"}
"#,
    );
    let expected = vec![("q#0.1.0".to_string(), 1.25), ("p#0.1.0".to_string(), 0.5)];
    if mixed.timings != expected {
        return Err(format!(
            "mixed timings={:?}, want {:?}",
            mixed.timings, expected
        ));
    }
    if mixed.skipped_non_json_lines != 2 {
        return Err(format!(
            "mixed skipped_non_json_lines={}, want 2",
            mixed.skipped_non_json_lines
        ));
    }

    Ok("self-test: ok")
}

pub(super) fn parse_cargo_timings_ndjson_text(input: &str) -> CargoTimingsParseResult {
    let mut timings = Vec::new();
    let mut skipped_non_json_lines = 0;

    for line in input.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let Ok(obj) = serde_json::from_str::<Value>(line) else {
            skipped_non_json_lines += 1;
            continue;
        };

        if obj.get("reason").and_then(Value::as_str) != Some("timing-info") {
            continue;
        }

        let package_id = obj
            .get("package_id")
            .map(cargo_timing_value_to_text)
            .unwrap_or_else(|| "unknown".to_string());
        let package = package_id
            .split_whitespace()
            .next()
            .filter(|part| !part.is_empty())
            .unwrap_or("unknown")
            .to_string();
        let duration = obj.get("duration").map_or(0.0, cargo_timing_duration);

        timings.push((package, duration));
    }

    timings.sort_by(|(_, left), (_, right)| {
        right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
    });

    CargoTimingsParseResult {
        timings,
        skipped_non_json_lines,
    }
}

fn cargo_timing_value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn cargo_timing_duration(value: &Value) -> f64 {
    match value {
        Value::Number(number) => number.as_f64().unwrap_or(0.0),
        Value::String(text) => text.parse::<f64>().unwrap_or(0.0),
        Value::Bool(true) => 1.0,
        Value::Bool(false) | Value::Null | Value::Array(_) | Value::Object(_) => 0.0,
    }
}
