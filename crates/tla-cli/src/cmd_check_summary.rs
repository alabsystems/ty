// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty check-summary` -- summarize saved `ty check --output json` output.

use std::fs;
use std::io::{self, Read};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::cli_schema::CheckSummaryOutputFormat;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CheckJsonSummary {
    status: String,
    error_type: String,
    error_code: String,
    violated_type: String,
    violated_name: String,
    states_found: u64,
    states_distinct: u64,
    has_counterexample: u8,
}

impl CheckJsonSummary {
    fn parse_error() -> Self {
        Self {
            status: "parse_error".to_string(),
            error_type: String::new(),
            error_code: String::new(),
            violated_type: String::new(),
            violated_name: String::new(),
            states_found: 0,
            states_distinct: 0,
            has_counterexample: 0,
        }
    }
}

pub(crate) fn cmd_check_summary(input: &str, format: CheckSummaryOutputFormat) -> Result<()> {
    let summary = read_summary(input);
    match format {
        CheckSummaryOutputFormat::Json => {
            println!("{}", serde_json::to_string(&summary)?);
        }
        CheckSummaryOutputFormat::Tsv => {
            println!("{}", render_tsv(&summary));
        }
    }
    Ok(())
}

fn read_summary(input: &str) -> CheckJsonSummary {
    let text = if input == "-" {
        let mut buf = String::new();
        match io::stdin().read_to_string(&mut buf) {
            Ok(_) => buf,
            Err(_) => return CheckJsonSummary::parse_error(),
        }
    } else {
        match fs::read_to_string(input) {
            Ok(text) => text,
            Err(_) => return CheckJsonSummary::parse_error(),
        }
    };

    let value: Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(_) => return CheckJsonSummary::parse_error(),
    };
    summarize_value(&value)
}

fn summarize_value(value: &Value) -> CheckJsonSummary {
    let Some(obj) = value.as_object() else {
        return CheckJsonSummary::parse_error();
    };

    let result = obj
        .get("result")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let statistics = obj
        .get("statistics")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let violated = result
        .get("violated_property")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let states_found = statistics
        .get("states_found")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let states_distinct = statistics
        .get("states_distinct")
        .and_then(Value::as_u64)
        .unwrap_or(states_found);

    CheckJsonSummary {
        status: string_field(result.get("status")),
        error_type: string_field(result.get("error_type")),
        error_code: string_field(result.get("error_code")),
        violated_type: string_field(violated.get("type").or_else(|| violated.get("prop_type"))),
        violated_name: string_field(violated.get("name")),
        states_found,
        states_distinct,
        has_counterexample: match obj.get("counterexample") {
            Some(Value::Null) | None => 0,
            Some(_) => 1,
        },
    }
}

fn string_field(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn render_tsv(summary: &CheckJsonSummary) -> String {
    [
        sanitize_field(&summary.status),
        sanitize_field(&summary.error_type),
        sanitize_field(&summary.error_code),
        sanitize_field(&summary.violated_type),
        sanitize_field(&summary.violated_name),
        summary.states_found.to_string(),
        summary.states_distinct.to_string(),
        summary.has_counterexample.to_string(),
    ]
    .join("\t")
}

fn sanitize_field(value: &str) -> String {
    value.replace(['\t', '\n'], " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_ok_extracts_states() {
        let value = serde_json::json!({
            "result": {"status": "ok"},
            "statistics": {"states_found": 123, "states_distinct": 122},
            "counterexample": null
        });

        let summary = summarize_value(&value);
        assert_eq!(summary.status, "ok");
        assert_eq!(summary.states_found, 123);
        assert_eq!(summary.states_distinct, 122);
        assert_eq!(summary.has_counterexample, 0);
    }

    #[test]
    fn summary_invariant_violation_extracts_property() {
        let value = serde_json::json!({
            "result": {
                "status": "error",
                "error_type": "invariant_violation",
                "error_code": "TLC_INVARIANT_VIOLATED",
                "violated_property": {"type": "invariant", "name": "Inv"}
            },
            "statistics": {"states_found": 9},
            "counterexample": {"length": 2, "states": []}
        });

        let summary = summarize_value(&value);
        assert_eq!(summary.status, "error");
        assert_eq!(summary.error_type, "invariant_violation");
        assert_eq!(summary.error_code, "TLC_INVARIANT_VIOLATED");
        assert_eq!(summary.violated_type, "invariant");
        assert_eq!(summary.violated_name, "Inv");
        assert_eq!(summary.states_found, 9);
        assert_eq!(summary.states_distinct, 9);
        assert_eq!(summary.has_counterexample, 1);
    }

    #[test]
    fn summary_accepts_legacy_prop_type_field() {
        let value = serde_json::json!({
            "result": {
                "status": "error",
                "violated_property": {"prop_type": "liveness", "name": "Live"}
            },
            "statistics": {"states_found": 5}
        });

        let summary = summarize_value(&value);
        assert_eq!(summary.violated_type, "liveness");
        assert_eq!(summary.violated_name, "Live");
    }

    #[test]
    fn non_object_json_is_parse_error() {
        assert_eq!(
            summarize_value(&serde_json::json!([1, 2, 3])),
            CheckJsonSummary::parse_error()
        );
    }

    #[test]
    fn tsv_output_uses_legacy_column_order() {
        let summary = CheckJsonSummary {
            status: "error".to_string(),
            error_type: "invariant\nviolation".to_string(),
            error_code: "TLC_INVARIANT_VIOLATED".to_string(),
            violated_type: "invariant".to_string(),
            violated_name: "Inv\tOne".to_string(),
            states_found: 9,
            states_distinct: 8,
            has_counterexample: 1,
        };

        assert_eq!(
            render_tsv(&summary),
            "error\tinvariant violation\tTLC_INVARIANT_VIOLATED\tinvariant\tInv One\t9\t8\t1"
        );
    }
}
