// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

mod common;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn check_summary_cli_dispatches_tsv_and_json_formats() {
    let dir = common::TempDir::new("check-summary-cli");
    let input = dir.path.join("check-output.json");
    common::write_file(
        &input,
        br#"{
  "result": {
	    "status": "error",
	    "error_type": "invariant_violation",
	    "error_code": "TLC_INVARIANT_VIOLATED",
	    "violated_property": {"type": "invariant", "name": "Safe"}
	  },
  "statistics": {"states_found": 7, "states_distinct": 5},
  "counterexample": {"length": 2, "states": []}
}"#,
    );
    let input = input.to_str().expect("utf-8 temp path");

    let (tsv_code, tsv_stdout, tsv_stderr) = common::run_tla_parsed(&["check-summary", input]);
    assert_eq!(
        tsv_code, 0,
        "check-summary TSV failed\nstdout:\n{tsv_stdout}\nstderr:\n{tsv_stderr}"
    );
    assert_eq!(
        tsv_stdout,
        "error\tinvariant_violation\tTLC_INVARIANT_VIOLATED\tinvariant\tSafe\t7\t5\t1\n"
    );

    let (json_code, json_stdout, json_stderr) =
        common::run_tla_parsed(&["check-summary", input, "--format", "json"]);
    assert_eq!(
        json_code, 0,
        "check-summary JSON failed\nstdout:\n{json_stdout}\nstderr:\n{json_stderr}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&json_stdout).expect("check-summary JSON stdout");
    assert_eq!(json["status"], "error");
    assert_eq!(json["error_type"], "invariant_violation");
    assert_eq!(json["error_code"], "TLC_INVARIANT_VIOLATED");
    assert_eq!(json["violated_type"], "invariant");
    assert_eq!(json["violated_name"], "Safe");
    assert_eq!(json["states_found"], 7);
    assert_eq!(json["states_distinct"], 5);
    assert_eq!(json["has_counterexample"], 1);
}
