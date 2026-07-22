// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::execution::collect_examination_for_model_inner;
use super::PreparedModel;
use crate::mcc_backend_evidence::{
    add_ltl_answer_lane_summary_evidence, add_mcc_setup_evidence,
    append_runtime_reachability_bmc_reports, maybe_emit_mcc_backend_evidence,
    mcc_backend_capability_report, EvidenceScope, MccRunStatus, MccSetupEvidence,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

pub(super) fn run_examination_for_model(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    config: &crate::explorer::ExplorationConfig,
    mut setup_evidence: Option<MccSetupEvidence>,
) {
    let mut backend_report = mcc_backend_capability_report(model, examination, config);

    // MCC runs are externally time-limited. Let property pipelines flush
    // resolved formula lines between phases so BenchKit keeps partial answers
    // if a later portfolio stage exceeds the confinement boundary.
    //
    // `EvidenceScope` is the single RAII boundary for the per-run evidence
    // accumulators: it bounds both the runtime reachability BMC reports and the
    // prepared-fingerprint-admission runtime consumption rows to this scope, so
    // leftover state cannot leak across runs and no half-borrowed thread-local
    // can persist past the run. The consumption accumulator is drained by the
    // downstream `take()` in `add_mcc_setup_evidence`; the scope guarantees it
    // is reset to a clean state on drop regardless.
    let mut evidence_scope = EvidenceScope::begin();
    let hot_started = Instant::now();
    let collected = catch_unwind(AssertUnwindSafe(|| {
        collect_examination_for_model_inner(model, examination, config, true)
    }));
    let hot_duration = hot_started.elapsed();
    let runtime_bmc_reports = evidence_scope.finish_runtime_reachability_bmc_reports();
    append_runtime_reachability_bmc_reports(&mut backend_report, runtime_bmc_reports);

    match collected {
        Ok(Ok((records, diagnostics))) => {
            if let Some(setup_evidence) = setup_evidence.as_mut() {
                setup_evidence.record_hot_execution(hot_duration);
                add_mcc_setup_evidence(
                    &mut backend_report,
                    model,
                    examination,
                    setup_evidence,
                    MccRunStatus::Completed {
                        records: records.len(),
                    },
                    None,
                );
            }
            backend_report.add_evidence(format!("mcc run completed records={}", records.len()));
            add_ltl_answer_lane_summary_evidence(
                &mut backend_report,
                examination,
                config,
                &records,
            );
            maybe_emit_mcc_backend_evidence(
                model,
                examination,
                config,
                &backend_report,
                setup_evidence.as_ref(),
                MccRunStatus::Completed {
                    records: records.len(),
                },
                None,
            );
            if let Some(load_diagnostics) = model.colored_load_diagnostics() {
                load_diagnostics.emit_stderr();
            }
            diagnostics.emit_stderr();
            let mut records = records;
            crate::examination::sort_records_by_formula_id(&mut records);
            for record in &records {
                crate::output::print_mcc_line(record.to_mcc_line());
            }
        }
        Ok(Err(error)) => {
            let exam_name = examination.as_str();
            backend_report.add_evidence(format!("mcc run error: {error}"));
            let error_message = error.to_string();
            if let Some(setup_evidence) = setup_evidence.as_mut() {
                setup_evidence.record_hot_execution(hot_duration);
                add_mcc_setup_evidence(
                    &mut backend_report,
                    model,
                    examination,
                    setup_evidence,
                    MccRunStatus::Error,
                    Some(&error_message),
                );
            }
            maybe_emit_mcc_backend_evidence(
                model,
                examination,
                config,
                &backend_report,
                setup_evidence.as_ref(),
                MccRunStatus::Error,
                Some(&error_message),
            );
            eprintln!("Warning: failed to parse {exam_name}.xml: {error}");
            print_cannot_compute_for_exam(model, examination);
        }
        Err(_) => {
            backend_report.add_evidence("mcc run panic".to_string());
            if let Some(setup_evidence) = setup_evidence.as_mut() {
                setup_evidence.record_hot_execution(hot_duration);
                add_mcc_setup_evidence(
                    &mut backend_report,
                    model,
                    examination,
                    setup_evidence,
                    MccRunStatus::Panic,
                    Some("internal panic"),
                );
            }
            maybe_emit_mcc_backend_evidence(
                model,
                examination,
                config,
                &backend_report,
                setup_evidence.as_ref(),
                MccRunStatus::Panic,
                Some("internal panic"),
            );
            eprintln!(
                "{}: CANNOT_COMPUTE after internal panic",
                examination.as_str()
            );
            print_cannot_compute_for_exam(model, examination);
        }
    }
}

fn print_cannot_compute_for_exam(
    model: &PreparedModel,
    examination: crate::examination::Examination,
) {
    if let Ok(xml_name) = examination.property_xml_name() {
        if let Ok(properties) = crate::property_xml::parse_properties(model.model_dir(), xml_name) {
            for property in properties {
                crate::output::print_mcc_line(crate::output::formula_line(
                    model.model_name(),
                    &property.id,
                    crate::output::Verdict::CannotCompute,
                ));
            }
            return;
        }
        if let Ok(ids) = crate::property_xml::parse_property_ids(model.model_dir(), xml_name) {
            for id in ids {
                crate::output::print_mcc_line(crate::output::formula_line(
                    model.model_name(),
                    &id,
                    crate::output::Verdict::CannotCompute,
                ));
            }
            return;
        }
    }

    crate::output::print_mcc_line(crate::output::cannot_compute_line(
        model.model_name(),
        examination.as_str(),
    ));
}
