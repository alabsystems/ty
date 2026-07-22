// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::Path;

use crate::error::PnmlError;

use super::diagnostics::ColoredLoadDiagnostics;
use super::{PreparedModel, PropertyAliases, SourceNetKind};

pub(super) fn load_model_dir(path: impl AsRef<Path>) -> Result<PreparedModel, PnmlError> {
    let dir = path.as_ref();
    let model_name = model_name_from_dir(dir);

    match crate::parser::parse_pnml_dir(dir) {
        Ok(net) => {
            let nupn = match crate::nupn::parse_nupn_file(&dir.join("model.pnml"), &net) {
                Ok(nupn) => nupn,
                Err(PnmlError::InvalidNupn { reason }) => {
                    eprintln!("Warning: ignoring invalid NUPN annotation: {reason}");
                    None
                }
                Err(error) => return Err(error),
            };
            Ok(build_pt_model(dir, model_name, net, nupn))
        }
        Err(PnmlError::UnsupportedNetType { .. }) => build_colored_model(dir, model_name),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
pub(crate) fn load_model_dir_no_colored_reduce(
    path: impl AsRef<Path>,
) -> Result<PreparedModel, PnmlError> {
    let dir = path.as_ref();
    let model_name = model_name_from_dir(dir);
    let colored = crate::hlpnml::parse_hlpnml_dir(dir)?;
    let colored_snapshot = colored.clone();
    let unfolded = crate::unfold::unfold_to_pt(&colored)?;
    Ok(PreparedModel::new(
        model_name,
        dir.to_path_buf(),
        SourceNetKind::SymmetricNet,
        unfolded.net,
        None,
        unfolded.aliases,
        Some(colored_snapshot),
        None,
    ))
}

fn model_name_from_dir(dir: &Path) -> String {
    dir.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn build_pt_model(
    dir: &Path,
    model_name: String,
    net: crate::petri_net::PetriNet,
    nupn: Option<crate::nupn::NupnStructure>,
) -> PreparedModel {
    PreparedModel::new(
        model_name,
        dir.to_path_buf(),
        SourceNetKind::Pt,
        net.clone(),
        nupn,
        PropertyAliases::identity(&net),
        None,
        None,
    )
}

fn build_colored_model(dir: &Path, model_name: String) -> Result<PreparedModel, PnmlError> {
    let mut colored = crate::hlpnml::parse_hlpnml_dir(dir)?;
    let uncollapsed_colored_source = colored.clone();
    let col_report = crate::colored_reduce::reduce_colored(&mut colored);
    let dead_report = crate::colored_dead_transitions::reduce(&mut colored);
    let colored_load_diagnostics = ColoredLoadDiagnostics::new(
        col_report.collapsed_places.len(),
        col_report.places_saved(),
        dead_report.transitions_removed,
    );

    // Reserve a fraction of the MCC wall-clock budget for load-time
    // unfolding so a near-cap colored model cannot starve the examination
    // phase (or get OS-killed mid-unfold). On expiry the unfold aborts with
    // the recoverable `ColoredUnfoldUnavailable`, handled below.
    let budget = crate::unfold::UnfoldBudget::new(colored_unfold_deadline());

    match crate::unfold::unfold_to_pt_with_budget(&colored, &budget) {
        Ok(unfolded) => Ok(PreparedModel::new(
            model_name,
            dir.to_path_buf(),
            SourceNetKind::SymmetricNet,
            unfolded.net,
            None,
            unfolded.aliases,
            Some(uncollapsed_colored_source),
            Some(colored_load_diagnostics),
        )),
        // RECOVERABLE: the net is too large / unfolding ran out of budget,
        // but the colored source is sound. Build a placeholder model so the
        // examination dispatcher can still run colored-source shortcuts
        // (e.g. OneSafe) and emit per-examination CANNOT_COMPUTE for the
        // rest, instead of collapsing the whole model to one CC.
        Err(PnmlError::ColoredUnfoldUnavailable { reason }) => {
            eprintln!(
                "colored unfolding over budget ({reason}); keeping colored source for \
                 structural shortcuts, net-dependent examinations -> CANNOT_COMPUTE"
            );
            let placeholder = crate::petri_net::PetriNet {
                name: colored.name.clone(),
                places: Vec::new(),
                transitions: Vec::new(),
                initial_marking: Vec::new(),
            };
            // `PropertyAliases` has no `Default`; `identity` on the empty
            // placeholder net yields an empty alias table (spec note 3).
            // These aliases are never consulted for a verdict — the
            // placeholder net is guarded out in `execution.rs`.
            let aliases = PropertyAliases::identity(&placeholder);
            Ok(PreparedModel::new(
                model_name,
                dir.to_path_buf(),
                SourceNetKind::SymmetricNet,
                placeholder,
                None,
                aliases,
                Some(uncollapsed_colored_source),
                Some(colored_load_diagnostics),
            )
            .with_colored_unfold_unavailable())
        }
        // UNRECOVERABLE: genuinely unsupported colored construct (or any
        // other error). Preserve existing behavior — propagate so the CLI
        // emits the whole-model CANNOT_COMPUTE it already does today.
        Err(error) => Err(error),
    }
}

/// Compute a load-time unfolding deadline from the MCC time budget.
///
/// Returns `None` (unbounded) when `BK_TIME_CONFINEMENT` is unset, so local
/// non-MCC runs and tests are unaffected. When set, reserves the first ~40%
/// of the budget for unfolding; the examination phase computes its own
/// deadline from the same env var afterwards.
fn colored_unfold_deadline() -> Option<std::time::Instant> {
    let secs: u64 = std::env::var("BK_TIME_CONFINEMENT").ok()?.parse().ok()?;
    let unfold_secs = (secs * 2) / 5; // 40% of the wall-clock budget
    Some(std::time::Instant::now() + std::time::Duration::from_secs(unfold_secs))
}
