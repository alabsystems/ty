// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! MCC examination dispatch.
//!
//! Maps examination names to exploration observers and formats output.

use std::path::Path;

use tla_bignum::{BigUint, ToPrimitive};

use crate::error::PnmlError;
use crate::explorer::ExplorationConfig;
use crate::model::PropertyAliases;
use crate::petri_net::PetriNet;

#[path = "examination_kind.rs"]
mod examination_kind;
#[path = "examination_non_property.rs"]
mod examination_non_property;
#[path = "examination_plan.rs"]
mod examination_plan;
#[path = "examination_techniques.rs"]
mod examination_techniques;

pub use self::examination_kind::Examination;
pub(crate) use self::examination_techniques::{
    ay_runtime_available, note_aiger_resolved_deadlock, techniques_for_examination,
};
pub use crate::output::{Technique, Techniques, Verdict};

#[cfg(test)]
pub(crate) use self::examination_non_property::{
    deadlock_verdict, liveness_verdict, liveness_verdict_with_groups, one_safe_verdict,
    quasi_liveness_verdict, quasi_liveness_verdict_with_groups, stable_marking_verdict,
    state_space_stats,
};

/// Test-only hook re-export (`doc(hidden)`, not advertised API): the Tier-1
/// StateSpace recognizer crosscheck entry point, reachable as
/// `tla_petri::examination::tier1_crosscheck_hook` from the crate's own
/// integration tests (`tests/tier1_crosscheck_bfs.rs`). Soundness-neutral. See
/// [`examination_non_property::tier1_crosscheck_hook`].
#[doc(hidden)]
pub use self::examination_non_property::tier1_crosscheck_hook;

// ---------------------------------------------------------------------------
// Typed result model
// ---------------------------------------------------------------------------

/// The value produced by one MCC examination formula or metric.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExaminationValue {
    /// Boolean verdict (TRUE / FALSE / CANNOT_COMPUTE).
    Verdict(Verdict),
    /// Numeric upper bound (or `None` for CANNOT_COMPUTE).
    OptionalBound(Option<u64>),
    /// State-space statistics (or `None` for CANNOT_COMPUTE).
    StateSpace(Option<StateSpaceReport>),
}

/// State-space statistics reported by the StateSpace examination.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StateSpaceReport {
    /// Total unique reachable markings, narrowed to `usize`. This is the
    /// back-compat field existing consumers read; it is exact whenever `|R|`
    /// fits `usize` (every explicit-BFS / BDD result). When the count exceeds
    /// `usize` this is the saturated `usize::MAX` coarse marker and the EXACT
    /// value is carried in [`Self::states_big`]; the MCC `STATE_SPACE STATES`
    /// line is always emitted from [`Self::states_exact`] (the bignum), so no
    /// precision is ever lost on the wire.
    pub states: usize,
    /// EXACT reachable-marking count as an arbitrary-precision [`tla_bignum::BigUint`].
    /// The authoritative count: carries counts BEYOND `u128` (e.g. FMS ≈1e47,
    /// Kanban/Philosophers ≈1e238) so a structurally-computable count is
    /// REPORTED at full precision rather than declining on a fixed-width cap.
    /// The MCC `STATE_SPACE STATES` row is emitted from this value.
    pub states_big: BigUint,
    /// Total transition firings explored, narrowed to `u128` (saturated marker
    /// `u128::MAX` when it does not fit; [`Self::edges_big`] is the source of
    /// truth). Retained for back-compat.
    pub edges: u128,
    /// Total transition firings explored, EXACT as an arbitrary-precision
    /// [`tla_bignum::BigUint`]. The `STATE_SPACE TRANSITIONS` row is emitted
    /// from this value at full precision.
    pub edges_big: BigUint,
    /// Maximum tokens in any single place.
    pub max_token_in_place: u64,
    /// Maximum sum of tokens across all places.
    pub max_token_sum: u64,
}

impl StateSpaceReport {
    /// Create a new state-space report with a `usize`-representable state count
    /// and a `u64` edge count (the explicit / BDD shape). The symbolic /
    /// structural lanes construct the struct directly (via [`Self::from_big`])
    /// with the wide fields.
    #[must_use]
    pub fn new(states: usize, edges: u64, max_token_in_place: u64, max_token_sum: u64) -> Self {
        Self {
            states,
            states_big: BigUint::from(states),
            edges: edges as u128,
            edges_big: BigUint::from(edges),
            max_token_in_place,
            max_token_sum,
        }
    }

    /// Create a report from the authoritative arbitrary-precision counts. The
    /// narrowed `usize` / `u128` back-compat fields are filled by narrowing
    /// fail-closed: exact when the value fits, otherwise the saturated marker
    /// (`usize::MAX` / `u128::MAX`). The bignum fields are the source of truth.
    #[must_use]
    pub fn from_big(
        states_big: BigUint,
        edges_big: BigUint,
        max_token_in_place: u64,
        max_token_sum: u64,
    ) -> Self {
        let states = states_big.to_usize().unwrap_or(usize::MAX);
        let edges = states_big_to_u128_marker(&edges_big);
        Self {
            states,
            states_big,
            edges,
            edges_big,
            max_token_in_place,
            max_token_sum,
        }
    }

    /// The exact reachable-marking count as a [`tla_bignum::BigUint`] — the
    /// single source of truth for the emitted `STATE_SPACE STATES` row.
    #[must_use]
    pub fn states_exact(&self) -> &BigUint {
        &self.states_big
    }

    /// The exact transition-firing count as a [`tla_bignum::BigUint`] — the
    /// single source of truth for the emitted `STATE_SPACE TRANSITIONS` row.
    #[must_use]
    pub fn edges_exact(&self) -> &BigUint {
        &self.edges_big
    }
}

/// Narrow a [`BigUint`] to `u128`, saturating to `u128::MAX` when it does not
/// fit (the back-compat field's coarse marker; the bignum field is exact).
#[must_use]
fn states_big_to_u128_marker(v: &BigUint) -> u128 {
    v.to_u128().unwrap_or(u128::MAX)
}

/// One result record from an MCC examination.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExaminationRecord {
    /// MCC formula identifier.
    ///
    /// GlobalProperties use the examination name (for example
    /// `"ReachabilityDeadlock"`). Property examinations use the formula ID
    /// read from the MCC XML.
    pub formula_id: String,
    /// The examination value.
    pub value: ExaminationValue,
    /// Which techniques were used to produce this result.
    pub techniques: Techniques,
}

impl ExaminationRecord {
    /// Create a new examination record with default (EXPLICIT) techniques.
    #[must_use]
    pub fn new(formula_id: String, value: ExaminationValue) -> Self {
        Self {
            formula_id,
            value,
            techniques: Techniques::default(),
        }
    }

    /// Create a new examination record with specific techniques.
    #[must_use]
    pub fn with_techniques(
        formula_id: String,
        value: ExaminationValue,
        techniques: Techniques,
    ) -> Self {
        Self {
            formula_id,
            value,
            techniques,
        }
    }

    /// Render this record as MCC output line(s).
    ///
    /// For most examinations this is a single `FORMULA` line. For `StateSpace`,
    /// the MCC specification requires four `STATE_SPACE` lines, one per
    /// metric. All keywords route through [`crate::mcc_keywords`].
    #[must_use]
    pub fn to_mcc_line(&self) -> String {
        use crate::mcc_keywords::{FORMULA, TECHNIQUES};
        use crate::output::{
            formula_cannot_compute_line_with, state_space_cannot_compute_line,
            state_space_metric_line, StateSpaceMetric,
        };
        match &self.value {
            ExaminationValue::Verdict(v) => {
                format!(
                    "{FORMULA} {id} {v} {TECHNIQUES} {tags}",
                    id = self.formula_id,
                    tags = self.techniques.as_mcc_str()
                )
            }
            ExaminationValue::OptionalBound(Some(b)) => {
                format!(
                    "{FORMULA} {id} {b} {TECHNIQUES} {tags}",
                    id = self.formula_id,
                    tags = self.techniques.as_mcc_str()
                )
            }
            ExaminationValue::OptionalBound(None) => {
                formula_cannot_compute_line_with(&self.formula_id, &self.techniques)
            }
            ExaminationValue::StateSpace(Some(ss)) => [
                state_space_metric_line(
                    StateSpaceMetric::States,
                    // EXACT arbitrary-precision count, so a structurally-
                    // computable `|R|` BEYOND `u128` is emitted at full decimal
                    // precision rather than truncated/declined.
                    ss.states_exact(),
                    &self.techniques,
                ),
                state_space_metric_line(
                    StateSpaceMetric::Transitions,
                    // EXACT bignum edge count (widened with |R|).
                    ss.edges_exact(),
                    &self.techniques,
                ),
                state_space_metric_line(
                    StateSpaceMetric::MaxTokenInPlace,
                    ss.max_token_in_place,
                    &self.techniques,
                ),
                state_space_metric_line(
                    StateSpaceMetric::MaxTokenPerMarking,
                    ss.max_token_sum,
                    &self.techniques,
                ),
            ]
            .join("\n"),
            ExaminationValue::StateSpace(None) => state_space_cannot_compute_line(&self.techniques),
        }
    }
}

/// Sort examination records into MCC formula-declaration order.
///
/// Property examinations resolve formulas concurrently across solver lanes, so
/// records arrive in completion order rather than declaration order. The MCC
/// organizer matches each `FORMULA` line by its id, so out-of-order emission is
/// not a scoring error — but emitting in declaration order is more robust and
/// keeps local CSV comparators (which align partial output positionally) from
/// mislabeling a correct-but-reordered result as a corpus mismatch.
///
/// Sort key is `(prefix, trailing-integer)` where the trailing integer is the
/// numeric suffix of the formula id (e.g. `...-2025-07` → `7`). Ids without a
/// numeric suffix (GlobalProperties / StateSpace, which use the examination
/// name) sort by the whole string with a sentinel suffix, so single-record
/// examinations are unaffected.
pub(crate) fn sort_records_by_formula_id(records: &mut [ExaminationRecord]) {
    fn split_key(id: &str) -> (&str, i64) {
        let trailing_digits = id
            .as_bytes()
            .iter()
            .rev()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if trailing_digits == 0 {
            return (id, -1);
        }
        let split = id.len() - trailing_digits;
        let num = id[split..].parse::<i64>().unwrap_or(-1);
        (&id[..split], num)
    }

    records.sort_by(|a, b| {
        let (ap, an) = split_key(&a.formula_id);
        let (bp, bn) = split_key(&b.formula_id);
        ap.cmp(bp).then(an.cmp(&bn))
    });
}

// ---------------------------------------------------------------------------
// Public collector API
// ---------------------------------------------------------------------------

/// Collect examination results as typed records without printing.
///
/// Returns structured [`ExaminationRecord`] values for all 13 MCC examination
/// kinds. Property-based examinations (UpperBounds, Reachability*, CTL*, LTL*)
/// parse their XML from `model_dir`. Non-property examinations ignore
/// `model_dir`.
///
/// Returns `Err(PnmlError)` only for real API/load failures (missing XML, XML
/// parse error, IO error). Computational incompleteness is represented as
/// `CANNOT_COMPUTE` inside the records.
pub fn collect_examination_with_dir(
    net: &PetriNet,
    model_name: &str,
    model_dir: &Path,
    examination: Examination,
    config: &ExplorationConfig,
) -> Result<Vec<ExaminationRecord>, PnmlError> {
    let aliases = PropertyAliases::identity(net);
    collect_examination_core(
        net,
        model_name,
        model_dir,
        &aliases,
        examination,
        config,
        false,
    )
}

pub(crate) fn collect_examination_core(
    net: &PetriNet,
    _model_name: &str,
    model_dir: &Path,
    aliases: &PropertyAliases,
    examination: Examination,
    config: &ExplorationConfig,
    flush: bool,
) -> Result<Vec<ExaminationRecord>, PnmlError> {
    collect_examination_core_with_nupn(
        net,
        _model_name,
        model_dir,
        aliases,
        examination,
        config,
        flush,
        None,
    )
}

pub(crate) fn collect_examination_core_with_nupn(
    net: &PetriNet,
    _model_name: &str,
    model_dir: &Path,
    aliases: &PropertyAliases,
    examination: Examination,
    config: &ExplorationConfig,
    flush: bool,
    nupn: Option<&crate::nupn::NupnStructure>,
) -> Result<Vec<ExaminationRecord>, PnmlError> {
    // Compute the technique set this examination's pipeline exercises. The
    // attribution is keyed on which engines the dispatcher wires below, with
    // runtime signals (ay availability, NUPN presence) narrowing the set so
    // we do not claim a technique we cannot actually run on this invocation.
    let techniques = techniques_for_examination(examination, ay_runtime_available(), nupn);

    match examination {
        // -- Non-property examinations (single record each) --
        Examination::ReachabilityDeadlock => {
            let v = examination_non_property::deadlock_verdict(net, config);
            Ok(vec![ExaminationRecord::with_techniques(
                examination.as_str().to_string(),
                ExaminationValue::Verdict(v),
                techniques,
            )])
        }
        Examination::OneSafe => {
            let colored_groups = aliases.colored_place_groups();
            let v = examination_non_property::one_safe_verdict_with_nupn(
                net,
                config,
                &colored_groups,
                nupn,
            );
            Ok(vec![ExaminationRecord::with_techniques(
                examination.as_str().to_string(),
                ExaminationValue::Verdict(v),
                techniques,
            )])
        }
        Examination::QuasiLiveness => {
            let colored_transition_groups = aliases.colored_transition_groups_as_usize();
            let v = examination_non_property::quasi_liveness_verdict_with_groups(
                net,
                config,
                &colored_transition_groups,
            );
            Ok(vec![ExaminationRecord::with_techniques(
                examination.as_str().to_string(),
                ExaminationValue::Verdict(v),
                techniques,
            )])
        }
        Examination::StableMarking => {
            let colored_groups = aliases.colored_place_groups();
            let v = examination_non_property::stable_marking_verdict(net, config, &colored_groups);
            Ok(vec![ExaminationRecord::with_techniques(
                examination.as_str().to_string(),
                ExaminationValue::Verdict(v),
                techniques,
            )])
        }
        Examination::Liveness => {
            let colored_transition_groups = aliases.colored_transition_groups_as_usize();
            let v = examination_non_property::liveness_verdict_with_groups(
                net,
                config,
                &colored_transition_groups,
            );
            Ok(vec![ExaminationRecord::with_techniques(
                examination.as_str().to_string(),
                ExaminationValue::Verdict(v),
                techniques,
            )])
        }
        Examination::StateSpace => {
            // `nupn` seeds the DD lane's variable order (performance-only;
            // all four metrics are NUPN-invariant by construction).
            let stats = examination_non_property::state_space_stats_with_nupn(net, config, nupn);
            Ok(vec![ExaminationRecord::with_techniques(
                examination.as_str().to_string(),
                ExaminationValue::StateSpace(stats.map(|s| {
                    // `s.states` / `s.edges` are EXACT `BigUint` (widened for the
                    // structural / symbolic lanes). `from_big` fills the narrowed
                    // back-compat fields fail-closed (saturated marker when the
                    // count exceeds them) while the bignum fields — emitted on the
                    // wire — carry the exact value at any magnitude.
                    StateSpaceReport::from_big(
                        s.states,
                        s.edges,
                        s.max_token_in_place,
                        s.max_token_sum,
                    )
                })),
                techniques,
            )])
        }

        // -- Property examinations (one record per property) --
        Examination::UpperBounds => {
            let properties = crate::property_xml::parse_properties(model_dir, "UpperBounds")?;
            // `nupn` seeds the DD fast-path variable order (performance-
            // only; bounds are NUPN-invariant by construction).
            let results =
                crate::examinations::upper_bounds::check_upper_bounds_properties_with_aliases_and_nupn(
                    net,
                    &properties,
                    aliases,
                    config,
                    nupn,
                );
            Ok(results
                .into_iter()
                .map(|(id, bound)| {
                    ExaminationRecord::with_techniques(
                        id,
                        ExaminationValue::OptionalBound(bound),
                        techniques.clone(),
                    )
                })
                .collect())
        }
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            let exam_name = examination.as_str();
            let properties = crate::property_xml::parse_properties(model_dir, exam_name)?;
            let results = if flush {
                crate::examinations::reachability::check_reachability_properties_with_flush_and_nupn(
                    net,
                    &properties,
                    aliases,
                    config,
                    nupn,
                )
            } else {
                crate::examinations::reachability::check_reachability_properties_with_aliases_and_nupn(
                    net,
                    &properties,
                    aliases,
                    config,
                    nupn,
                )
            };
            Ok(results
                .into_iter()
                .map(|(id, verdict)| {
                    ExaminationRecord::with_techniques(
                        id,
                        ExaminationValue::Verdict(verdict),
                        techniques.clone(),
                    )
                })
                .collect())
        }
        Examination::CTLCardinality | Examination::CTLFireability => {
            let exam_name = examination.as_str();
            let properties = crate::property_xml::parse_properties(model_dir, exam_name)?;
            let results = if flush {
                crate::examinations::ctl::check_ctl_properties_with_flush_and_techniques(
                    net,
                    &properties,
                    aliases,
                    config,
                    &techniques,
                )
            } else {
                crate::examinations::ctl::check_ctl_properties_with_aliases(
                    net,
                    &properties,
                    aliases,
                    config,
                )
            };
            Ok(results
                .into_iter()
                .map(|(id, verdict)| {
                    ExaminationRecord::with_techniques(
                        id,
                        ExaminationValue::Verdict(verdict),
                        techniques.clone(),
                    )
                })
                .collect())
        }
        Examination::LTLCardinality | Examination::LTLFireability => {
            let exam_name = examination.as_str();
            let properties = crate::property_xml::parse_properties(model_dir, exam_name)?;
            let results = if flush {
                crate::examinations::ltl::check_ltl_properties_with_flush_and_techniques(
                    net,
                    &properties,
                    aliases,
                    config,
                    &techniques,
                )
            } else {
                crate::examinations::ltl::check_ltl_properties_with_aliases(
                    net,
                    &properties,
                    aliases,
                    config,
                )
            };
            Ok(results
                .into_iter()
                .map(|(id, verdict)| {
                    ExaminationRecord::with_techniques(
                        id,
                        ExaminationValue::Verdict(verdict),
                        techniques.clone(),
                    )
                })
                .collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Compatibility helpers (existing public API)
// ---------------------------------------------------------------------------

/// Check UpperBounds properties and return `(property_id, optional_bound)` pairs.
///
/// Returns `Err` if the property XML cannot be parsed, or `Ok(results)` with
/// one entry per property. Each entry is `(id, Some(bound))` for resolved
/// properties or `(id, None)` for unresolved ones.
pub fn check_upper_bounds(
    net: &PetriNet,
    model_dir: &Path,
    config: &ExplorationConfig,
) -> Result<Vec<(String, Option<u64>)>, PnmlError> {
    let records =
        collect_examination_with_dir(net, "", model_dir, Examination::UpperBounds, config)?;
    Ok(records
        .into_iter()
        .map(|r| {
            let bound = match r.value {
                ExaminationValue::OptionalBound(b) => b,
                _ => None,
            };
            (r.formula_id, bound)
        })
        .collect())
}

/// Check reachability properties and return `(property_id, verdict_string)` pairs.
///
/// The `examination` parameter selects which reachability variant to check
/// (e.g., `ReachabilityCardinality` or `ReachabilityFireability`).
/// Verdict strings are `"TRUE"`, `"FALSE"`, or `"CANNOT_COMPUTE"`.
pub fn check_reachability(
    net: &PetriNet,
    model_dir: &Path,
    examination: Examination,
    config: &ExplorationConfig,
) -> Result<Vec<(String, String)>, PnmlError> {
    let records = collect_examination_with_dir(net, "", model_dir, examination, config)?;
    Ok(records
        .into_iter()
        .map(|r| {
            let verdict_str = match r.value {
                ExaminationValue::Verdict(v) => v.to_string(),
                _ => crate::mcc_keywords::CANNOT_COMPUTE.to_string(),
            };
            (r.formula_id, verdict_str)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Print wrappers (existing public API)
// ---------------------------------------------------------------------------

/// Run an examination on a Petri net and print MCC-format output to stdout.
///
/// For property-based examinations (UpperBounds), use
/// [`run_examination_with_dir`] instead. This function prints
/// `CANNOT_COMPUTE` for examinations that require model-directory property XML.
pub fn run_examination(
    net: &PetriNet,
    model_name: &str,
    examination: Examination,
    config: &ExplorationConfig,
) {
    if examination.needs_property_xml() {
        let exam_name = examination.as_str();
        eprintln!("{exam_name} requires model directory; use --examination with model dir");
        crate::output::print_mcc_line(crate::output::cannot_compute_line(model_name, exam_name));
        return;
    }
    // Diagnostic (opt-in via TY_MCC_DECOMP_REPORT): log the net's structural
    // decomposition — loosely-coupled components + the technique each would route
    // to. LOG-ONLY, never affects the verdict; it gives the built-but-unwired
    // decomposition analysis a sound production caller so a net's decomposability
    // is observable (the signal the assume-guarantee compose work needs).
    if std::env::var_os("TY_MCC_DECOMP_REPORT").is_some() {
        eprintln!("{}", crate::decomposition::decomposition_report(net, 2));
    }
    // Non-property examinations: collect and print.
    // `model_dir` is unused for non-property examinations — use a dummy path.
    let dummy = Path::new(".");
    match collect_examination_with_dir(net, model_name, dummy, examination, config) {
        Ok(mut records) => {
            sort_records_by_formula_id(&mut records);
            for record in &records {
                crate::output::print_mcc_line(record.to_mcc_line());
            }
        }
        Err(error) => {
            eprintln!(
                "{}: {} ({error})",
                examination.as_str(),
                crate::mcc_keywords::CANNOT_COMPUTE
            );
            crate::output::print_mcc_line(crate::output::cannot_compute_line(
                model_name,
                examination.as_str(),
            ));
        }
    }
}

/// Run an examination with access to the model directory.
///
/// Required for property-based examinations that read `<Examination>.xml`.
pub fn run_examination_with_dir(
    net: &PetriNet,
    model_name: &str,
    model_dir: &Path,
    examination: Examination,
    config: &ExplorationConfig,
) {
    let aliases = PropertyAliases::identity(net);
    match collect_examination_core(
        net,
        model_name,
        model_dir,
        &aliases,
        examination,
        config,
        true,
    ) {
        Ok(mut records) => {
            sort_records_by_formula_id(&mut records);
            for record in &records {
                crate::output::print_mcc_line(record.to_mcc_line());
            }
        }
        Err(error) => {
            let exam_name = examination.as_str();
            eprintln!("Warning: failed to parse {exam_name}.xml: {error}");
            crate::output::print_mcc_line(crate::output::cannot_compute_line(
                model_name, exam_name,
            ));
        }
    }
}

#[cfg(test)]
#[path = "examination_tests.rs"]
mod tests;
