// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::error::PnmlError;
use crate::petri_net::PetriNet;
use crate::simplification_report::SimplificationReport;

use super::diagnostics::{ColoredExecutionDiagnostics, ColoredPropertyDiagnostic};
use super::{PreparedModel, PropertyAliases, SourceNetKind};

/// Build the technique-attribution for an examination on a prepared model.
///
/// Mirrors the dispatcher logic in [`crate::examination::collect_examination_core_with_nupn`]
/// so colored-relevance and StateSpace fail-closed records report the same
/// technique vocabulary as the regular pipeline.
fn techniques_for_model(
    model: &PreparedModel,
    examination: crate::examination::Examination,
) -> crate::output::Techniques {
    crate::examination::techniques_for_examination(
        examination,
        crate::examination::ay_runtime_available(),
        model.nupn(),
    )
}

pub(super) fn collect_examination_for_model(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    config: &crate::explorer::ExplorationConfig,
) -> Result<Vec<crate::examination::ExaminationRecord>, PnmlError> {
    let (records, _) = collect_examination_for_model_inner(model, examination, config, false)?;
    Ok(records)
}

pub(super) fn collect_examination_for_model_inner(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    config: &crate::explorer::ExplorationConfig,
    flush: bool,
) -> Result<
    (
        Vec<crate::examination::ExaminationRecord>,
        ColoredExecutionDiagnostics,
    ),
    PnmlError,
> {
    if model.source_kind() == SourceNetKind::SymmetricNet
        && matches!(examination, crate::examination::Examination::StateSpace)
    {
        if model.colored_source.is_some() {
            return collect_on_uncollapsed_colored_source(model, examination, config, flush);
        }
        return Ok((
            uncollapsed_colored_cannot_compute_records(
                model,
                examination,
                techniques_for_model(model, examination),
            )?,
            ColoredExecutionDiagnostics::default(),
        ));
    }

    if model.source_kind() == SourceNetKind::SymmetricNet
        && matches!(examination, crate::examination::Examination::UpperBounds)
        && model.colored_source.is_some()
    {
        return collect_with_colored_relevance(model, examination, config);
    }

    // Colored OneSafe shortcut: when the source HLPN structurally forces
    // 1-safety (initial markings ≤ 1 per color AND every transition's
    // outputs are dominated by its inputs), we can answer TRUE without
    // unfolding. This unlocks colored families whose unfolded P/T net
    // would exceed the place cap, AND gives an O(arcs) answer for many
    // small colored models. On `None` we fall through unchanged — the
    // unfolding-based OneSafe path remains the source of truth and is the
    // only path that can issue `Verdict::False`.
    if model.source_kind() == SourceNetKind::SymmetricNet
        && matches!(examination, crate::examination::Examination::OneSafe)
    {
        if let Some(colored) = model.colored_source.as_ref() {
            if let Some(verdict) = super::colored_shortcuts::try_one_safe_colored_shortcut(colored)
            {
                return Ok((
                    vec![crate::examination::ExaminationRecord::with_techniques(
                        examination.as_str().to_string(),
                        crate::examination::ExaminationValue::Verdict(verdict),
                        techniques_for_model(model, examination)
                            .with(crate::output::Technique::Structural),
                    )],
                    ColoredExecutionDiagnostics::default(),
                ));
            }
        }
    }

    if model.source_kind() == SourceNetKind::SymmetricNet
        && matches!(
            examination,
            crate::examination::Examination::ReachabilityCardinality
                | crate::examination::Examination::ReachabilityFireability
                | crate::examination::Examination::CTLCardinality
                | crate::examination::Examination::CTLFireability
                | crate::examination::Examination::LTLCardinality
                | crate::examination::Examination::LTLFireability
                | crate::examination::Examination::OneSafe
        )
        && model.colored_source.is_some()
    {
        return collect_on_uncollapsed_colored_source(model, examination, config, flush);
    }

    // FAIL-CLOSED GUARD: for a colored model whose load-time unfolding aborted
    // over budget, `model.net()` is an empty PLACEHOLDER. The examinations
    // that reach this fallthrough for SymmetricNet (ReachabilityDeadlock,
    // QuasiLiveness, StableMarking, Liveness) have no colored-source-aware
    // path, so running them on the placeholder would fabricate a verdict
    // (e.g. an empty net is trivially deadlocked / trivially live). Emit
    // CANNOT_COMPUTE instead. (StateSpace / UpperBounds / Reachability* /
    // CTL* / LTL* / OneSafe never reach here — they are handled above and
    // re-abort to CC on their own re-unfold.)
    if model.colored_unfold_unavailable() {
        return Ok((
            net_dependent_cannot_compute_records(
                model,
                examination,
                techniques_for_model(model, examination),
            )?,
            ColoredExecutionDiagnostics::default(),
        ));
    }

    let records = crate::examination::collect_examination_core_with_nupn(
        model.net(),
        model.model_name(),
        model.model_dir(),
        model.aliases(),
        examination,
        config,
        flush,
        model.nupn(),
    )?;
    Ok((records, ColoredExecutionDiagnostics::default()))
}

/// Emit a single CANNOT_COMPUTE record for a net-dependent, single-verdict
/// examination whose colored model has no executable net (placeholder).
fn net_dependent_cannot_compute_records(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    techniques: crate::output::Techniques,
) -> Result<Vec<crate::examination::ExaminationRecord>, PnmlError> {
    use crate::examination::{ExaminationRecord, ExaminationValue, Verdict};
    let _ = model;
    Ok(vec![ExaminationRecord::with_techniques(
        examination.as_str().to_string(),
        ExaminationValue::Verdict(Verdict::CannotCompute),
        techniques,
    )])
}

fn collect_on_uncollapsed_colored_source(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    config: &crate::explorer::ExplorationConfig,
    flush: bool,
) -> Result<
    (
        Vec<crate::examination::ExaminationRecord>,
        ColoredExecutionDiagnostics,
    ),
    PnmlError,
> {
    let colored_source = model.colored_source.as_ref().expect("checked by caller");
    let base_techniques = techniques_for_model(model, examination);

    let uncollapsed = match crate::unfold::unfold_to_pt(colored_source) {
        Ok(unfolded) => unfolded,
        Err(error) => {
            // The explicit P/T unfold declined (out of the place-materialization
            // budget, or a per-binding cap). For StateSpace ONLY, try the
            // SYMBOLIC-COLORED engine: it builds the reachable-marking set as one
            // compact MDD over the unfolded `(place, color)` levels WITHOUT
            // materializing the P/T place/transition/alias tables, so it can
            // recover the four StateSpace metrics on nets whose unfolded P/T form
            // is too large to materialize but whose reachable set is a compact
            // MDD. Fail-closed: on decline / overflow / deadline / out-of-subclass
            // it returns `None` and we fall through to the existing CC unchanged.
            #[cfg(feature = "dd-backend")]
            if matches!(examination, crate::examination::Examination::StateSpace) {
                if let Some(record) =
                    try_symbolic_colored_state_space(colored_source, config, &base_techniques)
                {
                    return Ok((vec![record], ColoredExecutionDiagnostics::default()));
                }
            }
            eprintln!(
                "{}: uncollapsed colored semantic unfold failed ({error}) - CANNOT_COMPUTE",
                examination.as_str()
            );
            return Ok((
                uncollapsed_colored_cannot_compute_records(model, examination, base_techniques)?,
                ColoredExecutionDiagnostics::default(),
            ));
        }
    };

    let records = crate::examination::collect_examination_core_with_nupn(
        &uncollapsed.net,
        model.model_name(),
        model.model_dir(),
        &uncollapsed.aliases,
        examination,
        config,
        flush,
        None,
    )?;
    Ok((records, ColoredExecutionDiagnostics::default()))
}

/// Kill-switch for the symbolic-colored StateSpace lane. The lane is ON by
/// default; set `TY_MCC_ENABLE_SYMBOLIC_COLORED` to a FALSY value
/// (`0`/`off`/`false`/`no`) to disable it and fall through to the existing
/// CANNOT_COMPUTE behavior unchanged.
///
/// SOUNDNESS-NEUTRAL either way: disabling the lane can only turn a recovered
/// colored StateSpace count back into CANNOT_COMPUTE. It never changes a
/// published value (the lane only ever publishes the exact-by-construction
/// symbolic count, or withholds).
///
/// Parsed like the sibling `TY_MCC_ENABLE_*` flags: a flag that is set but
/// explicitly falsy disables; unset or any non-falsy value keeps the default
/// (ON).
#[cfg(feature = "dd-backend")]
fn symbolic_colored_disabled() -> bool {
    flag_explicitly_falsy("TY_MCC_ENABLE_SYMBOLIC_COLORED")
}

/// INDEPENDENT kill-switch for the BINDING-QUANTIFIED sub-path of the
/// symbolic-colored StateSpace lane. The sub-path is ON by default; set
/// `TY_MCC_ENABLE_BINDING_QUANTIFIED` to a FALSY value
/// (`0`/`off`/`false`/`no`) to disable ONLY the quantified fallback while
/// leaving the (primary) enumerate path of the colored lane intact.
///
/// SOUNDNESS-NEUTRAL: disabling the sub-path can only turn a quantified-recovered
/// colored StateSpace count back into CANNOT_COMPUTE; it never changes a
/// published value. When OFF (or when the enumerate path DECIDES), the lane's
/// behavior is EXACTLY the pre-PR-1 enumerate-only behavior
/// (publish-on-enumerate-decide, else CANNOT_COMPUTE).
///
/// Parsed identically to the sibling `TY_MCC_ENABLE_*` flags (unset / non-falsy
/// ⇒ default ON; explicitly falsy ⇒ OFF).
#[cfg(feature = "dd-backend")]
fn binding_quantified_disabled() -> bool {
    flag_explicitly_falsy("TY_MCC_ENABLE_BINDING_QUANTIFIED")
}

/// Shared parser for the `TY_MCC_ENABLE_*` kill-switch flags: returns `true`
/// (disabled) only when the flag is SET to an explicitly falsy value
/// (`0`/`off`/`false`/`no`); unset or any non-falsy value keeps the default
/// (ON ⇒ `false`).
#[cfg(feature = "dd-backend")]
fn flag_explicitly_falsy(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| {
        let v = v.trim();
        v == "0"
            || v.eq_ignore_ascii_case("off")
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("no")
    })
}

/// The outcome of one symbolic-colored worker attempt, distinguishing a clean
/// DECIDE (publishable bundle) from a recoverable DECLINE (fall through / try the
/// next path) and from a hard fail-closed boundary (spawn fail / panic / timeout
/// ⇒ DECLINE too, but logged distinctly). A DECLINE is NEVER a wrong count.
#[cfg(feature = "dd-backend")]
enum SymbolicColoredAttempt {
    /// The path DECIDED: an exact-by-construction metric bundle to publish.
    Decided(tla_mdd::MddStateSpaceMetrics),
    /// The path DECLINED (out-of-sub-class / binding-or-place cap / MDD
    /// fail-closed / budget / worker boundary) ⇒ CANNOT_COMPUTE for this path.
    Declined,
}

/// Run ONE symbolic-colored path (`run` = the enumerate or the quantified
/// engine) on a fresh worker thread with the big DD stack + an INTERNAL deadline,
/// preserving the fail-closed `recv_timeout` boundary VERBATIM. Any
/// out-of-sub-class / cap / overflow / budget / spawn-failure / panic is mapped
/// to [`SymbolicColoredAttempt::Declined`] (never a wrong count).
///
/// `which` names the path for the diagnostic log lines.
#[cfg(feature = "dd-backend")]
fn run_symbolic_colored_worker<F>(
    colored_source: &crate::hlpnml::ColoredNet,
    budget: std::time::Duration,
    which: &str,
    run: F,
) -> SymbolicColoredAttempt
where
    F: FnOnce(
            &crate::hlpnml::ColoredNet,
            Option<std::time::Instant>,
        ) -> Result<
            tla_mdd::MddStateSpaceMetrics,
            crate::symbolic_colored::SymbolicColoredError,
        > + Send
        + 'static,
{
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    // Run on a worker thread with the big DD stack (the MDD recursions descend
    // the per-`(place,color)`-level node chain, so a many-level net needs more
    // than the default 8 MiB stack) + the deadline installed INSIDE the worker
    // (the saturation engine declines, fail-closed, rather than overrun it). The
    // worker boundary turns any panic into a clean DECLINE.
    let (tx, rx) = mpsc::channel();
    let colored_for_thread = colored_source.clone();
    let inner_deadline = Some(Instant::now() + budget);
    let handle = std::thread::Builder::new()
        .name(format!("tla-symbolic-colored-statespace-{which}"))
        .stack_size(tla_dd::DD_WORKER_STACK_BYTES)
        .spawn(move || {
            let r = run(&colored_for_thread, inner_deadline);
            let _ = tx.send(r);
        });
    if handle.is_err() {
        eprintln!(
            "StateSpace: symbolic-colored ({which}) thread spawn failed — using CANNOT_COMPUTE"
        );
        return SymbolicColoredAttempt::Declined;
    }

    match rx.recv_timeout(budget + Duration::from_millis(1500)) {
        Ok(Ok(metrics)) => SymbolicColoredAttempt::Decided(metrics),
        Ok(Err(crate::symbolic_colored::SymbolicColoredError::OutOfSubclass(reason))) => {
            eprintln!(
                "StateSpace: symbolic-colored ({which}) declined (out of v1 sub-class: {reason}) — \
                 using CANNOT_COMPUTE"
            );
            SymbolicColoredAttempt::Declined
        }
        Ok(Err(crate::symbolic_colored::SymbolicColoredError::Mdd(err))) => {
            eprintln!(
                "StateSpace: symbolic-colored ({which}) declined (MDD fail-closed: {err:?}) — \
                 using CANNOT_COMPUTE"
            );
            SymbolicColoredAttempt::Declined
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "StateSpace: symbolic-colored ({which}) exceeded budget — using CANNOT_COMPUTE"
            );
            SymbolicColoredAttempt::Declined
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            eprintln!(
                "StateSpace: symbolic-colored ({which}) worker panicked — using CANNOT_COMPUTE"
            );
            SymbolicColoredAttempt::Declined
        }
    }
}

/// Attempt the SYMBOLIC-COLORED StateSpace engine on `colored_source` and, on
/// success, return the single populated StateSpace record. Returns `None`
/// (fall through to the existing CANNOT_COMPUTE) on the kill-switch, an
/// out-of-sub-class construct, a binding/place cap, an MDD overflow / node
/// budget / deadline, a worker spawn failure, or a worker panic.
///
/// # Dispatch (PR-1: binding-quantified fallback)
///
/// The ENUMERATE path (`colored_state_space_metrics`) stays PRIMARY: when it
/// DECIDES, its bundle is published UNCHANGED (behavior-preserving). Only when it
/// DECLINES (binding/place cap or out-of-sub-class) — and the independent
/// `TY_MCC_ENABLE_BINDING_QUANTIFIED` kill-switch is ON — do we ADDITIONALLY try
/// the BINDING-QUANTIFIED path (`colored_state_space_metrics_quantified`), which
/// branches binding variables symbolically instead of enumerating them and so
/// decides nets whose binding count blows the enumerate cap. If the quantified
/// path also declines, we return `None` (CANNOT_COMPUTE) EXACTLY as before.
///
/// SOUNDNESS: it is safe to try the quantified path on ANY enumerate decline
/// because the quantified path ITSELF fail-closes (`Err` ⇒ DECLINE) on anything
/// out of its sub-class — no reason-parsing is required. The differential battery
/// pins quantified == enumerate == oracle on every net the enumerate path
/// decides, so the enumerate path never decides where the quantified path would
/// decide a DIFFERENT value; the two cannot both decide here anyway (we only run
/// quantified AFTER an enumerate DECLINE), so no value is ever overwritten.
///
/// Mirrors the kill-switch + worker-thread (`DD_WORKER_STACK_BYTES`) + deadline
/// + fail-closed posture of
///   [`crate::examinations::reachability::mdd_fastpath`]. Both engines keep a
///   STRICT admission gate (the v1 sub-class + sound per-sort token-conservation
///   bounds), so any net either admits is exact-by-construction — pinned EQUAL to
///   the explicitly-unfolded P/T MDD StateSpace oracle by the differential battery
///   in `crate::symbolic_colored`'s tests. A wrong colored count is impossible: the
///   lane can only publish the exact count or withhold.
#[cfg(feature = "dd-backend")]
fn try_symbolic_colored_state_space(
    colored_source: &crate::hlpnml::ColoredNet,
    config: &crate::explorer::ExplorationConfig,
    base_techniques: &crate::output::Techniques,
) -> Option<crate::examination::ExaminationRecord> {
    use crate::examination::{ExaminationRecord, ExaminationValue, StateSpaceReport};
    use std::time::{Duration, Instant};

    if symbolic_colored_disabled() {
        return None;
    }

    // Wall-clock budget: the caller's remaining deadline if any, else a small
    // floor (matching the StateSpace MDD lane's no-deadline budget). A
    // non-positive remaining budget ⇒ decline rather than spawn a fresh
    // long-running computation past the budget.
    const SYMBOLIC_COLORED_NO_DEADLINE_BUDGET: Duration = Duration::from_secs(5);
    let budget = match config.deadline() {
        Some(d) => {
            let remaining = d.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            remaining
        }
        None => SYMBOLIC_COLORED_NO_DEADLINE_BUDGET,
    };

    // PRIMARY: the ENUMERATE path. When it DECIDES, publish its bundle UNCHANGED.
    let metrics = match run_symbolic_colored_worker(
        colored_source,
        budget,
        "enumerate",
        crate::symbolic_colored::colored_state_space_metrics,
    ) {
        SymbolicColoredAttempt::Decided(metrics) => metrics,
        // The enumerate path DECLINED (binding/place cap or out-of-sub-class /
        // budget). FALLBACK: try the BINDING-QUANTIFIED path on the SAME big-stack
        // worker + a fresh deadline derived from the REMAINING budget, gated by the
        // independent kill-switch. It fail-closes on anything out of its sub-class,
        // so trying it on any enumerate decline is sound.
        SymbolicColoredAttempt::Declined => {
            if binding_quantified_disabled() {
                return None;
            }
            // Recompute the remaining budget: the enumerate attempt may have spent
            // some of it. A non-positive remainder ⇒ decline rather than start a
            // fresh long-running computation past the budget.
            let remaining = match config.deadline() {
                Some(d) => {
                    let r = d.saturating_duration_since(Instant::now());
                    if r.is_zero() {
                        return None;
                    }
                    r
                }
                None => budget,
            };
            match run_symbolic_colored_worker(
                colored_source,
                remaining,
                "quantified",
                crate::symbolic_colored::colored_state_space_metrics_quantified,
            ) {
                SymbolicColoredAttempt::Decided(metrics) => {
                    eprintln!(
                        "StateSpace: symbolic-colored BINDING-QUANTIFIED fallback decided the \
                         net the enumerate path could not (binding cap / out-of-sub-class)"
                    );
                    metrics
                }
                // Both paths declined — CANNOT_COMPUTE exactly as before PR-1.
                SymbolicColoredAttempt::Declined => return None,
            }
        }
    };

    // Map the exact-by-construction MDD metric bundle into the StateSpace
    // record. The bundle now carries EXACT arbitrary-precision counts
    // (`state_count_big` / `edge_count_big`); `StateSpaceReport::from_big` fills
    // the narrowed back-compat fields fail-closed (saturated marker when the
    // count exceeds them) and emits the bignum on the wire, so a colored net
    // whose |R| exceeds `u128` is REPORTED at full precision (the colored MDD
    // engine no longer declines on count magnitude).
    eprintln!(
        "StateSpace: symbolic-colored lane recovered the colored net the P/T unfold could not \
         materialize (|R|={}, edges={}, max_in_place={}, max_sum={})",
        metrics.state_count_big,
        metrics.edge_count_big,
        metrics.max_token_in_place,
        metrics.max_token_sum,
    );
    Some(ExaminationRecord::with_techniques(
        // StateSpace is a single non-property examination: one record named after
        // the examination, identical to the regular P/T StateSpace path.
        crate::examination::Examination::StateSpace
            .as_str()
            .to_string(),
        ExaminationValue::StateSpace(Some(StateSpaceReport::from_big(
            metrics.state_count_big.clone(),
            metrics.edge_count_big.clone(),
            metrics.max_token_in_place,
            metrics.max_token_sum,
        ))),
        base_techniques.clone(),
    ))
}

fn uncollapsed_colored_cannot_compute_records(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    techniques: crate::output::Techniques,
) -> Result<Vec<crate::examination::ExaminationRecord>, PnmlError> {
    use crate::examination::{Examination, ExaminationRecord, ExaminationValue, Verdict};

    match examination {
        Examination::OneSafe => Ok(vec![ExaminationRecord::with_techniques(
            examination.as_str().to_string(),
            ExaminationValue::Verdict(Verdict::CannotCompute),
            techniques,
        )]),
        Examination::StateSpace => Ok(vec![ExaminationRecord::with_techniques(
            examination.as_str().to_string(),
            ExaminationValue::StateSpace(None),
            techniques,
        )]),
        Examination::ReachabilityCardinality
        | Examination::ReachabilityFireability
        | Examination::CTLCardinality
        | Examination::CTLFireability
        | Examination::LTLCardinality
        | Examination::LTLFireability => {
            let properties =
                crate::property_xml::parse_properties(model.model_dir(), examination.as_str())?;
            Ok(properties
                .into_iter()
                .map(|property| {
                    ExaminationRecord::with_techniques(
                        property.id,
                        ExaminationValue::Verdict(Verdict::CannotCompute),
                        techniques.clone(),
                    )
                })
                .collect())
        }
        _ => unreachable!("uncollapsed colored semantic path is not defined for {examination:?}"),
    }
}

pub(super) fn collect_simplification_report_for_model(
    model: &PreparedModel,
    examination: crate::examination::Examination,
) -> Result<SimplificationReport, PnmlError> {
    let xml_name = examination.property_xml_name()?;
    let properties = crate::property_xml::parse_properties(model.model_dir(), xml_name)?;
    let run = crate::formula_simplify::simplify_properties_with_report(
        model.net(),
        &properties,
        model.aliases(),
    );
    Ok(run.report)
}

fn collect_with_colored_relevance(
    model: &PreparedModel,
    examination: crate::examination::Examination,
    config: &crate::explorer::ExplorationConfig,
) -> Result<
    (
        Vec<crate::examination::ExaminationRecord>,
        ColoredExecutionDiagnostics,
    ),
    PnmlError,
> {
    let colored_source = model.colored_source.as_ref().expect("checked by caller");
    let exam_name = examination.as_str();
    let properties = crate::property_xml::parse_properties(model.model_dir(), exam_name)?;

    // Colored relevance always exercises STRUCTURAL reasoning (colored
    // reductions + relevance pruning) before unfolding to PT.
    let base_techniques =
        techniques_for_model(model, examination).with(crate::output::Technique::Structural);

    let mut records = Vec::with_capacity(properties.len());
    let mut diagnostics = ColoredExecutionDiagnostics::default();

    let uncollapsed = match crate::unfold::unfold_to_pt(colored_source) {
        Ok(unfolded) => unfolded,
        Err(error) => {
            eprintln!(
                "UpperBounds: uncollapsed colored baseline unfold failed ({error}) - \
                 CANNOT_COMPUTE",
            );
            for property in &properties {
                let mut diagnostic = ColoredPropertyDiagnostic::new(property.id.clone());
                diagnostic.set_fallback_reason(format!(
                    "uncollapsed colored baseline unfold failed: {error}"
                ));
                diagnostics.push(diagnostic);
                records.push(crate::examination::ExaminationRecord::with_techniques(
                    property.id.clone(),
                    crate::examination::ExaminationValue::OptionalBound(None),
                    base_techniques.clone(),
                ));
            }
            return Ok((records, diagnostics));
        }
    };

    for property in &properties {
        let refs = crate::colored_relevance::extract_refs(&property.formula);
        let has_refs = !refs.places.is_empty() || !refs.transitions.is_empty();
        let mut diagnostic = ColoredPropertyDiagnostic::new(property.id.clone());

        // `relevance_applied` records whether `colored_relevance::reduce`
        // actually trimmed places or transitions for this property. Used
        // below to opt the UpperBounds path into the Safety-Net-C
        // ground-truth fallback: colored_relevance is a backward-closure
        // pruning pass that is not UpperBounds-preserving in general (it
        // can remove producer transitions whose loss under-counts the
        // observed maximum, or remove drainer transitions whose loss
        // over-counts via spurious markings). When relevance has touched
        // the net, the per-property unfolded `net` cannot be trusted as
        // ground truth, so we cross-check on the uncollapsed colored
        // baseline.
        let mut relevance_applied = false;

        let (net, aliases) = if has_refs {
            let mut reduced = colored_source.clone();
            let report = crate::colored_relevance::reduce(&mut reduced, &property.formula);
            if report.is_reduction() {
                diagnostic.set_reduction(report.places_removed, report.transitions_removed);
                relevance_applied = true;
            }
            match crate::unfold::unfold_to_pt(&reduced) {
                Ok(unfolded) => (unfolded.net, unfolded.aliases),
                Err(error) => {
                    diagnostic.set_fallback_reason(error.to_string());
                    diagnostic.clear_reduction();
                    relevance_applied = false;
                    (uncollapsed.net.clone(), uncollapsed.aliases.clone())
                }
            }
        } else {
            (uncollapsed.net.clone(), uncollapsed.aliases.clone())
        };

        let one_property = std::slice::from_ref(property);
        let property_records = run_single_property_exam(
            &net,
            one_property,
            &aliases,
            config,
            examination,
            &base_techniques,
            // Only pass ground truth when colored_relevance ACTUALLY
            // reduced the net. When `relevance_applied` is false the per-
            // property `net` already is the uncollapsed baseline or a
            // relevance reduction that failed back to it, so the existing
            // PT-layer Safety-Net-A/B is sufficient.
            relevance_applied.then_some((&uncollapsed.net, &uncollapsed.aliases)),
            // The colored source enables the compact colored MDD UpperBounds lane
            // (build_colored_mdd_net) over the unfolded `net` the BDD lane blows
            // up on; same (place,color) slot encoding so query coeffs align.
            Some(colored_source),
        );
        records.extend(property_records);
        diagnostics.push(diagnostic);
    }

    Ok((records, diagnostics))
}

fn run_single_property_exam(
    net: &PetriNet,
    properties: &[crate::property_xml::Property],
    aliases: &PropertyAliases,
    config: &crate::explorer::ExplorationConfig,
    examination: crate::examination::Examination,
    techniques: &crate::output::Techniques,
    upper_bounds_ground_truth: Option<(&PetriNet, &PropertyAliases)>,
    colored: Option<&crate::hlpnml::ColoredNet>,
) -> Vec<crate::examination::ExaminationRecord> {
    use crate::examination::{Examination, ExaminationRecord, ExaminationValue};

    match examination {
        Examination::UpperBounds => {
            let ground_truth = upper_bounds_ground_truth.map(|(net, aliases)| {
                crate::examinations::upper_bounds::GroundTruthNet { net, aliases }
            });
            crate::examinations::upper_bounds::check_upper_bounds_properties_with_aliases_and_ground_truth(
                net, properties, aliases, config, ground_truth, colored,
            )
            .into_iter()
            .map(|(id, bound)| {
                ExaminationRecord::with_techniques(
                    id,
                    ExaminationValue::OptionalBound(bound),
                    techniques.clone(),
                )
            })
            .collect()
        }
        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            crate::examinations::reachability::check_reachability_properties_with_aliases(
                net, properties, aliases, config,
            )
            .into_iter()
            .map(|(id, verdict)| {
                ExaminationRecord::with_techniques(
                    id,
                    ExaminationValue::Verdict(verdict),
                    techniques.clone(),
                )
            })
            .collect()
        }
        Examination::CTLCardinality | Examination::CTLFireability => {
            crate::examinations::ctl::check_ctl_properties_with_aliases(
                net, properties, aliases, config,
            )
            .into_iter()
            .map(|(id, verdict)| {
                ExaminationRecord::with_techniques(
                    id,
                    ExaminationValue::Verdict(verdict),
                    techniques.clone(),
                )
            })
            .collect()
        }
        Examination::LTLCardinality | Examination::LTLFireability => {
            crate::examinations::ltl::check_ltl_properties_with_aliases(
                net, properties, aliases, config,
            )
            .into_iter()
            .map(|(id, verdict)| {
                ExaminationRecord::with_techniques(
                    id,
                    ExaminationValue::Verdict(verdict),
                    techniques.clone(),
                )
            })
            .collect()
        }
        _ => Vec::new(),
    }
}

// ===========================================================================
// PRODUCTION-LANE coverage for the binding-quantified StateSpace fallback.
// ===========================================================================
//
// The differential SOUNDNESS battery in `crate::symbolic_colored_tests` proves
// quantified == enumerate == oracle on the *engine* functions directly. This
// module adds the missing coverage of the *production dispatch*
// (`try_symbolic_colored_state_space`): a colored net whose binding PRODUCT
// exceeds the enumerate cap (so the enumerate path DECLINES) is RECOVERED by the
// binding-quantified fallback, populating the StateSpace record with the exact
// count instead of CANNOT_COMPUTE — and is CANNOT_COMPUTE again when the
// independent `TY_MCC_ENABLE_BINDING_QUANTIFIED` kill-switch is OFF.
#[cfg(all(test, feature = "dd-backend"))]
mod binding_quantified_lane_tests {
    use crate::explorer::ExplorationConfig;
    use crate::hlpnml::parse_hlpnml;
    use crate::output::Techniques;
    use tla_bignum::ToPrimitive as _;

    /// A colored net whose single transition has `vars` binding variables, ALL
    /// over a 4-color cyclic enum, ALL pinned to `c0` by a conjunctive guard.
    /// Binding PRODUCT = `4^vars` (chosen > the 50M enumerate cap), but the
    /// quantified prune cuts every `x_i != c0` sub-tree, leaving ONE surviving
    /// binding ⇒ O(vars) quantified work (fast). One token of color `c0` moves
    /// to `succ(c0) = c1` via the surviving binding ⇒ |R| = 2 (the c0-marking and
    /// the c1-marking), edges = 1 (the single surviving binding fires only at the
    /// c0-marking). The enumerate path / `unfold_to_pt` DECLINE at the binding
    /// cap; the quantified path DECIDES it.
    fn build_wide_binding_net(vars: usize) -> crate::hlpnml::ColoredNet {
        assert!(vars >= 1);
        // Guard: AND over all variables of `x_i = c0`. The first arc consumes
        // `1'x0`, the output produces `1'succ(x0)`; x1.. are pure spectators
        // (pinned by the guard but absent from the arcs).
        let guard_conjuncts: String = (0..vars)
            .map(|i| {
                format!(
                    "<subterm><equality>\
                       <subterm><variable refvariable=\"x{i}\"/></subterm>\
                       <subterm><useroperator declaration=\"c0\"/></subterm>\
                     </equality></subterm>"
                )
            })
            .collect();
        let var_decls: String = (0..vars)
            .map(|i| {
                format!("<variabledecl id=\"x{i}\" name=\"x{i}\"><usersort declaration=\"C\"/></variabledecl>")
            })
            .collect();
        let pnml = format!(
            r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="widebind" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="p">
        <type><structure><usersort declaration="C"/></structure></type>
        <hlinitialMarking><structure>
          <numberof><subterm><numberconstant value="1"/></subterm><subterm><useroperator declaration="c0"/></subterm></numberof>
        </structure></hlinitialMarking>
      </place>
      <transition id="move">
        <condition><structure>
          <and>{guard_conjuncts}</and>
        </structure></condition>
      </transition>
      <arc id="p2t" source="p" target="move">
        <hlinscription><structure><numberof><subterm><numberconstant value="1"/></subterm><subterm><variable refvariable="x0"/></subterm></numberof></structure></hlinscription>
      </arc>
      <arc id="t2p" source="move" target="p">
        <hlinscription><structure><numberof><subterm><numberconstant value="1"/></subterm><subterm><successor><subterm><variable refvariable="x0"/></subterm></successor></subterm></numberof></structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="C" name="C"><cyclicenumeration>
        <feconstant id="c0" name="c0"/><feconstant id="c1" name="c1"/><feconstant id="c2" name="c2"/><feconstant id="c3" name="c3"/>
      </cyclicenumeration></namedsort>
      {var_decls}
    </declarations></structure></declaration>
  </net>
</pnml>"#
        );
        parse_hlpnml(&pnml).expect("generated wide-binding PNML parses")
    }

    /// Read the StateSpace record's `(states, edges, max_in_place)` if it is
    /// POPULATED, or `None` if CANNOT_COMPUTE (StateSpace value `None`).
    fn statespace_of(rec: &crate::examination::ExaminationRecord) -> Option<(u128, u128, u64)> {
        match &rec.value {
            crate::examination::ExaminationValue::StateSpace(Some(r)) => Some((
                r.states_big.to_u128().unwrap_or(u128::MAX),
                r.edges,
                r.max_token_in_place,
            )),
            crate::examination::ExaminationValue::StateSpace(None) => None,
            other => panic!("expected a StateSpace record, got {other:?}"),
        }
    }

    /// PRODUCTION-LANE coverage: a net whose binding product (4^13 ≈ 67M) blows
    /// the 50M enumerate cap is RECOVERED through the production entry
    /// `try_symbolic_colored_state_space` by the binding-quantified fallback —
    /// populated with the exact count when the kill-switch is ON, CANNOT_COMPUTE
    /// (None) when it is OFF.
    #[test]
    fn production_lane_binding_quantified_recovers_over_cap_net() {
        // 4^13 = 67_108_864 > MAX_BINDING_ITERATIONS (50M): the enumerate path /
        // unfold_to_pt DECLINE at the binding cap.
        const VARS: usize = 13;
        assert!(
            4usize.pow(VARS as u32) > crate::unfold::MAX_BINDING_ITERATIONS,
            "binding product must exceed the enumerate cap"
        );
        let net = build_wide_binding_net(VARS);

        // Precondition: the explicit P/T unfold DECLINES at the binding cap (this
        // is the cell that is CANNOT_COMPUTE today without the quantified lane).
        match crate::unfold::unfold_to_pt(&net) {
            Err(crate::error::PnmlError::ColoredUnfoldUnavailable { .. }) => {}
            other => panic!(
                "expected unfold_to_pt to DECLINE at the binding cap, got {:?}",
                other
                    .map(|u| (u.net.num_places(), u.net.transitions.len()))
                    .err()
            ),
        }

        let config = ExplorationConfig::default();
        let techniques = Techniques::default();

        // KILL-SWITCH ON (default): the quantified fallback RECOVERS the net.
        crate::env_guard::remove_var("TY_MCC_ENABLE_BINDING_QUANTIFIED");
        crate::env_guard::remove_var("TY_MCC_ENABLE_SYMBOLIC_COLORED");
        let rec_on = super::try_symbolic_colored_state_space(&net, &config, &techniques)
            .expect("quantified fallback must POPULATE the StateSpace record (kill-switch ON)");
        let (states, edges, max_in_place) = statespace_of(&rec_on)
            .expect("StateSpace record must be POPULATED (not CANNOT_COMPUTE) with the switch ON");
        // Family invariant: only c0 moves (to c1) ⇒ |R| = 2, one token, one
        // surviving binding firing once at the c0-marking ⇒ edges = 1.
        assert_eq!(states, 2, "only c0 moves to c1 ⇒ |R| = 2");
        assert_eq!(edges, 1, "single surviving binding fires once ⇒ edges = 1");
        assert_eq!(max_in_place, 1, "one token ⇒ max-in-place 1");

        // KILL-SWITCH OFF: the quantified sub-path is disabled, the enumerate
        // path declines at the binding cap, and the lane returns None
        // (CANNOT_COMPUTE) — exactly the pre-PR-1 behavior.
        crate::env_guard::set_var("TY_MCC_ENABLE_BINDING_QUANTIFIED", "0");
        let rec_off = super::try_symbolic_colored_state_space(&net, &config, &techniques);
        crate::env_guard::remove_var("TY_MCC_ENABLE_BINDING_QUANTIFIED");
        assert!(
            rec_off.is_none(),
            "with the binding-quantified kill-switch OFF the over-cap net must be CANNOT_COMPUTE"
        );
    }

    /// Behavior-preserving check: a SMALL net the ENUMERATE path can decide is
    /// published by the enumerate path UNCHANGED, regardless of the
    /// binding-quantified kill-switch (the fallback is never consulted). Drives
    /// the production entry with the switch both ON and OFF; the populated record
    /// is identical.
    #[test]
    fn production_lane_enumerate_decides_unaffected_by_quantified_switch() {
        // 4^2 = 16 bindings ≪ the cap ⇒ the enumerate path DECIDES. One token of
        // color c0 moves c0->c1 ⇒ |R| = 2.
        let net = build_wide_binding_net(2);
        // unfold must SUCCEED here (under the cap) — the enumerate path is live.
        assert!(crate::unfold::unfold_to_pt(&net).is_ok());

        let config = ExplorationConfig::default();
        let techniques = Techniques::default();
        crate::env_guard::remove_var("TY_MCC_ENABLE_SYMBOLIC_COLORED");

        crate::env_guard::remove_var("TY_MCC_ENABLE_BINDING_QUANTIFIED");
        let on = super::try_symbolic_colored_state_space(&net, &config, &techniques)
            .and_then(|r| statespace_of(&r));
        crate::env_guard::set_var("TY_MCC_ENABLE_BINDING_QUANTIFIED", "0");
        let off = super::try_symbolic_colored_state_space(&net, &config, &techniques)
            .and_then(|r| statespace_of(&r));
        crate::env_guard::remove_var("TY_MCC_ENABLE_BINDING_QUANTIFIED");

        assert_eq!(
            on,
            Some((2, 1, 1)),
            "enumerate path decides the small net (|R|=2) with the switch ON"
        );
        assert_eq!(
            on, off,
            "the enumerate-decided record is identical regardless of the quantified switch \
             (behavior-preserving)"
        );
    }
}
