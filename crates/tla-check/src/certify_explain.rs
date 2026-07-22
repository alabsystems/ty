// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty certify` decline EXPLANATION — say EXACTLY why a spec is not certified.
//!
//! The north-star invariant is "wherever ty cannot deliver, it says so exactly". The
//! certification pipeline ([`crate::cert::certify_spec`] and
//! [`crate::explicit_fixpoint_cert::certify_explicit_state_spec_bounded`]) is fail-closed:
//! any stage outside its fragment returns `None`, and the CLI can only print a bare
//! `NOT CERTIFIED`. This module re-runs the SAME pipeline stages ONE BY ONE — with the
//! same live helpers (`extract_init_constraints`, `enumerate_states_from_constraint_branches`,
//! `enumerate_successors`, `value_cell_encode`, `recognize_pred_sorts`) — and reports every
//! stage that fails, with specifics (which variable, which value shape, which invariant,
//! which construct).
//!
//! Each report line is prefixed with a bracketed STAGE TAG (`[parse]`, `[config]`,
//! `[constants]`, `[init-form]`, `[init-enum]`, `[state-encoding]`, `[next-enum]`, `[bfs]`,
//! `[invariant]`, `[kernel]`) so downstream tooling can histogram first-failing stages, and —
//! where it is clear — with the north-star roadmap phase that unblocks it (R1 constants /
//! R3 functions / R4 quantifiers / R5 sequences / R6 instance; `docs/north-star-roadmap.md`).
//!
//! FAIL-CLOSED WORDING: these are DIAGNOSTIC reasons, not promises. An empty report means
//! every stage probed here passes — the decline (if any) then lives in a deeper kernel/prover
//! leg this probe does not re-run. A non-empty report does NOT promise the spec certifies
//! once the listed reasons are fixed.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::config::Config;
use tla_core::ast::{Expr, Module, OperatorDef, Unit};

/// Re-run the certification pipeline's stages one by one on an (already-flattened,
/// SPECIFICATION-resolved) module source, and report every stage that fails, with
/// specifics.
///
/// The caller is expected to hand in the SAME `(spec_src, config)` the certify entry
/// points saw: `spec_src` a single self-contained module (the CLI flattens the EXTENDS
/// closure first) and `config` with `init`/`next` resolved (the CLI decomposes
/// SPECIFICATION-form configs before certifying). This function does not redo either.
///
/// Returns one human-readable reason line per detected failure, in pipeline-stage order
/// (see the module docs for the stage tags). An EMPTY vector means every probed stage
/// passes; it is NOT a certification promise (the kernel/prover legs are not re-run here).
///
/// PANIC-ISOLATED: the probe deliberately walks states the fail-closed pipeline stops
/// short of (it keeps enumerating past encodability failures), so an evaluator panic the
/// pipeline never reaches can fire here. Such a panic is caught and reported as its own
/// `[probe]` reason line (with the reasons collected up to that point kept) — a diagnostic
/// pass must never crash where the pipeline itself declines gracefully.
pub fn explain_certify_decline(spec_src: &str, config: &Config) -> Vec<String> {
    let reasons = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&reasons);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        explain_stages(spec_src, config, &mut |line: String| {
            sink.lock().expect("reason sink poisoned").push(line);
        });
    }));
    let mut out = std::mem::take(&mut *reasons.lock().expect("reason sink poisoned"));
    if let Err(panic) = outcome {
        let msg = panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_string());
        out.push(format!(
            "[probe] the stage probe crashed while re-running the pipeline stages ({msg}) — \
             reasons above are the ones collected before the crash; this indicates a ty \
             evaluator bug, please report"
        ));
    }
    out
}

/// The stage-by-stage worker behind [`explain_certify_decline`]: pushes each reason line into
/// `report` as it is found (so a panic in a later stage keeps the earlier findings).
fn explain_stages(spec_src: &str, config: &Config, report: &mut dyn FnMut(String)) {
    // Push-through sink: every reason reaches the caller the moment it is found (panic-safe).
    let mut reasons = ReasonSink { report };

    // ── stage a: parse/lower; VARIABLES present; INIT/NEXT resolved ────────────────────────
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let Some(module) = lowered.module else {
        reasons.push(format!(
            "[parse] module failed to lower ({} error(s)) — nothing downstream can run",
            lowered.errors.len()
        ));
        return;
    };

    let var_names: Vec<Arc<str>> = module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    if var_names.is_empty() {
        reasons.push(
            "[parse] the module declares no VARIABLES — there is no state to certify".to_string(),
        );
        return;
    }

    let find_op = |name: &str| -> Option<&OperatorDef> {
        module.units.iter().find_map(|unit| match &unit.node {
            Unit::Operator(op) if op.name.node == name => Some(op),
            _ => None,
        })
    };

    let Some(init_name) = config.init.as_deref() else {
        reasons.push(
            "[config] no INIT resolved (neither INIT nor a decomposable SPECIFICATION) — \
             the certification lanes need an initial predicate"
                .to_string(),
        );
        return;
    };
    let Some(next_name) = config.next.as_deref() else {
        reasons.push(
            "[config] no NEXT resolved (neither NEXT nor a decomposable SPECIFICATION) — \
             the certification lanes need a next-state action"
                .to_string(),
        );
        return;
    };
    let Some(init_def) = find_op(init_name) else {
        reasons.push(format!(
            "[config] INIT operator `{init_name}` is not defined at the module's top level \
             (missing, or produced by INSTANCE — module composition is roadmap R6)"
        ));
        return;
    };
    let Some(next_def) = find_op(next_name) else {
        reasons.push(format!(
            "[config] NEXT operator `{next_name}` is not defined at the module's top level \
             (missing, or produced by INSTANCE — module composition is roadmap R6)"
        ));
        return;
    };
    if !init_def.params.is_empty() || !next_def.params.is_empty() {
        reasons.push(format!(
            "[config] INIT/NEXT must be nullary operators (`{init_name}` takes {}, \
             `{next_name}` takes {} parameter(s))",
            init_def.params.len(),
            next_def.params.len()
        ));
        return;
    }
    // Zero configured invariants ⇒ nothing to certify. MULTIPLE invariants are now conjoined
    // and certified together (`R ⊆ I_0 ∧ … ∧ I_{n-1}`), so a count > 1 is NOT a blocker.
    if config.invariants.is_empty() {
        reasons.push(
            "[config] no INVARIANT configured — nothing to certify (this spec's verdict, if \
             any, is a liveness property or a counterexample)"
                .to_string(),
        );
    }

    // ── stage b: configured CONSTANTs bind ──────────────────────────────────────────────────
    for unit in &module.units {
        if let Unit::Constant(decls) = &unit.node {
            for d in decls {
                let name = &d.name.node;
                if !config.constants.contains_key(name) {
                    reasons.push(format!(
                        "[constants] CONSTANT `{name}` has no configured value — cannot bind \
                         (configured-constant certification is roadmap R1)"
                    ));
                }
            }
        }
    }
    let build_ctx = || -> Result<crate::eval::EvalCtx, crate::error::EvalError> {
        let mut ctx = crate::eval::EvalCtx::new();
        ctx.load_module(&module);
        for v in &var_names {
            ctx.register_var(Arc::clone(v));
        }
        crate::constants::bind_constants_from_config(&mut ctx, config)?;
        Ok(ctx)
    };
    let ctx = match build_ctx() {
        Ok(ctx) => ctx,
        Err(e) => {
            reasons.push(format!(
                "[constants] binding the configured CONSTANTs failed: {e} (roadmap R1)"
            ));
            return;
        }
    };

    // ── stage c: Init constraint extraction ─────────────────────────────────────────────────
    use crate::enumerate::{enumerate_states_from_constraint_branches, extract_init_constraints};
    let Some(branches) = extract_init_constraints(&ctx, &init_def.body, &var_names, None) else {
        let hint = classify_unsupported(&init_def.body.node, &module)
            .map(|(desc, tag)| format!(" — {}{}", desc, roadmap_suffix(tag)))
            .unwrap_or_default();
        reasons.push(format!(
            "[init-form] Init `{init_name}` is not in constraint form \
             (equality/membership extraction failed){hint}"
        ));
        return;
    };

    // ── stage d: Init state enumeration ─────────────────────────────────────────────────────
    let init_states =
        match enumerate_states_from_constraint_branches(Some(&ctx), &var_names, &branches) {
            Err(e) => {
                reasons.push(format!("[init-enum] Init state enumeration failed: {e}"));
                return;
            }
            Ok(None) => {
                reasons.push(
                    "[init-enum] Init state enumeration is unsupported for this constraint shape \
                 (a state variable has no enumerable domain)"
                        .to_string(),
                );
                return;
            }
            Ok(Some(states)) if states.is_empty() => {
                reasons.push("[init-enum] Init enumerated ZERO initial states".to_string());
                return;
            }
            Ok(Some(states)) => states,
        };

    // ── stages e+f: per-column CELL encodability + bounded BFS enumerability ────────────────
    #[cfg(feature = "clean-cic")]
    let observed_sorts = probe_encoding_and_bfs(
        &build_ctx,
        &var_names,
        next_name,
        next_def,
        &init_states,
        &mut reasons,
    );
    #[cfg(not(feature = "clean-cic"))]
    {
        let _ = init_states;
        reasons.push(
            "[kernel] state-encoding and kernel-recognition stages not probed — this build \
             lacks the `clean-cic` feature"
                .to_string(),
        );
    }

    // ── stage g: INVARIANT recognition into the kernel predicate fragment ───────────────────
    // The pipeline resolves zero-arity operator references and Int-literal configured CONSTANTs
    // inside the recognized bodies first (`cert_inline`), so the probe recognizes and classifies
    // the SAME resolved predicate the certify lane sees.
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    for inv_name in &config.invariants {
        let Some(inv_def) = find_op(inv_name) else {
            reasons.push(format!(
                "[invariant] `{inv_name}` is not defined at the module's top level \
                 (missing, or produced by INSTANCE — module composition is roadmap R6)"
            ));
            continue;
        };
        let inv_body = inline_env.inline(&inv_def.body);
        // Kernel recognition needs the observed per-column sorts; when they are unavailable
        // (a `clean-cic`-free build, or a column that never encoded), fall back to the
        // syntactic classifier so the invariant still gets a specific reason where one exists.
        #[allow(unused_mut)] // mut is taken in the clean-cic block below
        let mut probed_by_kernel = false;
        #[cfg(feature = "clean-cic")]
        if let Some(sorts) = &observed_sorts {
            probed_by_kernel = true;
            let vars: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
            match crate::cleancic::recognize_pred_sorts(&inv_body.node, &vars, sorts) {
                None => {
                    let (desc, tag) =
                        classify_unsupported(&inv_body.node, &module).unwrap_or_else(|| {
                            (
                                "not recognized into the kernel predicate fragment (PredIR)"
                                    .to_string(),
                                None,
                            )
                        });
                    reasons.push(format!(
                        "[invariant] {inv_name}: {desc}{}",
                        roadmap_suffix(tag)
                    ));
                }
                // Recognized. The nonneg `⋀ x ≥ 0` shape rides the primary tuple R⊆Safety leg;
                // anything else rides the GENERAL embedded safety leg (`safety_general`) —
                // provided it passes that lane's fail-closed gates, mirrored here EXACTLY
                // (`explicit_fixpoint_cert`): truth-direction exactness + state predicate.
                Some(ir)
                    if !crate::cleancic::is_conjunctive_nonneg_safety(
                        &inv_body.node,
                        &vars,
                        sorts,
                    ) =>
                {
                    if !crate::refinement_cert::pred_exact(&ir, sorts) {
                        reasons.push(format!(
                            "[invariant] {inv_name}: embeds in the kernel predicate fragment, \
                             but through a truth-direction-INEXACT form (Nat-truncating \
                             subtraction/div/mod, Seq digit ops, or a set \
                             comprehension/quantifier fold) — the general R⊆Safety leg needs \
                             kernel-TRUE ⇒ TLA-TRUE exactness, so it declines (fail-closed)"
                        ));
                    } else if crate::refinement_cert::pred_mentions_prime(&ir) {
                        reasons.push(format!(
                            "[invariant] {inv_name}: mentions primed state — an invariant must \
                             be a STATE predicate for the general R⊆Safety leg"
                        ));
                    }
                    // else: certifiable via the general embedded safety leg — not a decline.
                    // (A state with NO Int column is admitted too, as of R3: its tuple safety
                    // leg is trivially true and the general leg carries the spec's claim.)
                }
                Some(_) => {} // the nonneg shape — the primary tuple leg proves it
            }
        }
        if !probed_by_kernel {
            match classify_unsupported(&inv_body.node, &module) {
                Some((desc, tag)) => reasons.push(format!(
                    "[invariant] {inv_name}: {desc}{}",
                    roadmap_suffix(tag)
                )),
                None => reasons.push(format!(
                    "[invariant] {inv_name}: kernel recognition not probed (state cell sorts \
                     unavailable — see the reasons above)"
                )),
            }
        }
    }
}

/// A push-through reason sink: each line is forwarded to the caller's report closure the
/// moment it is found, so a later-stage panic (caught in [`explain_certify_decline`]) keeps
/// every earlier finding.
struct ReasonSink<'a> {
    /// The caller's line consumer.
    report: &'a mut dyn FnMut(String),
}

impl ReasonSink<'_> {
    /// Report one reason line.
    fn push(&mut self, line: String) {
        (self.report)(line);
    }
}

/// Render a roadmap-phase tag as a report-line suffix (empty when no phase clearly applies).
fn roadmap_suffix(tag: Option<&'static str>) -> String {
    match tag {
        Some(t) => format!(" (roadmap {t})"),
        None => String::new(),
    }
}

// ═════════════════════════════════ stages e+f (clean-cic) ═══════════════════════════════════

/// Probe per-column cell encodability over every enumerated state and BFS enumerability under
/// the certification state cap, appending one reason line per detected failure. Returns the
/// observed per-column sorts when EVERY column encoded consistently (the precondition for the
/// stage-g kernel recognizer), `None` otherwise.
#[cfg(feature = "clean-cic")]
fn probe_encoding_and_bfs(
    build_ctx: &dyn Fn() -> Result<crate::eval::EvalCtx, crate::error::EvalError>,
    var_names: &[Arc<str>],
    next_name: &str,
    next_def: &OperatorDef,
    init_states: &[crate::state::State],
    reasons: &mut ReasonSink<'_>,
) -> Option<Vec<crate::explicit_fixpoint_cert::ColSort>> {
    use crate::explicit_fixpoint_cert::{memory_derived_state_cap, value_cell_encode, ColSort};
    use crate::state::State;

    // Mirror the mint-side reachable-set feasibility bound so the `[bfs]` diagnostic reports the
    // ACTUAL cap the certifier uses (memory-derived from the stable machine size), not a stale
    // fixed count.
    let cap = memory_derived_state_cap();
    // Per-variable FIRST failure line (one example per variable — the key diagnostic).
    let mut cell_failures: Vec<Option<String>> = vec![None; var_names.len()];
    // Per-variable sort-instability line (a column Int in one state, Bool in another).
    let mut sort_conflicts: Vec<Option<String>> = vec![None; var_names.len()];
    let mut col_sorts: Vec<Option<ColSort>> = vec![None; var_names.len()];

    // Two sorts that differ ONLY in the compound pack radix (the R3 per-column DERIVATION) are the
    // SAME column at different bases — the pipeline re-encodes the whole column at the LARGER derived
    // base (`max` over states of `compound_min_base`), so the probe records the larger-base sort
    // instead of flagging a conflict.
    let widened_join = |a: &ColSort, b: &ColSort| -> Option<ColSort> {
        match (a, b) {
            // A widen is the SAME column at a larger radix ONLY when the field set AND the per-position KIND
            // vector (`cells`) also agree — differing `cells` is a genuine kind CONFLICT (the discriminant),
            // not a radix widen, so it falls to `_ => None` and is flagged.
            (
                ColSort::Record {
                    base: ba,
                    fields: fa,
                    cells: ca,
                },
                ColSort::Record {
                    base: bb,
                    fields: fb,
                    cells: cb,
                },
            ) if fa == fb && ca == cb && ba != bb => Some(ColSort::Record {
                base: *ba.max(bb),
                fields: fa.clone(),
                cells: ca.clone(),
            }),
            (
                ColSort::Func {
                    base: ba,
                    arity: aa,
                    cells: ca,
                    dom: da,
                    dom_kind: ka,
                },
                ColSort::Func {
                    base: bb,
                    arity: ab,
                    cells: cb,
                    dom: db,
                    dom_kind: kb,
                },
            ) if aa == ab && ca == cb && da == db && ka == kb && ba != bb => Some(ColSort::Func {
                base: *ba.max(bb),
                arity: *aa,
                cells: ca.clone(),
                dom: da.clone(),
                dom_kind: *ka,
            }),
            _ => None,
        }
    };
    let record = |s: &State,
                  cell_failures: &mut Vec<Option<String>>,
                  sort_conflicts: &mut Vec<Option<String>>,
                  col_sorts: &mut Vec<Option<ColSort>>| {
        for (i, v) in var_names.iter().enumerate() {
            let Some(val) = s.get(v) else { continue };
            // Mirror the pipeline's per-column derivation rule: the floor compound radix first, then
            // the DERIVED minimal admitting base (`compound_min_base`) when the floor does not fit.
            let encoded = value_cell_encode(val).or_else(|| {
                crate::explicit_fixpoint_cert::compound_min_base(val)
                    .and_then(|b| crate::explicit_fixpoint_cert::value_cell_encode_at(val, b))
            });
            match encoded {
                Some((sort, _)) => match &col_sorts[i] {
                    Some(prev) if *prev != sort => {
                        if let Some(joined) = widened_join(prev, &sort) {
                            col_sorts[i] = Some(joined); // same column, widened radix
                        } else if sort_conflicts[i].is_none() {
                            sort_conflicts[i] = Some(format!(
                                "[state-encoding] {v}: cell sort differs across reachable states \
                                 ({prev:?} vs {sort:?}) — a column must keep ONE sort"
                            ));
                        }
                    }
                    Some(_) => {}
                    None => col_sorts[i] = Some(sort),
                },
                None => {
                    // A FUNCTION-of-ENUM cell (a function whose values are all labels, e.g.
                    // `pc: [0..N-1 -> {"a","b","Done"}]`) is NOT in `value_cell_encode`'s fragment,
                    // but the certify pipeline's enum encoder HANDLES it (as ColSort::FuncEnum over the
                    // column's observed label union) — so it is ENCODABLE, not a wall. The probe does not
                    // reconstruct the cross-state label union, so it leaves the column's sort
                    // undetermined (the invariant stage then falls back to the syntactic classifier), but
                    // it must NOT report a spurious cell decline for such a column.
                    if crate::explicit_fixpoint_cert::func_enum_view(val).is_some() {
                        continue;
                    }
                    // A FUNCTION-to-SET cell (a function `[D -> SUBSET E]` whose values are all atom sets,
                    // e.g. `alloc ∈ [Clients -> SUBSET Resources]`) is NOT in `value_cell_encode`'s fragment
                    // but the certify pipeline's F2 encoder HANDLES it (as `ColSort::FuncSetMask` over the
                    // column's observed value-atom universe) — so it is ENCODABLE, not a wall. The probe does
                    // not reconstruct the cross-state value universe, so it leaves the column's sort
                    // undetermined but must NOT report a spurious cell decline (mirrors the func_enum guard).
                    if crate::explicit_fixpoint_cert::funcsetmask_view(val).is_some() {
                        continue;
                    }
                    // A positional compound with a String/model-value ENUM position (the value-type-leaf
                    // Enum cell, e.g. `[mode |-> "a", n |-> 0]`) is ALSO encodable — the certify pipeline
                    // resolves each position against its cross-state label union. The probe does not
                    // reconstruct that union, so it leaves the sort undetermined but must NOT report a wall.
                    if crate::explicit_fixpoint_cert::compound_enum_view(val) {
                        continue;
                    }
                    // A SEQUENCE whose elements are all nonneg-Int / `Bool` / atom (`String`/model) LEAVES
                    // (within the length bound) is ENCODABLE by the certify loop's generalized
                    // `ColSort::Seq{elem}` (a self-delimiting pack over the element value-type leaf), so it
                    // is NOT a wall — mirror the func_enum_view/compound_enum_view guards. A NESTED element
                    // (tuple/record) or a negative/oversized Int still IS a wall (`cell_decline_line`).
                    if seq_of_leaves_encodable(val) {
                        continue;
                    }
                    if cell_failures[i].is_none() {
                        cell_failures[i] = Some(cell_decline_line(v, val));
                    }
                }
            }
        }
    };

    for s in init_states {
        record(s, &mut cell_failures, &mut sort_conflicts, &mut col_sorts);
    }

    // Bounded BFS over the live successor enumerator — the same primitive the pipeline uses.
    // Visited is keyed by the STATE (not the encoded tuple) so encodability failures do not
    // stop the reachability probe.
    let mut bfs_reason: Option<String> = None;
    // The certification cap applies to the INITIAL states too (they are reachable states; the
    // pipeline fail-closes on the enumerated-set size regardless of how a state was reached).
    if init_states.len() > cap {
        bfs_reason = Some(format!(
            "[bfs] the reachable set exceeds the {cap}-state certification cap ({} initial \
             states alone) — the cap is derived from this machine's memory budget; a spec whose \
             reachable set fits certifies",
            init_states.len()
        ));
    }
    match build_ctx() {
        Err(_) => {} // constants failed to bind; already reported in stage b
        Ok(_) if bfs_reason.is_some() => {} // already over the cap — skip the successor walk
        Ok(mut next_ctx) => {
            let mut visited: BTreeSet<State> = init_states.iter().cloned().collect();
            let mut frontier: Vec<State> = init_states.to_vec();
            'bfs: while let Some(cur) = frontier.pop() {
                match enumerate_successors_probe(&mut next_ctx, next_def, &cur, var_names) {
                    Err(e) => {
                        bfs_reason = Some(format!(
                            "[next-enum] successor enumeration failed under Next \
                             `{next_name}`: {e}"
                        ));
                        break 'bfs;
                    }
                    Ok(succs) => {
                        for succ in succs {
                            record(
                                &succ,
                                &mut cell_failures,
                                &mut sort_conflicts,
                                &mut col_sorts,
                            );
                            if visited.contains(&succ) {
                                continue;
                            }
                            visited.insert(succ.clone());
                            if visited.len() > cap {
                                bfs_reason = Some(format!(
                                    "[bfs] the reachable set exceeds the {cap}-state \
                                     certification cap (memory-derived from this machine's size; \
                                     BFS truncated — either a large finite state space, or an \
                                     infinite one outside the parametric unbounded affine fragment)"
                                ));
                                break 'bfs;
                            }
                            frontier.push(succ);
                        }
                    }
                }
            }
        }
    }

    for line in cell_failures.iter().chain(sort_conflicts.iter()).flatten() {
        reasons.push(line.clone());
    }
    if let Some(line) = bfs_reason {
        reasons.push(line);
    }

    let clean = cell_failures.iter().all(Option::is_none)
        && sort_conflicts.iter().all(Option::is_none)
        && col_sorts.iter().all(Option::is_some);
    clean.then(|| col_sorts.into_iter().flatten().collect())
}

/// Wrapper over the live successor enumerator, isolated so the probe's only `enumerate` import
/// surface is explicit.
#[cfg(feature = "clean-cic")]
fn enumerate_successors_probe(
    ctx: &mut crate::eval::EvalCtx,
    next_def: &OperatorDef,
    current: &crate::state::State,
    vars: &[Arc<str>],
) -> Result<Vec<crate::state::State>, crate::error::EvalError> {
    crate::enumerate::enumerate_successors(ctx, next_def, current, vars)
}

// ══════════════════════════════ value-shape diagnostics (stage e) ═══════════════════════════

/// The full `[state-encoding]` reason line for one variable's first non-encodable value:
/// the variable, the value's SHAPE, why the cell encoder declines it, and the roadmap phase
/// where one clearly applies.
#[cfg(feature = "clean-cic")]
fn cell_decline_line(var: &str, val: &crate::value::Value) -> String {
    let (why, tag) = cell_decline_why(val);
    format!(
        "[state-encoding] {var}: {} — {why}{}",
        value_shape(val),
        roadmap_suffix(tag)
    )
}

/// A compact structural SHAPE of a value for diagnostics (e.g. `Seq of Tuple(String,Int)`).
#[cfg(feature = "clean-cic")]
fn value_shape(val: &crate::value::Value) -> String {
    use crate::value::Value;
    match val {
        Value::Bool(_) => "Bool".to_string(),
        Value::SmallInt(_) | Value::Int(_) => "Int".to_string(),
        Value::String(_) => "String".to_string(),
        Value::ModelValue(_) => "ModelValue".to_string(),
        Value::Set(s) => match s.iter().next() {
            Some(e) => format!("Set of {}", value_shape(e)),
            None => "Set (empty)".to_string(),
        },
        Value::Interval(_) => "Set (Int interval)".to_string(),
        Value::Seq(s) => match s.iter().next() {
            Some(e) => format!("Seq of {}", value_shape(e)),
            None => "Seq (empty)".to_string(),
        },
        Value::Tuple(t) => {
            let inner: Vec<String> = t.iter().take(4).map(value_shape).collect();
            let ellipsis = if t.len() > 4 { ",…" } else { "" };
            format!("Tuple({}{ellipsis})", inner.join(","))
        }
        Value::Record(r) => {
            let fields: Vec<String> = r.iter_str().map(|(n, _)| n.to_string()).collect();
            format!("Record[{}]", fields.join(","))
        }
        Value::Func(f) => match f.iter().next() {
            Some((d, r)) => format!("Func({} -> {})", value_shape(d), value_shape(r)),
            None => "Func (empty)".to_string(),
        },
        Value::IntFunc(f) => match f.values().first() {
            Some(r) => format!("Func(Int -> {})", value_shape(r)),
            None => "Func (empty)".to_string(),
        },
        other => other.type_name().to_string(),
    }
}

/// Whether `val` is a SEQUENCE (`Value::Seq`/`Value::Tuple`) all of whose elements are ENCODABLE value-type
/// LEAVES — a nonneg Int, a `Bool`, or an atom (`String`/model value) — within the [`SEQ_MAX_LEN`] pack
/// bound, i.e. exactly what the certify loop's generalized [`crate::explicit_fixpoint_cert::ColSort::Seq`]
/// (with its per-element `elem` leaf) packs. Used by the decline probe to AVOID reporting a now-encodable
/// atom/`Bool` sequence as a cell wall (a NESTED element — tuple/record — or a negative/oversized Int
/// element is NOT a leaf ⇒ `false` ⇒ still surfaced as a wall).
#[cfg(feature = "clean-cic")]
fn seq_of_leaves_encodable(val: &crate::value::Value) -> bool {
    use crate::explicit_fixpoint_cert::SEQ_MAX_LEN;
    use crate::value::Value;
    let elems: Vec<&Value> = match val {
        Value::Seq(s) => s.iter().collect(),
        Value::Tuple(t) => t.iter().collect(),
        _ => return false,
    };
    if elems.len() > SEQ_MAX_LEN as usize {
        return false;
    }
    elems.iter().all(|e| {
        matches!(e, Value::Bool(_) | Value::String(_) | Value::ModelValue(_))
            || as_nonneg_u64(e).is_some()
    })
}

/// A nonneg small-Int extractor for the shape sub-diagnostics (mirrors the cell encoder's
/// `nonneg_small_int` leaf without widening its visibility).
#[cfg(feature = "clean-cic")]
fn as_nonneg_u64(val: &crate::value::Value) -> Option<u64> {
    use crate::value::Value;
    match val {
        Value::SmallInt(k) if *k >= 0 => Some(*k as u64),
        Value::Int(k) => {
            use num_traits::Signed;
            if k.is_negative() {
                None
            } else {
                u64::try_from(k.as_ref().clone()).ok()
            }
        }
        _ => None,
    }
}

/// WHY the cell encoder ([`crate::explicit_fixpoint_cert::value_cell_encode`]) declines this
/// value, with the roadmap phase where one clearly applies. Mirrors the encoder's fail-closed
/// arms case by case.
#[cfg(feature = "clean-cic")]
fn cell_decline_why(val: &crate::value::Value) -> (String, Option<&'static str>) {
    use crate::explicit_fixpoint_cert::{SEQ_MAX_LEN, SET_UNIVERSE_BITS};
    use crate::value::{IntIntervalFunc, Value};
    match val {
        Value::SmallInt(_) | Value::Int(_) => (
            "negative or oversized Int — the cell fragment is nonneg Int within u64".to_string(),
            None,
        ),
        Value::String(_) => (
            "outside the cell fragment — strings need an enumerated scalar sort, the same \
             machinery as roadmap R1 model values"
                .to_string(),
            Some("R1"),
        ),
        Value::ModelValue(_) => (
            "outside the cell fragment — model values as an enumerated sort".to_string(),
            Some("R1"),
        ),
        Value::Set(s) => {
            for e in s.iter() {
                match as_nonneg_u64(e) {
                    Some(v) if v < u64::from(SET_UNIVERSE_BITS) => {}
                    Some(v) => {
                        return (
                            format!(
                                "set element {v} exceeds the {SET_UNIVERSE_BITS}-bit bitmask \
                                 universe"
                            ),
                            None,
                        )
                    }
                    None => {
                        let (_, elem_tag) = cell_decline_why(e);
                        return (
                            format!(
                                "outside the bitmask set fragment — elements must be nonneg \
                                 Int < {SET_UNIVERSE_BITS}, this set holds {}",
                                value_shape(e)
                            ),
                            elem_tag,
                        );
                    }
                }
            }
            ("outside the bitmask set fragment".to_string(), None)
        }
        Value::Seq(_) | Value::Tuple(_) => {
            // The element radix is DERIVED per column (`max(SEQ_BASE, maxElement+1)`, the `seq_min_radix`
            // widening), so a nonneg-Int element is NEVER an element-bound decline on its own — the only
            // declines are: a length beyond the pack bound, a NON-nonneg-Int element (Bool/String/model
            // need a per-element enumerated leaf sort — the value-type-leaf refactor; or a negative Int),
            // or an all-nonneg-Int sequence whose DERIVED radix^len overflows the u64 pack.
            let (len, first_bad) = match val {
                Value::Seq(s) => (
                    s.len(),
                    s.iter().find(|e| as_nonneg_u64(e).is_none()).cloned(),
                ),
                Value::Tuple(t) => (
                    t.len(),
                    t.iter().find(|e| as_nonneg_u64(e).is_none()).cloned(),
                ),
                _ => unreachable!(),
            };
            if len > SEQ_MAX_LEN as usize {
                (
                    format!(
                        "length {len} exceeds the sequence pack bound max_len={SEQ_MAX_LEN} — \
                         realistic-length sequences"
                    ),
                    Some("R5"),
                )
            } else if let Some(e) = first_bad {
                // A Bool/String/model (or negative-Int) sequence element: the sequence pack encodes only
                // nonneg Int digits, so a non-Int leaf needs a per-element enumerated leaf sort — the
                // deferred value-type-leaf refactor (roadmap R5, the sequence phase), NOT the scalar-enum
                // R1: a Bool leaf packs to the same digit as the Int 0/1, so admitting it without a
                // distinct leaf sort would conflate `<<FALSE>>` with `<<0>>`.
                (
                    format!(
                        "sequence element holds {} — the sequence pack encodes only nonneg Int \
                         elements; Bool/String/model elements need a per-element enumerated leaf sort \
                         (the value-type-leaf refactor)",
                        value_shape(&e)
                    ),
                    Some("R5"),
                )
            } else {
                (
                    format!(
                        "sequence of length {len} — its derived pack radix^len overflows the u64 cell \
                         (the pack ceiling); shorter or smaller-valued elements would fit"
                    ),
                    Some("R3"),
                )
            }
        }
        Value::Record(r) => {
            for (name, v) in r.iter_str() {
                if as_nonneg_u64(v).is_none() {
                    return (
                        format!(
                            "record field `{name}` holds {} — non-Int field values need an \
                             enumerated scalar cell sort (roadmap R1); the pack encodes only \
                             nonneg Int field values",
                            value_shape(v)
                        ),
                        Some("R1"),
                    );
                }
            }
            // Every field is a nonneg Int but the DERIVED base^arity overflows the u64 pack.
            (
                format!(
                    "record of arity {} — its derived pack base^arity overflows the u64 cell \
                     (the pack ceiling); a smaller arity or field values would fit",
                    r.len()
                ),
                Some("R3"),
            )
        }
        Value::Func(f) => {
            // The domain must be the Int prefix `0..n-1` OR a homogeneous atom-key set (model values /
            // `String`s) — the `func_enum_domain_keys` shape now the Int/Bool/enum-valued pack ALSO carries.
            // A domain outside that (a compound key, a mixed-kind key set, a non-0 Int prefix) is the
            // genuine decline.
            if crate::explicit_fixpoint_cert::func_enum_domain_keys(f).is_none() {
                let first_bad = f
                    .domain_iter()
                    .find(|k| {
                        !matches!(
                            k,
                            Value::SmallInt(_)
                                | Value::Int(_)
                                | Value::String(_)
                                | Value::ModelValue(_)
                        )
                    })
                    .map(value_shape)
                    .unwrap_or_else(|| "a non-0-prefix / mixed-kind key set".to_string());
                return (
                    format!(
                        "function domain is not the 0..n-1 Int prefix nor a homogeneous atom-key set \
                         ({first_bad}) — general domains"
                    ),
                    Some("R3"),
                );
            }
            for v in f.mapping_values() {
                if as_nonneg_u64(v).is_none() {
                    return (
                        format!(
                            "function value holds {} — non-Int values need an enumerated scalar \
                             cell sort (roadmap R1); the pack encodes only nonneg Int values",
                            value_shape(v)
                        ),
                        Some("R1"),
                    );
                }
            }
            (
                format!(
                    "function of domain size {} — its derived pack base^arity overflows the u64 \
                     cell (the pack ceiling)",
                    f.domain_len()
                ),
                Some("R3"),
            )
        }
        Value::IntFunc(f) => {
            if IntIntervalFunc::min(f) != 0 {
                return (
                    "function domain is not the 0-based prefix 0..n-1 — general domains"
                        .to_string(),
                    Some("R3"),
                );
            }
            for v in f.values() {
                if as_nonneg_u64(v).is_none() {
                    return (
                        format!(
                            "function value holds {} — non-Int values need an enumerated scalar \
                             cell sort (roadmap R1); the pack encodes only nonneg Int values",
                            value_shape(v)
                        ),
                        Some("R1"),
                    );
                }
            }
            (
                format!(
                    "function of domain size {} — its derived pack base^arity overflows the u64 \
                     cell (the pack ceiling)",
                    f.len()
                ),
                Some("R3"),
            )
        }
        Value::LazyFunc(_) => (
            "function over a non-enumerable domain — general function state".to_string(),
            Some("R3"),
        ),
        Value::Closure(_) => (
            "operator/closure value — outside the cell fragment".to_string(),
            None,
        ),
        _ => ("outside the cell fragment".to_string(), None),
    }
}

// ═══════════════════════════ syntactic construct classifier (stages c/g) ════════════════════

/// A classifier finding: a human-readable description of the FIRST construct (pre-order) that
/// puts an expression outside the kernel predicate fragment, plus the roadmap phase where one
/// clearly applies. `None` in the option means "nothing notable found".
#[derive(Clone, Default)]
struct Finding(Option<(String, Option<&'static str>)>);

impl tla_core::VisitorOutput for Finding {
    fn combine(self, other: Self) -> Self {
        if self.0.is_some() {
            self
        } else {
            other
        }
    }
    fn is_terminal(&self) -> bool {
        self.0.is_some()
    }
}

/// The classifier visitor: walks pre-order and reports the first construct outside the
/// recognized fragment (see [`classify_unsupported`]).
struct UnsupportedClassifier<'a> {
    /// Declared CONSTANT names (an `Ident` hit here is an R1 reason).
    constants: BTreeSet<&'a str>,
    /// Top-level operator names (an `Ident`/`Apply` hit here means the recognizer would need
    /// operator expansion, which it does not perform).
    operators: BTreeSet<&'a str>,
}

/// Sequence-module operator names, flagged as R5 when applied.
const SEQ_OPS: &[&str] = &[
    "Len",
    "Head",
    "Tail",
    "Append",
    "SubSeq",
    "SelectSeq",
    "Seq",
];

impl tla_core::ExprVisitor for UnsupportedClassifier<'_> {
    type Output = Finding;

    fn visit_node(&mut self, expr: &Expr) -> Option<Finding> {
        let hit = |desc: String, tag: Option<&'static str>| Some(Finding(Some((desc, tag))));
        match expr {
            Expr::Forall(..) | Expr::Exists(..) => hit(
                "uses a quantifier fold over a general set".to_string(),
                Some("R4"),
            ),
            Expr::Choose(..) => hit("uses CHOOSE".to_string(), None),
            Expr::SetFilter(..) | Expr::SetBuilder(..) => {
                hit("uses a set comprehension".to_string(), Some("R4"))
            }
            Expr::Powerset(_) => hit("uses SUBSET (powerset)".to_string(), Some("R4")),
            Expr::BigUnion(_) => hit("uses UNION".to_string(), Some("R4")),
            Expr::In(..) | Expr::NotIn(..) | Expr::Subseteq(..) => hit(
                "uses set membership/⊆ over a non-bitmask-encodable set".to_string(),
                Some("R4"),
            ),
            Expr::Union(..) | Expr::Intersect(..) | Expr::SetMinus(..) => hit(
                "uses a set operation outside the bitmask fragment".to_string(),
                Some("R4"),
            ),
            Expr::FuncDef(..)
            | Expr::FuncApply(..)
            | Expr::Domain(_)
            | Expr::Except(..)
            | Expr::FuncSet(..) => hit(
                "uses function application/construction".to_string(),
                Some("R3"),
            ),
            Expr::Record(_) | Expr::RecordAccess(..) | Expr::RecordSet(_) => hit(
                "uses record construction/field access".to_string(),
                Some("R3"),
            ),
            Expr::Tuple(_) | Expr::Times(_) => {
                hit("uses tuple/sequence construction".to_string(), Some("R5"))
            }
            Expr::ModuleRef(..) | Expr::InstanceExpr(..) | Expr::SubstIn(..) => hit(
                "references an INSTANCE'd module operator".to_string(),
                Some("R6"),
            ),
            Expr::String(_) => hit(
                "uses a string literal (strings are not an encodable cell sort)".to_string(),
                Some("R1"),
            ),
            Expr::Always(_)
            | Expr::Eventually(_)
            | Expr::LeadsTo(..)
            | Expr::WeakFair(..)
            | Expr::StrongFair(..)
            | Expr::Enabled(_) => hit(
                "is temporal/action-level, not a state predicate".to_string(),
                None,
            ),
            Expr::If(..) => hit("uses IF/THEN/ELSE".to_string(), None),
            Expr::Case(..) => hit("uses CASE".to_string(), None),
            Expr::Let(..) => hit("uses LET".to_string(), None),
            Expr::Lambda(..) | Expr::OpRef(_) => {
                hit("uses a higher-order operator construct".to_string(), None)
            }
            Expr::Sub(..) | Expr::Neg(_) => hit(
                "uses integer subtraction/negation (only the narrowly-sound positive-equality \
                 form embeds)"
                    .to_string(),
                None,
            ),
            Expr::Div(..) | Expr::Pow(..) => hit(
                "uses real division / exponentiation (only +, *, \\div, % embed)".to_string(),
                None,
            ),
            Expr::Ident(name, _) => {
                if self.constants.contains(name.as_str()) {
                    hit(
                        format!("references CONSTANT `{name}` — constant-aware recognition"),
                        Some("R1"),
                    )
                } else if self.operators.contains(name.as_str()) {
                    hit(
                        format!(
                            "references helper operator `{name}` (the kernel recognizer reads \
                             the literal predicate body; no operator expansion)"
                        ),
                        None,
                    )
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn visit_apply(
        &mut self,
        op_expr: &tla_core::Spanned<Expr>,
        _args: &[tla_core::Spanned<Expr>],
    ) -> Option<Finding> {
        if let Expr::Ident(name, _) = &op_expr.node {
            if SEQ_OPS.contains(&name.as_str()) {
                return Some(Finding(Some((
                    format!("uses sequence operator `{name}`"),
                    Some("R5"),
                ))));
            }
            if self.operators.contains(name.as_str()) {
                return Some(Finding(Some((
                    format!(
                        "applies helper operator `{name}` (the kernel recognizer reads the \
                         literal predicate body; no operator expansion)"
                    ),
                    None,
                ))));
            }
        }
        None
    }
}

/// Classify WHY an expression falls outside the kernel predicate fragment: the first
/// (pre-order) unsupported construct, described, with the roadmap phase where one clearly
/// applies. `None` when nothing notable is found (the expression may still be unrecognized
/// for a subtler reason — fail-closed wording is the caller's job).
fn classify_unsupported(expr: &Expr, module: &Module) -> Option<(String, Option<&'static str>)> {
    let mut constants: BTreeSet<&str> = BTreeSet::new();
    let mut operators: BTreeSet<&str> = BTreeSet::new();
    for unit in &module.units {
        match &unit.node {
            Unit::Constant(decls) => {
                constants.extend(decls.iter().map(|d| d.name.node.as_str()));
            }
            Unit::Operator(op) => {
                operators.insert(op.name.node.as_str());
            }
            _ => {}
        }
    }
    let mut classifier = UnsupportedClassifier {
        constants,
        operators,
    };
    tla_core::walk_expr(&mut classifier, expr).0
}

// ═════════════════════════════════════════ tests ════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(src: &str) -> Config {
        Config::parse(src).expect("config parses")
    }

    /// A Seq-of-TUPLES state variable reports the cell-fragment reason NAMING the variable (the key
    /// per-column diagnostic), tagged with the sequence roadmap phase. (A seq-of-ATOMS is now ENCODABLE
    /// via the generalized `ColSort::Seq{elem}` and is deliberately NOT a wall — the probe's
    /// `seq_of_leaves_encodable` guard skips it; only a NESTED element — tuple/record — is still surfaced.)
    #[test]
    #[cfg(feature = "clean-cic")]
    fn seq_of_tuples_state_reports_cell_fragment_reason_naming_the_variable() {
        let src = "---- MODULE M ----\n\
                   VARIABLE waiting\n\
                   Init == waiting = <<<<1, 2>>>>\n\
                   Next == waiting' = waiting\n\
                   Safety == waiting = waiting\n\
                   ====\n";
        let config = cfg("INIT Init\nNEXT Next\nINVARIANT Safety\n");
        let reasons = explain_certify_decline(src, &config);
        let enc: Vec<&String> = reasons
            .iter()
            .filter(|r| r.starts_with("[state-encoding]"))
            .collect();
        assert_eq!(
            enc.len(),
            1,
            "expected exactly one encoding reason: {reasons:?}"
        );
        assert!(
            enc[0].contains("waiting:"),
            "names the variable: {}",
            enc[0]
        );
        // `<<<<1,2>>>>` is a sequence whose element is a Tuple — a NESTED (non-leaf) element ⇒ the shape
        // must surface the Tuple element inside the sequence/tuple pack diagnosis.
        assert!(
            enc[0].contains("Tuple"),
            "shows the element shape: {}",
            enc[0]
        );
        assert!(
            enc[0].contains("sequence") || enc[0].contains("Seq"),
            "names the sequence/tuple pack fragment: {}",
            enc[0]
        );
        assert!(
            enc[0].contains("R5"),
            "tags the sequence roadmap phase: {}",
            enc[0]
        );
    }

    /// A quantified invariant reports the per-invariant recognition reason with the
    /// quantifier roadmap tag.
    #[test]
    #[cfg(feature = "clean-cic")]
    fn quantified_invariant_reports_quantifier_reason() {
        let src = "---- MODULE M ----\n\
                   VARIABLE x\n\
                   Init == x = 0\n\
                   Next == x' = x\n\
                   Safety == \\A y \\in Nat : x <= y + x\n\
                   ====\n";
        let config = cfg("INIT Init\nNEXT Next\nINVARIANT Safety\n");
        let reasons = explain_certify_decline(src, &config);
        let inv: Vec<&String> = reasons
            .iter()
            .filter(|r| r.starts_with("[invariant]"))
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly one invariant reason: {reasons:?}"
        );
        assert!(
            inv[0].contains("Safety:"),
            "names the invariant: {}",
            inv[0]
        );
        assert!(
            inv[0].contains("quantifier"),
            "names the construct: {}",
            inv[0]
        );
        assert!(
            inv[0].contains("R4"),
            "tags the quantifier roadmap phase: {}",
            inv[0]
        );
    }

    /// A certifiable bounded counter passes every probed stage — the reason list is EMPTY
    /// (diagnostics say exactly nothing when there is nothing stage-level to say).
    #[test]
    #[cfg(feature = "clean-cic")]
    fn certifiable_counter_reports_no_reasons() {
        let src = "---- MODULE M ----\n\
                   VARIABLE x\n\
                   Init == x = 0\n\
                   Next == x' = x + 1 /\\ x < 3\n\
                   Safety == x >= 0\n\
                   ====\n";
        let config = cfg("INIT Init\nNEXT Next\nINVARIANT Safety\n");
        let reasons = explain_certify_decline(src, &config);
        assert!(
            reasons.is_empty(),
            "expected no stage-level reasons: {reasons:?}"
        );
    }

    /// An unconfigured CONSTANT is reported by NAME with the R1 tag (stage b), and the
    /// downstream enumeration failure names the same stage family.
    #[test]
    fn unbound_constant_reported_by_name() {
        let src = "---- MODULE M ----\n\
                   CONSTANT N\n\
                   VARIABLE x\n\
                   Init == x = N\n\
                   Next == x' = x\n\
                   Safety == x >= 0\n\
                   ====\n";
        let config = cfg("INIT Init\nNEXT Next\nINVARIANT Safety\n");
        let reasons = explain_certify_decline(src, &config);
        assert!(
            reasons
                .iter()
                .any(|r| r.starts_with("[constants]") && r.contains("`N`")),
            "expected a [constants] reason naming N: {reasons:?}"
        );
        assert!(
            reasons.iter().any(|r| r.contains("R1")),
            "expected the R1 roadmap tag: {reasons:?}"
        );
    }

    /// A missing INIT operator is a [config] reason, and probing stops there.
    #[test]
    fn missing_init_operator_is_a_config_reason() {
        let src = "---- MODULE M ----\n\
                   VARIABLE x\n\
                   Next == x' = x\n\
                   Safety == x >= 0\n\
                   ====\n";
        let config = cfg("INIT Init\nNEXT Next\nINVARIANT Safety\n");
        let reasons = explain_certify_decline(src, &config);
        assert_eq!(reasons.len(), 1, "{reasons:?}");
        assert!(reasons[0].starts_with("[config]"), "{}", reasons[0]);
        assert!(reasons[0].contains("`Init`"), "{}", reasons[0]);
    }
}
