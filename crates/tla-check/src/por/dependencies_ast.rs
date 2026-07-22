// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AST-level dependency extraction for canonical POR action analysis.
//!
//! # Fail-closed extraction (hole #3, 2026-07-07)
//!
//! Callers hand this walker an action body that went through
//! `expand_operators_with_primes`, but expansion is NOT total: the expander
//! deliberately declines to inline zero-arg FuncDef operators (the #2955 perf
//! guard), capture-unsafe applications, recursion re-entries, precomputed
//! constants, and builtin-overridden operators. The old walker silently
//! skipped that residue (`Expr::Ident(_, _) => {}` / `Expr::OpRef(_) => {}`),
//! extracting EMPTY deps from operator-shaped expressions — and
//! `IndependenceMatrix::compute` then landed the unanalyzable pairs on
//! INDEPENDENT, the unsound side (pinned false cleans:
//! `por_hidden_funcdef_dep_regression`, `por_hidden_capture_dep_regression`).
//!
//! This walker instead classifies every residual identifier and FAILS CLOSED:
//! anything it cannot prove benign marks the whole action OPAQUE
//! (`ActionDependencies::mark_opaque` — dependent on everything, visible to
//! every invariant). Benign residue is exactly:
//!
//! - **Bound variables** of enclosing binders (∀/∃/CHOOSE/set-builder/
//!   set-filter/function-def/LAMBDA/LET operator params), tracked with the
//!   same telescoping-domain scoping the expander uses.
//! - **LET operator names** in scope: the `Let` arm extracts every definition
//!   body's deps up front, so a residual reference to (or application of) a
//!   LET-defined operator adds nothing new.
//! - **State-variable identifiers** that resolve through the `EvalCtx` var
//!   registry (recorded as reads/writes, exactly like `Expr::StateVar`).
//! - **Concrete config `CONSTANT` values**. Operator replacements and
//!   expression-bearing values (closures/lazy functions/set predicates) are
//!   opaque because a bare token may execute later against live state.
//!   Operators folded by `precompute_constant_operators` are benign only when
//!   their stored result is concrete data; deferred expression values remain
//!   opaque.
//! - **Unshadowed builtin constant sets** (`Nat`, `Int`, `Real`, `Infinity`,
//!   `BOOLEAN`, `STRING`) and applications of name-dispatched PURE builtins
//!   (`Append`, `Len`, `Cardinality`, ...), which read state only through their
//!   walked arguments. Impure/context-dependent builtins (`TLCGet`/`TLCSet`,
//!   `IO*`, `Random*`, CSV/JSON deserialization, time) are opaque.
//! - **`INSTANCE ... WITH` substitution names** inside `SubstIn` bodies (their
//!   deps come from the walked substitution expressions).
//! - **The `@` old-value placeholder** inside an EXCEPT replacement value,
//!   whose reads are already covered by the EXCEPT base and path walked just
//!   before it (see `at_placeholder_is_bound`). An `@` anywhere else is
//!   unbound and opaque.
//!
//! Everything else — un-inlined operator references and applications, module
//! refs, instance expressions, primed non-variables, `UNCHANGED` over
//! non-variables, applications of bound (higher-order) operator parameters,
//! and identifiers that resolve to nothing — is opaque. When in doubt, opaque.
//!
//! # Definition-body resolution (2026-07-20) — DEFAULT-OFF since WP-25
//!
//! The largest source of opacity in practice is a RECURSIVE operator: the
//! expander refuses to inline a recursion re-entry, so `FindLeafNode(root,
//! key)` survives expansion and marks its whole action opaque even though
//! the definition body is right there and fully analyzable. Rather than give
//! up, `resolve_operator_body` walks the DEFINITION BODY in a fresh scope with
//! the formals bound, cutting call-graph cycles with an in-progress set. See
//! that function for the exactness argument and the list of shapes that still
//! fail closed (higher-order formals, `has_primed_param`, formals shadowing a
//! state-variable name, arity mismatch, depth/budget exhaustion).
//!
//! Both analyses are gated ([`ResolutionPolicy`]), but their DEFAULTS differ.
//! A more precise footprint is not only a POR input — it also widens the
//! hybrid native dispatcher's per-action admission — so each default is set
//! by measurement, not by symmetry with the other:
//!
//! - Operator-body resolution is **opt-in** (`TY_POR_RESOLVE_OPERATOR_BODIES=1`).
//!   Default-ON it routes `btree`'s `GetValue`/`UpdateReq` natively, and their
//!   compiled artifacts miscompile `ret'`; WP-25 turned it off.
//! - `@` resolution is **default-ON** (`TY_POR_RESOLVE_EXCEPT_AT=0` opts out).
//!   WP-25 disabled it as collateral in that same commit; WP-30 measured it
//!   alone as clean and restored it.
//!
//! `ResolutionPolicy::from_env` records the measured reason for each.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use tla_core::ast::{
    BoundVar, CaseArm, ExceptPathElement, ExceptSpec, Expr, ModuleTarget, OperatorDef,
};
use tla_core::{single_bound_var_names, FileId, Spanned};

use crate::eval::{has_complete_builtin_override, should_prefer_builtin_override, EvalCtx};
use crate::var_index::VarIndex;

use super::dependencies::ActionDependencies;

/// Extract dependencies from a canonical (expanded) action expression.
///
/// Fails closed: residue that cannot be analyzed marks the result opaque.
pub(super) fn extract_action_dependencies(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
) -> ActionDependencies {
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr(ctx, &expr.node, &mut deps);
    deps
}

/// Extract dependencies from an AST expression into `deps`.
///
/// `ctx` is consulted to classify residual identifiers (state variables,
/// config constants, precomputed constants, operator definitions). Residue
/// that cannot be proven benign marks `deps` opaque (fail closed).
pub(crate) fn extract_dependencies_ast_expr(
    ctx: &EvalCtx,
    expr: &Expr,
    deps: &mut ActionDependencies,
) {
    DepExtractor::new(ctx, ResolutionPolicy::from_env()).walk(expr, deps);
}

/// Extract dependencies under an EXPLICIT [`ResolutionPolicy`] instead of the
/// process default. Test-only: both arms of the two default-OFF gates are
/// pinned directly, without mutating process-wide env that other tests share.
#[cfg(test)]
pub(crate) fn extract_dependencies_ast_expr_with_policy(
    ctx: &EvalCtx,
    expr: &Expr,
    deps: &mut ActionDependencies,
    policy: ResolutionPolicy,
) {
    DepExtractor::new(ctx, policy).walk(expr, deps);
}

/// [`extract_action_dependencies`] under an explicit [`ResolutionPolicy`].
#[cfg(test)]
pub(crate) fn extract_action_dependencies_with_policy(
    ctx: &EvalCtx,
    expr: &Spanned<Expr>,
    policy: ResolutionPolicy,
) -> ActionDependencies {
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr_with_policy(ctx, &expr.node, &mut deps, policy);
    deps
}

/// Maximum NESTING depth of resolved operator bodies. Call-graph cycles are
/// already cut by the in-progress set; this bounds the non-cyclic chain so a
/// pathological definition tower degrades to opaque instead of blowing the
/// stack.
const MAX_OPERATOR_RESOLVE_DEPTH: usize = 32;

/// Maximum number of operator bodies walked per action extraction. Exhaustion
/// degrades to opaque (fail closed), never to a partial footprint.
const MAX_OPERATOR_RESOLVE_BODIES: usize = 512;

/// Which of the 2026-07-20 precision analyses this extraction may use.
///
/// WP-03a's operator-body resolution ships DEFAULT-OFF (WP-25); WP-13's `@`
/// resolution ships DEFAULT-ON (restored by WP-30). See
/// [`ResolutionPolicy::from_env`]. Carrying
/// them as data rather than reading the process env at each decision point is
/// what makes BOTH arms directly unit-testable: the OFF arm is the production
/// default surface, and pinning it needs a policy value, not a global env
/// mutation that would leak across tests sharing the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolutionPolicy {
    /// WP-03a: walk the DEFINITION BODY of an un-inlined operator application
    /// instead of failing closed (`TY_POR_RESOLVE_OPERATOR_BODIES=1`).
    operator_bodies: bool,
    /// WP-13: treat an `@` inside an EXCEPT replacement value as covered by the
    /// already-walked base and path (`TY_POR_RESOLVE_EXCEPT_AT=1`).
    except_at: bool,
}

impl ResolutionPolicy {
    /// Everything OFF — the pre-Wave-4 (2026-07-19) default surface: every
    /// un-inlined operator application and every `@` fails closed to opaque.
    #[cfg(test)]
    pub(crate) const OPAQUE: Self = Self {
        operator_bodies: false,
        except_at: false,
    };

    /// Everything ON — both precision analyses engaged.
    #[cfg(test)]
    pub(crate) const RESOLVED: Self = Self {
        operator_bodies: true,
        except_at: true,
    };

    /// The production policy, resolved ONCE per process.
    ///
    /// # Why WP-03a ships DEFAULT-OFF (WP-25) and WP-13 does not (WP-30)
    ///
    /// WP-03a (operator-body resolution) and WP-13 (`@` resolution) both make
    /// action FOOTPRINTS more precise, and a footprint is not only a POR input:
    /// the hybrid per-action native dispatcher
    /// (`check::model_checker::hybrid_dispatch`) admits an action iff its
    /// footprint is non-opaque and its writes are flat-admissible. Shipping
    /// them default-ON therefore changed the DEFAULT dispatch surface, which
    /// the campaign's invariant forbids — every change ships behind a
    /// default-OFF gate so the default surface stays byte-identical.
    ///
    /// That was not theoretical. On `test_specs/btree.tla` with the full hybrid
    /// gate set plus `TY_HYBRID_NATIVE_AUTHORITATIVE=1`, operator-body
    /// resolution lifts `GetValue` and `UpdateReq` out of opaque (hybrid
    /// eligible 6 -> 8) because their `LET ... == FindLeafNode(root, key)`
    /// survives expansion as a recursion re-entry. `GetValue`'s COMPILED
    /// artifact then miscomputes `ret'` on any parent whose tree has two
    /// levels, and the fail-closed differential fires:
    /// `native_unmatched_interp = native_residue = 15,093`,
    /// `mismatch_fallback = 30,186`, permanent authoritative fail-back. With
    /// the resolution OFF all of those are exactly 0. The miscompile is a
    /// lowering defect, NOT a footprint defect (the diverging variable `ret`
    /// IS in the action's declared write set, and the footprint this module
    /// computes for that shape is pinned exact by
    /// `test_btree_get_value_shape_resolves_exact_footprint`) — but until it is
    /// fixed, the default tree must not route those actions natively.
    ///
    /// That evidence indicts OPERATOR-BODY expansion specifically. WP-25
    /// nevertheless flipped `@` resolution off in the same commit, as
    /// collateral. WP-30 re-measured `@` resolution ALONE and restored it:
    /// `TY_POR_RESOLVE_EXCEPT_AT` is DEFAULT-ON again, `=0` opts out.
    /// `TY_POR_RESOLVE_OPERATOR_BODIES=1` opts back into WP-03a; `=0` (or
    /// unset) is its default. In-crate tests default to ON for both so the
    /// resolution rules keep their own coverage — only the production DEFAULT
    /// is the policy above.
    fn from_env() -> Self {
        static POLICY: std::sync::OnceLock<ResolutionPolicy> = std::sync::OnceLock::new();
        *POLICY.get_or_init(|| ResolutionPolicy {
            operator_bodies: gate_enabled("TY_POR_RESOLVE_OPERATOR_BODIES"),
            except_at: gate_enabled_default_on("TY_POR_RESOLVE_EXCEPT_AT"),
        })
    }
}

/// Read one default-OFF opt-in gate: `1` on, `0` (or unset) off, in-crate
/// tests on.
fn gate_enabled(name: &str) -> bool {
    match std::env::var(name).as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => cfg!(test),
    }
}

/// Read one default-ON gate: `0` off, `1` (or unset) on.
///
/// WP-30 restores WP-13 (`@` resolution) to default-ON after WP-25 flipped it
/// off as collateral in the commit that correctly disabled WP-03a. The two
/// analyses are independent: WP-03a's miscompile evidence above is entirely
/// about `GetValue`/`UpdateReq` being lifted out of opaque by OPERATOR-BODY
/// expansion. `@` resolution touches neither action, and the single-gate A/B
/// (5 authoritative `btree` runs, WP-30) measured it CLEAN — every divergence
/// counter 0, no authoritative fail-back, state counts and POR anchors exact.
fn gate_enabled_default_on(name: &str) -> bool {
    !matches!(std::env::var(name).as_deref(), Ok("0"))
}

/// Builtin constant sets that survive expansion as bare identifiers and are
/// genuinely state-independent.
const BUILTIN_CONSTANT_SETS: [&str; 6] = ["Nat", "Int", "Real", "Infinity", "BOOLEAN", "STRING"];

/// Name-dispatched builtins that are impure or context-dependent: their value
/// is not a pure function of their (walked) arguments, so an application must
/// mark the action opaque. Pure builtins (`Append`, `Len`, `Cardinality`,
/// `SubSeq`, folds, ...) read state only through arguments and stay benign.
fn is_impure_or_context_dependent_builtin(name: &str) -> bool {
    name.starts_with("IO")
        || name.starts_with("Random")
        || matches!(
            name,
            "TLCGet"
                | "TLCSet"
                | "Print"
                | "PrintT"
                | "Assert"
                | "AssertEq"
                | "AssertError"
                | "TLCGetOrDefault"
                | "TLCGetAndSet"
                | "TLCEval"
                | "TLCEvalDefinition"
                | "PickSuccessor"
                | "Trace"
                | "CounterExample"
                | "CSVRead"
                | "CSVWrite"
                | "CSVRecords"
                | "JsonSerialize"
                | "JsonDeserialize"
                | "ndJsonSerialize"
                | "ndJsonDeserialize"
                | "Serialize"
                | "Deserialize"
                | "SystemTime"
                | "JavaTime"
        )
}

#[cfg(test)]
mod impurity_tests {
    use super::is_impure_or_context_dependent_builtin;

    #[test]
    fn dynamic_and_contextual_tlcext_builtins_fail_closed() {
        for name in [
            "TLCEvalDefinition",
            "TLCGetOrDefault",
            "TLCGetAndSet",
            "Print",
            "PrintT",
            "Assert",
            "AssertEq",
            "AssertError",
            "PickSuccessor",
            "Trace",
            "CounterExample",
        ] {
            assert!(
                is_impure_or_context_dependent_builtin(name),
                "{name} must never produce a reusable pure-state projection"
            );
        }
    }
}

/// Scope-tracking, ctx-aware dependency walker.
///
/// Mirrors the scoping rules of `enumerate::expand_ops::ExpandOperators`
/// (telescoping binder domains, LET shadowing) so that the post-expansion
/// residue it sees is classified under the same name resolution the expander
/// (and the evaluator) use.
struct DepExtractor<'a> {
    ctx: &'a EvalCtx,
    /// Multiset of names bound by enclosing binders (quantifiers, CHOOSE,
    /// set-builders/filters, function defs, LAMBDA params, LET operator
    /// params). A bare reference is benign; an APPLICATION is opaque
    /// (higher-order operator parameter with unknowable behavior).
    bound: FxHashMap<String, u32>,
    /// Multiset of names whose dependencies are already fully accounted for
    /// in `deps`: LET operator names (definition bodies walked at the `Let`
    /// arm) and `SubstIn` substitution names (substituted expressions walked
    /// at the `SubstIn` arm). Both bare references and applications are
    /// benign.
    covered: FxHashMap<String, u32>,
    /// Number of enclosing EXCEPT replacement-value scopes that bind the `@`
    /// old-value placeholder. Zero means a residual `@` is unbound and fails
    /// closed. See `at_placeholder_is_bound` for the exactness argument.
    except_value_depth: u32,
    /// Global operators whose bodies are on the current resolution stack. A
    /// re-entry is a call-graph cycle: the body is already being accumulated
    /// into the same `deps` further up, so the re-entry contributes only its
    /// (separately walked) arguments.
    resolving: FxHashSet<String>,
    /// Global operators whose bodies are already fully walked into `deps`.
    /// Body contributions are scope-independent (walked in a fresh definition
    /// scope), so a later call site adds nothing beyond its arguments.
    resolved: FxHashSet<String>,
    /// Remaining body-walk budget for this extraction.
    resolve_budget: usize,
    /// Which precision analyses this extraction may use (WP-25: both are
    /// DEFAULT-OFF opt-in gates).
    policy: ResolutionPolicy,
}

impl<'a> DepExtractor<'a> {
    fn new(ctx: &'a EvalCtx, policy: ResolutionPolicy) -> Self {
        Self {
            ctx,
            bound: FxHashMap::default(),
            covered: FxHashMap::default(),
            except_value_depth: 0,
            resolving: FxHashSet::default(),
            resolved: FxHashSet::default(),
            resolve_budget: MAX_OPERATOR_RESOLVE_BODIES,
            policy,
        }
    }

    fn push_name(map: &mut FxHashMap<String, u32>, name: &str) {
        *map.entry(name.to_string()).or_insert(0) += 1;
    }

    fn pop_name(map: &mut FxHashMap<String, u32>, name: &str) {
        if let Some(count) = map.get_mut(name) {
            if *count <= 1 {
                map.remove(name);
            } else {
                *count -= 1;
            }
        }
    }

    /// Resolve a global operator definition the way the expander does.
    fn resolve_global_op(&self, name: &str) -> Option<&tla_core::ast::OperatorDef> {
        let resolved = self.ctx.resolve_op_name(name);
        self.ctx.get_op(resolved).map(|def| def.as_ref())
    }

    /// Resolve a global operator definition, keeping it alive independently of
    /// the `&mut self` borrow taken while walking its body.
    fn resolve_global_op_arc(&self, name: &str) -> Option<Arc<OperatorDef>> {
        let resolved = self.ctx.resolve_op_name(name);
        self.ctx.get_op(resolved).cloned()
    }

    /// Account for an un-inlined reference/application of global operator
    /// `name` by walking its DEFINITION BODY into `deps`.
    ///
    /// Returns `false` when the definition cannot be resolved EXACTLY; the
    /// caller must then mark the action opaque (fail closed). This analysis is
    /// shared with POR, where an under-approximated footprint prunes real
    /// transitions, so every branch below that cannot be discharged returns
    /// `false` rather than guessing.
    ///
    /// # Exactness
    ///
    /// For `Op(p_1, .., p_n) == body`, the application `Op(a_1, .., a_n)`
    /// evaluates `body` under `p_i := a_i`, so its footprint is exactly
    /// `footprint(body with the p_i unknown) ∪ ⋃_i footprint(a_i)`. The caller
    /// (`walk_apply`) walks every `a_i` at the call site; this walk supplies
    /// the first term by binding the formals, which makes a bare `p_i`
    /// contribute nothing and an APPLICATION of `p_i` opaque.
    ///
    /// The body is walked in a FRESH scope. A definition body is resolved in
    /// its own module scope, not the caller's, so leaving the caller's binder
    /// and LET names in scope would silently absolve a body identifier that
    /// merely shares a caller-local name — the unsound direction.
    ///
    /// # Termination
    ///
    /// `resolving` cuts every call-graph cycle after one walk of each body in
    /// the cycle, which is the least fixpoint: `deps` is a single monotone
    /// accumulator, so a cycle re-entry has nothing left to add beyond the
    /// arguments the caller walks. `MAX_OPERATOR_RESOLVE_DEPTH` bounds nesting
    /// and `resolve_budget` bounds total work; both degrade to opaque.
    fn resolve_operator_body(
        &mut self,
        name: &str,
        def: &OperatorDef,
        argc: usize,
        deps: &mut ActionDependencies,
    ) -> bool {
        if !self.policy.operator_bodies {
            return false;
        }
        if self.resolving.contains(name) || self.resolved.contains(name) {
            return true;
        }
        if def.params.len() != argc {
            return false;
        }
        if def.params.iter().any(|p| p.arity != 0) {
            // Higher-order definition: the body applies an operator supplied by
            // the caller, whose footprint is not this body's.
            return false;
        }
        if def.has_primed_param {
            // Call-by-name substitution: `p'` in the body primes the ARGUMENT,
            // so the write set belongs to the call site, not the body.
            return false;
        }
        if def
            .params
            .iter()
            .any(|p| self.ctx.var_registry().get(&p.name.node).is_some())
        {
            // A formal shadowing a state-variable name would be resolved as
            // that variable by `walk_prime`/`walk_unchanged`, which consult the
            // registry without consulting binder scope.
            return false;
        }
        if self.resolving.len() >= MAX_OPERATOR_RESOLVE_DEPTH || self.resolve_budget == 0 {
            return false;
        }
        self.resolve_budget -= 1;

        let saved_bound = std::mem::take(&mut self.bound);
        let saved_covered = std::mem::take(&mut self.covered);
        // A definition body is not lexically inside the caller's EXCEPT, so a
        // stray `@` in it is unbound here and must fail closed.
        let saved_except_value_depth = std::mem::take(&mut self.except_value_depth);
        for p in &def.params {
            Self::push_name(&mut self.bound, &p.name.node);
        }
        self.resolving.insert(name.to_string());

        self.walk(&def.body.node, deps);

        self.resolving.remove(name);
        self.bound = saved_bound;
        self.covered = saved_covered;
        self.except_value_depth = saved_except_value_depth;
        self.resolved.insert(name.to_string());
        true
    }

    /// Classify a zero-arg operator folded by
    /// `precompute_constant_operators`. `Some(true)` is concrete data;
    /// `Some(false)` is a deferred expression value (Closure/LazyFunc/SetPred)
    /// that may consult live state or runtime context when consumed later.
    fn precomputed_constant_is_concrete(&self, name: &str) -> Option<bool> {
        use tla_core::name_intern::lookup_name_id;
        lookup_name_id(name)
            .and_then(|id| self.ctx.precomputed_constants().get(&id))
            .map(|value| value.is_concrete_data())
    }

    /// Whether a residual `@` is bound by an enclosing EXCEPT replacement
    /// value, and therefore contributes nothing new to `deps`.
    ///
    /// # Exactness
    ///
    /// `@` denotes the OLD value at the enclosing spec's path, i.e.
    /// `base[i_1]..[i_n]`, so `footprint(@) ⊆ footprint(base) ∪ ⋃_k
    /// footprint(i_k)`. The `Expr::Except` arm walks `base` and every `i_k`
    /// into the SAME `deps` accumulator BEFORE it enters `value`, and `deps`
    /// only grows — so by the time any `@` in `value` is reached, everything
    /// `@` could read is already recorded. This holds for every spec in the
    /// list (each spec's own path is walked immediately before its value) and
    /// for multi-element and record-field paths (a `Field` carries no
    /// expression).
    ///
    /// Nesting is handled by construction rather than by resolving which level
    /// `@` binds to: in `[f EXCEPT ![i] = [g EXCEPT ![j] = @]]` both candidate
    /// levels (`f`,`i` and `g`,`j`) are fully walked before the inner value, so
    /// the conclusion is the same whichever one binds. `Field`/`Index` scoping
    /// still matters for the DEPTH bookkeeping, which mirrors
    /// `SubstituteAt::fold_except_specs`: base and path indices stay at the
    /// enclosing level, only `value` opens a new one.
    ///
    /// # Fail-closed
    ///
    /// Depth zero means no enclosing EXCEPT value was walked into this `deps`,
    /// so nothing bounds what `@` reads — the caller marks the action opaque.
    /// `resolve_operator_body` resets the depth for the same reason: a
    /// definition body is walked in a fresh scope, and a stray `@` there is not
    /// bound by the caller's EXCEPT.
    fn at_placeholder_is_bound(&self) -> bool {
        self.policy.except_at && self.except_value_depth > 0
    }

    /// Classify a BARE residual identifier (not an application head).
    fn classify_bare_ident(&mut self, name: &str, deps: &mut ActionDependencies) {
        if name == "@" {
            if self.at_placeholder_is_bound() {
                return;
            }
            deps.mark_opaque(
                "`@` old-value placeholder outside an EXCEPT replacement value".to_string(),
            );
            return;
        }
        if self.bound.contains_key(name) || self.covered.contains_key(name) {
            return; // binder-scoped variable / LET-covered operator
        }
        if let Some(var) = self.ctx.var_registry().get(name) {
            // State variable that was not rewritten to Expr::StateVar —
            // record the read instead of losing it.
            deps.add_read(var);
            return;
        }
        if self.ctx.op_replacements().contains_key(name) {
            // A bare operator token can be executed later by a higher-order
            // builtin (FoldFunction, RelationUnder, ...). Config replacement
            // bodies are not represented by the token's walked arguments.
            deps.mark_opaque(format!(
                "bare config-substituted operator `{name}` may be invoked dynamically"
            ));
            return;
        }
        if self.ctx.is_config_constant(name) {
            // Ordinary config values are state-independent, but the full
            // expression fallback also accepts LAMBDA values. Such a closure
            // can read the caller's state when a higher-order builtin invokes
            // this bare token. Only concrete data is therefore certifiably
            // benign here.
            let is_concrete = self
                .ctx
                .lookup(name)
                .map(|value| value.is_concrete_data())
                // Normal setup promotes env constants before rebuilding this
                // analysis, and the ident-hint fast path may then bypass env
                // lookup. Inspect the authoritative promoted value as well.
                .or_else(|| self.precomputed_constant_is_concrete(name))
                .unwrap_or(false);
            if is_concrete {
                return;
            }
            deps.mark_opaque(format!(
                "non-concrete config constant `{name}` may contain executable state-dependent data"
            ));
            return;
        }
        if let Some(is_concrete) = self.precomputed_constant_is_concrete(name) {
            if is_concrete {
                return;
            }
            deps.mark_opaque(format!(
                "non-concrete precomputed operator `{name}` may execute against live state"
            ));
            return;
        }
        // Operator-shaped or unknown residue: the expander declined to inline
        // it (zero-arg FuncDef #2955 guard, self-referential FuncDef,
        // recursion re-entry) or nothing resolves it. A NULLARY definition is
        // just a value, so walking its body is exact; a definition with
        // formals reached as a bare token can be applied later to arguments
        // this walk never sees, so it stays opaque.
        if let Some(def) = self.resolve_global_op_arc(name) {
            // Shared zero-arg operators normally win over builtin constants at
            // runtime. Preserve the genuine stdlib override only when the
            // evaluator's authoritative dispatch predicate says that this
            // exact definition is replaced by the builtin. In particular, do
            // not absolve a state-dependent user `Nat == FuncDef(...)` merely
            // because its spelling is also a builtin constant set.
            if BUILTIN_CONSTANT_SETS.contains(&name)
                && should_prefer_builtin_override(name, &def, 0, self.ctx)
            {
                return;
            }
            if def.params.is_empty() && self.resolve_operator_body(name, &def, 0, deps) {
                return;
            }
        } else if BUILTIN_CONSTANT_SETS.contains(&name) {
            return;
        }
        deps.mark_opaque(format!("un-inlined operator/unknown identifier `{name}`"));
    }

    /// Classify a bare operator reference (`Expr::OpRef`): builtin operator
    /// SYMBOLS (`+`, `\cup`, ...) are pure and benign; anything that resolves
    /// to an operator definition is un-inlined residue and opaque.
    fn classify_op_ref(&self, name: &str, deps: &mut ActionDependencies) {
        if name == "@" {
            // Not a builtin operator symbol: an `@` reaching first-class
            // operator position is unmodelled residue.
            deps.mark_opaque("`@` old-value placeholder as an operator reference".to_string());
            return;
        }
        if self.bound.contains_key(name) || self.covered.contains_key(name) {
            return;
        }
        if self.ctx.op_replacements().contains_key(name) {
            deps.mark_opaque(format!(
                "config-substituted first-class operator reference `{name}`"
            ));
            return;
        }
        if is_impure_or_context_dependent_builtin(name) {
            deps.mark_opaque(format!(
                "impure/context-dependent first-class builtin `{name}`"
            ));
            return;
        }
        if self.resolve_global_op(name).is_some() {
            deps.mark_opaque(format!("un-inlined operator reference `{name}`"));
        }
        // else: builtin operator symbol — pure; reads state only through the
        // arguments supplied at its (walked) application sites.
    }

    /// Walk an operator application, classifying the head fail-closed.
    fn walk_apply(
        &mut self,
        op: &Spanned<Expr>,
        args: &[Spanned<Expr>],
        deps: &mut ActionDependencies,
    ) {
        let head: Option<&str> = match &op.node {
            Expr::Ident(name, _) => Some(name.as_str()),
            Expr::OpRef(name) => Some(name.as_str()),
            _ => None,
        };
        match head {
            Some(name) => {
                if name == "@" {
                    // `@` is a VALUE, not an operator; `@[k]` lowers to
                    // `FuncApply`, so an operator-application head spelled `@`
                    // is a shape this walker does not model.
                    deps.mark_opaque(
                        "`@` old-value placeholder applied as an operator".to_string(),
                    );
                } else if self.covered.contains_key(name) {
                    // LET-defined operator: its body's deps were extracted at
                    // the Let arm (params scoped), so the application adds
                    // only the argument deps walked below.
                } else if self.bound.contains_key(name) {
                    // Higher-order operator PARAMETER applied — behavior
                    // unknowable at this site.
                    deps.mark_opaque(format!("application of bound operator parameter `{name}`"));
                } else if self.ctx.op_replacements().contains_key(name) {
                    // Runtime resolves the target before dispatch. Classifying
                    // only the source spelling can miss a context-dependent
                    // builtin target such as `Get <- TLCGet`.
                    deps.mark_opaque(format!(
                        "application of config-substituted operator `{name}`"
                    ));
                } else if self.ctx.is_config_constant(name) {
                    // Config values shadow same-named global definitions at
                    // runtime. An applied value may be an executable Closure.
                    deps.mark_opaque(format!("application of config-bound value `{name}`"));
                } else if let Some(def) = self.resolve_global_op_arc(name) {
                    // A definition exists but the expander declined to inline
                    // the call (capture-unsafe, recursion re-entry, builtin
                    // override kept the token). Benign when the runtime
                    // provably dispatches a complete pure Rust builtin instead
                    // of the definition body (module-scoped overrides are
                    // shadowed by same-named MAIN-module definitions, so
                    // require the def to come from a library module) —
                    // otherwise walk the definition body itself, and fail
                    // closed only if that cannot be done exactly.
                    let builtin_replaces_def = {
                        let resolved = self.ctx.resolve_op_name(name);
                        has_complete_builtin_override(resolved, args.len(), self.ctx)
                            && def.name.span.file != FileId(0)
                    };
                    if !builtin_replaces_def
                        && !self.resolve_operator_body(name, &def, args.len(), deps)
                    {
                        deps.mark_opaque(format!("un-inlined operator application `{name}`"));
                    }
                } else if is_impure_or_context_dependent_builtin(name) {
                    deps.mark_opaque(format!("impure/context-dependent builtin `{name}`"));
                }
                // else: name-dispatched PURE builtin (Append, Len,
                // Cardinality, ...) — reads state only through the walked
                // arguments — or an undefined operator, which fails
                // evaluation LOUDLY at runtime (never a silent false clean).
            }
            None => self.walk(&op.node, deps),
        }
        for arg in args {
            self.walk(&arg.node, deps);
        }
    }

    /// Walk a multi-bound binder with the expander's TELESCOPING scope:
    /// each domain is walked with all EARLIER bound names in scope, then the
    /// body is walked with every bound name in scope.
    fn walk_telescoping_binder(
        &mut self,
        bounds: &[BoundVar],
        body: &Spanned<Expr>,
        deps: &mut ActionDependencies,
    ) {
        let mut pushed: Vec<String> = Vec::new();
        for bound in bounds {
            if let Some(domain) = &bound.domain {
                self.walk(&domain.node, deps);
            }
            for name in single_bound_var_names(bound) {
                Self::push_name(&mut self.bound, &name);
                pushed.push(name);
            }
        }
        self.walk(&body.node, deps);
        for name in &pushed {
            Self::pop_name(&mut self.bound, name);
        }
    }

    /// Walk a primed expression. Only primes over state variables (and
    /// tuples thereof) have statically knowable write sets; anything else
    /// is opaque (fail closed — e.g. `f[i]' = e` partially writes `f`).
    fn walk_prime(&mut self, inner: &Spanned<Expr>, deps: &mut ActionDependencies) {
        match &inner.node {
            Expr::StateVar(_, idx, _) => deps.add_write(VarIndex(*idx)),
            Expr::Ident(name, _) => {
                if let Some(var) = self.ctx.var_registry().get(name) {
                    deps.add_write(var);
                } else {
                    deps.mark_opaque(format!("primed non-variable expression `{name}'`"));
                    self.walk(&inner.node, deps);
                }
            }
            Expr::Tuple(elems) | Expr::Times(elems) => {
                for elem in elems {
                    self.walk_prime(elem, deps);
                }
            }
            Expr::Label(label) => self.walk_prime(&label.body, deps),
            _ => {
                deps.mark_opaque(
                    "primed compound expression — written variables unknowable".to_string(),
                );
                self.walk(&inner.node, deps);
            }
        }
    }

    /// Walk an UNCHANGED target. `UNCHANGED x` / `UNCHANGED <<x, y>>` are
    /// identity writes; anything else (e.g. `UNCHANGED f[i]`, an un-inlined
    /// operator) is opaque — the preserved/constrained variables are
    /// unknowable (fail closed).
    fn walk_unchanged(&mut self, expr: &Expr, deps: &mut ActionDependencies) {
        match expr {
            Expr::StateVar(_, idx, _) => {
                // Only record as identity write. Do NOT add_read — the "read"
                // in UNCHANGED x is vacuous for commutativity purposes.
                deps.add_unchanged(VarIndex(*idx));
            }
            Expr::Ident(name, _) => {
                if let Some(var) = self.ctx.var_registry().get(name) {
                    deps.add_unchanged(var);
                } else {
                    deps.mark_opaque(format!("UNCHANGED over unresolved identifier `{name}`"));
                }
            }
            Expr::Tuple(items) | Expr::Times(items) => {
                for item in items {
                    self.walk_unchanged(&item.node, deps);
                }
            }
            Expr::Label(label) => self.walk_unchanged(&label.body.node, deps),
            other => {
                deps.mark_opaque(
                    "UNCHANGED over non-variable expression — preserved variables unknowable"
                        .to_string(),
                );
                self.walk(other, deps);
            }
        }
    }

    /// Extract state variable reads from condition expressions in branching
    /// constructs (IF/CASE) that were classified as identity-preserving.
    ///
    /// When `x' = IF cond THEN x ELSE x` is detected as an identity write,
    /// we still need to track any state variable reads in `cond` because
    /// the guard expression observes state. Part of #3993 Phase 11.
    fn walk_condition_reads_from_branching(
        &mut self,
        expr: &Spanned<Expr>,
        deps: &mut ActionDependencies,
    ) {
        match &expr.node {
            Expr::If(cond, then_e, else_e) => {
                self.walk(&cond.node, deps);
                self.walk_condition_reads_from_branching(then_e, deps);
                self.walk_condition_reads_from_branching(else_e, deps);
            }
            Expr::Case(arms, other) => {
                for CaseArm { guard, body } in arms {
                    self.walk(&guard.node, deps);
                    self.walk_condition_reads_from_branching(body, deps);
                }
                if let Some(other) = other {
                    self.walk_condition_reads_from_branching(other, deps);
                }
            }
            Expr::Let(defs, body) => {
                for def in defs {
                    self.walk(&def.body.node, deps);
                }
                self.walk_condition_reads_from_branching(body, deps);
            }
            Expr::Label(label) => {
                self.walk_condition_reads_from_branching(&label.body, deps);
            }
            // Terminal identity expression (StateVar) — no additional reads.
            _ => {}
        }
    }

    fn walk(&mut self, expr: &Expr, deps: &mut ActionDependencies) {
        match expr {
            Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => {}
            Expr::Ident(name, _) => self.classify_bare_ident(name, deps),
            Expr::StateVar(_, idx, _) => {
                deps.add_read(VarIndex(*idx));
            }
            Expr::OpRef(name) => self.classify_op_ref(name, deps),

            Expr::Apply(op, args) => self.walk_apply(op, args, deps),
            Expr::ModuleRef(target, _, args) => {
                // An un-inlined `M!Op(...)` reference: the instanced
                // operator's body is not visible here (fail closed).
                deps.mark_opaque("un-inlined module operator reference".to_string());
                match target {
                    ModuleTarget::Parameterized(_, params) => {
                        for p in params {
                            self.walk(&p.node, deps);
                        }
                    }
                    ModuleTarget::Chained(base) => {
                        self.walk(&base.node, deps);
                    }
                    ModuleTarget::Named(_) => {}
                }
                for arg in args {
                    self.walk(&arg.node, deps);
                }
            }
            Expr::InstanceExpr(_, subs) => {
                deps.mark_opaque("INSTANCE expression".to_string());
                for sub in subs {
                    self.walk(&sub.to.node, deps);
                }
            }
            Expr::SubstIn(subs, body) => {
                for sub in subs {
                    self.walk(&sub.to.node, deps);
                }
                // Substituted names inside the body are covered by the
                // substitution expressions walked above.
                let names: Vec<&str> = subs.iter().map(|s| s.from.node.as_str()).collect();
                for name in &names {
                    Self::push_name(&mut self.covered, name);
                }
                self.walk(&body.node, deps);
                for name in &names {
                    Self::pop_name(&mut self.covered, name);
                }
            }
            Expr::Lambda(params, body) => {
                for p in params {
                    Self::push_name(&mut self.bound, &p.node);
                }
                self.walk(&body.node, deps);
                for p in params {
                    Self::pop_name(&mut self.bound, &p.node);
                }
            }

            Expr::And(left, right)
            | Expr::Or(left, right)
            | Expr::Implies(left, right)
            | Expr::Equiv(left, right) => {
                self.walk(&left.node, deps);
                self.walk(&right.node, deps);
            }
            Expr::Not(inner) => {
                self.walk(&inner.node, deps);
            }

            Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
                self.walk_telescoping_binder(bounds, body, deps);
            }
            Expr::Choose(bound, body) => {
                self.walk_telescoping_binder(std::slice::from_ref(bound), body, deps);
            }

            Expr::SetEnum(elems) => {
                for e in elems {
                    self.walk(&e.node, deps);
                }
            }
            Expr::SetBuilder(body, bounds) => {
                self.walk_telescoping_binder(bounds, body, deps);
            }
            Expr::SetFilter(bound, body) => {
                self.walk_telescoping_binder(std::slice::from_ref(bound), body, deps);
            }
            Expr::In(left, right)
            | Expr::NotIn(left, right)
            | Expr::Subseteq(left, right)
            | Expr::Union(left, right)
            | Expr::Intersect(left, right)
            | Expr::SetMinus(left, right) => {
                self.walk(&left.node, deps);
                self.walk(&right.node, deps);
            }
            Expr::Powerset(inner) | Expr::BigUnion(inner) => {
                self.walk(&inner.node, deps);
            }

            Expr::FuncDef(bounds, body) => {
                self.walk_telescoping_binder(bounds, body, deps);
            }
            Expr::FuncApply(func, arg) => {
                self.walk(&func.node, deps);
                self.walk(&arg.node, deps);
            }
            Expr::Domain(inner) => {
                self.walk(&inner.node, deps);
            }
            Expr::Except(base, specs) => {
                // The base and the path index expressions sit at the ENCLOSING
                // `@` level (`SubstituteAt::fold_except_specs` folds them but
                // not `value`); only `value` opens a new binding level. Walking
                // them first is what makes `@` inside `value` free — see
                // `at_placeholder_is_bound`.
                self.walk(&base.node, deps);
                for ExceptSpec { path, value } in specs {
                    for elem in path {
                        if let ExceptPathElement::Index(idx) = elem {
                            self.walk(&idx.node, deps);
                        }
                    }
                    self.except_value_depth += 1;
                    self.walk(&value.node, deps);
                    self.except_value_depth -= 1;
                }
            }
            Expr::FuncSet(domain, range) => {
                self.walk(&domain.node, deps);
                self.walk(&range.node, deps);
            }

            Expr::Record(fields) | Expr::RecordSet(fields) => {
                for (_, value) in fields {
                    self.walk(&value.node, deps);
                }
            }
            Expr::RecordAccess(record, _) => {
                self.walk(&record.node, deps);
            }

            Expr::Tuple(elems) | Expr::Times(elems) => {
                for e in elems {
                    self.walk(&e.node, deps);
                }
            }

            Expr::Prime(inner) => self.walk_prime(inner, deps),
            Expr::Always(inner) | Expr::Eventually(inner) => {
                self.walk(&inner.node, deps);
            }
            Expr::LeadsTo(left, right)
            | Expr::WeakFair(left, right)
            | Expr::StrongFair(left, right) => {
                self.walk(&left.node, deps);
                self.walk(&right.node, deps);
            }

            Expr::Enabled(inner) => {
                self.walk(&inner.node, deps);
            }
            Expr::Unchanged(inner) => {
                self.walk_unchanged(&inner.node, deps);
            }

            Expr::If(cond, then_e, else_e) => {
                self.walk(&cond.node, deps);
                self.walk(&then_e.node, deps);
                self.walk(&else_e.node, deps);
            }
            Expr::Case(arms, other) => {
                for CaseArm { guard, body } in arms {
                    self.walk(&guard.node, deps);
                    self.walk(&body.node, deps);
                }
                if let Some(other) = other {
                    self.walk(&other.node, deps);
                }
            }
            Expr::Let(defs, body) => {
                // All LET definition names are in scope for every definition
                // body (mutual/forward references) and for the LET body. The
                // definition bodies' deps are extracted here up front, which
                // is what makes residual references to these names benign
                // (a sound over-approximation for definitions that are never
                // actually referenced).
                let names: Vec<&str> = defs.iter().map(|d| d.name.node.as_str()).collect();
                for name in &names {
                    Self::push_name(&mut self.covered, name);
                }
                for def in defs {
                    let params: Vec<&str> =
                        def.params.iter().map(|p| p.name.node.as_str()).collect();
                    for p in &params {
                        Self::push_name(&mut self.bound, p);
                    }
                    self.walk(&def.body.node, deps);
                    for p in &params {
                        Self::pop_name(&mut self.bound, p);
                    }
                }
                self.walk(&body.node, deps);
                for name in &names {
                    Self::pop_name(&mut self.covered, name);
                }
            }

            Expr::Eq(left, right) => {
                // Detect identity assignments: `x' = x` where both sides refer
                // to the same state variable. This is semantically equivalent to
                // `UNCHANGED x` — the identity function — and commutes with all
                // operations. Record as an identity write instead of a real write
                // to enable POR for specs that use explicit `x' = x` instead of
                // `UNCHANGED x`. Part of #3993.
                if let Some(identity_var) = detect_identity_assignment(left, right) {
                    deps.add_unchanged(identity_var);
                }
                // Part of #3993 Phase 11: detect identity through IF/THEN/ELSE and CASE.
                // `x' = IF cond THEN x ELSE x` is equivalent to UNCHANGED x.
                else if let Some(identity_var) = detect_identity_through_if(left, right) {
                    // The condition expression may read state variables — track those reads.
                    self.walk_condition_reads_from_branching(right, deps);
                    deps.add_unchanged(identity_var);
                }
                // Part of #3993 Phase 11: detect EXCEPT identity pattern.
                // `f' = [f EXCEPT ![k] = f[k]]` means the action reads f[k] and
                // writes back the same value — a no-op on f. Treat as identity.
                else if let Some(identity_var) = detect_except_identity(left, right) {
                    deps.add_unchanged(identity_var);
                }
                // Part of #3993 Phase 11: constant write detection.
                // `x' = 0` (state-independent RHS) means the action writes x but
                // does NOT read x. Record as a write without an implicit read.
                else if let Some(write_var) = detect_constant_write(left, right) {
                    deps.add_write(write_var);
                    // Do NOT add a read for the written variable — the value is constant.
                } else {
                    self.walk(&left.node, deps);
                    self.walk(&right.node, deps);
                }
            }
            Expr::Neq(left, right)
            | Expr::Lt(left, right)
            | Expr::Leq(left, right)
            | Expr::Gt(left, right)
            | Expr::Geq(left, right)
            | Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::IntDiv(left, right)
            | Expr::Mod(left, right)
            | Expr::Pow(left, right)
            | Expr::Range(left, right) => {
                self.walk(&left.node, deps);
                self.walk(&right.node, deps);
            }
            Expr::Neg(inner) => {
                self.walk(&inner.node, deps);
            }

            Expr::Label(label) => {
                self.walk(&label.body.node, deps);
            }
        }
    }
}

/// Detect the identity assignment pattern `x' = x` (or `x = x'`).
///
/// Returns `Some(VarIndex)` when the equality compares a primed state variable
/// with the same unprimed state variable — meaning the action preserves that
/// variable's value. This is semantically equivalent to `UNCHANGED x`.
///
/// The `EXCEPT` identity pattern (`f' = [f EXCEPT ![k] = f[k]]`) is handled
/// separately by `detect_except_identity`. This function only handles the
/// simple `x' = x` scalar case.
///
/// Part of #3993: many TLA+ specs use explicit `x' = x` instead of `UNCHANGED x`.
/// Without this detection, POR treats them as real writes, making virtually all
/// actions dependent and defeating reduction.
fn detect_identity_assignment(left: &Spanned<Expr>, right: &Spanned<Expr>) -> Option<VarIndex> {
    // Pattern 1: x' = x  (Prime(StateVar(idx)) = StateVar(idx))
    if let Expr::Prime(primed) = &left.node {
        if let Expr::StateVar(_, prime_idx, _) = &primed.node {
            if let Expr::StateVar(_, rhs_idx, _) = &right.node {
                if prime_idx == rhs_idx {
                    return Some(VarIndex(*prime_idx));
                }
            }
        }
    }

    // Pattern 2: x = x'  (StateVar(idx) = Prime(StateVar(idx)))
    // TLA+ equality is symmetric; some specs may write it this way.
    if let Expr::Prime(primed) = &right.node {
        if let Expr::StateVar(_, prime_idx, _) = &primed.node {
            if let Expr::StateVar(_, lhs_idx, _) = &left.node {
                if prime_idx == lhs_idx {
                    return Some(VarIndex(*prime_idx));
                }
            }
        }
    }

    None
}

/// Detect identity assignment through IF/THEN/ELSE:
/// `x' = IF cond THEN x ELSE x` is equivalent to `x' = x`.
///
/// This pattern arises in PlusCal-generated and hand-written specs where
/// an assignment conditionally updates a variable:
///   `x' = IF cond THEN expr ELSE x`
/// When BOTH branches resolve to the unprimed variable, it's an identity.
///
/// Part of #3993 Phase 11: strengthen identity detection through branching.
fn detect_identity_through_if(left: &Spanned<Expr>, right: &Spanned<Expr>) -> Option<VarIndex> {
    // Pattern: x' = IF cond THEN x ELSE x
    if let Expr::Prime(primed) = &left.node {
        if let Expr::StateVar(_, prime_idx, _) = &primed.node {
            if is_identity_preserving_expr(&right.node, *prime_idx) {
                return Some(VarIndex(*prime_idx));
            }
        }
    }

    // Symmetric: IF cond THEN x ELSE x = x'
    if let Expr::Prime(primed) = &right.node {
        if let Expr::StateVar(_, prime_idx, _) = &primed.node {
            if is_identity_preserving_expr(&left.node, *prime_idx) {
                return Some(VarIndex(*prime_idx));
            }
        }
    }

    None
}

/// Check if an expression always evaluates to the current value of a state
/// variable, making it an identity-preserving expression.
///
/// Returns true for:
/// - `StateVar(idx)` — direct reference to the variable
/// - `IF cond THEN <identity> ELSE <identity>` — both branches preserve
/// - `CASE ... -> <identity> [] ... -> <identity> [] OTHER -> <identity>`
/// - `LET ... IN <identity>` — body preserves through let bindings
///
/// Part of #3993 Phase 11.
fn is_identity_preserving_expr(expr: &Expr, var_idx: u16) -> bool {
    match expr {
        Expr::StateVar(_, idx, _) => *idx == var_idx,

        Expr::If(_, then_e, else_e) => {
            is_identity_preserving_expr(&then_e.node, var_idx)
                && is_identity_preserving_expr(&else_e.node, var_idx)
        }

        Expr::Case(arms, other) => {
            let arms_identity = arms
                .iter()
                .all(|arm| is_identity_preserving_expr(&arm.body.node, var_idx));
            let other_identity = other
                .as_ref()
                .map_or(true, |o| is_identity_preserving_expr(&o.node, var_idx));
            arms_identity && other_identity
        }

        Expr::Let(_, body) => is_identity_preserving_expr(&body.node, var_idx),

        Expr::Label(label) => is_identity_preserving_expr(&label.body.node, var_idx),

        _ => false,
    }
}

/// Detect the EXCEPT identity pattern: `f' = [f EXCEPT ![k] = f[k]]`.
///
/// This means the action updates function `f` at key `k` with the current
/// value of `f[k]` — a no-op. The function is preserved unchanged.
///
/// Returns `Some(VarIndex)` when ALL EXCEPT specs are identity updates:
/// each spec has a single `Index(key_expr)` path element and its value is
/// `FuncApply(StateVar(idx), key_expr)` where `key_expr` structurally
/// matches the path key. Mixed EXCEPT specs (some identity, some not) are
/// conservatively treated as non-identity (returns `None`).
///
/// Part of #3993 Phase 11: many TLA+ specs use `[f EXCEPT ![k] = f[k]]`
/// in action conjuncts to express "f is unchanged at key k". Without this
/// detection, POR treats these as real writes, blocking independence.
fn detect_except_identity(left: &Spanned<Expr>, right: &Spanned<Expr>) -> Option<VarIndex> {
    // Pattern: f' = [f EXCEPT ![k] = f[k]]
    if let Expr::Prime(primed) = &left.node {
        if let Expr::StateVar(_, prime_idx, _) = &primed.node {
            if let Expr::Except(base, specs) = &right.node {
                if let Expr::StateVar(_, base_idx, _) = &base.node {
                    if *prime_idx == *base_idx && is_except_all_identity(*base_idx, specs) {
                        return Some(VarIndex(*prime_idx));
                    }
                }
            }
        }
    }

    // Symmetric: [f EXCEPT ![k] = f[k]] = f'
    if let Expr::Prime(primed) = &right.node {
        if let Expr::StateVar(_, prime_idx, _) = &primed.node {
            if let Expr::Except(base, specs) = &left.node {
                if let Expr::StateVar(_, base_idx, _) = &base.node {
                    if *prime_idx == *base_idx && is_except_all_identity(*base_idx, specs) {
                        return Some(VarIndex(*prime_idx));
                    }
                }
            }
        }
    }

    None
}

/// Check whether ALL EXCEPT specs are identity updates for the given
/// state variable. An identity EXCEPT spec has the form:
///   `![k] = f[k]`
/// i.e., a single `Index(key_expr)` path element and a value of
/// `FuncApply(StateVar(var_idx), key_expr)` where key_expr matches
/// structurally.
///
/// Part of #3993 Phase 11.
fn is_except_all_identity(var_idx: u16, specs: &[ExceptSpec]) -> bool {
    if specs.is_empty() {
        return false; // Degenerate case — treat as non-identity
    }

    specs.iter().all(|spec| {
        // Only handle single-element index paths: ![k]
        // Multi-element paths like ![k1][k2] or field paths like !.field
        // are not handled (conservative: return false).
        if spec.path.len() != 1 {
            return false;
        }

        let key_expr = match &spec.path[0] {
            ExceptPathElement::Index(idx) => &idx.node,
            ExceptPathElement::Field(_) => return false, // Record field — not handled
        };

        // Value must be FuncApply(StateVar(var_idx), key_expr) — i.e., f[k]
        if let Expr::FuncApply(func, arg) = &spec.value.node {
            if let Expr::StateVar(_, func_idx, _) = &func.node {
                *func_idx == var_idx && exprs_structurally_equal(key_expr, &arg.node)
            } else {
                false
            }
        } else {
            false
        }
    })
}

/// Conservative structural equality check for AST expressions.
///
/// Returns `true` when `a` and `b` have the same shape and leaf values.
/// Only handles simple, commonly-occurring expression forms; returns
/// `false` for anything complex (conservative for soundness).
///
/// Part of #3993 Phase 11: used by EXCEPT identity detection to compare
/// the key expression in the path with the key expression in the value.
fn exprs_structurally_equal(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Bool(x), Expr::Bool(y)) => x == y,
        (Expr::Int(x), Expr::Int(y)) => x == y,
        (Expr::String(x), Expr::String(y)) => x == y,
        (Expr::Ident(name_a, _), Expr::Ident(name_b, _)) => name_a == name_b,
        (Expr::StateVar(_, idx_a, _), Expr::StateVar(_, idx_b, _)) => idx_a == idx_b,
        (Expr::OpRef(a), Expr::OpRef(b)) => a == b,
        (Expr::Add(la, ra), Expr::Add(lb, rb))
        | (Expr::Sub(la, ra), Expr::Sub(lb, rb))
        | (Expr::Mul(la, ra), Expr::Mul(lb, rb))
        | (Expr::Div(la, ra), Expr::Div(lb, rb)) => {
            exprs_structurally_equal(&la.node, &lb.node)
                && exprs_structurally_equal(&ra.node, &rb.node)
        }
        (Expr::Neg(ia), Expr::Neg(ib)) => exprs_structurally_equal(&ia.node, &ib.node),
        (Expr::FuncApply(fa, aa), Expr::FuncApply(fb, ab)) => {
            exprs_structurally_equal(&fa.node, &fb.node)
                && exprs_structurally_equal(&aa.node, &ab.node)
        }
        (Expr::Tuple(ea), Expr::Tuple(eb)) => {
            ea.len() == eb.len()
                && ea
                    .iter()
                    .zip(eb.iter())
                    .all(|(x, y)| exprs_structurally_equal(&x.node, &y.node))
        }
        // Conservative: anything else is not structurally equal
        _ => false,
    }
}

/// Detect a constant write pattern: `x' = <constant>` where the RHS
/// does not depend on any state variable.
///
/// Returns `Some(VarIndex)` when the LHS is a primed state variable and
/// the RHS is state-independent. This means the action writes to x but
/// does not READ x — the write value is determined entirely by constants.
///
/// This matters for POR because a normal `x' = x + 1` records both a
/// write AND a read of x. But `x' = 0` only needs a write, not a read.
/// Fewer read dependencies means more chances for independence.
///
/// Part of #3993 Phase 11.
fn detect_constant_write(left: &Spanned<Expr>, right: &Spanned<Expr>) -> Option<VarIndex> {
    // Pattern: x' = <constant>
    if let Expr::Prime(primed) = &left.node {
        if let Expr::StateVar(_, idx, _) = &primed.node {
            if is_state_independent_expr(&right.node) {
                return Some(VarIndex(*idx));
            }
        }
    }

    // Symmetric: <constant> = x'
    if let Expr::Prime(primed) = &right.node {
        if let Expr::StateVar(_, idx, _) = &primed.node {
            if is_state_independent_expr(&left.node) {
                return Some(VarIndex(*idx));
            }
        }
    }

    None
}

/// Check if an expression is state-independent (a pure constant that does
/// not read any state variable). Used to classify "constant writes" like
/// `x' = 0` or `x' = TRUE` which don't conflict with reads of x from
/// other actions (since the write value doesn't depend on x's current value).
///
/// Note: constant writes still conflict with other writes to the same variable
/// and with reads of x (because they change x's value). The real win: a
/// constant write `x' = 0` does NOT add a READ dependency on x, unlike
/// `x' = x + 1`.
///
/// Part of #3993 Phase 11.
fn is_state_independent_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) => true,
        Expr::Ident(_, _) | Expr::OpRef(_) => {
            // Identifiers could refer to constants or operators — conservatively
            // treat as potentially state-dependent since we can't resolve here.
            false
        }
        Expr::StateVar(_, _, _) => false,
        Expr::Prime(_) => false,

        Expr::Tuple(elems) | Expr::SetEnum(elems) | Expr::Times(elems) => {
            elems.iter().all(|e| is_state_independent_expr(&e.node))
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => fields
            .iter()
            .all(|(_, v)| is_state_independent_expr(&v.node)),

        Expr::Not(inner) | Expr::Neg(inner) => is_state_independent_expr(&inner.node),

        Expr::And(l, r)
        | Expr::Or(l, r)
        | Expr::Add(l, r)
        | Expr::Sub(l, r)
        | Expr::Mul(l, r)
        | Expr::Div(l, r)
        | Expr::IntDiv(l, r)
        | Expr::Mod(l, r)
        | Expr::Pow(l, r)
        | Expr::Range(l, r)
        | Expr::Eq(l, r)
        | Expr::Neq(l, r)
        | Expr::Lt(l, r)
        | Expr::Leq(l, r)
        | Expr::Gt(l, r)
        | Expr::Geq(l, r) => {
            is_state_independent_expr(&l.node) && is_state_independent_expr(&r.node)
        }

        Expr::If(cond, then_e, else_e) => {
            is_state_independent_expr(&cond.node)
                && is_state_independent_expr(&then_e.node)
                && is_state_independent_expr(&else_e.node)
        }

        Expr::Label(label) => is_state_independent_expr(&label.body.node),

        // Conservative: anything else could read state.
        _ => false,
    }
}
