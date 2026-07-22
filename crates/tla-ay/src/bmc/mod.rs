// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bounded Model Checking (BMC) using ay SMT solver
//!
//! Part of #542: BMC encodes k-step transition sequences as SAT formulas
//! to quickly find counterexamples without exhaustive state exploration.
//!
//! # BMC Formula
//!
//! For depth bound k, the BMC formula is:
//! ```text
//! Init(s0) ∧ Next(s0,s1) ∧ ... ∧ Next(sk-1,sk) ∧ (¬Safety(s0) ∨ ... ∨ ¬Safety(sk))
//! ```
//!
//! If SAT: counterexample trace exists where safety fails at some step j ≤ k.
//! If UNSAT: safety holds at all steps up to depth k.
//!
//! # Variable Unrolling
//!
//! Each state variable `x` becomes k+1 SMT variables: `x__0`, `x__1`, ..., `x__k`.
//!
//! | TLA+ | BMC encoding at step i |
//! |------|------------------------|
//! | `x` | `x__i` |
//! | `x'` | `x__i+1` |
//! | `UNCHANGED x` | `x__i+1 = x__i` |
//!
//! # Division and Modulo Restrictions (#556)
//!
//! BMC uses QF_LIA (Quantifier-Free Linear Integer Arithmetic) which does not
//! support native division or modulo. We linearize these operations as follows:
//!
//! - **Constant divisors only**: `x \div k` and `x % k` require `k` to be a literal.
//!   Variable divisors (e.g., `x \div y`) return `UntranslatableExpr` error.
//!
//! - **Positive divisors only**: Following TLC semantics, divisor must be > 0.
//!   Zero divisors return `DivisionByZero` error; negative divisors return
//!   `UnsupportedOp` error.
//!
//! - **Linearization**: `x \div k` and `x % k` introduce fresh variables `q` and `r`
//!   with constraints: `x = k*q + r ∧ 0 ≤ r < k` (Euclidean division).
//!
//! For arbitrary div/mod (non-constant or negative divisors), use the CHC/PDR
//! path in `translate.rs` which emits native solver terms.
//!
//! # Multiplication Restrictions (#771)
//!
//! QF_LIA does not support *non-linear* multiplication. In practice this means:
//! - `x * 2` is allowed (constant multiplication).
//! - `x * y` is rejected (both operands symbolic).

/// Compound type dispatch: sets, functions, cardinality in BMC context.
/// Part of #3778.
mod compound_dispatch;
mod funcset;
/// Incremental BMC loop for iterative deepening. Part of #3724.
pub mod incremental;
/// Pure SMT-level k-induction checker for safety properties. Part of #3722.
pub mod kinduction;
/// Record and tuple encoding: per-field/per-element SMT variables. Part of #3787.
mod record_encoder;
mod translate_bmc;
mod translate_expr_impl;

use std::collections::HashMap;

// Re-exported for tests.rs (which uses `use super::*`)
use ay_dpll::api::{
    Logic, Model, SolveDecisionProfileSummary, SolveResult, Solver, Sort, Term, UnsatProofArtifact,
};
#[cfg(test)]
pub(crate) use tla_core::ast::Expr;
#[cfg(test)]
pub(crate) use tla_core::Spanned;

use crate::error::{AYError, AYResult, MAX_BMC_BOUND};
use crate::TlaSort;

use record_encoder::{BmcRecordVarInfo, BmcTupleVarInfo};

/// Information about a variable across all BMC steps
#[derive(Debug)]
struct BmcVarInfo {
    /// Sort of the variable
    sort: TlaSort,
    /// ay terms for each step: index i has variable at step i
    terms: Vec<Term>,
}

/// A single state in a BMC trace
#[derive(Debug, Clone)]
pub struct BmcState {
    /// Step number (0-indexed)
    pub step: usize,
    /// Variable assignments at this step
    pub assignments: HashMap<String, BmcValue>,
}

/// A value in a BMC state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BmcValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Arbitrary-precision integer value (for values that overflow i64)
    ///
    /// Part of #3888: Avoid silently dropping big integers in extract_trace.
    BigInt(num_bigint::BigInt),
    /// String value.
    ///
    /// BMC represents `TlaSort::String` terms as interned integers internally
    /// (see `bmc_intern_string`); `extract_trace` decodes them back to the
    /// original string so cross-validation replays the real TLA+ string value
    /// (not its interned id) through the interpreter.
    String(String),
    /// Set value (finite set of elements)
    ///
    /// Part of #3778: Finite set encoding via SMT arrays.
    /// Stores the members of the set as a sorted list of values.
    Set(Vec<BmcValue>),
    /// Sequence value (ordered list of elements)
    ///
    /// Part of #3793: Sequence encoding in BMC translator.
    /// Stores elements in order (1-indexed when accessed in TLA+).
    Sequence(Vec<BmcValue>),
    /// Function value (finite mapping from Int keys to values)
    ///
    /// Part of #3786: Function encoding in BMC translator.
    /// Stores the mapping as a sorted list of (key, value) pairs.
    /// The domain is the set of keys.
    Function(Vec<(i64, BmcValue)>),
    /// Record value (finite mapping from field names to values)
    ///
    /// Part of #3787: Record encoding in BMC translator.
    /// Stores the fields as a sorted list of (field_name, value) pairs.
    Record(Vec<(String, BmcValue)>),
    /// Tuple value (ordered list of elements, 1-indexed in TLA+)
    ///
    /// Part of #3787: Tuple encoding in BMC translator.
    /// Stores elements in order (index 1 = first element).
    Tuple(Vec<BmcValue>),
}

/// Information about a function variable across all BMC steps.
///
/// Each function is encoded as two SMT arrays per step:
/// - `domain_terms[step]`: `(Array Int Bool)` — the domain membership set
/// - `mapping_terms[step]`: `(Array Int RangeSort)` — the value mapping
///
/// Part of #3786: Function encoding in BMC translator.
#[derive(Debug)]
struct BmcFuncVarInfo {
    /// Range sort of the function (retained for future introspection)
    #[allow(dead_code)]
    range_sort: TlaSort,
    /// Domain key sort. `Int` for integer-keyed functions; `String` for
    /// functions whose domain is a set of string literals. String-keyed
    /// functions are encoded with a *distinct* SMT `String` index sort so a
    /// string key can never alias an integer-literal key in the shared array
    /// encoding (soundness — see `declare_func_var_with_key_sort`).
    key_sort: TlaSort,
    /// Domain set terms per step: `(Array KeySort Bool)`.
    ///
    /// EMPTY for symbolic-domain (map-only) functions — their domain is the
    /// arithmetic fact `lo <= x <= N+offset`, carried in `symbolic_domain`
    /// below, not a membership array.
    domain_terms: Vec<Term>,
    /// Mapping array terms per step: `(Array KeySort RangeSort)`
    mapping_terms: Vec<Term>,
    /// Symbolic contiguous integer domain `lo..(const+offset)`, present only for
    /// [`TlaSort::FunctionSym`] functions. `None` for finite-domain functions
    /// (which carry a `domain_terms` membership array instead). When present,
    /// `x \in DOMAIN f` is translated as the arithmetic bound `lo <= x <= hi`.
    symbolic_domain: Option<(i64, String, i64)>,
}

/// Information about a sequence variable across all BMC steps.
///
/// Each sequence is encoded as an SMT array (index -> element) and a
/// length term per step.
///
/// Part of #3793: Sequence encoding in BMC translator.
#[derive(Debug)]
struct BmcSeqVarInfo {
    /// Element sort of the sequence (retained for future introspection)
    #[allow(dead_code)]
    element_sort: TlaSort,
    /// Maximum sequence length (for bounding).
    max_len: usize,
    /// Array terms per step: `(Array Int ElementSort)`.
    array_terms: Vec<Term>,
    /// Length terms per step (Int).
    length_terms: Vec<Term>,
}

/// BMC translator for k-step bounded model checking
///
/// This translates TLA+ Init/Next/Safety into a BMC formula and checks it.
pub struct BmcTranslator {
    /// The ay solver instance
    solver: Solver,
    /// Maximum bound k
    bound_k: usize,
    /// Variable info: name -> BmcVarInfo (with terms for all steps)
    vars: HashMap<String, BmcVarInfo>,
    /// Function variable info: name -> BmcFuncVarInfo (with domain+mapping per step)
    ///
    /// Part of #3786: Function encoding in BMC translator.
    func_vars: HashMap<String, BmcFuncVarInfo>,
    /// Sequence variable info: name -> BmcSeqVarInfo (array+length per step)
    ///
    /// Part of #3793: Sequence encoding in BMC translator.
    seq_vars: HashMap<String, BmcSeqVarInfo>,
    /// Record variable info: name -> BmcRecordVarInfo (per-field terms per step)
    ///
    /// Part of #3787: Record encoding in BMC translator.
    record_vars: HashMap<String, BmcRecordVarInfo>,
    /// Tuple variable info: name -> BmcTupleVarInfo (per-element terms per step)
    ///
    /// Part of #3787: Tuple encoding in BMC translator.
    tuple_vars: HashMap<String, BmcTupleVarInfo>,
    /// Current step being translated (for primed variable resolution)
    current_step: usize,
    /// Counter for generating unique auxiliary variable names (linearization)
    aux_var_counter: usize,
    /// Auxiliary DEFINITIONAL constraints side-asserted during translation
    /// (currently: the `x = k*q + r ∧ 0 ≤ r < k` Euclidean linearization of
    /// `\div`/`%`, see `linearize_div_mod`). These are asserted directly into the
    /// solver — so they DO appear in the exported proof bundle's assertion set —
    /// but are NOT part of the top-level `Term` list a `build_*` returns. The
    /// certificate's no-solve render-binding re-translates an obligation and must
    /// reconstruct the SAME assertion set the proof used; it renders these
    /// alongside the returned terms via `aux_asserted_canonical`. Empty for any
    /// spec without `\div`/`%`, so non-modulo paths are byte-identical.
    aux_asserted: Vec<Term>,
    /// Names of base state variables declared via `declare_var`.
    ///
    /// Used by `clear_temporary_vars` to distinguish base declarations from
    /// Skolem constants and other temporary variables injected during
    /// translation (quantifier expansion, CHOOSE, etc.). Only base vars
    /// survive `clear_temporary_vars`; temporaries are evicted.
    ///
    /// Part of #4006: prevent variable accumulation across cooperative seeds.
    base_var_names: Vec<String>,
    /// String literal -> stable interned integer id.
    ///
    /// BMC declares `TlaSort::String` variables as `Sort::Int` (the interned
    /// representation, see `TlaSort::to_ay`). For scalar string equality to be
    /// consistent, every distinct string literal must map to a distinct,
    /// stable integer. This table assigns those ids on demand. Ids start at a
    /// large negative base to avoid colliding with the small non-negative
    /// integers used as ordinary domain/range values.
    string_intern: HashMap<String, i64>,
    /// Integer values that appear in concrete set/function/sequence
    /// assertions (`assert_concrete_state`).
    ///
    /// These are merged into the finite universe used by subset / membership
    /// encodings so that a concretely-stored element outside the literal base
    /// set (e.g. `T = {1, 4}` versus `SUBSET {1, 2, 3}`) is still in scope
    /// for the `T \subseteq base` pointwise check. Without this, the subset
    /// constraint would only quantify over the literal base universe and
    /// silently allow stray elements through.
    pub(crate) tracked_universe_ints: std::collections::BTreeSet<i64>,
}

impl BmcTranslator {
    /// Create a new BMC translator for bound k
    ///
    /// This creates a solver instance that will check for violations up to depth k.
    /// Uses QF_LIA logic (quantifier-free linear integer arithmetic).
    ///
    /// # Errors
    /// Returns `BmcBoundTooLarge` if k exceeds `MAX_BMC_BOUND` (100,000).
    pub fn new(k: usize) -> AYResult<Self> {
        if k > MAX_BMC_BOUND {
            return Err(AYError::BmcBoundTooLarge {
                bound: k,
                max: MAX_BMC_BOUND,
            });
        }
        Ok(Self {
            solver: Solver::try_new(Logic::QfLia)?,
            bound_k: k,
            vars: HashMap::new(),
            func_vars: HashMap::new(),
            seq_vars: HashMap::new(),
            record_vars: HashMap::new(),
            tuple_vars: HashMap::new(),
            current_step: 0,
            aux_var_counter: 0,
            aux_asserted: Vec::new(),
            base_var_names: Vec::new(),
            string_intern: HashMap::new(),
            tracked_universe_ints: std::collections::BTreeSet::new(),
        })
    }

    /// Create a new BMC translator with array support for bound k.
    ///
    /// Uses QF_AUFLIA logic (arrays + uninterpreted functions + linear integer
    /// arithmetic), which is required when any state variable has `Set` or
    /// `Function` sort.
    ///
    /// Part of #3778: Finite set encoding via SMT arrays.
    /// Part of #3786: Function encoding via SMT arrays.
    ///
    /// # Errors
    /// Returns `BmcBoundTooLarge` if k exceeds `MAX_BMC_BOUND` (100,000).
    pub fn new_with_arrays(k: usize) -> AYResult<Self> {
        if k > MAX_BMC_BOUND {
            return Err(AYError::BmcBoundTooLarge {
                bound: k,
                max: MAX_BMC_BOUND,
            });
        }
        // #arr2lia-starvation opt-out: BMC unroll queries are UNSAT-expected
        // in the common (no-violation) case. Upstream ay (1da1732d) gates the
        // arrays->LIA rescue size floor on the ORIGINAL problem size, so small
        // one-shot sessions like this one bypass the floor natively — no
        // caller-side opt-out needed (the CHC portfolio keeps its floor).
        let solver = Solver::try_new(Logic::QfAuflia)?;
        Ok(Self {
            solver,
            bound_k: k,
            vars: HashMap::new(),
            func_vars: HashMap::new(),
            seq_vars: HashMap::new(),
            record_vars: HashMap::new(),
            tuple_vars: HashMap::new(),
            current_step: 0,
            aux_var_counter: 0,
            aux_asserted: Vec::new(),
            base_var_names: Vec::new(),
            string_intern: HashMap::new(),
            tracked_universe_ints: std::collections::BTreeSet::new(),
        })
    }

    /// Map a string literal to a stable, distinct interned integer id.
    ///
    /// `TlaSort::String` variables are declared as `Sort::Int` (the interned
    /// representation), so string literals and string-sorted terms must share
    /// one consistent id namespace for equality to be sound. Ids are negative
    /// to keep them disjoint from the small non-negative integers commonly used
    /// as concrete domain/range values, so a string literal can never alias an
    /// ordinary integer constant.
    pub(super) fn bmc_intern_string(&mut self, s: &str) -> i64 {
        if let Some(&id) = self.string_intern.get(s) {
            return id;
        }
        // Base far below any plausible literal int to avoid collision.
        let id = -1_000_000_007 - self.string_intern.len() as i64;
        self.string_intern.insert(s.to_string(), id);
        id
    }

    /// Declare a state variable for all k+1 steps
    ///
    /// Creates variables x__0, x__1, ..., x__k for the state variable x.
    /// Supports scalar types (Bool, Int, String), Set types, Function types,
    /// Record types, and Tuple types. Function, Sequence, Record, and Tuple sorts
    /// are delegated to their dedicated `declare_*` methods.
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `sort` is not a supported BMC type, or
    /// a propagated error (e.g. [`AYError::Solver`]) from the delegated declaration
    /// or sort conversion.
    pub fn declare_var(&mut self, name: &str, sort: TlaSort) -> AYResult<()> {
        // Record this as a base state variable so clear_temporary_vars knows
        // to preserve it. Part of #4006.
        self.base_var_names.push(name.to_string());

        // Delegate Function sort to dedicated method
        if let TlaSort::Function { range, .. } = &sort {
            return self.declare_func_var(name, (**range).clone());
        }

        // Delegate symbolic-domain Function sort to the MAP-ONLY declaration
        // (no `f__dom` membership array; the domain is the arithmetic fact
        // `lo <= x <= N+offset`). Part of the function-state all-N encoding.
        if let TlaSort::FunctionSym {
            domain_lo,
            domain_hi_const,
            domain_hi_offset,
            range,
        } = &sort
        {
            return self.declare_funcsym_var(
                name,
                *domain_lo,
                domain_hi_const.clone(),
                *domain_hi_offset,
                (**range).clone(),
            );
        }

        // Delegate Sequence sort to dedicated method
        if let TlaSort::Sequence {
            element_sort,
            max_len,
        } = &sort
        {
            return self.declare_seq_var(name, (**element_sort).clone(), *max_len);
        }

        // Delegate Record sort to dedicated method (Part of #3787)
        if let TlaSort::Record { field_sorts } = &sort {
            return self.declare_record_var(name, field_sorts.clone());
        }

        // Delegate Tuple sort to dedicated method (Part of #3787)
        if let TlaSort::Tuple { element_sorts } = &sort {
            return self.declare_tuple_var(name, element_sorts.clone());
        }

        if !sort.is_scalar() && !matches!(sort, TlaSort::Set { .. }) {
            return Err(AYError::UnsupportedOp(format!(
                "BMC only supports scalar, set, function, sequence, record, and tuple types, \
                 got {sort} for variable {name}"
            )));
        }

        let mut terms = Vec::with_capacity(self.bound_k + 1);

        // Create k+1 variables: x__0, x__1, ..., x__k
        for step in 0..=self.bound_k {
            let step_name = format!("{name}__{step}");
            let term = self.solver.declare_const(&step_name, sort.to_ay()?);
            terms.push(term);
        }

        self.vars
            .insert(name.to_string(), BmcVarInfo { sort, terms });
        Ok(())
    }

    /// Declare a RIGID constant (a `CONSTANT` kept symbolic for an all-N proof):
    /// a SINGLE scalar SMT variable SHARED across all steps, so `name__step`
    /// resolves to the SAME term for every step. The constant is therefore
    /// structurally rigid (`N' = N`) with NO extra equality asserted — which keeps
    /// the obligation proofs in AY's strict single-equality Farkas fragment (a
    /// separate `N' = N` equality would force a variable-elimination the strict
    /// checker demotes to a trust step). The constant is a free variable, so a
    /// proof that holds for it holds for ALL its values.
    ///
    /// Idempotent: re-declaring an existing name is a no-op.
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `sort` is not scalar, or a propagated
    /// [`AYError::Solver`] if the sort has no ay encoding.
    pub fn declare_rigid_const(&mut self, name: &str, sort: TlaSort) -> AYResult<()> {
        if !sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "rigid constant must be scalar, got {sort} for {name}"
            )));
        }
        if self.vars.contains_key(name) {
            return Ok(());
        }
        self.base_var_names.push(name.to_string());
        let term = self.solver.declare_const(name, sort.to_ay()?);
        let terms = vec![term; self.bound_k + 1];
        self.vars
            .insert(name.to_string(), BmcVarInfo { sort, terms });
        Ok(())
    }

    /// Declare a function state variable for all k+1 steps.
    ///
    /// Each function is encoded as two SMT arrays per step:
    /// - `{name}__dom__{step}`: `(Array Int Bool)` — domain membership set
    /// - `{name}__map__{step}`: `(Array Int RangeSort)` — value mapping
    ///
    /// The range sort must be scalar (Bool, Int, or String). Defaults to an
    /// integer-keyed domain; see
    /// [`declare_func_var_with_key_sort`](Self::declare_func_var_with_key_sort).
    ///
    /// Part of #3786: Function encoding in BMC translator.
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `range_sort` is not scalar, or a
    /// propagated [`AYError::Solver`] from the sort conversion.
    pub fn declare_func_var(&mut self, name: &str, range_sort: TlaSort) -> AYResult<()> {
        // Default to an integer-keyed domain. String-keyed functions are
        // upgraded in place (see `upgrade_func_key_sort_to_string`) once a
        // construction reveals the domain is string-typed, which the lossy
        // `TlaSort::Function { domain_keys }` cannot record on its own.
        self.declare_func_var_with_key_sort(name, TlaSort::Int, range_sort)
    }

    /// Declare a function variable with an explicit domain *key* sort.
    ///
    /// The key sort selects the SMT index sort of the domain/mapping arrays:
    /// `Int` -> `(Array Int _)`, `String` -> `(Array String _)`. Giving
    /// string-keyed functions a genuinely distinct `String` index sort is what
    /// keeps a string key from aliasing an integer-literal key: SMT array
    /// `select`/`store` over `(Array String _)` and `(Array Int _)` live in
    /// disjoint sorts, so no string constant can ever equal an integer literal
    /// key. (Part of #5 — string-keyed BMC function domains.)
    ///
    /// Idempotent: re-declaring an existing function name is a no-op.
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `range_sort` is not scalar or
    /// `key_sort` is neither `Int` nor `String`, or a propagated [`AYError::Solver`]
    /// from the array-sort conversion.
    pub fn declare_func_var_with_key_sort(
        &mut self,
        name: &str,
        key_sort: TlaSort,
        range_sort: TlaSort,
    ) -> AYResult<()> {
        if self.func_vars.contains_key(name) {
            return Ok(()); // Already declared
        }

        if !range_sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "BMC function range must be scalar, got {range_sort} for function {name}"
            )));
        }
        if !matches!(key_sort, TlaSort::Int | TlaSort::String) {
            return Err(AYError::UnsupportedOp(format!(
                "BMC function domain key sort must be Int or String, got {key_sort} for function {name}"
            )));
        }

        // NOTE: do NOT route the key sort through `TlaSort::to_ay()` — that maps
        // `String -> Sort::Int` (the interned-string representation), which is
        // exactly the aliasing we must avoid. Map the index sort explicitly so a
        // `String` key domain yields a genuine `(Array String _)`.
        let key_ay = match key_sort {
            TlaSort::Int => Sort::Int,
            TlaSort::String => Sort::String,
            _ => unreachable!("guarded above"),
        };
        let dom_sort = Sort::array(key_ay.clone(), Sort::Bool);
        let map_sort = Sort::array(key_ay, range_sort.to_ay()?);

        let mut domain_terms = Vec::with_capacity(self.bound_k + 1);
        let mut mapping_terms = Vec::with_capacity(self.bound_k + 1);

        for step in 0..=self.bound_k {
            let dom_name = format!("{name}__dom__{step}");
            let map_name = format!("{name}__map__{step}");
            domain_terms.push(self.solver.declare_const(&dom_name, dom_sort.clone()));
            mapping_terms.push(self.solver.declare_const(&map_name, map_sort.clone()));
        }

        self.func_vars.insert(
            name.to_string(),
            BmcFuncVarInfo {
                range_sort,
                key_sort,
                domain_terms,
                mapping_terms,
                symbolic_domain: None,
            },
        );
        Ok(())
    }

    /// Declare a symbolic-domain (map-only) function variable for all k+1 steps.
    ///
    /// The domain is the contiguous integer range `domain_lo ..
    /// (domain_hi_const + domain_hi_offset)` over an unbound symbolic
    /// `CONSTANT`. Only the mapping array `{name}__map__{step}` (an `(Array Int
    /// RangeSort)`) is declared — there is NO `{name}__dom__{step}` membership
    /// array, because the domain is the ARITHMETIC fact `lo <= x <= hi`, not an
    /// enumerable set. `x \in DOMAIN f` is translated to that bound
    /// (see `symbolic_func_domain_bound`).
    ///
    /// Part of the function-state all-N encoding
    /// (docs/cert/function-state-alln-design.md).
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `range_sort` is not scalar, or a
    /// propagated [`AYError::Solver`] from the array-sort conversion.
    pub fn declare_funcsym_var(
        &mut self,
        name: &str,
        domain_lo: i64,
        domain_hi_const: String,
        domain_hi_offset: i64,
        range_sort: TlaSort,
    ) -> AYResult<()> {
        if self.func_vars.contains_key(name) {
            return Ok(()); // Already declared
        }
        if !range_sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "BMC symbolic-domain function range must be scalar, got {range_sort} for {name}"
            )));
        }
        self.base_var_names.push(name.to_string());
        let map_sort = Sort::array(Sort::Int, range_sort.to_ay()?);
        let mut mapping_terms = Vec::with_capacity(self.bound_k + 1);
        for step in 0..=self.bound_k {
            let map_name = format!("{name}__map__{step}");
            mapping_terms.push(self.solver.declare_const(&map_name, map_sort.clone()));
        }
        self.func_vars.insert(
            name.to_string(),
            BmcFuncVarInfo {
                range_sort,
                key_sort: TlaSort::Int,
                domain_terms: Vec::new(),
                mapping_terms,
                symbolic_domain: Some((domain_lo, domain_hi_const, domain_hi_offset)),
            },
        );
        Ok(())
    }

    /// Get the domain key sort of a declared function variable.
    pub(crate) fn func_key_sort(&self, name: &str) -> Option<TlaSort> {
        self.func_vars.get(name).map(|i| i.key_sort.clone())
    }

    /// The symbolic domain `(lo, const, offset)` of a map-only function
    /// variable, or `None` for finite-domain (or unknown) functions.
    pub(crate) fn func_symbolic_domain(&self, name: &str) -> Option<(i64, String, i64)> {
        self.func_vars
            .get(name)
            .and_then(|i| i.symbolic_domain.clone())
    }

    /// Upgrade an already-declared, still-Int-keyed function variable to a
    /// `String`-keyed encoding by re-declaring its domain/mapping arrays with a
    /// `String` index sort.
    ///
    /// This is only safe to call before any constraint has referenced the
    /// function's arrays (e.g. at the first `f = [k \in {"a"} |-> ...]`
    /// construction in `Init`/`Next`), because it replaces the array terms. We
    /// therefore refuse the upgrade if the function is already `String`-keyed
    /// (idempotent no-op) and never downgrade. Fresh `String`-sorted array
    /// constants are introduced with distinct names so prior `Int`-indexed
    /// constants (if somehow referenced) remain well-sorted but unconstrained.
    /// (Part of #5.)
    pub(crate) fn upgrade_func_key_sort_to_string(&mut self, name: &str) -> AYResult<()> {
        let already_string = match self.func_vars.get(name) {
            Some(info) => matches!(info.key_sort, TlaSort::String),
            None => {
                return Err(AYError::UnknownVariable(format!(
                    "function {name} (upgrade to string-keyed)"
                )))
            }
        };
        if already_string {
            return Ok(());
        }
        let range_sort = self
            .func_vars
            .get(name)
            .expect("checked above")
            .range_sort
            .clone();
        let dom_sort = Sort::array(Sort::String, Sort::Bool);
        let map_sort = Sort::array(Sort::String, range_sort.to_ay()?);

        let mut domain_terms = Vec::with_capacity(self.bound_k + 1);
        let mut mapping_terms = Vec::with_capacity(self.bound_k + 1);
        for step in 0..=self.bound_k {
            let dom_name = format!("{name}__dom__str__{step}");
            let map_name = format!("{name}__map__str__{step}");
            domain_terms.push(self.solver.declare_const(&dom_name, dom_sort.clone()));
            mapping_terms.push(self.solver.declare_const(&map_name, map_sort.clone()));
        }

        let info = self.func_vars.get_mut(name).expect("checked above");
        info.key_sort = TlaSort::String;
        info.domain_terms = domain_terms;
        info.mapping_terms = mapping_terms;
        // String-keyed upgrade never applies to symbolic-domain functions.
        info.symbolic_domain = None;
        Ok(())
    }

    /// Get the mapping array term for a function variable at a specific step.
    ///
    /// Part of #3786.
    pub(crate) fn get_func_mapping_at_step(&self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .func_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("function {name} (at step {step})")))?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        Ok(info.mapping_terms[step])
    }

    /// Get the domain set term for a function variable at a specific step.
    ///
    /// Part of #3786.
    pub(crate) fn get_func_domain_at_step(&self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .func_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("function {name} (at step {step})")))?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        // Map-only (symbolic-domain) functions have NO domain membership array;
        // any operation that would need one (whole-function/domain equality,
        // domain-array comparison) is outside the pointwise-∀ fragment and must
        // fail closed (decline) rather than index an empty vector.
        if info.symbolic_domain.is_some() {
            return Err(AYError::UnsupportedOp(format!(
                "function {name} has a symbolic (map-only) domain with no domain array"
            )));
        }
        Ok(info.domain_terms[step])
    }

    /// Declare a sequence state variable for all k+1 steps.
    ///
    /// Each sequence is encoded as an SMT array + length per step:
    /// - `{name}__arr__{step}`: `(Array Int ElemSort)` — 1-indexed element storage
    /// - `{name}__len__{step}`: `Int` — current length
    ///
    /// The element sort must be scalar (Bool, Int, or String).
    /// Length is constrained to `0 <= len <= max_len` at each step.
    ///
    /// Idempotent: re-declaring an existing sequence name is a no-op.
    ///
    /// Part of #3793: Sequence encoding in BMC translator.
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `element_sort` is not scalar, or a
    /// propagated [`AYError::Solver`] from the array-sort conversion or length
    /// constraints.
    pub fn declare_seq_var(
        &mut self,
        name: &str,
        element_sort: TlaSort,
        max_len: usize,
    ) -> AYResult<()> {
        if self.seq_vars.contains_key(name) {
            return Ok(()); // Already declared
        }

        if !element_sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "BMC sequence element must be scalar, got {element_sort} for sequence {name}"
            )));
        }

        let arr_sort = Sort::array(Sort::Int, element_sort.to_ay()?);

        let mut array_terms = Vec::with_capacity(self.bound_k + 1);
        let mut length_terms = Vec::with_capacity(self.bound_k + 1);

        for step in 0..=self.bound_k {
            let arr_name = format!("{name}__arr__{step}");
            let len_name = format!("{name}__len__{step}");
            let arr = self.solver.declare_const(&arr_name, arr_sort.clone());
            let len = self.solver.declare_const(&len_name, Sort::Int);

            // Constrain: 0 <= len <= max_len
            let zero = self.solver.int_const(0);
            let max = self.solver.int_const(max_len as i64);
            let ge_zero = self.solver.try_ge(len, zero)?;
            let le_max = self.solver.try_le(len, max)?;
            self.solver
                .try_assert_term(ge_zero)
                .expect("invariant: ge is Bool-sorted");
            self.solver
                .try_assert_term(le_max)
                .expect("invariant: le is Bool-sorted");

            array_terms.push(arr);
            length_terms.push(len);
        }

        self.seq_vars.insert(
            name.to_string(),
            BmcSeqVarInfo {
                element_sort,
                max_len,
                array_terms,
                length_terms,
            },
        );
        Ok(())
    }

    /// Get the array term for a sequence variable at a specific step.
    ///
    /// Part of #3793.
    pub(crate) fn get_seq_array_at_step(&self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .seq_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("sequence {name} (at step {step})")))?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        Ok(info.array_terms[step])
    }

    /// Get the length term for a sequence variable at a specific step.
    ///
    /// Part of #3793.
    pub(crate) fn get_seq_length_at_step(&self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .seq_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("sequence {name} (at step {step})")))?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        Ok(info.length_terms[step])
    }

    /// Get the maximum length bound for a sequence variable.
    ///
    /// Part of #3793.
    pub(crate) fn get_seq_max_len(&self, name: &str) -> AYResult<usize> {
        let info = self
            .seq_vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("sequence {name}")))?;
        Ok(info.max_len)
    }

    /// Linearize div/mod for QF_LIA: x div k / x mod k for positive constant k.
    ///
    /// Introduces fresh variables q (quotient) and r (remainder) and asserts:
    ///   x = k*q + r ∧ 0 ≤ r < k
    ///
    /// This encodes floor division semantics (TLC-compatible):
    /// - For x = 7, k = 3:  7 = 3*2 + 1, so q = 2, r = 1
    /// - For x = -7, k = 3: -7 = 3*(-3) + 2, so q = -3, r = 2
    ///
    /// Returns (q_term, r_term) where q is the quotient and r is the remainder.
    fn linearize_div_mod(&mut self, x_term: Term, k: i64) -> AYResult<(Term, Term)> {
        let id = self.aux_var_counter;
        self.aux_var_counter += 1;

        // Declare fresh auxiliary variables
        let q_name = format!("__div_q_{id}");
        let r_name = format!("__div_r_{id}");
        let q = self.solver.declare_const(&q_name, Sort::Int);
        let r = self.solver.declare_const(&r_name, Sort::Int);

        // Assert: x = k*q + r
        let k_term = self.solver.int_const(k);
        let k_times_q = self.solver.try_mul(k_term, q)?;
        let k_q_plus_r = self.solver.try_add(k_times_q, r)?;
        let eq = self.solver.try_eq(x_term, k_q_plus_r)?;
        self.solver.try_assert_term(eq)?;

        // Assert: 0 <= r
        let zero = self.solver.int_const(0);
        let r_ge_0 = self.solver.try_ge(r, zero)?;
        self.solver.try_assert_term(r_ge_0)?;

        // Assert: r < k
        let r_lt_k = self.solver.try_lt(r, k_term)?;
        self.solver.try_assert_term(r_lt_k)?;

        // Record the definitional constraints so the certificate's no-solve
        // render-binding can reconstruct the SAME assertion set the proof used
        // (these are side-asserted, so they are NOT in the returned Term list).
        self.aux_asserted.push(eq);
        self.aux_asserted.push(r_ge_0);
        self.aux_asserted.push(r_lt_k);

        Ok((q, r))
    }

    /// Canonically render the auxiliary DEFINITIONAL constraints side-asserted
    /// during this translator's `build_*` (the `\div`/`%` linearization; see
    /// `aux_asserted`). The certificate's render-binding appends these to the
    /// returned obligation terms so the re-translated assertion set matches the
    /// proof bundle's (which includes them). Empty absent any `\div`/`%`.
    #[must_use]
    pub fn aux_asserted_canonical(&self) -> Vec<String> {
        self.aux_asserted
            .iter()
            .map(|&t| self.solver.render_term_canonical(t))
            .collect()
    }

    /// Get the ay term for a variable at a specific step
    fn get_var_at_step(&self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .vars
            .get(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("{name} (at step {step})")))?;

        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {} exceeds bound {}",
                step, self.bound_k
            )));
        }

        Ok(info.terms[step])
    }

    /// Assert a Boolean term in the solver.
    ///
    /// # Panics
    /// Panics if `term` is not Bool-sorted (only Boolean formulas may be asserted).
    pub fn assert(&mut self, term: Term) {
        self.solver
            .try_assert_term(term)
            .expect("invariant: assert requires Bool-sorted term");
    }

    /// Check satisfiability of the asserted BMC formula.
    ///
    /// Returns [`SolveResult::Unknown`] on timeout or solver panic; use
    /// [`try_check_sat`](Self::try_check_sat) to surface solver errors instead.
    pub fn check_sat(&mut self) -> SolveResult {
        self.solver.check_sat().into_inner()
    }

    /// Check satisfiability with panic protection.
    ///
    /// Uses the upstream `try_check_sat()` API to catch solver panics and
    /// return them as `AYError::Solver(SolverError::SolverPanic(...))`.
    /// Part of #2826.
    ///
    /// # Errors
    /// Returns [`AYError::Solver`] if the solver fails or panics internally; a
    /// timeout surfaces as `Ok(SolveResult::Unknown)`, not an error.
    pub fn try_check_sat(&mut self) -> AYResult<SolveResult> {
        Ok(self.solver.try_check_sat()?.into_inner())
    }

    /// Check satisfiability and return the typed AY decision/profile summary from the same solve.
    ///
    /// # Errors
    /// Returns [`AYError::Solver`] if the underlying solve fails or panics.
    pub fn try_check_sat_with_decision_profile_summary(
        &mut self,
    ) -> AYResult<(SolveResult, SolveDecisionProfileSummary)> {
        let details = self.solver.try_check_sat_with_details()?;
        let summary = details.decision_profile_summary();
        Ok((details.result.into_inner(), summary))
    }

    /// Get the consumer-accepted model if SAT.
    pub fn get_model(&mut self) -> Option<Model> {
        self.solver
            .model_for_consumer()
            .map(ay_dpll::VerifiedModel::into_inner)
    }

    /// Get the model with error reporting.
    ///
    /// Uses the upstream `try_get_model_for_consumer()` API so TY only
    /// accepts SAT witnesses after AY's validation boundary passes.
    /// Part of #2826, #4445.
    ///
    /// # Errors
    /// Returns [`AYError::Solver`] if there is no consumer-accepted model — e.g.
    /// the last result was not SAT, or the model failed AY's validation boundary.
    pub fn try_get_model(&self) -> AYResult<Model> {
        Ok(self.solver.try_get_model_for_consumer()?.into_inner())
    }

    /// Set a timeout for solver `check_sat` calls. Part of #2826.
    pub fn set_timeout(&mut self, timeout: Option<std::time::Duration>) {
        self.solver.set_timeout(timeout);
    }

    /// Cross-thread interrupt handle for the underlying SMT solver.
    ///
    /// Storing `true` into the returned flag makes an in-flight `check_sat`
    /// (and any subsequent one on this solver instance) return `Unknown`
    /// with reason `Interrupted` at its next internal control check, without
    /// waiting for the per-call timeout. Used by ty's fused/CDEMC orchestrator
    /// to tear symbolic lanes down the moment another lane resolves the
    /// verdict. The handle is per-solver: harvest a fresh one after
    /// recreating the translator. Interruption is cooperative and best-effort
    /// (solver phases that do not poll controls finish their current step
    /// first); callers must not rely on it alone for bounded-latency shutdown.
    pub fn interrupt_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.solver.interrupt_handle()
    }

    /// Enable proof production for the next `check_sat`. After UNSAT, retrieve
    /// the (strict-checked) Alethe/LRAT/Farkas artifact via
    /// [`Self::export_last_unsat_artifact`]. Used by certifying verification to
    /// carry AY's own re-checkable proof of each obligation in the certificate.
    pub fn set_produce_proofs(&mut self, enabled: bool) {
        self.solver.set_produce_proofs(enabled);
    }

    /// Export the last UNSAT proof as a portable [`UnsatProofArtifact`] (rendered
    /// Alethe text scoped to the asserted problem, a strict-checked verdict, an
    /// optional LRAT backbone, and Farkas certificates). `None` if the last result
    /// was not UNSAT or proof production was not enabled.
    #[must_use]
    pub fn export_last_unsat_artifact(&self) -> Option<UnsatProofArtifact> {
        self.solver.export_last_unsat_artifact()
    }

    /// Export the last UNSAT proof as a portable, checker-only
    /// [`SerializableProofBundle`](ay_dpll::api::SerializableProofBundle) that can
    /// be re-checked OFFLINE via `ay_proof::re_check_bundle_strict` — with no
    /// solver run and without trusting this solver. `None` if the last result was
    /// not UNSAT or proof production was not enabled. Mirrors
    /// [`Self::export_last_unsat_artifact`]; carries a producer-solver-independent
    /// proof object for Leg D of certifying verification.
    #[must_use]
    pub fn export_last_unsat_bundle(&self) -> Option<ay_dpll::api::SerializableProofBundle> {
        self.solver.export_last_unsat_bundle()
    }

    /// Leg D part-2 (NO-SOLVE): render the asserted `terms` canonically
    /// (variables by NAME, store-INDEPENDENT) against THIS translator's current
    /// term store, so renders are comparable across two different stores. Does NOT
    /// call `check_sat`. Returns one S-expr string per term, in the given order;
    /// the caller sorts into a multiset for the assume-coverage equality.
    #[must_use]
    pub fn render_terms_canonical(&self, terms: &[Term]) -> Vec<String> {
        terms
            .iter()
            .map(|&t| self.solver.render_term_canonical(t))
            .collect()
    }

    /// Get the current timeout setting.
    pub fn get_timeout(&self) -> Option<std::time::Duration> {
        self.solver.timeout()
    }

    /// Get the structured reason for the last `Unknown` result. Part of #2826.
    pub fn last_unknown_reason(&self) -> Option<ay_dpll::UnknownReason> {
        self.solver.unknown_reason()
    }

    /// Push a new assertion scope for incremental solving. Part of #3724.
    ///
    /// # Errors
    /// Returns [`AYError::Solver`] if the solver rejects the push.
    pub fn push_scope(&mut self) -> AYResult<()> {
        Ok(self.solver.try_push()?)
    }

    /// Pop the most recent assertion scope. Part of #3724.
    ///
    /// # Errors
    /// Returns [`AYError::Solver`] if there is no scope to pop or the solver
    /// rejects the pop.
    pub fn pop_scope(&mut self) -> AYResult<()> {
        Ok(self.solver.try_pop()?)
    }

    /// Remove temporary variables accumulated during translation.
    ///
    /// When the BMC translator is reused across multiple seeds in the
    /// cooperative engine, Skolem constants and other auxiliary variables
    /// are inserted into `self.vars` by quantifier expansion, CHOOSE
    /// translation, and function construction. These entries persist
    /// across `push_scope`/`pop_scope` pairs because they are HashMap
    /// entries, not solver assertions. Over many seeds, this causes the
    /// `vars` map to grow unboundedly.
    ///
    /// This method evicts all `vars` entries that are NOT in the base
    /// variable set (those declared via `declare_var`). It should be
    /// called after `pop_scope` at the end of each seed to keep the
    /// translator's variable map clean.
    ///
    /// Note: This does NOT reset `aux_var_counter`, which must remain
    /// monotonically increasing to ensure unique variable names in the
    /// solver. The solver-level declarations are harmless (the solver
    /// tracks them internally and they are lightweight); the problem is
    /// the translator-side HashMap growing without bound.
    ///
    /// Part of #4006: prevent variable accumulation across cooperative seeds.
    pub fn clear_temporary_vars(&mut self) {
        if self.base_var_names.is_empty() {
            return;
        }
        self.vars
            .retain(|name, _| self.base_var_names.contains(name));
    }

    /// Get the number of temporary variables currently in the `vars` map.
    ///
    /// Returns the count of entries in `vars` that are NOT base state
    /// variables. Useful for diagnostics and testing.
    ///
    /// Part of #4006.
    pub fn temporary_var_count(&self) -> usize {
        self.vars
            .keys()
            .filter(|name| !self.base_var_names.contains(name))
            .count()
    }

    /// Get the total number of entries in the `vars` map (base + temporary).
    ///
    /// Part of #4006.
    pub fn total_var_count(&self) -> usize {
        self.vars.len()
    }

    /// Assert concrete variable assignments at a given BMC step.
    ///
    /// Used by CDEMC to seed BMC from BFS frontier instead of Init.
    /// Each `(name, value)` pair constrains the named variable at `step`
    /// to the given concrete value. This replaces `translate_init()` when
    /// BMC starts from a known concrete state rather than the Init predicate.
    ///
    /// Part of #3765, Epic #3762.
    ///
    /// # Errors
    /// Returns [`AYError::UntranslatableExpr`] if `step` exceeds the configured
    /// bound `k`, [`AYError::UnknownVariable`] for an unassignable name,
    /// [`AYError::IntegerOverflow`] for a `BigInt` value too large for the solver,
    /// or a propagated [`AYError::Solver`] from building the equality constraint.
    pub fn assert_concrete_state(
        &mut self,
        assignments: &[(String, BmcValue)],
        step: usize,
    ) -> AYResult<()> {
        for (name, value) in assignments {
            match value {
                BmcValue::Bool(b) => {
                    let var_term = self.get_var_at_step(name, step)?;
                    let val_term = self.solver.bool_const(*b);
                    let eq = self.solver.try_eq(var_term, val_term)?;
                    self.assert(eq);
                }
                BmcValue::Int(i) => {
                    let var_term = self.get_var_at_step(name, step)?;
                    let val_term = self.solver.int_const(*i);
                    let eq = self.solver.try_eq(var_term, val_term)?;
                    self.assert(eq);
                }
                BmcValue::BigInt(n) => {
                    use num_traits::ToPrimitive;
                    let var_term = self.get_var_at_step(name, step)?;
                    let i = n.to_i64().ok_or_else(|| {
                        AYError::IntegerOverflow(format!(
                            "BigInt value {n} for variable '{name}' too large for solver"
                        ))
                    })?;
                    let val_term = self.solver.int_const(i);
                    let eq = self.solver.try_eq(var_term, val_term)?;
                    self.assert(eq);
                }
                BmcValue::Set(members) => {
                    let var_term = self.get_var_at_step(name, step)?;
                    // Encode a concrete set: build (store ... (const false) ... true)
                    // then assert equality with the variable.
                    let false_val = self.solver.bool_const(false);
                    let true_val = self.solver.bool_const(true);
                    let mut arr = self.solver.try_const_array(Sort::Int, false_val)?;
                    for member in members {
                        let member_int = match member {
                            BmcValue::Int(i) => *i,
                            _ => {
                                return Err(AYError::UnsupportedOp(
                                    "BMC set members must be integers".to_string(),
                                ));
                            }
                        };
                        // Track concretely-stored set members so subset/membership
                        // encodings include them in the finite universe.
                        self.tracked_universe_ints.insert(member_int);
                        let member_term = self.solver.int_const(member_int);
                        arr = self.solver.try_store(arr, member_term, true_val)?;
                    }
                    let eq = self.solver.try_eq(var_term, arr)?;
                    self.assert(eq);
                }
                BmcValue::Sequence(elements) => {
                    // Encode a concrete sequence: store elements at 1-based indices,
                    // constrain length.
                    let arr_term = self.get_seq_array_at_step(name, step)?;
                    let len_term = self.get_seq_length_at_step(name, step)?;

                    // Assert length
                    let len_val = self.solver.int_const(elements.len() as i64);
                    let len_eq = self.solver.try_eq(len_term, len_val)?;
                    self.assert(len_eq);

                    // Assert each element at its 1-based index
                    for (i, elem) in elements.iter().enumerate() {
                        let idx = self.solver.int_const((i + 1) as i64);
                        let elem_term = match elem {
                            BmcValue::Int(v) => self.solver.int_const(*v),
                            BmcValue::Bool(b) => {
                                // Bool encoded as Int: true=1, false=0
                                self.solver.int_const(if *b { 1i64 } else { 0i64 })
                            }
                            BmcValue::String(s) => {
                                let id = self.bmc_intern_string(s);
                                self.solver.int_const(id)
                            }
                            _ => {
                                return Err(AYError::UnsupportedOp(
                                    "BMC sequence elements must be scalars".to_string(),
                                ));
                            }
                        };
                        let selected = self.solver.try_select(arr_term, idx)?;
                        let eq = self.solver.try_eq(selected, elem_term)?;
                        self.assert(eq);
                    }
                }
                BmcValue::Function(entries) => {
                    // Encode a concrete function: constrain domain membership and
                    // mapping values. Part of #3786.
                    let dom_term = self.get_func_domain_at_step(name, step)?;
                    let map_term = self.get_func_mapping_at_step(name, step)?;

                    let false_val = self.solver.bool_const(false);
                    let true_val = self.solver.bool_const(true);

                    // Build domain: (store ... (const false) ... true)
                    let mut expected_dom = self.solver.try_const_array(Sort::Int, false_val)?;
                    for &(key, _) in entries {
                        let key_term = self.solver.int_const(key);
                        expected_dom = self.solver.try_store(expected_dom, key_term, true_val)?;
                    }
                    let dom_eq = self.solver.try_eq(dom_term, expected_dom)?;
                    self.assert(dom_eq);

                    // Constrain mapping values at each key
                    for (key, value) in entries {
                        let key_term = self.solver.int_const(*key);
                        let val_term = match value {
                            BmcValue::Int(v) => self.solver.int_const(*v),
                            BmcValue::Bool(b) => {
                                self.solver.int_const(if *b { 1i64 } else { 0i64 })
                            }
                            BmcValue::String(s) => {
                                let id = self.bmc_intern_string(s);
                                self.solver.int_const(id)
                            }
                            _ => {
                                return Err(AYError::UnsupportedOp(
                                    "BMC function values must be scalars".to_string(),
                                ));
                            }
                        };
                        let selected = self.solver.try_select(map_term, key_term)?;
                        let eq = self.solver.try_eq(selected, val_term)?;
                        self.assert(eq);
                    }
                }
                BmcValue::Record(fields) => {
                    // Encode a concrete record: constrain per-field variables.
                    // Part of #3787: Record encoding in BMC translator.
                    for (field_name, value) in fields {
                        let val_term = match value {
                            BmcValue::Int(v) => self.solver.int_const(*v),
                            BmcValue::Bool(b) => {
                                self.solver.int_const(if *b { 1i64 } else { 0i64 })
                            }
                            BmcValue::String(s) => {
                                let id = self.bmc_intern_string(s);
                                self.solver.int_const(id)
                            }
                            _ => {
                                return Err(AYError::UnsupportedOp(
                                    "BMC record field values must be scalars".to_string(),
                                ));
                            }
                        };
                        let field_term = self.get_record_field_at_step(name, field_name, step)?;
                        let eq = self.solver.try_eq(field_term, val_term)?;
                        self.assert(eq);
                    }
                }
                BmcValue::Tuple(elements) => {
                    // Encode a concrete tuple: constrain per-element variables.
                    // Part of #3787: Tuple encoding in BMC translator.
                    for (i, elem) in elements.iter().enumerate() {
                        let val_term = match elem {
                            BmcValue::Int(v) => self.solver.int_const(*v),
                            BmcValue::Bool(b) => {
                                self.solver.int_const(if *b { 1i64 } else { 0i64 })
                            }
                            BmcValue::String(s) => {
                                let id = self.bmc_intern_string(s);
                                self.solver.int_const(id)
                            }
                            _ => {
                                return Err(AYError::UnsupportedOp(
                                    "BMC tuple element values must be scalars".to_string(),
                                ));
                            }
                        };
                        let elem_term = self.get_tuple_element_at_step(name, i + 1, step)?;
                        let eq = self.solver.try_eq(elem_term, val_term)?;
                        self.assert(eq);
                    }
                }
                BmcValue::String(s) => {
                    // Strings are interned to integers (see `bmc_intern_string`);
                    // pin the variable to the interned id.
                    let var_term = self.get_var_at_step(name, step)?;
                    let id = self.bmc_intern_string(s);
                    let val_term = self.solver.int_const(id);
                    let eq = self.solver.try_eq(var_term, val_term)?;
                    self.assert(eq);
                }
            }
        }
        Ok(())
    }

    /// Assert a blocking clause excluding a concrete state at its BMC step.
    ///
    /// Encodes `NOT(var1 = value1 AND var2 = value2 AND ...)` using the same
    /// value-equality helper as wavefront/state encodings. Empty states fail
    /// closed because they cannot produce a meaningful blocking clause.
    ///
    /// # Errors
    /// Returns [`AYError::UntranslatableExpr`] if `state` has no assignments, plus
    /// any error from encoding a per-variable equality (e.g.
    /// [`AYError::UnknownVariable`] for an unassignable name or
    /// [`AYError::Solver`]).
    pub fn block_concrete_state(&mut self, state: &BmcState) -> AYResult<()> {
        if state.assignments.is_empty() {
            return Err(AYError::UntranslatableExpr(format!(
                "cannot block empty concrete state at step {}",
                state.step
            )));
        }

        let mut assignments: Vec<_> = state.assignments.iter().collect();
        assignments.sort_by_key(|(left, _)| *left);

        let mut equalities = Vec::with_capacity(assignments.len());
        for (name, value) in assignments {
            equalities.push(self.make_value_eq(name, value, state.step)?);
        }

        let mut state_eq = equalities
            .pop()
            .expect("invariant: empty assignments returned before equality construction");
        for equality in equalities.into_iter().rev() {
            state_eq = self.solver.try_and(equality, state_eq)?;
        }

        let blocking_clause = self.solver.try_not(state_eq)?;
        self.assert(blocking_clause);
        Ok(())
    }

    /// Assert a disjunctive wavefront formula at a given BMC step.
    ///
    /// Encodes the wavefront as:
    /// ```text
    /// shared_constraints(step) /\ (disjunct_1(step) \/ disjunct_2(step) \/ ...)
    /// ```
    ///
    /// Each shared constraint becomes `var_step = value`. Each disjunct is
    /// a conjunction of per-variable assignments for varying variables.
    /// The disjunction of all state-conjuncts encodes "the system is in
    /// one of these frontier states".
    ///
    /// This replaces N separate `assert_concrete_state` calls with a single
    /// disjunctive formula, enabling the solver to reason about the entire
    /// frontier simultaneously rather than sequentially.
    ///
    /// Part of #3794.
    ///
    /// # Errors
    /// Returns an [`AYError`] if any shared constraint or disjunct cannot be encoded
    /// as a value equality (e.g. [`AYError::UnknownVariable`] for an unassignable
    /// name or a propagated [`AYError::Solver`]).
    pub fn assert_wavefront_formula(
        &mut self,
        shared: &[(String, BmcValue)],
        disjuncts: &[Vec<(String, BmcValue)>],
        step: usize,
    ) -> AYResult<()> {
        // 1. Assert shared constraints directly (they hold in ALL states).
        for (name, value) in shared {
            let eq = self.make_value_eq(name, value, step)?;
            self.assert(eq);
        }

        // 2. Build the disjunction of per-state conjuncts.
        if disjuncts.is_empty() {
            return Ok(());
        }

        // Build each conjunct as AND of per-variable equalities.
        let mut disjunct_terms: Vec<Term> = Vec::with_capacity(disjuncts.len());
        for state_assignments in disjuncts {
            if state_assignments.is_empty() {
                // Empty conjunct = TRUE (all vars are shared).
                disjunct_terms.push(self.solver.bool_const(true));
                continue;
            }

            let mut conjunct_parts: Vec<Term> = Vec::with_capacity(state_assignments.len());
            for (name, value) in state_assignments {
                let eq = self.make_value_eq(name, value, step)?;
                conjunct_parts.push(eq);
            }

            // AND the parts together.
            let mut conjunct = conjunct_parts
                .pop()
                .expect("invariant: non-empty conjunct_parts");
            for part in conjunct_parts.into_iter().rev() {
                conjunct = self.solver.try_and(part, conjunct)?;
            }
            disjunct_terms.push(conjunct);
        }

        // OR the disjuncts together.
        if disjunct_terms.len() == 1 {
            // Single disjunct: assert it directly (no OR needed).
            self.assert(disjunct_terms.pop().expect("invariant: len checked == 1"));
        } else {
            let mut disjunction = disjunct_terms
                .pop()
                .expect("invariant: non-empty disjunct_terms");
            for term in disjunct_terms.into_iter().rev() {
                disjunction = self.solver.try_or(term, disjunction)?;
            }
            self.assert(disjunction);
        }

        Ok(())
    }

    /// Build an equality term `var_at_step = value` for a BmcValue.
    ///
    /// Helper for [`assert_wavefront_formula`] — handles Bool, Int, BigInt,
    /// Set, Sequence, Function, Record, and Tuple.
    ///
    /// For compound types, builds a conjunction of per-element equalities
    /// following the same encoding patterns used in [`assert_concrete_state`].
    ///
    /// Part of #3794, extended for compound types.
    fn make_value_eq(&mut self, name: &str, value: &BmcValue, step: usize) -> AYResult<Term> {
        match value {
            BmcValue::Bool(b) => {
                let var_term = self.get_var_at_step(name, step)?;
                let val_term = self.solver.bool_const(*b);
                Ok(self.solver.try_eq(var_term, val_term)?)
            }
            BmcValue::Int(i) => {
                let var_term = self.get_var_at_step(name, step)?;
                let val_term = self.solver.int_const(*i);
                Ok(self.solver.try_eq(var_term, val_term)?)
            }
            BmcValue::String(s) => {
                let var_term = self.get_var_at_step(name, step)?;
                let id = self.bmc_intern_string(s);
                let val_term = self.solver.int_const(id);
                Ok(self.solver.try_eq(var_term, val_term)?)
            }
            BmcValue::BigInt(n) => {
                use num_traits::ToPrimitive;
                let var_term = self.get_var_at_step(name, step)?;
                if let Some(i) = n.to_i64() {
                    let val_term = self.solver.int_const(i);
                    Ok(self.solver.try_eq(var_term, val_term)?)
                } else {
                    Err(AYError::IntegerOverflow(format!(
                        "BigInt value {n} for variable '{name}' too large for wavefront encoding"
                    )))
                }
            }
            BmcValue::Set(members) => {
                // Build (store ... (const false) ... true) and assert equality
                let var_term = self.get_var_at_step(name, step)?;
                let false_val = self.solver.bool_const(false);
                let true_val = self.solver.bool_const(true);
                let mut arr = self.solver.try_const_array(Sort::Int, false_val)?;
                for member in members {
                    let member_term = match member {
                        BmcValue::Int(i) => self.solver.int_const(*i),
                        _ => {
                            return Err(AYError::UnsupportedOp(
                                "wavefront set members must be integers".to_string(),
                            ));
                        }
                    };
                    arr = self.solver.try_store(arr, member_term, true_val)?;
                }
                Ok(self.solver.try_eq(var_term, arr)?)
            }
            BmcValue::Sequence(elements) => {
                // Constrain array elements and length
                let arr_term = self.get_seq_array_at_step(name, step)?;
                let len_term = self.get_seq_length_at_step(name, step)?;

                let len_val = self.solver.int_const(elements.len() as i64);
                let mut conjuncts = vec![self.solver.try_eq(len_term, len_val)?];

                for (i, elem) in elements.iter().enumerate() {
                    let idx = self.solver.int_const((i + 1) as i64);
                    let elem_term = match elem {
                        BmcValue::Int(v) => self.solver.int_const(*v),
                        BmcValue::Bool(b) => self.solver.int_const(if *b { 1i64 } else { 0i64 }),
                        BmcValue::String(s) => {
                            let id = self.bmc_intern_string(s);
                            self.solver.int_const(id)
                        }
                        _ => {
                            return Err(AYError::UnsupportedOp(
                                "wavefront sequence elements must be scalars".to_string(),
                            ));
                        }
                    };
                    let selected = self.solver.try_select(arr_term, idx)?;
                    conjuncts.push(self.solver.try_eq(selected, elem_term)?);
                }

                // Fold into conjunction
                let mut result = conjuncts.pop().expect("invariant: at least length eq");
                for c in conjuncts.into_iter().rev() {
                    result = self.solver.try_and(c, result)?;
                }
                Ok(result)
            }
            BmcValue::Function(entries) => {
                // Constrain domain and mapping
                let dom_term = self.get_func_domain_at_step(name, step)?;
                let map_term = self.get_func_mapping_at_step(name, step)?;

                let false_val = self.solver.bool_const(false);
                let true_val = self.solver.bool_const(true);

                let mut expected_dom = self.solver.try_const_array(Sort::Int, false_val)?;
                for &(key, _) in entries {
                    let key_term = self.solver.int_const(key);
                    expected_dom = self.solver.try_store(expected_dom, key_term, true_val)?;
                }
                let mut conjuncts = vec![self.solver.try_eq(dom_term, expected_dom)?];

                for (key, value) in entries {
                    let key_term = self.solver.int_const(*key);
                    let val_term = match value {
                        BmcValue::Int(v) => self.solver.int_const(*v),
                        BmcValue::Bool(b) => self.solver.int_const(if *b { 1i64 } else { 0i64 }),
                        BmcValue::String(s) => {
                            let id = self.bmc_intern_string(s);
                            self.solver.int_const(id)
                        }
                        _ => {
                            return Err(AYError::UnsupportedOp(
                                "wavefront function values must be scalars".to_string(),
                            ));
                        }
                    };
                    let selected = self.solver.try_select(map_term, key_term)?;
                    conjuncts.push(self.solver.try_eq(selected, val_term)?);
                }

                let mut result = conjuncts.pop().expect("invariant: at least domain eq");
                for c in conjuncts.into_iter().rev() {
                    result = self.solver.try_and(c, result)?;
                }
                Ok(result)
            }
            BmcValue::Record(fields) => {
                // Constrain per-field variables using record encoder
                let mut conjuncts = Vec::with_capacity(fields.len());
                for (field_name, value) in fields {
                    let val_term = match value {
                        BmcValue::Int(v) => self.solver.int_const(*v),
                        BmcValue::Bool(b) => self.solver.int_const(if *b { 1i64 } else { 0i64 }),
                        BmcValue::String(s) => {
                            let id = self.bmc_intern_string(s);
                            self.solver.int_const(id)
                        }
                        _ => {
                            return Err(AYError::UnsupportedOp(
                                "wavefront record field values must be scalars".to_string(),
                            ));
                        }
                    };
                    let field_term = self.get_record_field_at_step(name, field_name, step)?;
                    conjuncts.push(self.solver.try_eq(field_term, val_term)?);
                }
                if conjuncts.is_empty() {
                    Ok(self.solver.bool_const(true))
                } else {
                    let mut result = conjuncts.pop().expect("invariant: non-empty");
                    for c in conjuncts.into_iter().rev() {
                        result = self.solver.try_and(c, result)?;
                    }
                    Ok(result)
                }
            }
            BmcValue::Tuple(elements) => {
                // Constrain per-element variables (1-indexed) using tuple encoder
                let mut conjuncts = Vec::with_capacity(elements.len());
                for (i, elem) in elements.iter().enumerate() {
                    let val_term = match elem {
                        BmcValue::Int(v) => self.solver.int_const(*v),
                        BmcValue::Bool(b) => self.solver.int_const(if *b { 1i64 } else { 0i64 }),
                        BmcValue::String(s) => {
                            let id = self.bmc_intern_string(s);
                            self.solver.int_const(id)
                        }
                        _ => {
                            return Err(AYError::UnsupportedOp(
                                "wavefront tuple element values must be scalars".to_string(),
                            ));
                        }
                    };
                    let elem_term = self.get_tuple_element_at_step(name, i + 1, step)?;
                    conjuncts.push(self.solver.try_eq(elem_term, val_term)?);
                }
                if conjuncts.is_empty() {
                    Ok(self.solver.bool_const(true))
                } else {
                    let mut result = conjuncts.pop().expect("invariant: non-empty");
                    for c in conjuncts.into_iter().rev() {
                        result = self.solver.try_and(c, result)?;
                    }
                    Ok(result)
                }
            }
        }
    }
}

// BMC translation methods extracted to translate_bmc.rs
// TranslateExpr trait impl extracted to translate_expr_impl.rs

#[cfg(test)]
mod tests;

#[cfg(test)]
mod concrete_state_blocking_tests {
    use super::*;

    #[test]
    fn block_concrete_state_excludes_each_scalar_model() {
        let mut translator = BmcTranslator::new(0).expect("translator");
        translator
            .declare_var("x", TlaSort::Int)
            .expect("declare x");
        translator
            .assert_wavefront_formula(
                &[],
                &[
                    vec![("x".to_string(), BmcValue::Int(0))],
                    vec![("x".to_string(), BmcValue::Int(1))],
                ],
                0,
            )
            .expect("two-state domain");

        assert_eq!(translator.try_check_sat().expect("sat"), SolveResult::Sat);
        let first_model = translator.try_get_model().expect("first model");
        let first_state = translator
            .extract_trace(&first_model)
            .into_iter()
            .find(|state| state.step == 0)
            .expect("step 0");
        translator
            .block_concrete_state(&first_state)
            .expect("block first state");

        assert_eq!(
            translator.try_check_sat().expect("sat after first block"),
            SolveResult::Sat
        );
        let second_model = translator.try_get_model().expect("second model");
        let second_state = translator
            .extract_trace(&second_model)
            .into_iter()
            .find(|state| state.step == 0)
            .expect("step 0 after first block");
        assert_ne!(first_state.assignments, second_state.assignments);
        translator
            .block_concrete_state(&second_state)
            .expect("block second state");

        assert!(matches!(
            translator.try_check_sat().expect("unsat after both blocks"),
            SolveResult::Unsat(_)
        ));
    }

    #[test]
    fn bmc_try_get_model_flows_through_ay_consumer_boundary() {
        let mut translator = BmcTranslator::new(0).expect("translator");
        translator
            .declare_var("x", TlaSort::Int)
            .expect("declare x");
        translator
            .assert_wavefront_formula(&[], &[vec![("x".to_string(), BmcValue::Int(3))]], 0)
            .expect("single-state domain");

        assert_eq!(translator.try_check_sat().expect("sat"), SolveResult::Sat);
        let direct = translator
            .solver
            .try_get_model_for_consumer()
            .expect("AY consumer boundary should accept validated SAT")
            .into_inner();
        let wrapped = translator
            .try_get_model()
            .expect("BMC translator should consume AY consumer model");

        assert_eq!(direct.int_val("x__0").cloned(), Some(3.into()));
        assert_eq!(wrapped.int_val("x__0").cloned(), Some(3.into()));
        assert!(translator.get_model().is_some());
    }

    #[test]
    fn block_concrete_state_rejects_empty_state() {
        let mut translator = BmcTranslator::new(0).expect("translator");
        let state = BmcState {
            step: 0,
            assignments: HashMap::new(),
        };

        assert!(matches!(
            translator.block_concrete_state(&state),
            Err(AYError::UntranslatableExpr(message))
                if message == "cannot block empty concrete state at step 0"
        ));
    }
}
