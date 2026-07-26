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
//! Each state variable `x` becomes k+1 collision-free SMT symbols produced by
//! [`BmcTranslator::state_step_symbol`].
//!
//! | TLA+ | BMC encoding at step i |
//! |------|------------------------|
//! | `x` | `state_step_symbol("x", i)` |
//! | `x'` | `state_step_symbol("x", i + 1)` |
//! | `UNCHANGED x` | equality between the symbols at `i + 1` and `i` |
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

use std::collections::{HashMap, HashSet};

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

/// A decoded scalar/set SMT symbol emitted by [`BmcTranslator`].
///
/// Consumers that inspect rendered proofs or models must use
/// [`BmcTranslator::parse_scalar_symbol`] instead of reconstructing the symbol
/// spelling. Source names are length-delimited and hex encoded so adversarial
/// TLA+ identifiers cannot collide with carrier or step delimiters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BmcScalarSymbol {
    /// A step-indexed state variable (scalar or set carrier).
    State {
        /// Original TLA+ source name.
        name: String,
        /// BMC unrolling step.
        step: usize,
    },
    /// A rigid scalar constant shared by every BMC step.
    Rigid {
        /// Original TLA+ source name.
        name: String,
    },
}

/// The mutually exclusive solver-carrier families used by public BMC
/// declarations. A source-level name must belong to exactly one family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BmcCarrierKind {
    ScalarState,
    RigidScalar,
    Function,
    Sequence,
    Record,
    Tuple,
}

impl BmcCarrierKind {
    fn label(self) -> &'static str {
        match self {
            Self::ScalarState => "scalar/set state variable",
            Self::RigidScalar => "rigid scalar constant",
            Self::Function => "function variable",
            Self::Sequence => "sequence variable",
            Self::Record => "record variable",
            Self::Tuple => "tuple variable",
        }
    }
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
    /// Function value with native TLA+ string keys.
    ///
    /// String-keyed BMC functions use SMT `String` array indices rather than
    /// the integer interning used for String-valued scalar carriers. Keeping a
    /// distinct variant prevents trace extraction from silently presenting
    /// native string keys as TLA+ integers.
    StringFunction(Vec<(String, BmcValue)>),
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
/// Each finite-domain function is encoded as two SMT arrays per step:
/// - `domain_terms[step]`: `(Array KeySort Bool)` — domain membership
/// - `mapping_terms[step]`: `(Array KeySort RangeSort)` — value mapping
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
    /// Exact finite domain keys when the public declaration carried a complete
    /// `TlaSort::Function::domain_keys` shape. `None` means the domain array is
    /// genuinely symbolic/unknown and whole-function equality must fail closed:
    /// QF_AUFLIA cannot quantify over every live key without a finite cover.
    finite_domain_keys: Option<Vec<BmcFunctionKey>>,
    /// Whether any translated term or caller has observed this function's
    /// current domain/mapping carriers. An Int->String key upgrade replaces the
    /// arrays, so it is sound only while this remains false.
    carrier_referenced: bool,
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

/// A statically exact finite function-domain key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BmcFunctionKey {
    Int(i64),
    String(String),
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
    /// Names in `vars` whose one SMT carrier is shared across every step.
    rigid_const_names: HashSet<String>,
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
            rigid_const_names: HashSet::new(),
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
            rigid_const_names: HashSet::new(),
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
    /// one consistent id namespace for same-kind equality. The numeric ids can
    /// coincide with legal TLA+ integers, so every comparison and model decode
    /// must also retain the declared TLA+ value kind; the id alone is not a type
    /// discriminator.
    pub(super) fn bmc_intern_string(&mut self, s: &str) -> i64 {
        if let Some(&id) = self.string_intern.get(s) {
            return id;
        }
        // A stable sparse-looking base keeps rendered terms recognizable. Type
        // safety comes from the TLA-kind gates, not from numeric disjointness.
        let id = -1_000_000_007 - self.string_intern.len() as i64;
        self.string_intern.insert(s.to_string(), id);
        id
    }

    /// Reject a public declaration when `name` already belongs to another
    /// solver-carrier family. Keeping the per-kind maps mutually exclusive is
    /// required because expression dispatch probes those maps independently;
    /// one name in two maps can silently route an expression to the wrong
    /// carrier.
    pub(super) fn ensure_declaration_carrier(
        &self,
        name: &str,
        requested: BmcCarrierKind,
    ) -> AYResult<()> {
        let mut existing = Vec::with_capacity(2);
        if self.vars.contains_key(name) {
            existing.push(if self.rigid_const_names.contains(name) {
                BmcCarrierKind::RigidScalar
            } else {
                BmcCarrierKind::ScalarState
            });
        }
        if self.func_vars.contains_key(name) {
            existing.push(BmcCarrierKind::Function);
        }
        if self.seq_vars.contains_key(name) {
            existing.push(BmcCarrierKind::Sequence);
        }
        if self.record_vars.contains_key(name) {
            existing.push(BmcCarrierKind::Record);
        }
        if self.tuple_vars.contains_key(name) {
            existing.push(BmcCarrierKind::Tuple);
        }

        if existing.is_empty() || (existing.len() == 1 && existing[0] == requested) {
            return Ok(());
        }
        let actual = existing
            .iter()
            .map(|kind| kind.label())
            .collect::<Vec<_>>()
            .join(" + ");
        Err(AYError::TypeMismatch {
            name: name.to_string(),
            expected: actual,
            actual: requested.label().to_string(),
        })
    }

    fn register_base_var_name(&mut self, name: &str) {
        if !self.base_var_names.iter().any(|existing| existing == name) {
            self.base_var_names.push(name.to_string());
        }
    }

    /// Injectively encode one source component inside an SMT symbol. The byte
    /// length makes boundaries unambiguous even for adversarial identifiers
    /// containing this module's separators.
    fn symbol_component(value: &str) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(value.len().saturating_mul(2).saturating_add(24));
        encoded.push_str(&value.len().to_string());
        encoded.push('_');
        for byte in value.bytes() {
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
        encoded
    }

    fn parse_symbol_component(encoded: &str) -> Option<String> {
        let (length, hex) = encoded.split_once('_')?;
        let byte_len: usize = length.parse().ok()?;
        if byte_len.to_string() != length || hex.len() != byte_len.checked_mul(2)? {
            return None;
        }
        let mut bytes = Vec::with_capacity(byte_len);
        for pair in hex.as_bytes().chunks_exact(2) {
            let nibble = |byte: u8| match byte {
                b'0'..=b'9' => Some(byte - b'0'),
                b'a'..=b'f' => Some(byte - b'a' + 10),
                _ => None,
            };
            bytes.push((nibble(pair[0])? << 4) | nibble(pair[1])?);
        }
        let decoded = String::from_utf8(bytes).ok()?;
        (Self::symbol_component(&decoded) == encoded).then_some(decoded)
    }

    /// Return the canonical collision-free SMT symbol for a state variable at
    /// one BMC unrolling step.
    #[must_use]
    pub fn state_step_symbol(name: &str, step: usize) -> String {
        format!(
            "__ty_bmc_state_{}_step_{step}",
            Self::symbol_component(name)
        )
    }

    /// Return the canonical collision-free SMT symbol for a rigid constant.
    #[must_use]
    pub fn rigid_const_symbol(name: &str) -> String {
        format!("__ty_bmc_rigid_{}", Self::symbol_component(name))
    }

    /// Decode a canonical state-variable or rigid-constant SMT symbol.
    ///
    /// Returns `None` for auxiliary/compound carrier symbols, malformed input,
    /// and non-canonical spellings such as a step with leading zeroes.
    #[must_use]
    pub fn parse_scalar_symbol(symbol: &str) -> Option<BmcScalarSymbol> {
        if let Some(encoded) = symbol.strip_prefix("__ty_bmc_rigid_") {
            return Self::parse_symbol_component(encoded)
                .map(|name| BmcScalarSymbol::Rigid { name });
        }

        let rest = symbol.strip_prefix("__ty_bmc_state_")?;
        let (encoded, step_text) = rest.rsplit_once("_step_")?;
        let step: usize = step_text.parse().ok()?;
        if step.to_string() != step_text {
            return None;
        }
        let name = Self::parse_symbol_component(encoded)?;
        (Self::state_step_symbol(&name, step) == symbol)
            .then_some(BmcScalarSymbol::State { name, step })
    }

    pub(super) fn function_domain_symbol(name: &str, string_keys: bool, step: usize) -> String {
        let key_tag = if string_keys { "string" } else { "int" };
        format!(
            "__ty_bmc_function_{key_tag}_domain_{}_step_{step}",
            Self::symbol_component(name)
        )
    }

    pub(super) fn function_mapping_symbol(name: &str, string_keys: bool, step: usize) -> String {
        let key_tag = if string_keys { "string" } else { "int" };
        format!(
            "__ty_bmc_function_{key_tag}_mapping_{}_step_{step}",
            Self::symbol_component(name)
        )
    }

    pub(super) fn symbolic_function_mapping_symbol(name: &str, step: usize) -> String {
        format!(
            "__ty_bmc_function_symbolic_mapping_{}_step_{step}",
            Self::symbol_component(name)
        )
    }

    pub(super) fn sequence_array_symbol(name: &str, step: usize) -> String {
        format!(
            "__ty_bmc_sequence_array_{}_step_{step}",
            Self::symbol_component(name)
        )
    }

    pub(super) fn sequence_length_symbol(name: &str, step: usize) -> String {
        format!(
            "__ty_bmc_sequence_length_{}_step_{step}",
            Self::symbol_component(name)
        )
    }

    pub(super) fn record_field_symbol(name: &str, field: &str, step: usize) -> String {
        format!(
            "__ty_bmc_record_{}_field_{}_step_{step}",
            Self::symbol_component(name),
            Self::symbol_component(field)
        )
    }

    pub(super) fn tuple_element_symbol(name: &str, index: usize, step: usize) -> String {
        format!(
            "__ty_bmc_tuple_{}_element_{index}_step_{step}",
            Self::symbol_component(name)
        )
    }

    /// Declare an auxiliary SMT constant in a namespace disjoint from every
    /// source carrier. The global monotonic id makes names deterministic and
    /// unique across scopes; the encoded purpose is diagnostic only. A caller
    /// may also use the returned name as a temporary translator binding, so
    /// skip any adversarial source name already present in a carrier map.
    pub(super) fn declare_internal_const(&mut self, purpose: &str, sort: Sort) -> (String, Term) {
        loop {
            let id = self.aux_var_counter;
            self.aux_var_counter = self
                .aux_var_counter
                .checked_add(1)
                .expect("BMC auxiliary symbol counter overflow");
            let name = format!(
                "__ty_bmc_aux_{id}_purpose_{}",
                Self::symbol_component(purpose)
            );
            if self.vars.contains_key(&name)
                || self.func_vars.contains_key(&name)
                || self.seq_vars.contains_key(&name)
                || self.record_vars.contains_key(&name)
                || self.tuple_vars.contains_key(&name)
            {
                continue;
            }
            let term = self.solver.declare_const(&name, sort);
            return (name, term);
        }
    }

    /// Declare a state variable for all k+1 steps
    ///
    /// Creates one canonical [`Self::state_step_symbol`] for every step 0..=k.
    /// Supports scalar types (Bool, Int, String), Set types, Function types,
    /// Record types, and Tuple types. Function, Sequence, Record, and Tuple sorts
    /// are delegated to their dedicated `declare_*` methods.
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if `sort` is not a supported BMC type, or
    /// a propagated error (e.g. [`AYError::Solver`]) from the delegated declaration
    /// or sort conversion.
    pub fn declare_var(&mut self, name: &str, sort: TlaSort) -> AYResult<()> {
        // Set arrays use Int carrier indices. Until a coherent Bool-to-Int
        // index encoding exists across declaration, membership, equality, and
        // model extraction, reject Set(Bool) before registering a carrier or
        // declaring any solver symbol.
        if matches!(
            &sort,
            TlaSort::Set { element_sort }
                if (**element_sort).clone().canonicalized() == TlaSort::Bool
        ) {
            return Err(AYError::UnsupportedOp(format!(
                "BMC variable {name} cannot use Set(Bool): no Bool-index encoding is defined"
            )));
        }

        // Delegate Function sort to dedicated method
        if let TlaSort::Function { domain_keys, range } = &sort {
            self.declare_func_var_with_exact_domain(name, domain_keys, (**range).clone())?;
            self.register_base_var_name(name);
            return Ok(());
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
            self.declare_funcsym_var(
                name,
                *domain_lo,
                domain_hi_const.clone(),
                *domain_hi_offset,
                (**range).clone(),
            )?;
            self.register_base_var_name(name);
            return Ok(());
        }

        // Delegate Sequence sort to dedicated method
        if let TlaSort::Sequence {
            element_sort,
            max_len,
        } = &sort
        {
            self.declare_seq_var(name, (**element_sort).clone(), *max_len)?;
            self.register_base_var_name(name);
            return Ok(());
        }

        // Delegate Record sort to dedicated method (Part of #3787)
        if let TlaSort::Record { field_sorts } = &sort {
            self.declare_record_var(name, field_sorts.clone())?;
            self.register_base_var_name(name);
            return Ok(());
        }

        // Delegate Tuple sort to dedicated method (Part of #3787)
        if let TlaSort::Tuple { element_sorts } = &sort {
            self.declare_tuple_var(name, element_sorts.clone())?;
            self.register_base_var_name(name);
            return Ok(());
        }

        if !sort.is_scalar() && !matches!(sort, TlaSort::Set { .. }) {
            return Err(AYError::UnsupportedOp(format!(
                "BMC only supports scalar, set, function, sequence, record, and tuple types, \
                 got {sort} for variable {name}"
            )));
        }

        self.ensure_declaration_carrier(name, BmcCarrierKind::ScalarState)?;
        if let Some(existing) = self.vars.get(name) {
            if existing.sort.clone().canonicalized() == sort.clone().canonicalized() {
                self.register_base_var_name(name);
                return Ok(());
            }
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: existing.sort.to_string(),
                actual: sort.to_string(),
            });
        }

        let ay_sort = sort.to_ay()?;

        let mut terms = Vec::with_capacity(self.bound_k + 1);

        // Create one collision-free carrier for every BMC step.
        for step in 0..=self.bound_k {
            let step_name = Self::state_step_symbol(name, step);
            let term = self.solver.declare_const(&step_name, ay_sort.clone());
            terms.push(term);
        }

        self.vars
            .insert(name.to_string(), BmcVarInfo { sort, terms });
        self.register_base_var_name(name);
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
    /// Idempotent only when re-declared as the same rigid scalar sort.
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
        self.ensure_declaration_carrier(name, BmcCarrierKind::RigidScalar)?;
        if let Some(existing) = self.vars.get(name) {
            if existing.sort.clone().canonicalized() == sort.clone().canonicalized() {
                self.register_base_var_name(name);
                return Ok(());
            }
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: existing.sort.to_string(),
                actual: sort.to_string(),
            });
        }
        let ay_sort = sort.to_ay()?;
        let symbol = Self::rigid_const_symbol(name);
        let term = self.solver.declare_const(&symbol, ay_sort);
        let terms = vec![term; self.bound_k + 1];
        self.vars
            .insert(name.to_string(), BmcVarInfo { sort, terms });
        self.rigid_const_names.insert(name.to_string());
        self.register_base_var_name(name);
        Ok(())
    }

    /// Declare a function state variable for all k+1 steps.
    ///
    /// Each function is encoded as two canonically named SMT arrays per step,
    /// produced by [`Self::function_domain_symbol`] and
    /// [`Self::function_mapping_symbol`].
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
        // A generic declaration has an unknown, integer-keyed domain. Legacy
        // callers that later assign a String-domain construction can upgrade
        // this unused carrier once; declarations with `TlaSort::Function`
        // should instead preserve their exact typed domain from the outset.
        if !range_sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "BMC function range must be scalar, got {range_sort} for function {name}"
            )));
        }
        self.ensure_declaration_carrier(name, BmcCarrierKind::Function)?;
        if let Some(existing) = self.func_vars.get(name) {
            if existing.symbolic_domain.is_none()
                && existing.range_sort.clone().canonicalized() == range_sort.clone().canonicalized()
            {
                return Ok(());
            }
        }
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
    /// Idempotent only for the same finite function key/range shape. Generic
    /// declarations preserve an existing one-way String-key upgrade.
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
        let range_ay = range_sort.to_ay()?;
        self.ensure_declaration_carrier(name, BmcCarrierKind::Function)?;
        if let Some(existing) = self.func_vars.get(name) {
            if existing.symbolic_domain.is_none()
                && existing.key_sort.clone().canonicalized() == key_sort.clone().canonicalized()
                && existing.range_sort.clone().canonicalized() == range_sort.clone().canonicalized()
            {
                return Ok(());
            }
            let existing_kind = if existing.symbolic_domain.is_some() {
                "symbolic-domain function".to_string()
            } else {
                format!("[{} -> {}]", existing.key_sort, existing.range_sort)
            };
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: existing_kind,
                actual: format!("[{key_sort} -> {range_sort}]"),
            });
        }

        // NOTE: do NOT route the key sort through `TlaSort::to_ay()` — that maps
        // `String -> Sort::Int` (the interned-string representation), which is
        // exactly the aliasing we must avoid. Map the index sort explicitly so a
        // `String` key domain yields a genuine `(Array String _)`.
        let string_keys = matches!(&key_sort, TlaSort::String);
        let key_ay = match &key_sort {
            TlaSort::Int => Sort::Int,
            TlaSort::String => Sort::String,
            _ => unreachable!("guarded above"),
        };
        let dom_sort = Sort::array(key_ay.clone(), Sort::Bool);
        let map_sort = Sort::array(key_ay, range_ay);

        let mut domain_terms = Vec::with_capacity(self.bound_k + 1);
        let mut mapping_terms = Vec::with_capacity(self.bound_k + 1);

        for step in 0..=self.bound_k {
            let dom_name = Self::function_domain_symbol(name, string_keys, step);
            let map_name = Self::function_mapping_symbol(name, string_keys, step);
            domain_terms.push(self.solver.declare_const(&dom_name, dom_sort.clone()));
            mapping_terms.push(self.solver.declare_const(&map_name, map_sort.clone()));
        }

        self.func_vars.insert(
            name.to_string(),
            BmcFuncVarInfo {
                range_sort,
                key_sort,
                finite_domain_keys: None,
                carrier_referenced: false,
                domain_terms,
                mapping_terms,
                symbolic_domain: None,
            },
        );
        Ok(())
    }

    /// Declare a finite function whose complete domain is part of its TLA sort.
    /// Unlike [`Self::declare_func_var`], this gives equality/UNCHANGED a finite
    /// and exact key cover, and pins every domain array to precisely those keys.
    fn declare_func_var_with_exact_domain(
        &mut self,
        name: &str,
        encoded_keys: &[String],
        range_sort: TlaSort,
    ) -> AYResult<()> {
        let (key_sort, keys) = Self::decode_finite_function_domain_keys(encoded_keys)?;
        self.ensure_declaration_carrier(name, BmcCarrierKind::Function)?;

        if let Some(existing) = self.func_vars.get(name) {
            if existing.symbolic_domain.is_none()
                && existing.key_sort.clone().canonicalized() == key_sort.clone().canonicalized()
                && existing.range_sort.clone().canonicalized() == range_sort.clone().canonicalized()
                && existing.finite_domain_keys.as_ref() == Some(&keys)
            {
                return Ok(());
            }
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: format!(
                    "finite function [{:?} -> {}]",
                    existing.finite_domain_keys, existing.range_sort
                ),
                actual: format!("finite function [{keys:?} -> {range_sort}]"),
            });
        }

        self.declare_func_var_with_key_sort(name, key_sort.clone(), range_sort)?;
        let domain_terms = self
            .func_vars
            .get(name)
            .expect("function was just declared")
            .domain_terms
            .clone();

        for domain_term in domain_terms {
            let false_default = if matches!(key_sort, TlaSort::String) {
                // AY currently interns const-arrays by the default value alone.
                // Give native-String domains a distinct false term so an earlier
                // `(Array Int Bool)` const-array cannot be reused at the wrong sort.
                let (_, fresh_false) =
                    self.declare_internal_const("exact string function domain false", Sort::Bool);
                let false_term = self.solver.bool_const(false);
                let pinned = self.solver.try_eq(fresh_false, false_term)?;
                self.solver
                    .try_assert_term(pinned)
                    .expect("invariant: equality is Bool-sorted");
                fresh_false
            } else {
                self.solver.bool_const(false)
            };
            let key_ay = if matches!(key_sort, TlaSort::String) {
                Sort::String
            } else {
                Sort::Int
            };
            let mut exact_domain = self.solver.try_const_array(key_ay, false_default)?;
            let present = self.solver.bool_const(true);
            for key in &keys {
                let key_term = match key {
                    BmcFunctionKey::Int(value) => self.solver.int_const(*value),
                    BmcFunctionKey::String(value) => self.solver.string_const(value),
                };
                exact_domain = self.solver.try_store(exact_domain, key_term, present)?;
            }
            let exact = self.solver.try_eq(domain_term, exact_domain)?;
            self.solver
                .try_assert_term(exact)
                .expect("invariant: equality is Bool-sorted");
        }

        let info = self
            .func_vars
            .get_mut(name)
            .expect("function was just declared");
        info.finite_domain_keys = Some(keys);
        info.carrier_referenced = true;
        Ok(())
    }

    /// Decode the typed domain-key spelling emitted by tla-check's sort
    /// inference. Legacy raw integer/string spellings remain accepted for the
    /// public `TlaSort` API, but mixed key kinds and unsupported Bool/model-value
    /// keys fail before any solver mutation.
    fn decode_finite_function_domain_keys(
        encoded_keys: &[String],
    ) -> AYResult<(TlaSort, Vec<BmcFunctionKey>)> {
        let mut keys = Vec::with_capacity(encoded_keys.len());
        for encoded in encoded_keys {
            let key = if let Some(value) = encoded.strip_prefix("int:") {
                BmcFunctionKey::Int(value.parse::<i64>().map_err(|_| {
                    AYError::UnsupportedOp(format!(
                        "BMC function domain has non-i64 integer key '{encoded}'"
                    ))
                })?)
            } else if let Some(value) = encoded.strip_prefix("str:") {
                BmcFunctionKey::String(value.to_string())
            } else if encoded.starts_with("bool:") || encoded.starts_with("id:") {
                return Err(AYError::UnsupportedOp(format!(
                    "BMC finite function domain key '{encoded}' is not an Int or String literal"
                )));
            } else if let Ok(value) = encoded.parse::<i64>() {
                BmcFunctionKey::Int(value)
            } else {
                BmcFunctionKey::String(encoded.clone())
            };
            keys.push(key);
        }
        keys.sort();
        keys.dedup();
        let has_int = keys.iter().any(|key| matches!(key, BmcFunctionKey::Int(_)));
        let has_string = keys
            .iter()
            .any(|key| matches!(key, BmcFunctionKey::String(_)));
        if has_int && has_string {
            return Err(AYError::UnsupportedOp(
                "BMC finite function domain mixes Int and String keys".to_string(),
            ));
        }
        let key_sort = if has_string {
            TlaSort::String
        } else {
            // The empty function has no observable key kind. Use the existing
            // Int carrier; logical equality special-cases its empty domain.
            TlaSort::Int
        };
        Ok((key_sort, keys))
    }

    /// Declare a symbolic-domain (map-only) function variable for all k+1 steps.
    ///
    /// The domain is the contiguous integer range `domain_lo ..
    /// (domain_hi_const + domain_hi_offset)` over an unbound symbolic
    /// `CONSTANT`. Only the canonically named mapping array is declared — there
    /// is no domain-membership array, because the domain is the arithmetic fact
    /// `lo <= x <= hi`, not an enumerable set. `x \in DOMAIN f` is translated to that bound
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
        if !range_sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "BMC symbolic-domain function range must be scalar, got {range_sort} for {name}"
            )));
        }
        let range_ay = range_sort.to_ay()?;
        self.ensure_declaration_carrier(name, BmcCarrierKind::Function)?;
        if let Some(existing) = self.func_vars.get(name) {
            let requested_domain = (domain_lo, domain_hi_const.clone(), domain_hi_offset);
            if existing.symbolic_domain.as_ref() == Some(&requested_domain)
                && existing.key_sort == TlaSort::Int
                && existing.range_sort.clone().canonicalized() == range_sort.clone().canonicalized()
            {
                self.register_base_var_name(name);
                return Ok(());
            }
            let expected = match &existing.symbolic_domain {
                Some((lo, hi, offset)) => {
                    format!(
                        "symbolic function [{lo}..{hi}{offset:+} -> {}]",
                        existing.range_sort
                    )
                }
                None => format!(
                    "finite function [{} -> {}]",
                    existing.key_sort, existing.range_sort
                ),
            };
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected,
                actual: format!(
                    "symbolic function [{domain_lo}..{domain_hi_const}{domain_hi_offset:+} -> \
                     {range_sort}]"
                ),
            });
        }
        let map_sort = Sort::array(Sort::Int, range_ay);
        let mut mapping_terms = Vec::with_capacity(self.bound_k + 1);
        for step in 0..=self.bound_k {
            let map_name = Self::symbolic_function_mapping_symbol(name, step);
            mapping_terms.push(self.solver.declare_const(&map_name, map_sort.clone()));
        }
        self.func_vars.insert(
            name.to_string(),
            BmcFuncVarInfo {
                range_sort,
                key_sort: TlaSort::Int,
                finite_domain_keys: None,
                carrier_referenced: false,
                domain_terms: Vec::new(),
                mapping_terms,
                symbolic_domain: Some((domain_lo, domain_hi_const, domain_hi_offset)),
            },
        );
        self.register_base_var_name(name);
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

    /// Compare two function mapping representations only at their exact live
    /// finite domain keys. Mapping-array cells outside DOMAIN are ghosts and
    /// cannot participate in TLA+ function equality or UNCHANGED.
    pub(super) fn translate_func_logical_mapping_eq(
        &mut self,
        left_name: &str,
        left_mapping: Term,
        right_name: &str,
        right_mapping: Term,
    ) -> AYResult<Term> {
        let left = self
            .func_vars
            .get(left_name)
            .ok_or_else(|| AYError::UnknownVariable(format!("function {left_name}")))?;
        let right = self
            .func_vars
            .get(right_name)
            .ok_or_else(|| AYError::UnknownVariable(format!("function {right_name}")))?;
        let left_keys = left.finite_domain_keys.clone().ok_or_else(|| {
            AYError::UnsupportedOp(format!(
                "BMC whole-function equality for {left_name} requires an exact finite domain"
            ))
        })?;
        let right_keys = right.finite_domain_keys.clone().ok_or_else(|| {
            AYError::UnsupportedOp(format!(
                "BMC whole-function equality for {right_name} requires an exact finite domain"
            ))
        })?;
        let left_range = left.range_sort.clone().canonicalized();
        let right_range = right.range_sort.clone().canonicalized();

        if left_keys != right_keys {
            return Ok(self.solver.bool_const(false));
        }
        if left_keys.is_empty() {
            // There is exactly one empty function, independent of inferred key
            // or range metadata, and no mapping cell is observable.
            return Ok(self.solver.bool_const(true));
        }
        if left_range != right_range {
            // Every live scalar value has a disjoint TLA+ kind.
            return Ok(self.solver.bool_const(false));
        }

        let mut result = self.solver.bool_const(true);
        for key in left_keys {
            let key_term = match key {
                BmcFunctionKey::Int(value) => self.solver.int_const(value),
                BmcFunctionKey::String(value) => self.solver.string_const(&value),
            };
            let left_value = self.solver.try_select(left_mapping, key_term)?;
            let right_value = self.solver.try_select(right_mapping, key_term)?;
            let value_eq = self.solver.try_eq(left_value, right_value)?;
            result = self.solver.try_and(result, value_eq)?;
        }
        Ok(result)
    }

    /// Upgrade an already-declared, still-Int-keyed function variable to a
    /// `String`-keyed encoding by re-declaring its domain/mapping arrays with a
    /// `String` index sort.
    ///
    /// This is only safe to call before any constraint has referenced the
    /// function's arrays (e.g. at the first `f = [k \in {"a"} |-> ...]`
    /// construction in `Init`/`Next`), because it replaces the array terms. We
    /// therefore make an already-`String` function an idempotent no-op, reject
    /// any referenced or symbolic carrier, and never downgrade. Fresh `String`-sorted array
    /// constants are introduced with distinct names so prior `Int`-indexed
    /// constants (if somehow referenced) remain well-sorted but unconstrained.
    /// (Part of #5.)
    pub(crate) fn upgrade_func_key_sort_to_string(&mut self, name: &str) -> AYResult<()> {
        match self.func_vars.get(name) {
            Some(info) => {
                if matches!(info.key_sort, TlaSort::String) {
                    return Ok(());
                }
                if info.symbolic_domain.is_some() {
                    return Err(AYError::UnsupportedOp(format!(
                        "cannot replace symbolic-domain function {name} with a String-keyed carrier"
                    )));
                }
                if info.carrier_referenced {
                    return Err(AYError::UnsupportedOp(format!(
                        "cannot upgrade function {name} to String keys after its Int carrier was referenced"
                    )));
                }
            }
            None => {
                return Err(AYError::UnknownVariable(format!(
                    "function {name} (upgrade to string-keyed)"
                )))
            }
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
            let dom_name = Self::function_domain_symbol(name, true, step);
            let map_name = Self::function_mapping_symbol(name, true, step);
            domain_terms.push(self.solver.declare_const(&dom_name, dom_sort.clone()));
            mapping_terms.push(self.solver.declare_const(&map_name, map_sort.clone()));
        }

        let info = self.func_vars.get_mut(name).expect("checked above");
        info.key_sort = TlaSort::String;
        info.finite_domain_keys = None;
        info.domain_terms = domain_terms;
        info.mapping_terms = mapping_terms;
        Ok(())
    }

    /// Get the mapping array term for a function variable at a specific step.
    ///
    /// Part of #3786.
    pub(crate) fn get_func_mapping_at_step(&mut self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .func_vars
            .get_mut(name)
            .ok_or_else(|| AYError::UnknownVariable(format!("function {name} (at step {step})")))?;
        if step > self.bound_k {
            return Err(AYError::UntranslatableExpr(format!(
                "step {step} exceeds bound {}",
                self.bound_k
            )));
        }
        info.carrier_referenced = true;
        Ok(info.mapping_terms[step])
    }

    /// Get the domain set term for a function variable at a specific step.
    ///
    /// Part of #3786.
    pub(crate) fn get_func_domain_at_step(&mut self, name: &str, step: usize) -> AYResult<Term> {
        let info = self
            .func_vars
            .get_mut(name)
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
        info.carrier_referenced = true;
        Ok(info.domain_terms[step])
    }

    /// Declare a sequence state variable for all k+1 steps.
    ///
    /// Each sequence is encoded as a canonically named SMT array plus length per
    /// step, produced by [`Self::sequence_array_symbol`] and
    /// [`Self::sequence_length_symbol`].
    ///
    /// The element sort must be scalar (Bool, Int, or String).
    /// Length is constrained to `0 <= len <= max_len` at each step.
    ///
    /// Idempotent only for the same element sort and maximum length.
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
        if !element_sort.is_scalar() {
            return Err(AYError::UnsupportedOp(format!(
                "BMC sequence element must be scalar, got {element_sort} for sequence {name}"
            )));
        }

        let arr_sort = Sort::array(Sort::Int, element_sort.to_ay()?);
        let max_len_i64 = i64::try_from(max_len).map_err(|_| {
            AYError::UnsupportedOp(format!(
                "BMC sequence maximum length {max_len} exceeds the SMT integer bound"
            ))
        })?;
        self.ensure_declaration_carrier(name, BmcCarrierKind::Sequence)?;
        if let Some(existing) = self.seq_vars.get(name) {
            if existing.element_sort.clone().canonicalized() == element_sort.clone().canonicalized()
                && existing.max_len == max_len
            {
                return Ok(());
            }
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: format!("Seq({}, max={})", existing.element_sort, existing.max_len),
                actual: format!("Seq({element_sort}, max={max_len})"),
            });
        }

        let mut array_terms = Vec::with_capacity(self.bound_k + 1);
        let mut length_terms = Vec::with_capacity(self.bound_k + 1);

        for step in 0..=self.bound_k {
            let arr_name = Self::sequence_array_symbol(name, step);
            let len_name = Self::sequence_length_symbol(name, step);
            let arr = self.solver.declare_const(&arr_name, arr_sort.clone());
            let len = self.solver.declare_const(&len_name, Sort::Int);

            // Constrain: 0 <= len <= max_len
            let zero = self.solver.int_const(0);
            let max = self.solver.int_const(max_len_i64);
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
        // Declare fresh auxiliary variables
        let (_, q) = self.declare_internal_const("division quotient", Sort::Int);
        let (_, r) = self.declare_internal_const("division remainder", Sort::Int);

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
            let equality = self.make_value_eq(name, value, step)?;
            self.assert(equality);
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

        // Blocking a projection would exclude every full state sharing that
        // projection and can silently skip reachable models. Require exactly
        // the public base carrier set; temporary Skolem bindings are not state
        // variables, while direct compound declarations are base carriers even
        // when they did not pass through `declare_var`.
        let mut expected = std::collections::BTreeSet::new();
        expected.extend(self.base_var_names.iter().map(String::as_str));
        expected.extend(self.func_vars.keys().map(String::as_str));
        expected.extend(self.seq_vars.keys().map(String::as_str));
        expected.extend(self.record_vars.keys().map(String::as_str));
        expected.extend(self.tuple_vars.keys().map(String::as_str));
        let actual: std::collections::BTreeSet<&str> =
            state.assignments.keys().map(String::as_str).collect();
        if actual != expected {
            let missing = expected
                .difference(&actual)
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
            let unexpected = actual
                .difference(&expected)
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(AYError::UntranslatableExpr(format!(
                "cannot block partial concrete state at step {}: missing [{}], unexpected [{}]",
                state.step, missing, unexpected
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

    /// Stable diagnostic name for a concrete BMC value kind.
    fn bmc_value_kind(value: &BmcValue) -> &'static str {
        match value {
            BmcValue::Bool(_) => "Bool",
            BmcValue::Int(_) | BmcValue::BigInt(_) => "Int",
            BmcValue::String(_) => "String",
            BmcValue::Set(_) => "Set",
            BmcValue::Sequence(_) => "Sequence",
            BmcValue::Function(_) => "Int-keyed Function",
            BmcValue::StringFunction(_) => "String-keyed Function",
            BmcValue::Record(_) => "Record",
            BmcValue::Tuple(_) => "Tuple",
        }
    }

    fn value_type_mismatch(name: &str, expected: &TlaSort, value: &BmcValue) -> AYError {
        AYError::TypeMismatch {
            name: name.to_string(),
            expected: expected.to_string(),
            actual: Self::bmc_value_kind(value).to_string(),
        }
    }

    fn scalar_value_term(
        &mut self,
        name: &str,
        value: &BmcValue,
        sort: &TlaSort,
    ) -> AYResult<Term> {
        match (sort, value) {
            (TlaSort::Bool, BmcValue::Bool(value)) => Ok(self.solver.bool_const(*value)),
            (TlaSort::Int, BmcValue::Int(value)) => Ok(self.solver.int_const(*value)),
            (TlaSort::Int, BmcValue::BigInt(value)) => Ok(self.solver.int_const_bigint(value)),
            (TlaSort::String, BmcValue::String(value)) => {
                let id = self.bmc_intern_string(value);
                Ok(self.solver.int_const(id))
            }
            _ => Err(Self::value_type_mismatch(name, sort, value)),
        }
    }

    fn set_value_term(
        &mut self,
        name: &str,
        members: &[BmcValue],
        element_sort: &TlaSort,
    ) -> AYResult<Term> {
        let false_value = self.solver.bool_const(false);
        let true_value = self.solver.bool_const(true);
        let mut array = self.solver.try_const_array(Sort::Int, false_value)?;
        for member in members {
            let key_term = match (element_sort, member) {
                (TlaSort::Int, BmcValue::Int(value)) => {
                    self.tracked_universe_ints.insert(*value);
                    self.solver.int_const(*value)
                }
                (TlaSort::Int, BmcValue::BigInt(value)) => {
                    use num_traits::ToPrimitive;
                    if let Some(value) = value.to_i64() {
                        self.tracked_universe_ints.insert(value);
                        self.solver.int_const(value)
                    } else {
                        self.solver.int_const_bigint(value)
                    }
                }
                (TlaSort::String, BmcValue::String(value)) => {
                    let key = self.bmc_intern_string(value);
                    self.tracked_universe_ints.insert(key);
                    self.solver.int_const(key)
                }
                _ => {
                    return Err(AYError::TypeMismatch {
                        name: name.to_string(),
                        expected: format!("set member of sort {element_sort}"),
                        actual: Self::bmc_value_kind(member).to_string(),
                    });
                }
            };
            array = self.solver.try_store(array, key_term, true_value)?;
        }
        Ok(array)
    }

    fn value_term_for_sort(
        &mut self,
        name: &str,
        value: &BmcValue,
        sort: &TlaSort,
    ) -> AYResult<Term> {
        match (sort, value) {
            (TlaSort::Set { element_sort }, BmcValue::Set(members)) => {
                self.set_value_term(name, members, element_sort)
            }
            _ => self.scalar_value_term(name, value, sort),
        }
    }

    fn conjunction(&mut self, mut terms: Vec<Term>) -> AYResult<Term> {
        let Some(mut result) = terms.pop() else {
            return Ok(self.solver.bool_const(true));
        };
        for term in terms.into_iter().rev() {
            result = self.solver.try_and(term, result)?;
        }
        Ok(result)
    }

    fn make_int_function_value_eq(
        &mut self,
        name: &str,
        entries: &[(i64, BmcValue)],
        step: usize,
    ) -> AYResult<Term> {
        let (key_sort, range_sort, symbolic_domain) = {
            let info = self
                .func_vars
                .get(name)
                .ok_or_else(|| AYError::UnknownVariable(format!("function {name}")))?;
            (
                info.key_sort.clone(),
                info.range_sort.clone(),
                info.symbolic_domain.clone(),
            )
        };
        if key_sort != TlaSort::Int {
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: format!("{key_sort}-keyed function"),
                actual: "Int-keyed Function".to_string(),
            });
        }

        let map_term = self.get_func_mapping_at_step(name, step)?;
        let mut conjuncts = Vec::with_capacity(entries.len() + 1);
        let mut seen = std::collections::BTreeSet::new();
        for (key, _) in entries {
            if !seen.insert(*key) {
                return Err(AYError::UnsupportedOp(format!(
                    "concrete function '{name}' repeats key {key}"
                )));
            }
        }
        if let Some((domain_lo, domain_hi_const, domain_hi_offset)) = symbolic_domain {
            let hi_info = self.vars.get(&domain_hi_const).ok_or_else(|| {
                AYError::UnknownVariable(format!(
                    "rigid upper-bound constant {domain_hi_const} for symbolic function {name}"
                ))
            })?;
            if !self.rigid_const_names.contains(&domain_hi_const) || hi_info.sort != TlaSort::Int {
                return Err(AYError::TypeMismatch {
                    name: domain_hi_const,
                    expected: "rigid Int constant".to_string(),
                    actual: hi_info.sort.to_string(),
                });
            }

            // A map-only symbolic function has the exact logical domain
            // `domain_lo..(N + offset)`. Constraining only the listed map cells
            // would accept a partial concrete function. Prove that the supplied
            // keys cover that entire interval by checking their static shape and
            // tying its upper endpoint to the rigid bound.
            if let Some((first, _)) = entries.first() {
                if *first != domain_lo
                    || entries
                        .windows(2)
                        .any(|pair| pair[0].0.checked_add(1) != Some(pair[1].0))
                {
                    return Err(AYError::UnsupportedOp(format!(
                        "concrete symbolic-domain function '{name}' keys must be exactly contiguous from {domain_lo}"
                    )));
                }
            }

            let hi_base = self.get_var_at_step(&domain_hi_const, step)?;
            let offset = self.solver.int_const(domain_hi_offset);
            let symbolic_hi = self.solver.try_add(hi_base, offset)?;
            if let Some((last, _)) = entries.last() {
                let concrete_hi = self.solver.int_const(*last);
                conjuncts.push(self.solver.try_eq(symbolic_hi, concrete_hi)?);
            } else {
                let concrete_lo = self.solver.int_const(domain_lo);
                conjuncts.push(self.solver.try_lt(symbolic_hi, concrete_lo)?);
            }
        } else {
            let domain_term = self.get_func_domain_at_step(name, step)?;
            let false_value = self.solver.bool_const(false);
            let true_value = self.solver.bool_const(true);
            let mut expected_domain = self.solver.try_const_array(Sort::Int, false_value)?;
            for (key, _) in entries {
                let key_term = self.solver.int_const(*key);
                expected_domain = self
                    .solver
                    .try_store(expected_domain, key_term, true_value)?;
            }
            conjuncts.push(self.solver.try_eq(domain_term, expected_domain)?);
        }

        for (key, value) in entries {
            let key_term = self.solver.int_const(*key);
            let selected = self.solver.try_select(map_term, key_term)?;
            let value_term = self.scalar_value_term(name, value, &range_sort)?;
            conjuncts.push(self.solver.try_eq(selected, value_term)?);
        }
        self.conjunction(conjuncts)
    }

    fn make_string_function_value_eq(
        &mut self,
        name: &str,
        entries: &[(String, BmcValue)],
        step: usize,
    ) -> AYResult<Term> {
        let (key_sort, range_sort, symbolic) = {
            let info = self
                .func_vars
                .get(name)
                .ok_or_else(|| AYError::UnknownVariable(format!("function {name}")))?;
            (
                info.key_sort.clone(),
                info.range_sort.clone(),
                info.symbolic_domain.is_some(),
            )
        };
        if symbolic || key_sort != TlaSort::String {
            return Err(AYError::TypeMismatch {
                name: name.to_string(),
                expected: format!("{key_sort}-keyed function"),
                actual: "String-keyed Function".to_string(),
            });
        }

        let domain_term = self.get_func_domain_at_step(name, step)?;
        let map_term = self.get_func_mapping_at_step(name, step)?;
        // AY currently interns const-arrays by the default value alone. A
        // shared `false` default can therefore alias an earlier Int-indexed
        // domain array. Give this String-indexed array a fresh Bool carrier and
        // pin it to false, matching native String-domain declaration.
        let (_, false_value) =
            self.declare_internal_const("concrete string function domain false", Sort::Bool);
        let shared_false = self.solver.bool_const(false);
        let false_value_is_false = self.solver.try_eq(false_value, shared_false)?;
        self.solver.try_assert_term(false_value_is_false)?;
        let true_value = self.solver.bool_const(true);
        let mut expected_domain = self.solver.try_const_array(Sort::String, false_value)?;
        let mut conjuncts = Vec::with_capacity(entries.len() + 1);
        let mut seen = std::collections::BTreeSet::new();
        for (key, _) in entries {
            if !seen.insert(key.as_str()) {
                return Err(AYError::UnsupportedOp(format!(
                    "concrete function '{name}' repeats key {key:?}"
                )));
            }
            let key_term = self.solver.string_const(key);
            expected_domain = self
                .solver
                .try_store(expected_domain, key_term, true_value)?;
        }
        conjuncts.push(self.solver.try_eq(domain_term, expected_domain)?);

        for (key, value) in entries {
            let key_term = self.solver.string_const(key);
            let selected = self.solver.try_select(map_term, key_term)?;
            let value_term = self.scalar_value_term(name, value, &range_sort)?;
            conjuncts.push(self.solver.try_eq(selected, value_term)?);
        }
        self.conjunction(conjuncts)
    }

    /// Build an equality term between one declared carrier and a concrete value.
    /// Every compound component is encoded according to declaration metadata;
    /// Bool carriers never pass through an Int 0/1 surrogate.
    fn make_value_eq(&mut self, name: &str, value: &BmcValue, step: usize) -> AYResult<Term> {
        match value {
            BmcValue::Bool(_) | BmcValue::Int(_) | BmcValue::BigInt(_) | BmcValue::String(_) => {
                let sort = self
                    .vars
                    .get(name)
                    .ok_or_else(|| AYError::UnknownVariable(name.to_string()))?
                    .sort
                    .clone();
                let variable = self.get_var_at_step(name, step)?;
                let concrete = self.scalar_value_term(name, value, &sort)?;
                Ok(self.solver.try_eq(variable, concrete)?)
            }
            BmcValue::Set(members) => {
                let sort = self
                    .vars
                    .get(name)
                    .ok_or_else(|| AYError::UnknownVariable(name.to_string()))?
                    .sort
                    .clone();
                let TlaSort::Set { element_sort } = sort else {
                    return Err(Self::value_type_mismatch(name, &sort, value));
                };
                let variable = self.get_var_at_step(name, step)?;
                let concrete = self.set_value_term(name, members, &element_sort)?;
                Ok(self.solver.try_eq(variable, concrete)?)
            }
            BmcValue::Sequence(elements) => {
                let (element_sort, max_len) = {
                    let info = self
                        .seq_vars
                        .get(name)
                        .ok_or_else(|| AYError::UnknownVariable(format!("sequence {name}")))?;
                    (info.element_sort.clone(), info.max_len)
                };
                if elements.len() > max_len {
                    return Err(AYError::UnsupportedOp(format!(
                        "concrete sequence '{name}' length {} exceeds declared maximum {max_len}",
                        elements.len()
                    )));
                }
                let length = i64::try_from(elements.len()).map_err(|_| {
                    AYError::UnsupportedOp(format!(
                        "concrete sequence '{name}' length cannot be encoded as SMT Int"
                    ))
                })?;
                let array = self.get_seq_array_at_step(name, step)?;
                let length_term = self.get_seq_length_at_step(name, step)?;
                let length_value = self.solver.int_const(length);
                let mut conjuncts = vec![self.solver.try_eq(length_term, length_value)?];
                for (offset, value) in elements.iter().enumerate() {
                    let index = self.solver.int_const((offset + 1) as i64);
                    let selected = self.solver.try_select(array, index)?;
                    let concrete = self.scalar_value_term(name, value, &element_sort)?;
                    conjuncts.push(self.solver.try_eq(selected, concrete)?);
                }
                self.conjunction(conjuncts)
            }
            BmcValue::Function(entries) => self.make_int_function_value_eq(name, entries, step),
            BmcValue::StringFunction(entries) => {
                self.make_string_function_value_eq(name, entries, step)
            }
            BmcValue::Record(fields) => {
                let field_sorts = self
                    .record_vars
                    .get(name)
                    .ok_or_else(|| AYError::UnknownVariable(format!("record {name}")))?
                    .field_sorts
                    .clone();
                if fields.len() != field_sorts.len() {
                    return Err(AYError::TypeMismatch {
                        name: name.to_string(),
                        expected: format!("record with {} fields", field_sorts.len()),
                        actual: format!("record with {} fields", fields.len()),
                    });
                }
                let mut seen = std::collections::HashSet::with_capacity(fields.len());
                let mut conjuncts = Vec::with_capacity(fields.len());
                for (field_name, value) in fields {
                    if !seen.insert(field_name.as_str()) {
                        return Err(AYError::UnsupportedOp(format!(
                            "concrete record '{name}' repeats field '{field_name}'"
                        )));
                    }
                    let sort = field_sorts
                        .iter()
                        .find(|(candidate, _)| candidate == field_name)
                        .map(|(_, sort)| sort)
                        .ok_or_else(|| {
                            AYError::UnsupportedOp(format!(
                                "concrete record '{name}' has unknown field '{field_name}'"
                            ))
                        })?;
                    let field = self.get_record_field_at_step(name, field_name, step)?;
                    let concrete = self.value_term_for_sort(name, value, sort)?;
                    conjuncts.push(self.solver.try_eq(field, concrete)?);
                }
                self.conjunction(conjuncts)
            }
            BmcValue::Tuple(elements) => {
                let element_sorts = self
                    .tuple_vars
                    .get(name)
                    .ok_or_else(|| AYError::UnknownVariable(format!("tuple {name}")))?
                    .element_sorts
                    .clone();
                if elements.len() != element_sorts.len() {
                    return Err(AYError::TypeMismatch {
                        name: name.to_string(),
                        expected: format!("tuple with {} elements", element_sorts.len()),
                        actual: format!("tuple with {} elements", elements.len()),
                    });
                }
                let mut conjuncts = Vec::with_capacity(elements.len());
                for (offset, (value, sort)) in elements.iter().zip(&element_sorts).enumerate() {
                    let element = self.get_tuple_element_at_step(name, offset + 1, step)?;
                    let concrete = self.scalar_value_term(name, value, sort)?;
                    conjuncts.push(self.solver.try_eq(element, concrete)?);
                }
                self.conjunction(conjuncts)
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

        let step_symbol = BmcTranslator::state_step_symbol("x", 0);
        assert_eq!(direct.int_val(&step_symbol).cloned(), Some(3.into()));
        assert_eq!(wrapped.int_val(&step_symbol).cloned(), Some(3.into()));
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

    #[test]
    fn block_concrete_state_rejects_partial_base_carrier_projection() {
        let mut translator = BmcTranslator::new(0).expect("translator");
        translator
            .declare_var("x", TlaSort::Int)
            .expect("declare x");
        translator
            .declare_var("ready", TlaSort::Bool)
            .expect("declare ready");
        let state = BmcState {
            step: 0,
            assignments: HashMap::from([("x".to_string(), BmcValue::Int(1))]),
        };

        assert!(matches!(
            translator.block_concrete_state(&state),
            Err(AYError::UntranslatableExpr(message))
                if message.contains("missing [ready]") && message.contains("unexpected []")
        ));
    }

    #[test]
    fn arbitrary_precision_int_round_trips_and_can_be_blocked() {
        let mut translator = BmcTranslator::new(0).expect("translator");
        translator
            .declare_var("x", TlaSort::Int)
            .expect("declare x");
        let large = num_bigint::BigInt::from(i64::MAX) + 1u8;
        translator
            .assert_concrete_state(&[("x".to_string(), BmcValue::BigInt(large.clone()))], 0)
            .expect("assert arbitrary-precision state");

        assert_eq!(
            translator.try_check_sat().expect("solver result"),
            SolveResult::Sat
        );
        let model = translator.try_get_model().expect("model");
        let state = translator
            .extract_trace(&model)
            .into_iter()
            .next()
            .expect("step zero");
        assert_eq!(state.assignments.get("x"), Some(&BmcValue::BigInt(large)));

        translator
            .block_concrete_state(&state)
            .expect("block arbitrary-precision state");
        assert!(matches!(
            translator
                .try_check_sat()
                .expect("solver result after block"),
            SolveResult::Unsat(_)
        ));
    }

    #[test]
    fn arbitrary_precision_set_member_round_trips_and_can_be_blocked() {
        let mut translator = BmcTranslator::new_with_arrays(0).expect("translator");
        translator
            .declare_var(
                "set",
                TlaSort::Set {
                    element_sort: Box::new(TlaSort::Int),
                },
            )
            .expect("declare Int set");
        let large = num_bigint::BigInt::from(i64::MAX) + 1u8;
        let value = BmcValue::Set(vec![BmcValue::BigInt(large)]);
        translator
            .assert_concrete_state(&[("set".to_string(), value.clone())], 0)
            .expect("assert arbitrary-precision set");

        assert_eq!(
            translator.try_check_sat().expect("solver result"),
            SolveResult::Sat
        );
        let model = translator.try_get_model().expect("model");
        let state = translator
            .extract_trace(&model)
            .into_iter()
            .next()
            .expect("step zero");
        assert_eq!(state.assignments.get("set"), Some(&value));

        translator
            .block_concrete_state(&state)
            .expect("block arbitrary-precision set state");
        assert!(matches!(
            translator
                .try_check_sat()
                .expect("solver result after block"),
            SolveResult::Unsat(_)
        ));
    }

    #[test]
    fn symbolic_function_concrete_value_rejects_partial_domain() {
        let mut translator = BmcTranslator::new_with_arrays(0).expect("translator");
        translator
            .declare_rigid_const("N", TlaSort::Int)
            .expect("declare rigid bound");
        translator
            .declare_funcsym_var("f", 1, "N".to_string(), 0, TlaSort::Bool)
            .expect("declare symbolic-domain function");
        translator
            .assert_concrete_state(
                &[
                    ("N".to_string(), BmcValue::Int(2)),
                    (
                        "f".to_string(),
                        BmcValue::Function(vec![(1, BmcValue::Bool(true))]),
                    ),
                ],
                0,
            )
            .expect("encode partial concrete function");

        assert!(matches!(
            translator.try_check_sat().expect("solver result"),
            SolveResult::Unsat(_)
        ));
    }

    #[test]
    fn symbolic_function_concrete_value_rejects_gapped_keys_before_solving() {
        let mut translator = BmcTranslator::new_with_arrays(0).expect("translator");
        translator
            .declare_rigid_const("N", TlaSort::Int)
            .expect("declare rigid bound");
        translator
            .declare_funcsym_var("f", 1, "N".to_string(), 0, TlaSort::Int)
            .expect("declare symbolic-domain function");

        assert!(matches!(
            translator.assert_concrete_state(
                &[(
                    "f".to_string(),
                    BmcValue::Function(vec![
                        (1, BmcValue::Int(10)),
                        (3, BmcValue::Int(30)),
                    ]),
                )],
                0,
            ),
            Err(AYError::UnsupportedOp(message))
                if message.contains("keys must be exactly contiguous from 1")
        ));
    }

    #[test]
    fn string_function_concrete_domain_stays_disjoint_from_int_const_array_cache() {
        let mut translator = BmcTranslator::new_with_arrays(0).expect("translator");
        translator
            .declare_func_var_with_key_sort("ints", TlaSort::Int, TlaSort::Bool)
            .expect("declare Int-keyed function");
        translator
            .declare_func_var_with_key_sort("strings", TlaSort::String, TlaSort::Bool)
            .expect("declare String-keyed function");
        let ints = BmcValue::Function(vec![(1, BmcValue::Bool(true))]);
        let strings = BmcValue::StringFunction(vec![("one".to_string(), BmcValue::Bool(false))]);
        translator
            .assert_concrete_state(
                &[
                    ("ints".to_string(), ints.clone()),
                    ("strings".to_string(), strings.clone()),
                ],
                0,
            )
            .expect("encode both index sorts");

        assert_eq!(
            translator.try_check_sat().expect("solver result"),
            SolveResult::Sat
        );
        let model = translator.try_get_model().expect("model");
        let state = translator
            .extract_trace(&model)
            .into_iter()
            .next()
            .expect("step zero");
        assert_eq!(state.assignments.get("ints"), Some(&ints));
        assert_eq!(state.assignments.get("strings"), Some(&strings));
    }
}
