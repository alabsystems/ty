// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode instruction set for TLA+ evaluation.
//!
//! Register-based instruction set covering non-temporal TLA+ operations.
//! Each instruction is a fixed-size enum variant — no variable-length encoding.
//! This keeps dispatch simple and makes the bytecode suitable for trust-ir/trust_cg
//! lowering.

pub use super::opcode_support::{
    BuiltinOp, ConstIdx, FieldIdx, JumpOffset, OpIdx, Register, VarIdx,
};

/// Bytecode instruction for the TLA+ VM.
///
/// Each variant is a single operation. The VM executes instructions linearly
/// unless a jump/branch redirects the program counter.
///
/// Register conventions:
/// - `rd`: destination register
/// - `r1`, `r2`: source registers
/// - `rs`: single source register
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    // =================================================================
    // Value Loading
    // =================================================================
    /// Load an immediate i64 constant into a register.
    /// Covers the common case of small integer literals.
    LoadImm {
        /// Destination register the immediate is written to.
        rd: Register,
        /// The immediate integer literal to load.
        value: i64,
    },

    /// Load a boolean constant.
    LoadBool {
        /// Destination register the boolean is written to.
        rd: Register,
        /// The boolean literal to load.
        value: bool,
    },

    /// Load a constant from the constant pool (for compound values, big
    /// integers, strings, etc.).
    LoadConst {
        /// Destination register the pooled constant is written to.
        rd: Register,
        /// Constant-pool index of the value to load.
        idx: ConstIdx,
    },

    /// Load a state variable by its pre-computed index.
    LoadVar {
        /// Destination register the variable value is written to.
        rd: Register,
        /// Index of the state variable to read (from current state, or
        /// next-state when the VM is in prime mode).
        var_idx: VarIdx,
    },

    /// Load a primed state variable (from the successor state).
    LoadPrime {
        /// Destination register the primed value is written to.
        rd: Register,
        /// Index of the state variable to read from the successor state.
        var_idx: VarIdx,
    },

    /// Store a value into the working successor state.
    StoreVar {
        /// Index of the successor-state variable to write.
        var_idx: VarIdx,
        /// Source register holding the value to store.
        rs: Register,
    },

    /// Copy a register value to another register.
    Move {
        /// Destination register the value is copied to.
        rd: Register,
        /// Source register the value is copied from.
        rs: Register,
    },

    // =================================================================
    // Integer Arithmetic (i64 fast path)
    // =================================================================
    /// rd = r1 + r2 (integer addition).
    AddInt {
        /// Destination register the sum is written to.
        rd: Register,
        /// Register holding the left addend.
        r1: Register,
        /// Register holding the right addend.
        r2: Register,
    },

    /// rd = r1 - r2 (integer subtraction).
    SubInt {
        /// Destination register the difference is written to.
        rd: Register,
        /// Register holding the minuend.
        r1: Register,
        /// Register holding the subtrahend.
        r2: Register,
    },

    /// rd = r1 * r2 (integer multiplication).
    MulInt {
        /// Destination register the product is written to.
        rd: Register,
        /// Register holding the first factor.
        r1: Register,
        /// Register holding the second factor.
        r2: Register,
    },

    /// rd = r1 / r2 (real division — TLA+ `/` is real, but in model checking
    /// context usually integer. Produces an error if not exact.)
    DivInt {
        /// Destination register the quotient is written to.
        rd: Register,
        /// Register holding the dividend.
        r1: Register,
        /// Register holding the divisor.
        r2: Register,
    },

    /// rd = r1 \div r2 (Euclidean integer division, TLC semantics).
    IntDiv {
        /// Destination register the quotient is written to.
        rd: Register,
        /// Register holding the dividend.
        r1: Register,
        /// Register holding the divisor.
        r2: Register,
    },

    /// rd = r1 % r2 (Euclidean modulus, TLC semantics).
    ModInt {
        /// Destination register the remainder is written to.
        rd: Register,
        /// Register holding the dividend.
        r1: Register,
        /// Register holding the divisor.
        r2: Register,
    },

    /// rd = -rs (integer negation).
    NegInt {
        /// Destination register the negated value is written to.
        rd: Register,
        /// Register holding the operand to negate.
        rs: Register,
    },

    /// rd = r1 ^ r2 (integer exponentiation).
    PowInt {
        /// Destination register the power is written to.
        rd: Register,
        /// Register holding the base.
        r1: Register,
        /// Register holding the exponent.
        r2: Register,
    },

    // =================================================================
    // Comparison
    // =================================================================
    /// rd = (r1 = r2), polymorphic equality.
    Eq {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    /// rd = (value = <<value[1], value[2]>>), fused tuple-shape test.
    ///
    /// The VM may decide this from the length of a direct tuple/sequence when
    /// both projections are in-domain. Every other representation follows the
    /// ordinary projection-then-equality path, preserving projection errors.
    Tuple2SelfEq {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the value used by both projections and equality.
        value: Register,
    },

    /// Fused exact conjunction
    /// `value = <<value[1], value[2]>> /\ {value[1], value[2]} \subseteq set_var`.
    ///
    /// This VM-only opcode preserves conjunction short-circuiting internally:
    /// `set_var_idx` is not read unless the tuple-shape equality succeeds.
    /// Direct two-element tuples/sequences reuse their elements; every other
    /// representation preserves the ordinary ordered projection path.
    Tuple2SelfSubseteq {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the value used by the equality and subset test.
        value: Register,
        /// State-variable slot containing the right-hand set.
        set_var_idx: VarIdx,
    },

    /// rd = (r1 /= r2), polymorphic inequality.
    Neq {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    /// rd = (r1 < r2), integer comparison.
    LtInt {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    /// rd = (r1 <= r2), integer comparison.
    LeInt {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    /// rd = (r1 > r2), integer comparison.
    GtInt {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    /// rd = (r1 >= r2), integer comparison.
    GeInt {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    // =================================================================
    // Boolean Operations
    // =================================================================
    /// rd = r1 /\ r2 (conjunction). NOT short-circuit — use JumpFalse for
    /// short-circuit evaluation.
    And {
        /// Destination register the conjunction result is written to.
        rd: Register,
        /// Register holding the left conjunct.
        r1: Register,
        /// Register holding the right conjunct.
        r2: Register,
    },

    /// rd = r1 \/ r2 (disjunction). NOT short-circuit.
    Or {
        /// Destination register the disjunction result is written to.
        rd: Register,
        /// Register holding the left disjunct.
        r1: Register,
        /// Register holding the right disjunct.
        r2: Register,
    },

    /// rd = ~rs (boolean negation).
    Not {
        /// Destination register the negated boolean is written to.
        rd: Register,
        /// Register holding the boolean to negate.
        rs: Register,
    },

    /// rd = (r1 => r2) (logical implication).
    Implies {
        /// Destination register the implication result is written to.
        rd: Register,
        /// Register holding the antecedent.
        r1: Register,
        /// Register holding the consequent.
        r2: Register,
    },

    /// rd = (r1 <=> r2) (logical equivalence).
    Equiv {
        /// Destination register the equivalence result is written to.
        rd: Register,
        /// Register holding the left operand.
        r1: Register,
        /// Register holding the right operand.
        r2: Register,
    },

    // =================================================================
    // Control Flow
    // =================================================================
    /// Unconditional jump.
    Jump {
        /// Signed offset added to this instruction's PC to form the jump target.
        offset: JumpOffset,
    },

    /// Jump if register is TRUE.
    JumpTrue {
        /// Register holding the boolean condition tested.
        rs: Register,
        /// Signed offset added to this instruction's PC, taken when `rs` is TRUE.
        offset: JumpOffset,
    },

    /// Jump if register is FALSE.
    JumpFalse {
        /// Register holding the boolean condition tested.
        rs: Register,
        /// Signed offset added to this instruction's PC, taken when `rs` is FALSE.
        offset: JumpOffset,
    },

    /// Call a user-defined operator. `argc` arguments are in registers
    /// starting at `args_start`. Result goes to `rd`.
    Call {
        /// Destination register the call's return value is written to.
        rd: Register,
        /// Index of the callee operator in the chunk's function table.
        op_idx: OpIdx,
        /// First register of the contiguous argument block.
        args_start: Register,
        /// Number of arguments passed (and the callee's expected arity).
        argc: u8,
    },

    /// Apply a runtime value as an operator/function.
    ///
    /// Used for higher-order `Apply` where the callee is not a statically
    /// resolved operator name. Closures use the full `argc` argument vector;
    /// ordinary function-like values accept exactly one argument.
    ValueApply {
        /// Destination register the application result is written to.
        rd: Register,
        /// Register holding the callable value (closure or function-like value).
        func: Register,
        /// First register of the contiguous argument block.
        args_start: Register,
        /// Number of arguments passed.
        argc: u8,
    },

    /// Return from the current function, yielding the value in `rs`.
    Ret {
        /// Register holding the value to return.
        rs: Register,
    },

    // =================================================================
    // Set Operations
    // =================================================================
    /// Build a set from `count` consecutive registers starting at `start`.
    SetEnum {
        /// Destination register the built set is written to.
        rd: Register,
        /// First register of the contiguous element block.
        start: Register,
        /// Number of element registers to gather into the set.
        count: u8,
    },

    /// rd = (r_elem \in r_set), set membership.
    SetIn {
        /// Destination register the boolean membership result is written to.
        rd: Register,
        /// Register holding the candidate element.
        elem: Register,
        /// Register holding the set to test membership in.
        set: Register,
    },

    /// rd = (<<r_first, r_second>> \in r_set), fused two-element tuple membership.
    ///
    /// This is semantically equivalent to constructing a two-element tuple and
    /// applying [`Opcode::SetIn`], without materializing the tuple first.
    Tuple2SetIn {
        /// Destination register the boolean membership result is written to.
        rd: Register,
        /// Register holding the first tuple element.
        first: Register,
        /// Register holding the second tuple element.
        second: Register,
        /// Register holding the set to test membership in.
        set: Register,
    },

    /// rd = ({r_start, ...} \subseteq r_set), fused set-enum subset test.
    ///
    /// This is semantically equivalent to constructing a set with
    /// [`Opcode::SetEnum`] and applying [`Opcode::Subseteq`], without
    /// materializing the left set when the right operand is a concrete set.
    SetEnumSubseteq {
        /// Destination register the boolean subset result is written to.
        rd: Register,
        /// First register of the contiguous left-set element block.
        start: Register,
        /// Number of element registers in the left-set block.
        count: u8,
        /// Register holding the candidate superset.
        set: Register,
    },

    /// rd = r1 \union r2.
    SetUnion {
        /// Destination register the union set is written to.
        rd: Register,
        /// Register holding the left set operand.
        r1: Register,
        /// Register holding the right set operand.
        r2: Register,
    },

    /// rd = r1 \intersect r2.
    SetIntersect {
        /// Destination register the intersection set is written to.
        rd: Register,
        /// Register holding the left set operand.
        r1: Register,
        /// Register holding the right set operand.
        r2: Register,
    },

    /// rd = r1 \ r2 (set difference).
    SetDiff {
        /// Destination register the difference set is written to.
        rd: Register,
        /// Register holding the set to subtract from.
        r1: Register,
        /// Register holding the set of elements to remove.
        r2: Register,
    },

    /// rd = (r1 \subseteq r2).
    Subseteq {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the candidate subset.
        r1: Register,
        /// Register holding the candidate superset.
        r2: Register,
    },

    /// Exact VM-only fusion for the one-binder FORALL body
    ///
    /// `Round(child) = Round(parent) - 1`,
    ///
    /// where both calls resolve directly to the same complete global
    /// definition `Round(p) == IF p = <<>> THEN 0 ELSE p[2]`, `child` is the
    /// current quantifier binding, and `parent` is an already-bound outer
    /// register. The VM evaluates the child call first, then the parent call,
    /// then applies ordinary integer subtraction and equality semantics.
    ///
    /// Native lowering deliberately rejects this opcode. It may be emitted
    /// only by an explicitly opted-in bytecode-VM compiler.
    RoundStepEq {
        /// Destination register receiving the boolean predicate result.
        rd: Register,
        /// Register holding the current child value.
        child: Register,
        /// Register holding the directly bound parent value.
        parent: Register,
    },

    /// rd = SUBSET(rs) (powerset).
    Powerset {
        /// Destination register the powerset value is written to.
        rd: Register,
        /// Register holding the base set.
        rs: Register,
    },

    /// rd = UNION(rs) (big union / flatten).
    BigUnion {
        /// Destination register the flattened union set is written to.
        rd: Register,
        /// Register holding the set-of-sets to flatten.
        rs: Register,
    },

    /// rd = KSubset(base, k) — k-element subsets of base set.
    /// Constructs a lazy KSubsetValue representing C(n,k) subsets without
    /// enumerating all 2^n subsets of the powerset. Part of #3907.
    KSubset {
        /// Destination register the lazy k-subset value is written to.
        rd: Register,
        /// Register holding the base set.
        base: Register,
        /// Register holding the subset size `k` (a non-negative integer).
        k: Register,
    },

    /// rd = lo..hi (integer interval set).
    Range {
        /// Destination register the interval set is written to.
        rd: Register,
        /// Register holding the inclusive lower bound.
        lo: Register,
        /// Register holding the inclusive upper bound.
        hi: Register,
    },

    // =================================================================
    // Quantifiers
    //
    // Quantifier loops use a two-instruction pattern:
    //   QuantBegin: initialize iterator, jump to end if domain is empty
    //   QuantNext: advance iterator, jump back to body or fall through
    // =================================================================
    /// Begin a FORALL quantifier over the domain in `r_domain`.
    /// `r_binding` receives each element. If domain is empty, jumps to
    /// `loop_end` (offset from this instruction).
    ForallBegin {
        /// Result register, seeded with TRUE and finalized by `ForallNext`.
        rd: Register,
        /// Register the current bound element is written to before each body run.
        r_binding: Register,
        /// Register holding the domain set to iterate.
        r_domain: Register,
        /// Forward offset (from this PC) taken when the domain is empty.
        loop_end: JumpOffset,
    },

    /// Advance the FORALL iterator. If the body produced FALSE, short-circuit
    /// to the end. Otherwise, bind the next element and jump to `loop_begin`.
    ForallNext {
        /// Result register written FALSE on short-circuit, TRUE when exhausted.
        rd: Register,
        /// Register the next bound element is written to before re-entering the body.
        r_binding: Register,
        /// Register holding the boolean value the body just produced.
        r_body: Register,
        /// Backward offset (from this PC) to re-enter the loop body.
        loop_begin: JumpOffset,
    },

    /// Begin an EXISTS quantifier (analogous to ForallBegin).
    ExistsBegin {
        /// Result register, seeded with FALSE and finalized by `ExistsNext`.
        rd: Register,
        /// Register the current bound element is written to before each body run.
        r_binding: Register,
        /// Register holding the domain set to iterate.
        r_domain: Register,
        /// Forward offset (from this PC) taken when the domain is empty.
        loop_end: JumpOffset,
    },

    /// Advance the EXISTS iterator. If the body produced TRUE, short-circuit
    /// to the end. Otherwise, bind next element and jump to `loop_begin`.
    ExistsNext {
        /// Result register written TRUE on short-circuit, FALSE when exhausted.
        rd: Register,
        /// Register the next bound element is written to before re-entering the body.
        r_binding: Register,
        /// Register holding the boolean value the body just produced.
        r_body: Register,
        /// Backward offset (from this PC) to re-enter the loop body.
        loop_begin: JumpOffset,
    },

    // =================================================================
    // Records / Functions / Tuples
    // =================================================================
    /// Build a record from `count` (field_id, value) pairs.
    /// Field IDs come from the constant pool, values from consecutive registers.
    RecordNew {
        /// Destination register the built record is written to.
        rd: Register,
        /// Constant-pool index of the first field-name entry (names are consecutive).
        fields_start: ConstIdx,
        /// First register of the contiguous field-value block (parallel to names).
        values_start: Register,
        /// Number of (field-name, value) pairs.
        count: u8,
    },

    /// rd = rs.field (record field access by pre-interned field ID).
    RecordGet {
        /// Destination register the field value is written to.
        rd: Register,
        /// Register holding the record to read from.
        rs: Register,
        /// Index of the pre-interned field name being accessed.
        field_idx: FieldIdx,
    },

    /// rd = rs[r_arg] (function application).
    FuncApply {
        /// Destination register the applied value is written to.
        rd: Register,
        /// Register holding the function value.
        func: Register,
        /// Register holding the application argument (the key).
        arg: Register,
    },

    /// rd = DOMAIN(rs).
    Domain {
        /// Destination register the domain set is written to.
        rd: Register,
        /// Register holding the function (or function-like value).
        rs: Register,
    },

    /// rd = [rs EXCEPT ![r_path] = r_val].
    /// Single-element EXCEPT (most common case).
    FuncExcept {
        /// Destination register the updated function is written to.
        rd: Register,
        /// Register holding the original function.
        func: Register,
        /// Register holding the path/key being overridden.
        path: Register,
        /// Register holding the replacement value at that path.
        val: Register,
    },

    /// Build a tuple from `count` consecutive registers starting at `start`.
    TupleNew {
        /// Destination register the built tuple is written to.
        rd: Register,
        /// First register of the contiguous element block.
        start: Register,
        /// Number of element registers to gather into the tuple.
        count: u8,
    },

    /// rd = rs[idx] (tuple element access, 1-indexed per TLA+ convention).
    TupleGet {
        /// Destination register the selected element is written to.
        rd: Register,
        /// Register holding the tuple to index into.
        rs: Register,
        /// Static 1-based position of the element to read.
        idx: u16,
    },

    /// Build a function `[x \in domain |-> body]`.
    /// `r_domain` has the domain set. For each element, bind to `r_binding`,
    /// evaluate body, collect into function.
    FuncDef {
        /// Destination register the built function is written to.
        rd: Register,
        /// Register holding the domain set.
        r_domain: Register,
        /// Register the current domain element is bound to while evaluating the body.
        r_binding: Register,
    },

    /// Build a function set `[S -> T]`.
    FuncSet {
        /// Destination register the function-set value is written to.
        rd: Register,
        /// Register holding the domain set `S`.
        domain: Register,
        /// Register holding the codomain set `T`.
        range: Register,
    },

    /// Build a record set `[f1: S1, f2: S2, ...]`.
    RecordSet {
        /// Destination register the record-set value is written to.
        rd: Register,
        /// Constant-pool index of the first field-name entry (names are consecutive).
        fields_start: ConstIdx,
        /// First register of the contiguous field-set block (parallel to names).
        values_start: Register,
        /// Number of (field-name, field-set) pairs.
        count: u8,
    },

    /// Build a cross product `S1 \X S2 \X ...`.
    Times {
        /// Destination register the cross-product value is written to.
        rd: Register,
        /// First register of the contiguous component-set block.
        start: Register,
        /// Number of component sets in the product.
        count: u8,
    },

    // =================================================================
    // Sequences
    // =================================================================
    /// Build a sequence from `count` consecutive registers.
    SeqNew {
        /// Destination register the built sequence is written to.
        rd: Register,
        /// First register of the contiguous element block.
        start: Register,
        /// Number of element registers to gather into the sequence.
        count: u8,
    },

    // =================================================================
    // String Operations
    // =================================================================
    /// rd = r1 \o r2 (string concatenation).
    StrConcat {
        /// Destination register the concatenated string is written to.
        rd: Register,
        /// Register holding the left string operand.
        r1: Register,
        /// Register holding the right string operand.
        r2: Register,
    },

    // =================================================================
    // Special
    // =================================================================
    /// rd = IF cond THEN rs ELSE rd (conditional move).
    /// Used for simple IF-THEN-ELSE without control flow.
    CondMove {
        /// Destination register overwritten with `rs` only when `cond` is TRUE.
        rd: Register,
        /// Register holding the boolean condition.
        cond: Register,
        /// Source register copied into `rd` when the condition holds.
        rs: Register,
    },

    /// UNCHANGED <<v1, v2, ...>>: compare primed vars equal unprimed.
    /// `start` and `count` refer to VarIdx entries in the constant pool.
    /// Writes `TRUE` to `rd` iff all listed vars match between state and next_state.
    Unchanged {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Constant-pool index of the first variable-index entry (entries are consecutive).
        start: ConstIdx,
        /// Number of variable indices to compare across the state pair.
        count: u8,
    },

    /// Begin a CHOOSE quantifier: iterate `r_domain`, evaluate predicate body,
    /// return first element where predicate is TRUE.
    /// If domain is empty, halts (TLA+ CHOOSE with no match is a runtime error).
    ChooseBegin {
        /// Result register; seeded FALSE here and set to the chosen element by `ChooseNext`.
        rd: Register,
        /// Register the current candidate element is bound to before each predicate run.
        r_binding: Register,
        /// Register holding the domain set to search.
        r_domain: Register,
        /// Forward offset reserved for the loop-end target (unused; CHOOSE on an
        /// empty domain is a runtime error rather than a jump).
        loop_end: JumpOffset,
    },

    /// Advance the CHOOSE iterator. If `r_body` is TRUE, set `rd = r_binding`
    /// and exit the loop. Otherwise, advance to next element. If domain is
    /// exhausted without finding a match, halts.
    ChooseNext {
        /// Result register set to the chosen element when the predicate holds.
        rd: Register,
        /// Register holding the current candidate (copied to `rd` on a match) and
        /// rebound to the next candidate otherwise.
        r_binding: Register,
        /// Register holding the boolean predicate value the body just produced.
        r_body: Register,
        /// Backward offset (from this PC) to re-enter the predicate body.
        loop_begin: JumpOffset,
    },

    /// Set comprehension: `{body : x \in S}`.
    /// Iterates over `r_domain`, binds to `r_binding`, evaluates body (in
    /// following instructions up to a SetBuilderEnd), collects into set.
    SetBuilderBegin {
        /// Result register; filled with the collected set by the terminating `LoopNext`
        /// (set to the empty set here when the domain is empty).
        rd: Register,
        /// Register the current domain element is bound to before each body run.
        r_binding: Register,
        /// Register holding the domain set to iterate.
        r_domain: Register,
        /// Forward offset (from this PC) taken when the domain is empty.
        loop_end: JumpOffset,
    },

    /// Set filter: `{x \in S : P(x)}`.
    /// Iterates over `r_domain`, binds to `r_binding`, evaluates predicate
    /// body, collects elements where predicate is TRUE.
    SetFilterBegin {
        /// Result register; filled with the filtered set by the terminating `LoopNext`
        /// (set to the empty set here when the domain is empty).
        rd: Register,
        /// Register the current domain element is bound to before each predicate run.
        r_binding: Register,
        /// Register holding the domain set to iterate.
        r_domain: Register,
        /// Forward offset (from this PC) taken when the domain is empty.
        loop_end: JumpOffset,
    },

    /// Function definition body loop: for each domain element, evaluate body
    /// and collect (key, value) pair.
    FuncDefBegin {
        /// Result register; filled with the built function by the terminating `LoopNext`
        /// (set to the empty function here when the domain is empty).
        rd: Register,
        /// Register the current domain element (the key) is bound to before each body run.
        r_binding: Register,
        /// Register holding the domain set to iterate.
        r_domain: Register,
        /// Forward offset (from this PC) taken when the domain is empty.
        loop_end: JumpOffset,
    },

    /// End of a quantifier/builder/filter loop body. Advances the iterator
    /// and jumps back to the loop start if more elements remain.
    LoopNext {
        /// Register the next domain element is rebound to when continuing the loop.
        r_binding: Register,
        /// Register holding the body value just produced (collected per the active builder/filter).
        r_body: Register,
        /// Backward offset (from this PC) to re-enter the loop body.
        loop_begin: JumpOffset,
    },

    /// Set VM prime mode: when enabled, `LoadVar` reads from next-state
    /// instead of current state. Used by the UNCHANGED general fallback to
    /// compile `expr = expr'` where `expr` may contain Call opcodes that
    /// jump to pre-compiled functions (which use LoadVar, not LoadPrime).
    SetPrimeMode {
        /// Whether to enable (`true`) or disable (`false`) next-state reads for `LoadVar`.
        enable: bool,
    },

    /// Build a closure value from a template, capturing runtime register values.
    ///
    /// `template_idx` points to the template `Value::Closure` in the constant
    /// pool. The next `capture_count` constant pool entries (starting at
    /// `template_idx + 1`) are `Value::Str` capture-name keys. The corresponding
    /// values come from consecutive registers starting at `captures_start`.
    /// The resulting closure's `env` maps each capture name to its register value.
    MakeClosure {
        /// Destination register the built closure value is written to.
        rd: Register,
        /// Constant-pool index of the template `Value::Closure`; the following
        /// `capture_count` pool entries hold the capture-name keys.
        template_idx: ConstIdx,
        /// First register of the contiguous capture-value block.
        captures_start: Register,
        /// Number of captured variables to bind into the closure environment.
        capture_count: u8,
    },

    /// Call an external (non-compiled) operator by name, falling back to
    /// the TIR tree-walker at runtime.
    ///
    /// Used for INSTANCE-imported operators that cannot be pre-compiled
    /// into bytecode. `name_idx` points to a `Value::String` in the
    /// constant pool holding the operator name. `args_start`/`argc`
    /// carry arguments (zero for zero-arg operators).
    ///
    /// At execution time, the VM calls back into `EvalCtx::eval_op`
    /// (zero-arg) or `apply_user_op_with_values` (with args).
    CallExternal {
        /// Destination register the external operator's result is written to.
        rd: Register,
        /// Constant-pool index of the `Value::String` holding the operator name.
        name_idx: ConstIdx,
        /// First register of the contiguous argument block (ignored when `argc` is 0).
        args_start: Register,
        /// Number of arguments passed (0 for a zero-arg operator).
        argc: u8,
        /// Compiler-authenticated provenance that this exact site is the
        /// containing function's recursive self-call. Set only when the
        /// on-demand recursion guard rejects that same function name/arity;
        /// generic, forced, unsupported, and qualified external fallbacks
        /// must leave it false. This is metadata for static consumers, not a
        /// request for ordinary VM execution to change CallExternal semantics.
        self_recursive: bool,
    },

    /// rd = r1 \o r2 (sequence/string concatenation via the `\o` or `\circ` operator).
    ///
    /// Distinguished from `StrConcat` because `\o` is polymorphic: it concatenates
    /// sequences as well as strings. The VM dispatches based on operand types.
    /// Part of #3789: stdlib operator compilation.
    Concat {
        /// Destination register the concatenated value is written to.
        rd: Register,
        /// Register holding the left sequence/string operand.
        r1: Register,
        /// Register holding the right sequence/string operand.
        r2: Register,
    },

    /// Call a standard-library builtin operator by tag.
    ///
    /// Used for operators from EXTENDS modules (Sequences, FiniteSets, TLC)
    /// that have dedicated implementations in the VM. Arguments are in
    /// consecutive registers starting at `args_start`.
    /// Part of #3789: cross-module identifier resolution for stdlib operators.
    CallBuiltin {
        /// Destination register the builtin's result is written to.
        rd: Register,
        /// Tag identifying which standard-library builtin to invoke.
        builtin: BuiltinOp,
        /// First register of the contiguous argument block.
        args_start: Register,
        /// Number of arguments passed to the builtin.
        argc: u8,
    },

    /// Fused `rd = (lhs = [func EXCEPT ![path] = val])` — the equality of a
    /// value against a single-path EXCEPT result, WITHOUT materializing the
    /// intermediate function.
    ///
    /// SEMANTIC CONTRACT: byte-identical to executing
    /// `FuncExcept { tmp, func, path, val }` followed by
    /// `Eq { rd, r1: lhs, r2: tmp }` where `tmp` is dead after the `Eq`
    /// (guaranteed by the compiler: fusion only replaces an expression-temp
    /// producer that is the immediately preceding instruction). The VM
    /// handler decides the equality structurally on fast shapes and falls
    /// back to literally constructing the EXCEPT result and comparing on
    /// every other shape — including every shape where the construction
    /// itself would error, so errors and verdicts are identical.
    ///
    /// Emitted ONLY by the implied-action term compile
    /// (`enable_eq_fusion`); action/invariant/constraint bytecode consumed by
    /// the trust-cg native lowering never contains this opcode.
    EqFuncExcept {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left-hand comparison value.
        lhs: Register,
        /// Register holding the original function.
        func: Register,
        /// Register holding the path/key being overridden.
        path: Register,
        /// Register holding the replacement value at that path.
        val: Register,
    },

    /// Fused `rd = (lhs = [f1 |-> v1, ..., fn |-> vn])` — the equality of a
    /// value against a record constructor, WITHOUT materializing the record.
    ///
    /// SEMANTIC CONTRACT: byte-identical to `RecordNew { tmp, ... }` followed
    /// by `Eq { rd, r1: lhs, r2: tmp }` with `tmp` dead after the `Eq` (same
    /// compiler guarantee as `EqFuncExcept`). Field layout matches
    /// `RecordNew`: `fields_start` names consecutive `Value::String` pool
    /// entries, `values_start` a contiguous register block.
    ///
    /// Emitted ONLY by the implied-action term compile (`enable_eq_fusion`).
    EqRecordNew {
        /// Destination register the boolean result is written to.
        rd: Register,
        /// Register holding the left-hand comparison value.
        lhs: Register,
        /// Constant-pool index of the first field-name entry (names are consecutive).
        fields_start: ConstIdx,
        /// First register of the contiguous field-value block (parallel to names).
        values_start: Register,
        /// Number of (field-name, value) pairs.
        count: u8,
    },

    /// No operation (used for alignment / patching).
    Nop,

    /// Halt execution with an error.
    Halt,
}
