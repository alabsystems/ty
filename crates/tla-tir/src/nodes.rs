// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::types::TirType;
use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::{NameId, Spanned};
use tla_value::{TirBody, Value};

/// Opaque wrapper for a preserved AST body carried alongside a TIR Lambda.
///
/// Part of #3163: The AST body is needed for `ClosureValue` construction
/// during the Phase 2c transition. It is NOT part of TIR semantic identity —
/// the TIR `body` field is canonical. Equality/hashing ignore this wrapper
/// so `TirExpr`'s derived `Eq`/`PartialEq` remain based on the TIR content.
/// This wrapper will be removed in Phase 2d when `ClosureValue` no longer
/// requires an AST body.
#[derive(Debug, Clone)]
pub struct PreservedAstBody(pub Arc<Spanned<Expr>>);

impl PartialEq for PreservedAstBody {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for PreservedAstBody {}

/// Opaque lowered TIR body stored inside closure values.
///
/// This lives in `tla-tir` so any TIR-producing path, including the bytecode
/// compiler, can attach the lowered body without depending on `tla-eval`.
///
/// Part of #4113: uses `Arc` instead of `Box` so cloning is O(1) refcount
/// bump instead of deep-cloning the entire TIR expression tree. This is
/// critical for performance because closure values carrying TIR bodies are
/// cloned frequently during model checking.
#[derive(Debug, Clone)]
pub struct StoredTirBody(pub Arc<Spanned<TirExpr>>);

impl StoredTirBody {
    /// Wrap a lowered TIR expression in a fresh `Arc`.
    #[must_use]
    pub fn new(expr: Spanned<TirExpr>) -> Self {
        Self(Arc::new(expr))
    }

    /// Create from a pre-existing `Arc` without extra allocation.
    ///
    /// Part of #4113: when the caller already holds an `Arc<Spanned<TirExpr>>`
    /// (e.g., from the operator cache), this avoids an unnecessary clone + re-Arc.
    #[must_use]
    pub fn from_arc(expr: Arc<Spanned<TirExpr>>) -> Self {
        Self(expr)
    }

    /// Borrow the stored expression.
    #[must_use]
    pub fn expr(&self) -> &Spanned<TirExpr> {
        &self.0
    }

    /// Return a cheap clone of the inner `Arc`.
    ///
    /// Part of #4113: callers that need to store or propagate TIR bodies
    /// can use `Arc::clone()` instead of deep-cloning the expression tree.
    #[must_use]
    pub fn arc(&self) -> Arc<Spanned<TirExpr>> {
        Arc::clone(&self.0)
    }
}

impl TirBody for StoredTirBody {
    fn clone_box(&self) -> Box<dyn TirBody> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Path segment for a resolved module/instance target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirModuleRefSegment {
    /// Module/instance name of this path segment.
    pub name: String,
    /// Interned identifier for the segment name.
    pub name_id: NameId,
    /// Instantiation arguments applied at this segment (empty if none).
    pub args: Vec<Spanned<TirExpr>>,
}

/// Lowered operator reference with a flattened INSTANCE/module path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirOperatorRef {
    /// Module/instance path leading to the operator (outermost first).
    pub path: Vec<TirModuleRefSegment>,
    /// Name of the referenced operator.
    pub operator: String,
    /// Interned identifier for the operator name.
    pub operator_id: NameId,
    /// Arguments applied to the operator.
    pub args: Vec<Spanned<TirExpr>>,
}

/// Variable identity in TIR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TirNameKind {
    /// An ordinary identifier (operator parameter, LET binding, constant, etc.).
    Ident,
    /// A model state variable, identified by its variable index.
    StateVar {
        /// Index of the state variable in the model's variable table.
        index: u16,
    },
}

/// Variable reference in TIR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirNameRef {
    /// The variable's source name.
    pub name: String,
    /// Interned identifier for the name.
    pub name_id: NameId,
    /// Whether this is a plain identifier or a state variable.
    pub kind: TirNameKind,
    /// The inferred type of the reference.
    pub ty: TirType,
}

/// Boolean operators preserved in TIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TirBoolOp {
    /// Conjunction (`/\`).
    And,
    /// Disjunction (`\/`).
    Or,
    /// Implication (`=>`).
    Implies,
    /// Equivalence (`<=>`).
    Equiv,
}

/// Comparison operators preserved in TIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TirCmpOp {
    /// Equality (`=`).
    Eq,
    /// Inequality (`/=`).
    Neq,
    /// Less-than (`<`).
    Lt,
    /// Less-than-or-equal (`<=`).
    Leq,
    /// Greater-than (`>`).
    Gt,
    /// Greater-than-or-equal (`>=`).
    Geq,
}

/// Arithmetic operators preserved in TIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TirArithOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Real division (`/`).
    Div,
    /// Integer division (`\div`).
    IntDiv,
    /// Modulus (`%`).
    Mod,
    /// Exponentiation (`^`).
    Pow,
}

/// The two kinds of TLA+ action subscripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TirActionSubscriptKind {
    /// A box subscript `[A]_v` (stuttering allowed).
    Box,
    /// An angle subscript `<<A>>_v` (a real, non-stuttering step).
    Angle,
}

/// A record field name with a pre-interned `NameId` for O(1) runtime lookup.
///
/// This is the TIR-native equivalent of `tla_core::ast::RecordFieldName`. TIR
/// types depend on `NameId` and `Spanned`, not on AST helper structs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirFieldName {
    /// The field name as written in source.
    pub name: String,
    /// Interned identifier for the field name (for O(1) lookup).
    pub field_id: NameId,
}

/// An element in an EXCEPT path: either an index expression or a record field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TirExceptPathElement {
    /// A function/sequence index step (`![expr]`).
    Index(Box<Spanned<TirExpr>>),
    /// A record field step (`!.field`).
    Field(TirFieldName),
}

/// A single EXCEPT specification: one path/value pair.
///
/// Multiple specs in `[f EXCEPT ![a] = 1, ![b] = 2]` are applied in source
/// order to the evolving result, matching TLC semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirExceptSpec {
    /// The path of index/field steps locating the element to override.
    pub path: Vec<TirExceptPathElement>,
    /// The replacement value at that path.
    pub value: Spanned<TirExpr>,
}

/// Destructuring pattern for a bound variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TirBoundPattern {
    /// A simple single-name binding (name and interned id).
    Var(String, NameId),
    /// A tuple-destructuring binding (one name/id pair per component).
    Tuple(Vec<(String, NameId)>),
}

/// A bound variable with optional domain and optional destructuring pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirBoundVar {
    /// The bound variable's source name.
    pub name: String,
    /// Interned identifier for the name.
    pub name_id: NameId,
    /// The domain set the variable ranges over (`None` for unbounded binders).
    pub domain: Option<Box<Spanned<TirExpr>>>,
    /// Optional tuple-destructuring pattern for the binding.
    pub pattern: Option<TirBoundPattern>,
}

/// A local operator definition inside a LET block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirLetDef {
    /// The local operator's name.
    pub name: String,
    /// Interned identifier for the name.
    pub name_id: NameId,
    /// Parameter names (empty for zero-arity definitions).
    pub params: Vec<String>,
    /// The definition body.
    pub body: Spanned<TirExpr>,
}

/// A case arm: `guard -> body`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TirCaseArm {
    /// The arm's guard predicate.
    pub guard: Spanned<TirExpr>,
    /// The value taken when the guard holds.
    pub body: Spanned<TirExpr>,
}

/// Set binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TirSetOp {
    /// Set union (`\union`).
    Union,
    /// Set intersection (`\intersect`).
    Intersect,
    /// Set difference (`\`).
    Minus,
}

/// Current supported TIR expression surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TirExpr {
    /// A literal constant value with its inferred type.
    Const {
        /// The constant value.
        value: Value,
        /// The constant's type.
        ty: TirType,
    },
    /// A variable or identifier reference.
    Name(TirNameRef),
    /// A reference to an operator reached through an INSTANCE/module path.
    OperatorRef(TirOperatorRef),
    /// A binary arithmetic operation.
    ArithBinOp {
        /// The left operand.
        left: Box<Spanned<TirExpr>>,
        /// The arithmetic operator.
        op: TirArithOp,
        /// The right operand.
        right: Box<Spanned<TirExpr>>,
    },
    /// Arithmetic negation (`-x`).
    ArithNeg(Box<Spanned<TirExpr>>),
    /// A binary boolean connective.
    BoolBinOp {
        /// The left operand.
        left: Box<Spanned<TirExpr>>,
        /// The boolean operator.
        op: TirBoolOp,
        /// The right operand.
        right: Box<Spanned<TirExpr>>,
    },
    /// Boolean negation (`~x`).
    BoolNot(Box<Spanned<TirExpr>>),
    /// A comparison between two operands.
    Cmp {
        /// The left operand.
        left: Box<Spanned<TirExpr>>,
        /// The comparison operator.
        op: TirCmpOp,
        /// The right operand.
        right: Box<Spanned<TirExpr>>,
    },
    /// Set membership (`elem \in set`).
    In {
        /// The candidate element.
        elem: Box<Spanned<TirExpr>>,
        /// The set tested for membership.
        set: Box<Spanned<TirExpr>>,
    },
    /// Subset relation (`left \subseteq right`).
    Subseteq {
        /// The candidate subset.
        left: Box<Spanned<TirExpr>>,
        /// The candidate superset.
        right: Box<Spanned<TirExpr>>,
    },
    /// `UNCHANGED v` — the wrapped expression's variables keep their values.
    Unchanged(Box<Spanned<TirExpr>>),
    /// A subscripted action (`[A]_v` or `<<A>>_v`).
    ActionSubscript {
        /// Whether the subscript is a box (`[A]_v`) or angle (`<<A>>_v`).
        kind: TirActionSubscriptKind,
        /// The action predicate `A`.
        action: Box<Spanned<TirExpr>>,
        /// The subscript expression `v`.
        subscript: Box<Spanned<TirExpr>>,
    },
    /// Temporal "always" (`[]P`).
    Always(Box<Spanned<TirExpr>>),
    /// Temporal "eventually" (`<>P`).
    Eventually(Box<Spanned<TirExpr>>),
    /// Record field access (`record.field`).
    RecordAccess {
        /// The record being accessed.
        record: Box<Spanned<TirExpr>>,
        /// The field being read.
        field: TirFieldName,
    },
    /// `[base EXCEPT ...]` — functional update with one or more specs.
    Except {
        /// The original function/record being updated.
        base: Box<Spanned<TirExpr>>,
        /// The override specifications, applied in source order.
        specs: Vec<TirExceptSpec>,
    },
    /// The `@` placeholder for the prior value inside an EXCEPT right-hand side.
    ExceptAt,
    /// An integer interval (`lo..hi`).
    Range {
        /// The inclusive lower bound.
        lo: Box<Spanned<TirExpr>>,
        /// The inclusive upper bound.
        hi: Box<Spanned<TirExpr>>,
    },
    /// `IF cond THEN then_ ELSE else_`.
    If {
        /// The condition.
        cond: Box<Spanned<TirExpr>>,
        /// The value when the condition holds.
        then_: Box<Spanned<TirExpr>>,
        /// The value when the condition does not hold.
        else_: Box<Spanned<TirExpr>>,
    },

    // === Quantifiers ===
    /// Universal quantification (`\A vars : body`).
    Forall {
        /// The bound variables (with their domains).
        vars: Vec<TirBoundVar>,
        /// The quantified body.
        body: Box<Spanned<TirExpr>>,
    },
    /// Existential quantification (`\E vars : body`).
    Exists {
        /// The bound variables (with their domains).
        vars: Vec<TirBoundVar>,
        /// The quantified body.
        body: Box<Spanned<TirExpr>>,
    },
    /// `CHOOSE var : body` — choose an element satisfying the predicate.
    Choose {
        /// The bound variable (with its domain).
        var: TirBoundVar,
        /// The selection predicate.
        body: Box<Spanned<TirExpr>>,
    },

    // === Sets ===
    /// Explicit set enumeration (`{e1, e2, ...}`).
    SetEnum(Vec<Spanned<TirExpr>>),
    /// Set filter (`{x \in S : P(x)}`).
    SetFilter {
        /// The bound variable and its domain set.
        var: TirBoundVar,
        /// The filter predicate.
        body: Box<Spanned<TirExpr>>,
    },
    /// Set comprehension (`{body : x \in S, ...}`).
    SetBuilder {
        /// The element-producing body expression.
        body: Box<Spanned<TirExpr>>,
        /// The bound variables and their domains.
        vars: Vec<TirBoundVar>,
    },
    /// A binary set operation (union/intersection/difference).
    SetBinOp {
        /// The left set operand.
        left: Box<Spanned<TirExpr>>,
        /// The set operator.
        op: TirSetOp,
        /// The right set operand.
        right: Box<Spanned<TirExpr>>,
    },
    /// Powerset (`SUBSET S`).
    Powerset(Box<Spanned<TirExpr>>),
    /// Big union (`UNION S`).
    BigUnion(Box<Spanned<TirExpr>>),
    /// K-subset: {x \in SUBSET(S) : Cardinality(x) = k} optimized to direct
    /// C(n,k) generation via KSubsetValue. Avoids 2^n powerset enumeration.
    /// Part of #3907.
    KSubset {
        /// The base set `S`.
        base: Box<Spanned<TirExpr>>,
        /// The subset size `k`.
        k: Box<Spanned<TirExpr>>,
    },

    // === Functions ===
    /// Function constructor (`[x \in S |-> body]`).
    FuncDef {
        /// The bound variables and their domains.
        vars: Vec<TirBoundVar>,
        /// The body mapping each domain element to a value.
        body: Box<Spanned<TirExpr>>,
    },
    /// Function application (`func[arg]`).
    FuncApply {
        /// The function being applied.
        func: Box<Spanned<TirExpr>>,
        /// The application argument.
        arg: Box<Spanned<TirExpr>>,
    },
    /// Function set (`[domain -> range]`).
    FuncSet {
        /// The domain set.
        domain: Box<Spanned<TirExpr>>,
        /// The range (codomain) set.
        range: Box<Spanned<TirExpr>>,
    },
    /// Function domain (`DOMAIN f`).
    Domain(Box<Spanned<TirExpr>>),

    // === Records/Tuples ===
    /// Record constructor (`[f1 |-> e1, ...]`), as field/value pairs.
    Record(Vec<(TirFieldName, Spanned<TirExpr>)>),
    /// Record-set constructor (`[f1: S1, ...]`), as field/set pairs.
    RecordSet(Vec<(TirFieldName, Spanned<TirExpr>)>),
    /// Tuple constructor (`<<e1, e2, ...>>`).
    Tuple(Vec<Spanned<TirExpr>>),
    /// Cross product (`S1 \X S2 \X ...`).
    Times(Vec<Spanned<TirExpr>>),

    // === Control ===
    /// `LET defs IN body`.
    Let {
        /// The local definitions.
        defs: Vec<TirLetDef>,
        /// The body evaluated under those definitions.
        body: Box<Spanned<TirExpr>>,
    },
    /// `CASE arm1 [] arm2 ... [OTHER -> other]`.
    Case {
        /// The guarded arms, in source order.
        arms: Vec<TirCaseArm>,
        /// The optional `OTHER` fallthrough value.
        other: Option<Box<Spanned<TirExpr>>>,
    },

    // === Actions ===
    /// Prime (`expr'`) — the value of `expr` in the successor state.
    Prime(Box<Spanned<TirExpr>>),

    // === Generic operator application ===
    /// Generic application of an operator-valued expression to arguments.
    Apply {
        /// The operator-valued expression being applied.
        op: Box<Spanned<TirExpr>>,
        /// The arguments.
        args: Vec<Spanned<TirExpr>>,
    },

    // === Lambda / higher-order ===
    /// `LAMBDA x, y : body` — a closure with named parameters and a body.
    ///
    /// Part of #3163: Carries both the lowered TIR body and the original AST body.
    /// The AST body is needed for `ClosureValue` construction (which lives in
    /// `tla-value` and cannot depend on `tla-tir`). The TIR body is attached
    /// via the `TirBody` trait for TIR-native evaluation at application time.
    /// The `ast_body` will be removed in Phase 2d when `ClosureValue` is migrated.
    Lambda {
        /// The lambda's parameter names.
        params: Vec<String>,
        /// The lowered TIR body.
        body: Box<Spanned<TirExpr>>,
        /// Original AST body preserved for `ClosureValue` construction.
        ast_body: PreservedAstBody,
    },
    /// Bare operator reference (e.g., `+`, `\cup`) used as a higher-order value.
    OpRef(String),
    /// Labeled subexpression (`P0:: expr`). Transparent wrapper with no runtime effect.
    Label {
        /// The label name.
        name: String,
        /// The labeled body.
        body: Box<Spanned<TirExpr>>,
    },

    // === Temporal (extended) ===
    /// Leads-to (`left ~> right`).
    LeadsTo {
        /// The antecedent state predicate.
        left: Box<Spanned<TirExpr>>,
        /// The consequent state predicate.
        right: Box<Spanned<TirExpr>>,
    },
    /// Weak fairness (`WF_vars(action)`).
    WeakFair {
        /// The subscript variables.
        vars: Box<Spanned<TirExpr>>,
        /// The action the fairness condition applies to.
        action: Box<Spanned<TirExpr>>,
    },
    /// Strong fairness (`SF_vars(action)`).
    StrongFair {
        /// The subscript variables.
        vars: Box<Spanned<TirExpr>>,
        /// The action the fairness condition applies to.
        action: Box<Spanned<TirExpr>>,
    },
    /// `ENABLED action` — the action is possible in the current state.
    Enabled(Box<Spanned<TirExpr>>),
}

impl TirExpr {
    /// Infer the static [`TirType`] of this expression.
    ///
    /// Returns [`TirType::Dyn`] for expressions whose type cannot be determined
    /// structurally (generic application, higher-order references, etc.).
    #[must_use]
    pub fn ty(&self) -> TirType {
        match self {
            Self::Const { ty, .. } => ty.clone(),
            Self::Name(name) => name.ty.clone(),
            Self::OperatorRef(_) | Self::ExceptAt => TirType::Dyn,
            Self::ArithBinOp { .. } | Self::ArithNeg(_) => TirType::Int,
            Self::BoolBinOp { .. }
            | Self::BoolNot(_)
            | Self::Cmp { .. }
            | Self::In { .. }
            | Self::Subseteq { .. }
            | Self::Unchanged(_)
            | Self::ActionSubscript { .. }
            | Self::Always(_)
            | Self::Eventually(_)
            | Self::Forall { .. }
            | Self::Exists { .. }
            | Self::LeadsTo { .. }
            | Self::WeakFair { .. }
            | Self::StrongFair { .. }
            | Self::Enabled(_) => TirType::Bool,
            Self::RecordAccess { record, field } => match record.node.ty() {
                TirType::Record(fields) | TirType::OpenRecord(fields, _) => fields
                    .iter()
                    .find(|(id, _)| *id == field.field_id)
                    .map_or(TirType::Dyn, |(_, ty)| ty.clone()),
                _ => TirType::Dyn,
            },
            Self::Except { base, .. } => base.node.ty(),

            // --- Set expressions with element type inference ---

            // a..b is Set(Int)
            Self::Range { .. } => TirType::Set(Box::new(TirType::Int)),

            // {e1, e2, ...}: join element types
            Self::SetEnum(elems) => {
                if elems.is_empty() {
                    TirType::Set(Box::new(TirType::Dyn))
                } else {
                    let mut elem_ty = elems[0].node.ty();
                    for e in &elems[1..] {
                        elem_ty = elem_ty.join(&e.node.ty());
                    }
                    TirType::Set(Box::new(elem_ty))
                }
            }

            // S \union T, S \intersect T, S \ T: join element types of operands
            Self::SetBinOp { left, right, .. } => {
                let lt = left.node.ty();
                let rt = right.node.ty();
                match (lt, rt) {
                    (TirType::Set(a), TirType::Set(b)) => TirType::Set(Box::new(a.join(&b))),
                    _ => TirType::Set(Box::new(TirType::Dyn)),
                }
            }

            // {x \in S : P(x)}: same element type as S
            Self::SetFilter { var, .. } => {
                if let Some(domain) = &var.domain {
                    let dom_ty = domain.node.ty();
                    if let TirType::Set(inner) = dom_ty {
                        TirType::Set(inner)
                    } else {
                        TirType::Set(Box::new(TirType::Dyn))
                    }
                } else {
                    TirType::Set(Box::new(TirType::Dyn))
                }
            }

            // {body : x \in S}: Set(typeof(body))
            Self::SetBuilder { body, .. } => TirType::Set(Box::new(body.node.ty())),

            // SUBSET S: Set(Set(elem_type))
            Self::Powerset(inner) => {
                let inner_ty = inner.node.ty();
                TirType::Set(Box::new(inner_ty))
            }

            // KSubset(S, k): Set(Set(elem_type_of_S))
            // The result is a set of k-element subsets of S.
            Self::KSubset { base, .. } => {
                let base_ty = base.node.ty();
                TirType::Set(Box::new(base_ty))
            }

            // UNION S: if S : Set(Set(T)) then Set(T), else Set(Dyn)
            Self::BigUnion(inner) => {
                let inner_ty = inner.node.ty();
                if let TirType::Set(mid) = inner_ty {
                    if let TirType::Set(_) = mid.as_ref() {
                        *mid
                    } else {
                        TirType::Set(Box::new(TirType::Dyn))
                    }
                } else {
                    TirType::Set(Box::new(TirType::Dyn))
                }
            }

            // [S -> T]: Set(Func(elem(S), elem(T)))
            Self::FuncSet { domain, range } => {
                let dom_elem = match domain.node.ty() {
                    TirType::Set(inner) => *inner,
                    _ => TirType::Dyn,
                };
                let rng_elem = match range.node.ty() {
                    TirType::Set(inner) => *inner,
                    _ => TirType::Dyn,
                };
                TirType::Set(Box::new(TirType::Func(
                    Box::new(dom_elem),
                    Box::new(rng_elem),
                )))
            }

            // DOMAIN f: if f : Func(D, R) then Set(D), else Set(Dyn)
            Self::Domain(inner) => {
                let inner_ty = inner.node.ty();
                if let TirType::Func(dom, _) = inner_ty {
                    TirType::Set(dom)
                } else {
                    TirType::Set(Box::new(TirType::Dyn))
                }
            }

            // [f1: S1, f2: S2, ...]: Set of records where field types are elem(Si)
            Self::RecordSet(fields) => {
                let rec_fields: Vec<(NameId, TirType)> = fields
                    .iter()
                    .map(|(f, e)| {
                        let elem_ty = match e.node.ty() {
                            TirType::Set(inner) => *inner,
                            _ => TirType::Dyn,
                        };
                        (f.field_id, elem_ty)
                    })
                    .collect();
                TirType::Set(Box::new(TirType::Record(rec_fields)))
            }

            // S1 \X S2 \X ...: Set of tuples where ith element type is elem(Si)
            Self::Times(elems) => {
                let tuple_elems: Vec<TirType> = elems
                    .iter()
                    .map(|e| match e.node.ty() {
                        TirType::Set(inner) => *inner,
                        _ => TirType::Dyn,
                    })
                    .collect();
                TirType::Set(Box::new(TirType::Tuple(tuple_elems)))
            }

            // --- Records and tuples ---
            Self::Record(fields) => TirType::Record(
                fields
                    .iter()
                    .map(|(f, e)| (f.field_id, e.node.ty()))
                    .collect(),
            ),
            Self::Tuple(elems) => TirType::Tuple(elems.iter().map(|e| e.node.ty()).collect()),

            // --- Control flow ---
            Self::If { then_, else_, .. } => {
                let then_ty = then_.node.ty();
                let else_ty = else_.node.ty();
                then_ty.join(&else_ty)
            }
            Self::Let { body, .. } => body.node.ty(),
            Self::Label { body, .. } => body.node.ty(),

            // CASE arms: join all arm body types
            Self::Case { arms, other } => {
                let mut result = TirType::Dyn;
                for arm in arms {
                    result = result.join(&arm.body.node.ty());
                }
                if let Some(o) = other {
                    result = result.join(&o.node.ty());
                }
                result
            }

            // CHOOSE x \in S : P(x): element type of the domain
            Self::Choose { var, .. } => {
                if let Some(domain) = &var.domain {
                    let dom_ty = domain.node.ty();
                    if let TirType::Set(inner) = dom_ty {
                        *inner
                    } else {
                        TirType::Dyn
                    }
                } else {
                    TirType::Dyn
                }
            }

            // --- Functions ---

            // [x \in S |-> body]: Func(elem(S), typeof(body))
            Self::FuncDef { vars, body } => {
                let dom_ty = if vars.len() == 1 {
                    if let Some(d) = &vars[0].domain {
                        match d.node.ty() {
                            TirType::Set(inner) => *inner,
                            _ => TirType::Dyn,
                        }
                    } else {
                        TirType::Dyn
                    }
                } else {
                    // Multi-variable: domain is a tuple of element types
                    let tup: Vec<TirType> = vars
                        .iter()
                        .map(|v| {
                            if let Some(d) = &v.domain {
                                match d.node.ty() {
                                    TirType::Set(inner) => *inner,
                                    _ => TirType::Dyn,
                                }
                            } else {
                                TirType::Dyn
                            }
                        })
                        .collect();
                    TirType::Tuple(tup)
                };
                TirType::Func(Box::new(dom_ty), Box::new(body.node.ty()))
            }

            // f[x]: if f : Func(D, R) then R, else Dyn
            Self::FuncApply { func, .. } => {
                let func_ty = func.node.ty();
                if let TirType::Func(_, range) = func_ty {
                    *range
                } else {
                    TirType::Dyn
                }
            }

            // x' has the same type as x
            Self::Prime(inner) => inner.node.ty(),

            // Generic application and higher-order: stay Dyn
            Self::Apply { .. } | Self::Lambda { .. } | Self::OpRef(_) => TirType::Dyn,
        }
    }
}

/// Placeholder action surface for the TIR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TirAction {
    /// An action expressed as a boolean guard predicate.
    Guard(Spanned<TirExpr>),
    /// An `UNCHANGED` action over the listed variables.
    Unchanged(Vec<TirNameRef>),
}

/// Placeholder temporal surface for the TIR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TirTemporal {
    /// A state predicate.
    StatePred(Spanned<TirExpr>),
    /// An action predicate.
    ActionPred(Spanned<TirExpr>),
    /// Temporal "always" over a nested temporal formula.
    Always(Box<TirTemporal>),
    /// Temporal "eventually" over a nested temporal formula.
    Eventually(Box<TirTemporal>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::intern_name;
    use tla_core::span::{FileId, Span};
    use tla_value::Value;

    fn span() -> Span {
        Span::new(FileId(0), 0, 0)
    }

    fn spanned(expr: TirExpr) -> Spanned<TirExpr> {
        Spanned::new(expr, span())
    }

    fn int_const(n: i64) -> Spanned<TirExpr> {
        spanned(TirExpr::Const {
            value: Value::int(n),
            ty: TirType::Int,
        })
    }

    fn bool_const(b: bool) -> Spanned<TirExpr> {
        spanned(TirExpr::Const {
            value: Value::bool(b),
            ty: TirType::Bool,
        })
    }

    fn str_const(s: &str) -> Spanned<TirExpr> {
        spanned(TirExpr::Const {
            value: Value::string(s),
            ty: TirType::Str,
        })
    }

    fn set_of_ints() -> Spanned<TirExpr> {
        spanned(TirExpr::SetEnum(vec![int_const(1), int_const(2)]))
    }

    fn set_of_bools() -> Spanned<TirExpr> {
        spanned(TirExpr::SetEnum(vec![bool_const(true), bool_const(false)]))
    }

    // ---- Range ----

    #[test]
    fn test_range_is_set_int() {
        let expr = TirExpr::Range {
            lo: Box::new(int_const(1)),
            hi: Box::new(int_const(10)),
        };
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    // ---- SetEnum ----

    #[test]
    fn test_set_enum_int_elements() {
        let expr = TirExpr::SetEnum(vec![int_const(1), int_const(2), int_const(3)]);
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    #[test]
    fn test_set_enum_empty() {
        let expr = TirExpr::SetEnum(vec![]);
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Dyn)));
    }

    #[test]
    fn test_set_enum_mixed_dyn() {
        // {1, dyn_val}: Dyn joins with Int = Int
        let dyn_const = spanned(TirExpr::Const {
            value: Value::int(0),
            ty: TirType::Dyn,
        });
        let expr = TirExpr::SetEnum(vec![int_const(1), dyn_const]);
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    // ---- SetFilter ----

    #[test]
    fn test_set_filter_preserves_element_type() {
        let bound = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        let expr = TirExpr::SetFilter {
            var: bound,
            body: Box::new(bool_const(true)),
        };
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    // ---- SetBuilder ----

    #[test]
    fn test_set_builder_body_type() {
        let bound = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        // {TRUE : x \in S} has type Set(Bool) because body is Bool
        let expr = TirExpr::SetBuilder {
            body: Box::new(bool_const(true)),
            vars: vec![bound],
        };
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Bool)));
    }

    // ---- SetBinOp ----

    #[test]
    fn test_set_union_preserves_element_type() {
        let expr = TirExpr::SetBinOp {
            left: Box::new(set_of_ints()),
            op: TirSetOp::Union,
            right: Box::new(set_of_ints()),
        };
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    // ---- Powerset ----

    #[test]
    fn test_powerset_type() {
        // SUBSET {1,2} : Set(Set(Int))
        let expr = TirExpr::Powerset(Box::new(set_of_ints()));
        assert_eq!(
            expr.ty(),
            TirType::Set(Box::new(TirType::Set(Box::new(TirType::Int))))
        );
    }

    // ---- BigUnion ----

    #[test]
    fn test_big_union_unwraps_set() {
        // UNION {{1,2},{3}} : if inner is Set(Set(Int)), result is Set(Int)
        let inner = spanned(TirExpr::SetEnum(vec![set_of_ints(), set_of_ints()]));
        let expr = TirExpr::BigUnion(Box::new(inner));
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    // ---- FuncSet ----

    #[test]
    fn test_func_set_type() {
        // [Int -> Bool] : Set(Func(Int, Bool))
        let expr = TirExpr::FuncSet {
            domain: Box::new(set_of_ints()),
            range: Box::new(set_of_bools()),
        };
        assert_eq!(
            expr.ty(),
            TirType::Set(Box::new(TirType::Func(
                Box::new(TirType::Int),
                Box::new(TirType::Bool),
            )))
        );
    }

    // ---- Domain ----

    #[test]
    fn test_domain_of_func() {
        // DOMAIN [x \in S |-> body]: Set(elem(S))
        let bound = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        let func = spanned(TirExpr::FuncDef {
            vars: vec![bound],
            body: Box::new(bool_const(true)),
        });
        let expr = TirExpr::Domain(Box::new(func));
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }

    // ---- FuncDef ----

    #[test]
    fn test_func_def_single_var() {
        let bound = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        let expr = TirExpr::FuncDef {
            vars: vec![bound],
            body: Box::new(bool_const(true)),
        };
        assert_eq!(
            expr.ty(),
            TirType::Func(Box::new(TirType::Int), Box::new(TirType::Bool))
        );
    }

    #[test]
    fn test_func_def_multi_var() {
        let b1 = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        let b2 = TirBoundVar {
            name: "y".to_string(),
            name_id: intern_name("y"),
            domain: Some(Box::new(set_of_bools())),
            pattern: None,
        };
        let expr = TirExpr::FuncDef {
            vars: vec![b1, b2],
            body: Box::new(str_const("hello")),
        };
        assert_eq!(
            expr.ty(),
            TirType::Func(
                Box::new(TirType::Tuple(vec![TirType::Int, TirType::Bool])),
                Box::new(TirType::Str),
            )
        );
    }

    // ---- FuncApply ----

    #[test]
    fn test_func_apply_extracts_range() {
        let bound = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        let func = spanned(TirExpr::FuncDef {
            vars: vec![bound],
            body: Box::new(bool_const(true)),
        });
        let expr = TirExpr::FuncApply {
            func: Box::new(func),
            arg: Box::new(int_const(1)),
        };
        assert_eq!(expr.ty(), TirType::Bool);
    }

    // ---- Choose ----

    #[test]
    fn test_choose_element_type() {
        let bound = TirBoundVar {
            name: "x".to_string(),
            name_id: intern_name("x"),
            domain: Some(Box::new(set_of_ints())),
            pattern: None,
        };
        let expr = TirExpr::Choose {
            var: bound,
            body: Box::new(bool_const(true)),
        };
        assert_eq!(expr.ty(), TirType::Int);
    }

    // ---- Case ----

    #[test]
    fn test_case_joins_arm_types() {
        let arm1 = TirCaseArm {
            guard: bool_const(true),
            body: int_const(1),
        };
        let arm2 = TirCaseArm {
            guard: bool_const(false),
            body: int_const(2),
        };
        let expr = TirExpr::Case {
            arms: vec![arm1, arm2],
            other: None,
        };
        assert_eq!(expr.ty(), TirType::Int);
    }

    // ---- RecordSet ----

    #[test]
    fn test_record_set_type() {
        let id_x = intern_name("x");
        let id_y = intern_name("y");
        let expr = TirExpr::RecordSet(vec![
            (
                TirFieldName {
                    name: "x".to_string(),
                    field_id: id_x,
                },
                set_of_ints(),
            ),
            (
                TirFieldName {
                    name: "y".to_string(),
                    field_id: id_y,
                },
                set_of_bools(),
            ),
        ]);
        assert_eq!(
            expr.ty(),
            TirType::Set(Box::new(TirType::Record(vec![
                (id_x, TirType::Int),
                (id_y, TirType::Bool),
            ])))
        );
    }

    // ---- Times ----

    #[test]
    fn test_times_type() {
        // S1 \X S2: Set(<<elem(S1), elem(S2)>>)
        let expr = TirExpr::Times(vec![set_of_ints(), set_of_bools()]);
        assert_eq!(
            expr.ty(),
            TirType::Set(Box::new(TirType::Tuple(vec![TirType::Int, TirType::Bool])))
        );
    }

    // ---- Prime ----

    #[test]
    fn test_prime_preserves_type() {
        let expr = TirExpr::Prime(Box::new(int_const(42)));
        assert_eq!(expr.ty(), TirType::Int);
    }

    // ---- If with join ----

    #[test]
    fn test_if_joins_branches() {
        // IF TRUE THEN Set({1,2}) ELSE Set({Dyn}) => join gives Set(Int)
        let then_branch = set_of_ints();
        let else_dyn = spanned(TirExpr::SetEnum(vec![spanned(TirExpr::Const {
            value: Value::int(0),
            ty: TirType::Dyn,
        })]));
        let expr = TirExpr::If {
            cond: Box::new(bool_const(true)),
            then_: Box::new(then_branch),
            else_: Box::new(else_dyn),
        };
        // Dyn element joins with Int => Set(Int)
        assert_eq!(expr.ty(), TirType::Set(Box::new(TirType::Int)));
    }
}
