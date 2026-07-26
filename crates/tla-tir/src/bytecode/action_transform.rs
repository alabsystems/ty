// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Action predicate -> next-state function transformation.
//!
//! TLA+ actions are boolean predicates (`x' = expr /\ UNCHANGED y`).
//! The JIT next-state cache requires bytecode functions that produce
//! successor states using `StoreVar` opcodes.
//!
//! This module rewrites simple prime-equality patterns:
//!
//! ```text
//! LoadPrime { rd: rp, var_idx: x }   -.
//! Eq { rd: req, r1: rp, r2: rexpr }  -+-> LoadBool { rd: req, value: true }
//!                                        StoreVar { var_idx: x, rs: rexpr }
//! ```
//!
//! The rewrite preserves instruction count and jump targets by emitting the
//! action-enabled flag (`LoadBool(true)`) at the original `LoadPrime` PC and
//! delaying the `StoreVar` until the original `Eq` PC, after the RHS register
//! has definitely been computed.
//!
//! `Unchanged` opcodes are kept as-is. They rely on the caller seeding the
//! successor buffer from the predecessor state before execution.

use rustc_hash::FxHashSet;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use tla_value::Rp;

use num_traits::ToPrimitive;
use tla_core::{intern_name, NameId};
use tla_value::Value;

use super::opcode::{ConstIdx, Opcode, Register, VarIdx};
use super::ConstantPool;

/// Packed `Option<bool>` facts for all 256 bytecode registers.
///
/// Semantically identical to `[Option<bool>; 256]` (per-register tri-state),
/// but stored as two 256-bit bitmaps (known + value). The unknown-value bit is
/// kept normalized to 0 so `Eq`/`Hash` stay canonical. This keeps path-state
/// hashing in the safety analyses below at 64 bytes per state instead of
/// SipHashing a 256-element `Option<bool>` array — the dominant fixed cost of
/// the action transform on action-heavy specs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct KnownBools {
    known: [u64; 4],
    value: [u64; 4],
}

impl KnownBools {
    #[inline]
    fn get(&self, reg: Register) -> Option<bool> {
        let w = (reg >> 6) as usize;
        let b = reg & 63;
        if (self.known[w] >> b) & 1 == 0 {
            None
        } else {
            Some((self.value[w] >> b) & 1 == 1)
        }
    }

    #[inline]
    fn set(&mut self, reg: Register, fact: Option<bool>) {
        let w = (reg >> 6) as usize;
        let b = reg & 63;
        match fact {
            None => {
                self.known[w] &= !(1u64 << b);
                self.value[w] &= !(1u64 << b);
            }
            Some(true) => {
                self.known[w] |= 1u64 << b;
                self.value[w] |= 1u64 << b;
            }
            Some(false) => {
                self.known[w] |= 1u64 << b;
                self.value[w] &= !(1u64 << b);
            }
        }
    }
}

/// Result of attempting to rewrite an action predicate into next-state bytecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionTransformOutcome {
    /// The action was safely rewritten for next-state execution.
    Transformed(Vec<Opcode>),
    /// No next-state rewrite pattern was found.
    NoRewrite,
    /// The action used primed values in a way the next-state ABI cannot
    /// represent soundly.
    Unsafe(String),
}

/// Scalar binding/domain element with enough type information to avoid raw-i64
/// collisions between strings, model values, booleans, and integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionScalarValue {
    /// An integer scalar.
    Int(i64),
    /// A boolean scalar.
    Bool(bool),
    /// A string literal, identified by its interned name id.
    String(NameId),
    /// A model value (uninterpreted constant), identified by its interned name id.
    ModelValue(NameId),
}

impl ActionScalarValue {
    /// The raw value used by the compact bytecode/JIT ABI for scalar lanes.
    #[must_use]
    pub fn as_jit_i64(self) -> i64 {
        match self {
            Self::Int(value) => value,
            Self::Bool(value) => i64::from(value),
            Self::String(name) | Self::ModelValue(name) => i64::from(name.0),
        }
    }
}

/// One exact finite-domain element.
///
/// Residual quantifiers may range over scalars (`x \in Proc`) or over finite
/// set values (`S \in SUBSET Resources`). Keeping set-valued elements typed
/// prevents model values from collapsing into raw integer/name-id metadata.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActionDomainElement {
    /// A scalar element (e.g. `x \in Proc`).
    Scalar(ActionScalarValue),
    /// A set-valued element (e.g. `S \in SUBSET Resources`).
    Set(Vec<ActionScalarValue>),
}

impl From<ActionScalarValue> for ActionDomainElement {
    fn from(value: ActionScalarValue) -> Self {
        Self::Scalar(value)
    }
}

/// Where an exact finite binding/domain shape came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionDomainSource {
    /// The value came directly from action-split binding metadata.
    BindingValue,
    /// A constant-pool finite set or interval.
    ConstantPool {
        /// Constant-pool index of the finite set/interval value.
        idx: ConstIdx,
    },
    /// A bytecode `{...}` set enumeration.
    SetEnum {
        /// Program counter of the `SetEnum` instruction.
        pc: usize,
        /// First register of the enumerated element block.
        start: Register,
        /// Number of enumerated elements.
        count: u8,
    },
    /// A bytecode integer interval.
    Range {
        /// Program counter of the `Range` instruction.
        pc: usize,
    },
    /// A finite-domain set difference whose operands were both exact.
    SetDiff {
        /// Program counter of the `SetDiff` instruction.
        pc: usize,
    },
    /// A finite-domain intersection whose operands were both exact.
    SetIntersect {
        /// Program counter of the `SetIntersect` instruction.
        pc: usize,
    },
    /// A finite-domain union whose operands were both exact.
    SetUnion {
        /// Program counter of the `SetUnion` instruction.
        pc: usize,
    },
    /// A finite powerset whose base set was exact.
    Powerset {
        /// Program counter of the `Powerset` instruction.
        pc: usize,
    },
    /// A finite k-subset set whose base set and k were exact.
    KSubset {
        /// Program counter of the `KSubset` instruction.
        pc: usize,
    },
    /// External proof that a register is an exact compact finite set.
    CompactProof {
        /// Human-readable description of the supplied proof.
        description: String,
    },
}

/// Exact finite set/domain shape used by split-action and residual-quantifier
/// lowering metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionFiniteDomain {
    /// The exact, enumerated elements of the domain.
    pub elements: Vec<ActionDomainElement>,
    /// Where this finite domain shape was derived from.
    pub source: ActionDomainSource,
}

/// Shape-preserving action binding value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionBindingValue {
    /// A single scalar binding value.
    Scalar(ActionScalarValue),
    /// A finite-domain binding value (a quantified set).
    FiniteDomain(ActionFiniteDomain),
}

/// Binding metadata that does not collapse finite domains into scalar-only
/// `BindingSpec` data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionBindingSpec {
    /// Name of the bound variable.
    pub name: Arc<str>,
    /// The value (scalar or finite domain) bound to the variable.
    pub value: ActionBindingValue,
}

/// Residual quantifier kind whose domain can be preserved for later lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualQuantifierKind {
    /// An existential quantifier (`\E`).
    Exists,
    /// A universal quantifier (`\A`).
    Forall,
}

/// Exact domain metadata for an EXISTS/FORALL left in action bytecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualQuantifierDomain {
    /// Program counter of the quantifier's `*Begin` instruction.
    pub begin_pc: usize,
    /// Whether the residual quantifier is an EXISTS or a FORALL.
    pub kind: ResidualQuantifierKind,
    /// The quantifier's result register.
    pub rd: Register,
    /// The quantifier's bound-variable register.
    pub r_binding: Register,
    /// The quantifier's domain-set register.
    pub r_domain: Register,
    /// The exact finite domain the quantifier ranges over.
    pub domain: ActionFiniteDomain,
}

/// A trusted compact-domain proof supplied by a caller that has layout/source
/// metadata unavailable in raw bytecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDomainHint {
    /// Program counter the domain register is known-exact immediately before.
    pub before_pc: usize,
    /// Register asserted to hold an exact compact finite set.
    pub domain_reg: Register,
    /// The exact finite domain asserted for that register.
    pub domain: ActionFiniteDomain,
}

/// Fail-closed reason while preserving action binding/domain metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionDomainShapeError {
    /// A binding value's shape is not representable by the metadata model.
    UnsupportedBinding {
        /// Name of the offending bound variable.
        name: String,
        /// Kind of value that could not be represented.
        value_kind: &'static str,
    },
    /// A finite domain exceeded the maximum enumerable size.
    DomainTooLarge {
        /// Context describing which domain overflowed.
        context: String,
        /// Actual element count encountered.
        len: usize,
        /// Maximum element count allowed.
        max: usize,
    },
    /// A residual quantifier's domain could not be resolved to an exact shape.
    UnknownQuantifierDomain {
        /// Program counter of the quantifier's `*Begin` instruction.
        begin_pc: usize,
        /// Whether the quantifier is an EXISTS or FORALL.
        kind: ResidualQuantifierKind,
        /// The unresolved domain-set register.
        r_domain: Register,
    },
}

impl fmt::Display for ActionDomainShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBinding { name, value_kind } => {
                write!(
                    f,
                    "binding '{name}' has unsupported non-finite value kind {value_kind}"
                )
            }
            Self::DomainTooLarge { context, len, max } => {
                write!(f, "{context} finite domain has {len} elements, max {max}")
            }
            Self::UnknownQuantifierDomain {
                begin_pc,
                kind,
                r_domain,
            } => write!(
                f,
                "{kind:?} at pc {begin_pc} has unknown or dynamic finite domain r{r_domain}"
            ),
        }
    }
}

impl std::error::Error for ActionDomainShapeError {}

/// Convert split-action bindings into shape-preserving metadata.
///
/// Scalars remain scalar. Exact finite sets/intervals remain finite domains
/// with typed elements, so model-value/process sets are not reduced to an
/// opaque materialized value or a raw scalar-only `BindingSpec` vector.
///
/// Unknown, dynamic, unbounded, or non-scalar-element domains return an error
/// rather than guessing a lowering shape.
pub fn action_binding_specs_from_values(
    bindings: &[(Arc<str>, Value)],
) -> Result<Vec<ActionBindingSpec>, ActionDomainShapeError> {
    bindings
        .iter()
        .map(|(name, value)| {
            let value = if let Some(scalar) = scalar_value_from_value(value) {
                ActionBindingValue::Scalar(scalar)
            } else if let Some(domain) =
                finite_domain_from_value(value, ActionDomainSource::BindingValue)
            {
                ensure_domain_size(
                    &domain,
                    format!("binding '{name}'"),
                    MAX_ACTION_DOMAIN_METADATA_SIZE,
                )?;
                ActionBindingValue::FiniteDomain(domain)
            } else {
                return Err(ActionDomainShapeError::UnsupportedBinding {
                    name: name.to_string(),
                    value_kind: value_kind_name(value),
                });
            };
            Ok(ActionBindingSpec {
                name: Arc::clone(name),
                value,
            })
        })
        .collect()
}

/// Collect exact finite-domain metadata for residual EXISTS/FORALL bytecode.
///
/// This is intentionally fail-closed: every residual quantifier in the stream
/// must have a domain that can be resolved from prior bytecode or from an
/// explicit compact-domain hint.
pub fn collect_residual_quantifier_domains(
    instructions: &[Opcode],
    constants: Option<&ConstantPool>,
) -> Result<Vec<ResidualQuantifierDomain>, ActionDomainShapeError> {
    collect_residual_quantifier_domains_with_hints(instructions, constants, &[])
}

/// Collect residual EXISTS/FORALL domains, accepting explicit compact-domain
/// hints from callers that own layout/source proofs outside bytecode.
pub fn collect_residual_quantifier_domains_with_hints(
    instructions: &[Opcode],
    constants: Option<&ConstantPool>,
    hints: &[ActionDomainHint],
) -> Result<Vec<ResidualQuantifierDomain>, ActionDomainShapeError> {
    let mut scalars: HashMap<Register, ActionScalarValue> = HashMap::new();
    let mut domains: HashMap<Register, ActionFiniteDomain> = HashMap::new();
    let mut result = Vec::new();

    for (pc, op) in instructions.iter().copied().enumerate() {
        match op {
            Opcode::ExistsBegin {
                rd,
                r_binding,
                r_domain,
                ..
            } => {
                let domain = domain_for_quantifier(pc, r_domain, &domains, hints).ok_or(
                    ActionDomainShapeError::UnknownQuantifierDomain {
                        begin_pc: pc,
                        kind: ResidualQuantifierKind::Exists,
                        r_domain,
                    },
                )?;
                ensure_domain_size(
                    &domain,
                    format!("EXISTS at pc {pc}"),
                    MAX_ACTION_DOMAIN_METADATA_SIZE,
                )?;
                result.push(ResidualQuantifierDomain {
                    begin_pc: pc,
                    kind: ResidualQuantifierKind::Exists,
                    rd,
                    r_binding,
                    r_domain,
                    domain,
                });
                clear_register_shape(rd, &mut scalars, &mut domains);
                clear_register_shape(r_binding, &mut scalars, &mut domains);
            }
            Opcode::ForallBegin {
                rd,
                r_binding,
                r_domain,
                ..
            } => {
                let domain = domain_for_quantifier(pc, r_domain, &domains, hints).ok_or(
                    ActionDomainShapeError::UnknownQuantifierDomain {
                        begin_pc: pc,
                        kind: ResidualQuantifierKind::Forall,
                        r_domain,
                    },
                )?;
                ensure_domain_size(
                    &domain,
                    format!("FORALL at pc {pc}"),
                    MAX_ACTION_DOMAIN_METADATA_SIZE,
                )?;
                result.push(ResidualQuantifierDomain {
                    begin_pc: pc,
                    kind: ResidualQuantifierKind::Forall,
                    rd,
                    r_binding,
                    r_domain,
                    domain,
                });
                clear_register_shape(rd, &mut scalars, &mut domains);
                clear_register_shape(r_binding, &mut scalars, &mut domains);
            }
            _ => transfer_domain_shapes(pc, op, constants, &mut scalars, &mut domains),
        }
    }

    Ok(result)
}

const MAX_ACTION_DOMAIN_METADATA_SIZE: usize = 100;

fn ensure_domain_size(
    domain: &ActionFiniteDomain,
    context: String,
    max: usize,
) -> Result<(), ActionDomainShapeError> {
    if domain.elements.len() <= max {
        Ok(())
    } else {
        Err(ActionDomainShapeError::DomainTooLarge {
            context,
            len: domain.elements.len(),
            max,
        })
    }
}

fn domain_for_quantifier(
    pc: usize,
    r_domain: Register,
    domains: &HashMap<Register, ActionFiniteDomain>,
    hints: &[ActionDomainHint],
) -> Option<ActionFiniteDomain> {
    domains.get(&r_domain).cloned().or_else(|| {
        hints
            .iter()
            .find(|hint| hint.before_pc == pc && hint.domain_reg == r_domain)
            .map(|hint| hint.domain.clone())
    })
}

fn transfer_domain_shapes(
    pc: usize,
    op: Opcode,
    constants: Option<&ConstantPool>,
    scalars: &mut HashMap<Register, ActionScalarValue>,
    domains: &mut HashMap<Register, ActionFiniteDomain>,
) {
    match op {
        Opcode::LoadImm { rd, value } => {
            set_scalar_shape(rd, ActionScalarValue::Int(value), scalars, domains);
        }
        Opcode::LoadBool { rd, value } => {
            set_scalar_shape(rd, ActionScalarValue::Bool(value), scalars, domains);
        }
        Opcode::LoadConst { rd, idx } => {
            if let Some(pool) = constants {
                let value = pool.get_value(idx);
                if let Some(scalar) = scalar_value_from_value(value) {
                    set_scalar_shape(rd, scalar, scalars, domains);
                } else if let Some(domain) =
                    finite_domain_from_value(value, ActionDomainSource::ConstantPool { idx })
                {
                    set_domain_shape(rd, domain, scalars, domains);
                } else {
                    clear_register_shape(rd, scalars, domains);
                }
            } else {
                clear_register_shape(rd, scalars, domains);
            }
        }
        Opcode::Move { rd, rs } => {
            if let Some(scalar) = scalars.get(&rs).copied() {
                set_scalar_shape(rd, scalar, scalars, domains);
            } else if let Some(domain) = domains.get(&rs).cloned() {
                set_domain_shape(rd, domain, scalars, domains);
            } else {
                clear_register_shape(rd, scalars, domains);
            }
        }
        Opcode::SetEnum { rd, start, count } => {
            let elements = (0..count)
                .map(|offset| {
                    start
                        .checked_add(offset)
                        .and_then(|reg| scalars.get(&reg).copied())
                        .map(ActionDomainElement::Scalar)
                })
                .collect::<Option<Vec<_>>>();
            if let Some(elements) = elements {
                set_domain_shape(
                    rd,
                    ActionFiniteDomain {
                        elements,
                        source: ActionDomainSource::SetEnum { pc, start, count },
                    },
                    scalars,
                    domains,
                );
            } else {
                clear_register_shape(rd, scalars, domains);
            }
        }
        Opcode::Range { rd, lo, hi } => {
            let domain = match (scalars.get(&lo).copied(), scalars.get(&hi).copied()) {
                (Some(ActionScalarValue::Int(lo)), Some(ActionScalarValue::Int(hi)))
                    if hi >= lo =>
                {
                    let count = hi.checked_sub(lo).and_then(|n| n.checked_add(1));
                    count
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|&n| n <= MAX_ACTION_DOMAIN_METADATA_SIZE)
                        .map(|_| ActionFiniteDomain {
                            elements: (lo..=hi)
                                .map(ActionScalarValue::Int)
                                .map(ActionDomainElement::Scalar)
                                .collect(),
                            source: ActionDomainSource::Range { pc },
                        })
                }
                _ => None,
            };
            if let Some(domain) = domain {
                set_domain_shape(rd, domain, scalars, domains);
            } else {
                clear_register_shape(rd, scalars, domains);
            }
        }
        Opcode::SetDiff { rd, r1, r2 } => {
            set_binary_domain_shape(
                rd,
                r1,
                r2,
                domains,
                scalars,
                ActionDomainSource::SetDiff { pc },
                |left, right| {
                    let right: HashSet<_> = right.iter().cloned().collect();
                    left.iter()
                        .filter(|value| !right.contains(*value))
                        .cloned()
                        .collect()
                },
            );
        }
        Opcode::SetIntersect { rd, r1, r2 } => {
            set_binary_domain_shape(
                rd,
                r1,
                r2,
                domains,
                scalars,
                ActionDomainSource::SetIntersect { pc },
                |left, right| {
                    let right: HashSet<_> = right.iter().cloned().collect();
                    left.iter()
                        .filter(|value| right.contains(*value))
                        .cloned()
                        .collect()
                },
            );
        }
        Opcode::SetUnion { rd, r1, r2 } => {
            set_binary_domain_shape(
                rd,
                r1,
                r2,
                domains,
                scalars,
                ActionDomainSource::SetUnion { pc },
                |left, right| {
                    let mut values = left.to_vec();
                    values.extend_from_slice(right);
                    values.sort_unstable();
                    values.dedup();
                    values
                },
            );
        }
        Opcode::Powerset { rd, rs } => {
            let domain = domains.get(&rs).and_then(|base| {
                powerset_domain_from_scalar_domain(base, ActionDomainSource::Powerset { pc })
            });
            if let Some(domain) = domain {
                set_domain_shape(rd, domain, scalars, domains);
            } else {
                clear_register_shape(rd, scalars, domains);
            }
        }
        Opcode::KSubset { rd, base, k } => {
            let domain = domains
                .get(&base)
                .zip(scalars.get(&k))
                .and_then(|(base, k)| match k {
                    ActionScalarValue::Int(k) if *k >= 0 => {
                        usize::try_from(*k).ok().and_then(|k| {
                            ksubset_domain_from_scalar_domain(
                                base,
                                k,
                                ActionDomainSource::KSubset { pc },
                            )
                        })
                    }
                    _ => None,
                });
            if let Some(domain) = domain {
                set_domain_shape(rd, domain, scalars, domains);
            } else {
                clear_register_shape(rd, scalars, domains);
            }
        }
        _ => {
            if let Some(rd) = op.dest_register() {
                clear_register_shape(rd, scalars, domains);
            }
            if let Some(r_binding) = op.binding_register() {
                clear_register_shape(r_binding, scalars, domains);
            }
        }
    }
}

fn set_binary_domain_shape(
    rd: Register,
    r1: Register,
    r2: Register,
    domains: &mut HashMap<Register, ActionFiniteDomain>,
    scalars: &mut HashMap<Register, ActionScalarValue>,
    source: ActionDomainSource,
    combine: impl FnOnce(&[ActionDomainElement], &[ActionDomainElement]) -> Vec<ActionDomainElement>,
) {
    let domain = domains
        .get(&r1)
        .zip(domains.get(&r2))
        .map(|(left, right)| ActionFiniteDomain {
            elements: combine(&left.elements, &right.elements),
            source,
        });
    if let Some(domain) = domain {
        set_domain_shape(rd, domain, scalars, domains);
    } else {
        clear_register_shape(rd, scalars, domains);
    }
}

fn set_scalar_shape(
    rd: Register,
    scalar: ActionScalarValue,
    scalars: &mut HashMap<Register, ActionScalarValue>,
    domains: &mut HashMap<Register, ActionFiniteDomain>,
) {
    scalars.insert(rd, scalar);
    domains.remove(&rd);
}

fn set_domain_shape(
    rd: Register,
    domain: ActionFiniteDomain,
    scalars: &mut HashMap<Register, ActionScalarValue>,
    domains: &mut HashMap<Register, ActionFiniteDomain>,
) {
    scalars.remove(&rd);
    domains.insert(rd, domain);
}

fn clear_register_shape(
    rd: Register,
    scalars: &mut HashMap<Register, ActionScalarValue>,
    domains: &mut HashMap<Register, ActionFiniteDomain>,
) {
    scalars.remove(&rd);
    domains.remove(&rd);
}

fn finite_domain_from_value(
    value: &Value,
    source: ActionDomainSource,
) -> Option<ActionFiniteDomain> {
    let elements = match value {
        Value::Set(set) => set
            .iter()
            .map(scalar_value_from_value)
            .map(|value| value.map(ActionDomainElement::Scalar))
            .collect::<Option<Vec<_>>>()?,
        Value::Interval(interval) => {
            let lo = interval.low().to_i64()?;
            let hi = interval.high().to_i64()?;
            if hi < lo {
                return Some(ActionFiniteDomain {
                    elements: Vec::new(),
                    source,
                });
            }
            let len = usize::try_from(hi.checked_sub(lo)?.checked_add(1)?).ok()?;
            if len > MAX_ACTION_DOMAIN_METADATA_SIZE {
                return None;
            }
            (lo..=hi)
                .map(ActionScalarValue::Int)
                .map(ActionDomainElement::Scalar)
                .collect()
        }
        Value::Subset(subset) => {
            let base = finite_domain_from_value(subset.base(), source.clone())?;
            return powerset_domain_from_scalar_domain(&base, source);
        }
        Value::KSubset(ksubset) => {
            let base = finite_domain_from_value(ksubset.base(), source.clone())?;
            return ksubset_domain_from_scalar_domain(&base, ksubset.k(), source);
        }
        _ => return None,
    };
    Some(ActionFiniteDomain { elements, source })
}

fn powerset_domain_from_scalar_domain(
    base: &ActionFiniteDomain,
    source: ActionDomainSource,
) -> Option<ActionFiniteDomain> {
    let base_scalars = base
        .elements
        .iter()
        .map(|element| match element {
            ActionDomainElement::Scalar(value) => Some(*value),
            ActionDomainElement::Set(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let subset_count = checked_powerset_len(base_scalars.len())?;
    if subset_count > MAX_ACTION_DOMAIN_METADATA_SIZE {
        return None;
    }

    let mut elements = Vec::with_capacity(subset_count);
    elements.push(ActionDomainElement::Set(Vec::new()));
    for k in 1..=base_scalars.len() {
        let mut indices: Vec<usize> = (0..k).collect();
        loop {
            elements.push(ActionDomainElement::Set(
                indices.iter().map(|&idx| base_scalars[idx]).collect(),
            ));

            let mut i = k;
            while i > 0 {
                i -= 1;
                if indices[i] < base_scalars.len() - k + i {
                    break;
                }
            }
            if i == 0 && indices[0] >= base_scalars.len() - k {
                break;
            }
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
        }
    }

    Some(ActionFiniteDomain { elements, source })
}

fn ksubset_domain_from_scalar_domain(
    base: &ActionFiniteDomain,
    k: usize,
    source: ActionDomainSource,
) -> Option<ActionFiniteDomain> {
    let base_scalars = base
        .elements
        .iter()
        .map(|element| match element {
            ActionDomainElement::Scalar(value) => Some(*value),
            ActionDomainElement::Set(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let subset_count =
        checked_combination_len_capped(base_scalars.len(), k, MAX_ACTION_DOMAIN_METADATA_SIZE)?;
    if subset_count > MAX_ACTION_DOMAIN_METADATA_SIZE {
        return None;
    }

    let mut elements = Vec::with_capacity(subset_count);
    push_k_subset_domain_elements(&base_scalars, k, &mut elements);
    Some(ActionFiniteDomain { elements, source })
}

fn push_k_subset_domain_elements(
    base_scalars: &[ActionScalarValue],
    k: usize,
    elements: &mut Vec<ActionDomainElement>,
) {
    if k > base_scalars.len() {
        return;
    }
    if k == 0 {
        elements.push(ActionDomainElement::Set(Vec::new()));
        return;
    }

    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        elements.push(ActionDomainElement::Set(
            indices.iter().map(|&idx| base_scalars[idx]).collect(),
        ));

        let mut i = k;
        while i > 0 {
            i -= 1;
            if indices[i] < base_scalars.len() - k + i {
                break;
            }
        }
        if i == 0 && indices[0] >= base_scalars.len() - k {
            break;
        }
        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

fn checked_powerset_len(base_len: usize) -> Option<usize> {
    if base_len >= usize::BITS as usize {
        return None;
    }
    1usize.checked_shl(u32::try_from(base_len).ok()?)
}

fn checked_combination_len_capped(n: usize, k: usize, cap: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        result = result.checked_mul((n - k + i) as u128)? / (i as u128);
        if result > cap as u128 {
            return cap.checked_add(1);
        }
    }
    usize::try_from(result).ok()
}

fn scalar_value_from_value(value: &Value) -> Option<ActionScalarValue> {
    match value {
        Value::SmallInt(value) => Some(ActionScalarValue::Int(*value)),
        Value::Int(value) => value.to_i64().map(ActionScalarValue::Int),
        Value::Bool(value) => Some(ActionScalarValue::Bool(*value)),
        Value::String(value) => Some(ActionScalarValue::String(intern_name(value.as_ref()))),
        Value::ModelValue(value) => {
            Some(ActionScalarValue::ModelValue(intern_name(value.as_ref())))
        }
        _ => None,
    }
}

fn value_kind_name(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "Bool",
        Value::SmallInt(_) | Value::Int(_) => "Int",
        Value::String(_) => "String",
        Value::Set(_) => "Set",
        Value::Interval(_) => "Interval",
        Value::Subset(_) => "Subset",
        Value::FuncSet(_) => "FuncSet",
        Value::RecordSet(_) => "RecordSet",
        Value::TupleSet(_) => "TupleSet",
        Value::SetCup(_) => "SetCup",
        Value::SetCap(_) => "SetCap",
        Value::SetDiff(_) => "SetDiff",
        Value::SetPred(_) => "SetPred",
        Value::KSubset(_) => "KSubset",
        Value::BigUnion(_) => "BigUnion",
        Value::Func(_) => "Func",
        Value::IntFunc(_) => "IntFunc",
        Value::LazyFunc(_) => "LazyFunc",
        Value::Seq(_) => "Seq",
        Value::Record(_) => "Record",
        Value::Tuple(_) => "Tuple",
        Value::ModelValue(_) => "ModelValue",
        Value::Closure(_) => "Closure",
        Value::StringSet => "StringSet",
        Value::AnySet => "AnySet",
        Value::SeqSet(_) => "SeqSet",
        _ => "Other",
    }
}

/// Transform an action predicate's bytecode into next-state function bytecode.
///
/// The transform is intentionally conservative. It rejects actions that read
/// residual primed values before the successor slot has definitely been
/// materialized, actions that would write the same primed variable multiple
/// times on one control-flow path, and any function that would retain
/// next-state-only machinery without a proof.
pub fn transform_action_to_next_state(instructions: &[Opcode]) -> ActionTransformOutcome {
    transform_action_to_next_state_impl(instructions, None)
}

/// Transform an action predicate's bytecode with access to constant-pool
/// metadata for `Unchanged` opcodes.
///
/// The plain [`transform_action_to_next_state`] wrapper can prove residual
/// `LoadPrime` reads only from prior `StoreVar` opcodes. This variant can also
/// treat a preceding `Unchanged` opcode as a must-write proof for the specific
/// variables listed in the constant pool.
pub fn transform_action_to_next_state_with_constants(
    instructions: &[Opcode],
    constants: &ConstantPool,
) -> ActionTransformOutcome {
    transform_action_to_next_state_impl(instructions, Some(constants))
}

fn transform_action_to_next_state_impl(
    instructions: &[Opcode],
    constants: Option<&ConstantPool>,
) -> ActionTransformOutcome {
    if instructions.is_empty() {
        return ActionTransformOutcome::NoRewrite;
    }
    if instructions
        .iter()
        .any(|op| matches!(op, Opcode::RoundStepEq { .. }))
    {
        return ActionTransformOutcome::Unsafe(
            "RoundStepEq is VM-only and cannot enter action/native transformation".to_string(),
        );
    }
    if instructions
        .iter()
        .any(|op| matches!(op, Opcode::EdgeFilter { .. }))
    {
        return ActionTransformOutcome::Unsafe(
            "EdgeFilter is VM-only and cannot enter action/native transformation".to_string(),
        );
    }

    let mut prime_defs: Vec<Option<(VarIdx, usize)>> = vec![None; 256];
    let mut candidates: Vec<PrimeEqRewrite> = Vec::new();
    let mut total_prime_loads = 0usize;
    let mut has_unchanged = false;

    for (pc, op) in instructions.iter().enumerate() {
        match *op {
            Opcode::LoadPrime { rd, var_idx } => {
                prime_defs[rd as usize] = Some((var_idx, pc));
                total_prime_loads += 1;
            }
            Opcode::Eq { rd, r1, r2 } => {
                if let Some((var_idx, prime_pc)) = prime_defs[r1 as usize] {
                    candidates.push(PrimeEqRewrite {
                        prime_pc,
                        eq_pc: pc,
                        var_idx,
                        expr_reg: r2,
                        eq_rd: rd,
                    });
                    prime_defs[r1 as usize] = None;
                } else if let Some((var_idx, prime_pc)) = prime_defs[r2 as usize] {
                    candidates.push(PrimeEqRewrite {
                        prime_pc,
                        eq_pc: pc,
                        var_idx,
                        expr_reg: r1,
                        eq_rd: rd,
                    });
                    prime_defs[r2 as usize] = None;
                }
                prime_defs[rd as usize] = None;
            }
            Opcode::Unchanged { .. } => {
                has_unchanged = true;
                clear_register_defs(*op, &mut prime_defs);
            }
            _ => {
                clear_register_defs(*op, &mut prime_defs);
            }
        }
    }

    let provisional = apply_prime_eq_rewrites(instructions, &candidates);
    let mut rewrites = Vec::new();
    for candidate in candidates {
        match load_prime_has_must_write_before_pc(
            &provisional,
            constants,
            candidate.prime_pc,
            candidate.var_idx,
        ) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(reason) => return ActionTransformOutcome::Unsafe(reason),
        }
        if eq_result_interferes_before_rewrite(
            instructions,
            candidate.eq_rd,
            candidate.expr_reg,
            candidate.prime_pc,
            candidate.eq_pc,
        ) {
            return ActionTransformOutcome::Unsafe(format!(
                "Eq destination r{} conflicts with live RHS before rewrite",
                candidate.eq_rd
            ));
        }
        if let Some(quant) = enclosing_quantifier_body(instructions, candidate.eq_pc) {
            // A `v' = e` prime-equality lowered inside a quantifier loop body is
            // only the binder's witness when its Eq result actually flows into
            // the quantifier's body truth value (`r_body`). When the Eq result
            // is discarded -- e.g. the body register is forced to a constant
            // `LoadBool false` -- the existential is unsatisfiable, yet rewriting
            // the prime-eq to an unconditional StoreVar would (a) fire a disabled
            // action and (b) pick arbitrarily among conflicting writes to the
            // same primed variable. Decline the rewrite in that case.
            if !eq_result_feeds_quantifier_body(
                instructions,
                candidate.eq_rd,
                candidate.eq_pc,
                quant,
            ) {
                return ActionTransformOutcome::Unsafe(format!(
                    "duplicate writes to primed var {}: prime-equality Eq result r{} is \
                     discarded by the enclosing quantifier body (not the binder witness)",
                    candidate.var_idx, candidate.eq_rd
                ));
            }
        }
        rewrites.push(candidate);
    }

    let result = apply_prime_eq_rewrites(instructions, &rewrites);

    if let Err(reason) = validate_residual_load_primes_after_must_write(&result, constants) {
        return ActionTransformOutcome::Unsafe(reason);
    }

    if let Some(pc) = residual_set_prime_mode_pc(&result) {
        return ActionTransformOutcome::Unsafe(format!(
            "SetPrimeMode remains after action rewrite at pc {pc}"
        ));
    }

    if let Some(var_idx) = duplicate_store_on_same_path(&result) {
        // Include the post-rewrite listing so the fail-closed reason is
        // actionable (which two StoreVars share a path, and what produced
        // them).
        let listing: Vec<String> = result
            .iter()
            .enumerate()
            .map(|(pc, op)| format!("pc {pc}: {op:?}"))
            .collect();
        return ActionTransformOutcome::Unsafe(format!(
            "duplicate writes to primed var {var_idx}. post-rewrite body: [{}]",
            listing.join("; ")
        ));
    }

    if rewrites.is_empty() {
        // UNCHANGED-only actions are still executable next-state actions. The
        // caller seeds state_out from state_in before native execution, so the
        // unchanged successor is represented without StoreVar opcodes.
        if has_unchanged || total_prime_loads > 0 {
            return ActionTransformOutcome::Transformed(result);
        }
        return ActionTransformOutcome::NoRewrite;
    }

    ActionTransformOutcome::Transformed(result)
}

fn residual_set_prime_mode_pc(instructions: &[Opcode]) -> Option<usize> {
    instructions
        .iter()
        .position(|op| matches!(op, Opcode::SetPrimeMode { .. }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrimeEqRewrite {
    prime_pc: usize,
    eq_pc: usize,
    var_idx: VarIdx,
    expr_reg: Register,
    eq_rd: Register,
}

fn apply_prime_eq_rewrites(instructions: &[Opcode], rewrites: &[PrimeEqRewrite]) -> Vec<Opcode> {
    let mut result = instructions.to_vec();
    for rewrite in rewrites {
        result[rewrite.prime_pc] = Opcode::LoadBool {
            rd: rewrite.eq_rd,
            value: true,
        };
        result[rewrite.eq_pc] = Opcode::StoreVar {
            var_idx: rewrite.var_idx,
            rs: rewrite.expr_reg,
        };
    }
    result
}

fn clear_register_defs(op: Opcode, prime_defs: &mut [Option<(VarIdx, usize)>]) {
    if let Some(rd) = op.dest_register() {
        prime_defs[rd as usize] = None;
    }
    if let Some(r_binding) = op.binding_register() {
        prime_defs[r_binding as usize] = None;
    }
}

/// A `QuantBegin .. QuantNext` loop, identified structurally from a `*Next`
/// opcode's back-edge offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantifierBody {
    /// First instruction of the loop body (target of the `*Next` back-edge).
    body_start: usize,
    /// PC of the `*Next` opcode that closes the loop.
    next_pc: usize,
    /// Register holding the per-iteration body truth value read by `*Next`.
    body_reg: Register,
}

/// Return the innermost quantifier loop body whose body span `[body_start,
/// next_pc)` contains `target_pc`, if any.
///
/// A `*Next` opcode at `next_pc` jumps back to `next_pc + loop_begin` (a
/// negative offset patched by the compiler), so the body occupies
/// `[body_start, next_pc)`. The matching `*Begin` sits at `body_start - 1`.
/// Innermost is selected by largest `body_start` so nested quantifiers resolve
/// to the loop closest to the prime-equality.
fn enclosing_quantifier_body(instructions: &[Opcode], target_pc: usize) -> Option<QuantifierBody> {
    let mut best: Option<QuantifierBody> = None;
    for (next_pc, op) in instructions.iter().enumerate() {
        let (r_body, loop_begin) = match *op {
            Opcode::ExistsNext {
                r_body, loop_begin, ..
            }
            | Opcode::ForallNext {
                r_body, loop_begin, ..
            }
            | Opcode::ChooseNext {
                r_body, loop_begin, ..
            }
            | Opcode::LoopNext {
                r_body, loop_begin, ..
            } => (r_body, loop_begin),
            _ => continue,
        };
        let Some(body_start) = jump_target(next_pc, loop_begin, instructions.len()) else {
            continue;
        };
        if body_start <= target_pc && target_pc < next_pc {
            let candidate = QuantifierBody {
                body_start,
                next_pc,
                body_reg: r_body,
            };
            if best.is_none_or(|prev| candidate.body_start > prev.body_start) {
                best = Some(candidate);
            }
        }
    }
    best
}

/// Decide whether the value computed by a prime-equality `Eq` (whose result is
/// in `eq_rd` at `eq_pc`) flows into the enclosing quantifier's body truth
/// value (`quant.body_reg`) -- i.e. whether the prime-equality is the binder's
/// witness rather than a discarded sub-expression.
///
/// This is a forward, monotone (may-reach) taint propagation over the loop body
/// that reuses the crate's exhaustive [`opcode_reads_tainted`] read model. We
/// seed the taint at `eq_rd` and walk the body span in pc order; whenever an
/// instruction reads a tainted register, its destination (and binding register,
/// for loop opcodes) is also marked tainted. Taint is never cleared: distinct
/// quantifier-body branches (e.g. the two arms of a disjunction) are exclusive
/// at runtime, so a definition on one arm must not erase a witness flow that
/// occurs on another. The result is a sound over-approximation of "the witness
/// value may reach `body_reg`": it never misses a genuine flow, so it never
/// spuriously declines a real witness and therefore cannot over-reject valid
/// actions. It only ever returns `false` when `body_reg` is computed entirely
/// independently of the prime-equality (the discarded / unsatisfiable case).
fn eq_result_feeds_quantifier_body(
    instructions: &[Opcode],
    eq_rd: Register,
    eq_pc: usize,
    quant: QuantifierBody,
) -> bool {
    let mut tainted = [false; 256];
    tainted[eq_rd as usize] = true;

    let start = eq_pc.saturating_add(1).max(quant.body_start);
    for op in &instructions[start..quant.next_pc] {
        if opcode_reads_tainted(op, &tainted) {
            if let Some(rd) = op.dest_register() {
                tainted[rd as usize] = true;
            }
            if let Some(r_binding) = op.binding_register() {
                tainted[r_binding as usize] = true;
            }
        }
    }

    tainted[quant.body_reg as usize]
}

fn duplicate_store_on_same_path(instructions: &[Opcode]) -> Option<VarIdx> {
    let mut stored_var_writes: BTreeMap<VarIdx, Vec<usize>> = BTreeMap::new();
    for (pc, op) in instructions.iter().enumerate() {
        let Opcode::StoreVar { var_idx, .. } = *op else {
            continue;
        };
        if stored_var_writes.get(&var_idx).is_some_and(|writes| {
            writes
                .iter()
                .any(|&prev_pc| writes_can_share_path(instructions, prev_pc, pc))
        }) {
            return Some(var_idx);
        }
        stored_var_writes.entry(var_idx).or_default().push(pc);
    }
    None
}

fn writes_can_share_path(instructions: &[Opcode], a_pc: usize, b_pc: usize) -> bool {
    can_execute_after_write(instructions, a_pc, b_pc)
        || can_execute_after_write(instructions, b_pc, a_pc)
}

/// Symbolic provenance of a boolean register, tracked up to negation.
///
/// `base` identifies the syntactic boolean value a register holds; `negated`
/// records polarity relative to that base. Two registers with the same `base`
/// but opposite `negated` are provably complementary (`G` vs `NOT G`). This is
/// computed as a path-insensitive single-definition pre-pass (see
/// [`compute_guard_provenance`]) so it can be consulted during the path
/// analysis WITHOUT enlarging the BFS state: it only ever lets a proven branch
/// fact (`rs = v`) refine a complementary register's `known_bool`, which is the
/// same kind of refinement the analysis already performs for the branched
/// register itself.
type GuardKey = Option<(u32, bool)>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BoolPathState {
    pc: usize,
    seen_first_write: bool,
    known_bools: KnownBools,
}

/// Build a path-insensitive guard-provenance map keyed by register.
///
/// A register is assigned a `GuardKey` only when it is defined EXACTLY ONCE in
/// the whole instruction stream (SSA-like), so the symbolic identity it carries
/// is independent of which path reached a use. `Not` flips polarity around its
/// source's base; `Move` copies it; every other boolean-producing opcode mints
/// a fresh leaf base keyed by its defining pc. Multiply-defined registers get
/// `None` (no provenance) and are treated conservatively. This guarantees that
/// if registers `a` and `b` share a base with opposite polarity, then on every
/// concrete execution `b == NOT a`, so a learned value for one soundly implies
/// the complementary value for the other.
fn compute_guard_provenance(instructions: &[Opcode]) -> [GuardKey; 256] {
    let mut def_count = [0u16; 256];
    for op in instructions {
        if let Some(rd) = op.dest_register() {
            def_count[rd as usize] = def_count[rd as usize].saturating_add(1);
        }
        if let Some(r_binding) = op.binding_register() {
            def_count[r_binding as usize] = def_count[r_binding as usize].saturating_add(1);
        }
    }

    let mut keys: [GuardKey; 256] = [None; 256];
    for (pc, op) in instructions.iter().enumerate() {
        match *op {
            Opcode::Not { rd, rs } => {
                if def_count[rd as usize] == 1 {
                    // Derive from the (already-computed) single-def source if it
                    // has a base, otherwise seed `rs` as a fresh base so the
                    // pair stays linked. Only single-def sources are trusted.
                    keys[rd as usize] = match keys[rs as usize] {
                        Some((base, negated)) => Some((base, !negated)),
                        None if def_count[rs as usize] == 1 => {
                            keys[rs as usize] = Some((pc as u32, false));
                            Some((pc as u32, true))
                        }
                        None => None,
                    };
                }
            }
            Opcode::Move { rd, rs } => {
                if def_count[rd as usize] == 1 {
                    keys[rd as usize] = keys[rs as usize];
                }
            }
            _ => {
                if let Some(rd) = op.dest_register() {
                    if def_count[rd as usize] == 1 {
                        keys[rd as usize] = Some((pc as u32, false));
                    }
                }
            }
        }
    }
    keys
}

fn can_execute_after_write(
    instructions: &[Opcode],
    first_write: usize,
    later_write: usize,
) -> bool {
    if first_write >= instructions.len() || later_write >= instructions.len() {
        return false;
    }

    let provenance = compute_guard_provenance(instructions);
    let mut seen = FxHashSet::default();
    let mut pending = vec![BoolPathState {
        pc: 0,
        seen_first_write: false,
        known_bools: KnownBools::default(),
    }];

    while let Some(mut state) = pending.pop() {
        if state.pc >= instructions.len() {
            continue;
        }
        if state.seen_first_write && state.pc == later_write {
            return true;
        }
        if state.pc == first_write {
            state.seen_first_write = true;
        }
        if !seen.insert(state) {
            continue;
        }

        let Some(successors) = instruction_successors(instructions, &provenance, state) else {
            return true;
        };
        pending.extend(successors);
    }

    false
}

fn instruction_successors(
    instructions: &[Opcode],
    provenance: &[GuardKey; 256],
    mut state: BoolPathState,
) -> Option<Vec<BoolPathState>> {
    let len = instructions.len();
    let pc = state.pc;
    let mut successors = Vec::with_capacity(2);
    match instructions[pc] {
        Opcode::Jump { offset } => {
            successors.push(next_state(state, jump_target(pc, offset, len)?));
        }
        Opcode::JumpTrue { .. } | Opcode::JumpFalse { .. } => {
            push_conditional_successors(&mut successors, provenance, state, instructions[pc], len)?;
        }
        Opcode::ForallBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(true));
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_end, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::ExistsBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::ChooseBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(false));
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_end, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::SetBuilderBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::FuncDefBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, None);
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_end, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                push_fallthrough(&mut successors, state, len);
            }
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                push_fallthrough(&mut successors, state, len);
            }
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, None);
                push_fallthrough(&mut successors, state, len);
            }
            Some(false) | None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state;
                loop_state.known_bools.set(r_binding, None);
                successors.push(next_state(loop_state, jump_target(pc, loop_begin, len)?));
                push_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::LoopNext {
            r_binding,
            loop_begin,
            ..
        } => {
            state.known_bools.set(r_binding, None);
            successors.push(next_state(state, jump_target(pc, loop_begin, len)?));
            push_fallthrough(&mut successors, state, len);
        }
        Opcode::Ret { .. } | Opcode::Halt => {}
        _ => {
            transfer_bool_facts(&mut state.known_bools, instructions[pc]);
            push_fallthrough(&mut successors, state, len);
        }
    }
    Some(successors)
}

fn push_conditional_successors(
    successors: &mut Vec<BoolPathState>,
    provenance: &[GuardKey; 256],
    state: BoolPathState,
    op: Opcode,
    len: usize,
) -> Option<()> {
    let (rs, offset, jump_on) = match op {
        Opcode::JumpTrue { rs, offset } => (rs, offset, true),
        Opcode::JumpFalse { rs, offset } => (rs, offset, false),
        _ => unreachable!("conditional successor helper called for non-branch"),
    };
    let pc = state.pc;
    match state.known_bools.get(rs) {
        Some(value) if value == jump_on => {
            successors.push(next_state_with_bool_fact(
                state,
                provenance,
                jump_target(pc, offset, len)?,
                rs,
                jump_on,
            ));
        }
        Some(_) => {
            push_fallthrough_with_bool_fact(successors, provenance, state, len, rs, !jump_on)
        }
        None => {
            successors.push(next_state_with_bool_fact(
                state,
                provenance,
                jump_target(pc, offset, len)?,
                rs,
                jump_on,
            ));
            push_fallthrough_with_bool_fact(successors, provenance, state, len, rs, !jump_on);
        }
    }
    Some(())
}

/// Record `known[reg] = value`, then propagate the fact to every register that
/// shares `reg`'s symbolic base (per the static `provenance` map): identical
/// polarity registers learn the same value, opposite-polarity registers
/// (`NOT G`) learn the negation. This is the soundness-preserving relaxation:
/// we only ever assert a derived fact when two registers are provably the same
/// boolean up to negation, which is what the single-definition provenance
/// guarantees.
fn apply_bool_fact_with_provenance(
    known: &mut KnownBools,
    provenance: &[GuardKey; 256],
    reg: Register,
    value: bool,
) {
    known.set(reg, Some(value));
    let Some((base, neg)) = provenance[reg as usize] else {
        return;
    };
    for other in 0..=255u8 {
        if other == reg {
            continue;
        }
        if let Some((other_base, other_neg)) = provenance[other as usize] {
            if other_base == base {
                // Same base, same polarity => same value; opposite polarity =>
                // negated value.
                known.set(other, Some(value ^ (other_neg != neg)));
            }
        }
    }
}

fn transfer_bool_facts(known: &mut KnownBools, op: Opcode) {
    match op {
        Opcode::LoadBool { rd, value } => known.set(rd, Some(value)),
        Opcode::Move { rd, rs } => known.set(rd, known.get(rs)),
        Opcode::Not { rd, rs } => known.set(rd, known.get(rs).map(|value| !value)),
        Opcode::And { rd, r1, r2 } => {
            known.set(
                rd,
                match (known.get(r1), known.get(r2)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                },
            );
        }
        Opcode::Or { rd, r1, r2 } => {
            known.set(
                rd,
                match (known.get(r1), known.get(r2)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                },
            );
        }
        Opcode::Implies { rd, r1, r2 } => {
            known.set(
                rd,
                match (known.get(r1), known.get(r2)) {
                    (Some(false), _) | (_, Some(true)) => Some(true),
                    (Some(true), Some(false)) => Some(false),
                    _ => None,
                },
            );
        }
        Opcode::Equiv { rd, r1, r2 } => {
            known.set(rd, known.get(r1).zip(known.get(r2)).map(|(a, b)| a == b));
        }
        Opcode::Eq { rd, r1, r2 } if r1 == r2 => known.set(rd, Some(true)),
        _ => {
            if let Some(rd) = op.dest_register() {
                known.set(rd, None);
            }
            if let Some(r_binding) = op.binding_register() {
                known.set(r_binding, None);
            }
        }
    }
}

fn push_fallthrough(successors: &mut Vec<BoolPathState>, state: BoolPathState, len: usize) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_state(state, next_pc));
    }
}

fn push_fallthrough_with_bool_fact(
    successors: &mut Vec<BoolPathState>,
    provenance: &[GuardKey; 256],
    state: BoolPathState,
    len: usize,
    reg: Register,
    value: bool,
) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_state_with_bool_fact(
            state, provenance, next_pc, reg, value,
        ));
    }
}

fn next_state(mut state: BoolPathState, pc: usize) -> BoolPathState {
    state.pc = pc;
    state
}

fn next_state_with_bool_fact(
    mut state: BoolPathState,
    provenance: &[GuardKey; 256],
    pc: usize,
    reg: Register,
    value: bool,
) -> BoolPathState {
    apply_bool_fact_with_provenance(&mut state.known_bools, provenance, reg, value);
    state.pc = pc;
    state
}

fn jump_target(pc: usize, offset: i32, len: usize) -> Option<usize> {
    let target = (pc as i64).checked_add(i64::from(offset))?;
    let target = usize::try_from(target).ok()?;
    (target < len).then_some(target)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct MustWritePathState {
    pc: usize,
    written_vars: BTreeSet<VarIdx>,
    known_bools: KnownBools,
}

fn validate_residual_load_primes_after_must_write(
    instructions: &[Opcode],
    constants: Option<&ConstantPool>,
) -> Result<(), String> {
    explore_must_write_paths(instructions, constants, None).map(|_| ())
}

fn load_prime_has_must_write_before_pc(
    instructions: &[Opcode],
    constants: Option<&ConstantPool>,
    target_pc: usize,
    var_idx: VarIdx,
) -> Result<bool, String> {
    explore_must_write_paths(instructions, constants, Some((target_pc, var_idx)))
}

fn explore_must_write_paths(
    instructions: &[Opcode],
    constants: Option<&ConstantPool>,
    target: Option<(usize, VarIdx)>,
) -> Result<bool, String> {
    let mut saw_target = false;
    let mut seen = FxHashSet::default();
    let mut pending = vec![MustWritePathState {
        pc: 0,
        written_vars: BTreeSet::new(),
        known_bools: KnownBools::default(),
    }];

    while let Some(state) = pending.pop() {
        if state.pc >= instructions.len() {
            continue;
        }
        let pc = state.pc;
        if let Some((target_pc, var_idx)) = target {
            if pc == target_pc {
                saw_target = true;
                if !state.written_vars.contains(&var_idx) {
                    return Ok(false);
                }
                continue;
            }
        }
        if !seen.insert(state.clone()) {
            continue;
        }
        if target.is_none() {
            if let Opcode::LoadPrime { var_idx, .. } = instructions[pc] {
                if !state.written_vars.contains(&var_idx) {
                    return Err(format!(
                        "residual LoadPrime for primed var {var_idx} at pc {pc} has no definite prior StoreVar/UNCHANGED proof"
                    ));
                }
            }
        }

        let successors = must_write_instruction_successors(instructions, state, constants)?;
        pending.extend(successors);
    }

    Ok(saw_target)
}

fn must_write_instruction_successors(
    instructions: &[Opcode],
    mut state: MustWritePathState,
    constants: Option<&ConstantPool>,
) -> Result<Vec<MustWritePathState>, String> {
    let len = instructions.len();
    let pc = state.pc;
    let mut successors = Vec::with_capacity(2);
    match instructions[pc] {
        Opcode::Jump { offset } => {
            let target = jump_target(pc, offset, len)
                .ok_or_else(|| format!("invalid jump target from pc {pc} with offset {offset}"))?;
            successors.push(next_must_write_state(state, target));
        }
        Opcode::JumpTrue { .. } | Opcode::JumpFalse { .. } => {
            push_must_write_conditional_successors(&mut successors, state, instructions[pc], len)?;
        }
        Opcode::ForallBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(true));
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_end, len).ok_or_else(|| {
                format!("invalid ForallBegin loop_end from pc {pc} with offset {loop_end}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::ExistsBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::ChooseBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, Some(false));
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_end, len).ok_or_else(|| {
                format!("invalid quantifier loop_end from pc {pc} with offset {loop_end}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::SetBuilderBegin {
            rd,
            r_binding,
            loop_end,
            ..
        }
        | Opcode::FuncDefBegin {
            rd,
            r_binding,
            loop_end,
            ..
        } => {
            state.known_bools.set(rd, None);
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_end, len).ok_or_else(|| {
                format!("invalid builder loop_end from pc {pc} with offset {loop_end}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::ForallNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ForallNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ForallNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ExistsNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, Some(true));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            Some(false) => {
                state.known_bools.set(rd, Some(false));
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ExistsNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
            None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ExistsNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::ChooseNext {
            rd,
            r_binding,
            r_body,
            loop_begin,
        } => match state.known_bools.get(r_body) {
            Some(true) => {
                state.known_bools.set(rd, None);
                push_must_write_fallthrough(&mut successors, state, len);
            }
            Some(false) | None => {
                state.known_bools.set(rd, None);
                let mut loop_state = state.clone();
                loop_state.known_bools.set(r_binding, None);
                let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                    format!("invalid ChooseNext loop_begin from pc {pc} with offset {loop_begin}")
                })?;
                successors.push(next_must_write_state(loop_state, target));
                push_must_write_fallthrough(&mut successors, state, len);
            }
        },
        Opcode::LoopNext {
            r_binding,
            loop_begin,
            ..
        } => {
            state.known_bools.set(r_binding, None);
            let target = jump_target(pc, loop_begin, len).ok_or_else(|| {
                format!("invalid LoopNext loop_begin from pc {pc} with offset {loop_begin}")
            })?;
            successors.push(next_must_write_state(state.clone(), target));
            push_must_write_fallthrough(&mut successors, state, len);
        }
        Opcode::Ret { .. } | Opcode::Halt => {}
        op => {
            apply_must_write_effect(&mut state, op, constants, pc)?;
            push_must_write_fallthrough(&mut successors, state, len);
        }
    }
    Ok(successors)
}

fn apply_must_write_effect(
    state: &mut MustWritePathState,
    op: Opcode,
    constants: Option<&ConstantPool>,
    pc: usize,
) -> Result<(), String> {
    match op {
        Opcode::StoreVar { var_idx, .. } => {
            state.written_vars.insert(var_idx);
        }
        Opcode::Unchanged { .. } => {
            for var_idx in unchanged_var_indices(op, constants, pc)? {
                state.written_vars.insert(var_idx);
            }
        }
        _ => {}
    }
    transfer_bool_facts(&mut state.known_bools, op);
    Ok(())
}

fn unchanged_var_indices(
    op: Opcode,
    constants: Option<&ConstantPool>,
    pc: usize,
) -> Result<Vec<VarIdx>, String> {
    let Opcode::Unchanged { start, count, .. } = op else {
        return Ok(Vec::new());
    };
    let Some(constants) = constants else {
        return Ok(Vec::new());
    };

    let mut vars = Vec::with_capacity(count as usize);
    for offset in 0..count as u16 {
        let value = constants.get_value(start + offset);
        let Value::SmallInt(raw_var_idx) = value else {
            return Err(format!(
                "Unchanged metadata at pc {pc} does not decode to SmallInt var indices"
            ));
        };
        let var_idx = VarIdx::try_from(*raw_var_idx).map_err(|_| {
            format!("Unchanged metadata at pc {pc} has out-of-range var index {raw_var_idx}")
        })?;
        vars.push(var_idx);
    }
    Ok(vars)
}

fn push_must_write_conditional_successors(
    successors: &mut Vec<MustWritePathState>,
    state: MustWritePathState,
    op: Opcode,
    len: usize,
) -> Result<(), String> {
    let (rs, offset, jump_on) = match op {
        Opcode::JumpTrue { rs, offset } => (rs, offset, true),
        Opcode::JumpFalse { rs, offset } => (rs, offset, false),
        _ => unreachable!("conditional successor helper called for non-branch"),
    };
    let pc = state.pc;
    match state.known_bools.get(rs) {
        Some(value) if value == jump_on => {
            let target = jump_target(pc, offset, len).ok_or_else(|| {
                format!("invalid conditional jump from pc {pc} with offset {offset}")
            })?;
            successors.push(next_must_write_state_with_bool_fact(
                state, target, rs, jump_on,
            ));
        }
        Some(_) => push_must_write_fallthrough_with_bool_fact(successors, state, len, rs, !jump_on),
        None => {
            let target = jump_target(pc, offset, len).ok_or_else(|| {
                format!("invalid conditional jump from pc {pc} with offset {offset}")
            })?;
            successors.push(next_must_write_state_with_bool_fact(
                state.clone(),
                target,
                rs,
                jump_on,
            ));
            push_must_write_fallthrough_with_bool_fact(successors, state, len, rs, !jump_on);
        }
    }
    Ok(())
}

fn push_must_write_fallthrough(
    successors: &mut Vec<MustWritePathState>,
    state: MustWritePathState,
    len: usize,
) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_must_write_state(state, next_pc));
    }
}

fn push_must_write_fallthrough_with_bool_fact(
    successors: &mut Vec<MustWritePathState>,
    state: MustWritePathState,
    len: usize,
    reg: Register,
    value: bool,
) {
    if state.pc + 1 < len {
        let next_pc = state.pc + 1;
        successors.push(next_must_write_state_with_bool_fact(
            state, next_pc, reg, value,
        ));
    }
}

fn next_must_write_state(mut state: MustWritePathState, pc: usize) -> MustWritePathState {
    state.pc = pc;
    state
}

fn next_must_write_state_with_bool_fact(
    mut state: MustWritePathState,
    pc: usize,
    reg: Register,
    value: bool,
) -> MustWritePathState {
    state.known_bools.set(reg, Some(value));
    state.pc = pc;
    state
}

fn eq_result_interferes_before_rewrite(
    instructions: &[Opcode],
    eq_rd: Register,
    expr_reg: Register,
    prime_pc: usize,
    eq_pc: usize,
) -> bool {
    if eq_rd == expr_reg {
        return true;
    }

    instructions[prime_pc + 1..eq_pc]
        .iter()
        .any(|op| opcode_reads_register(op, eq_rd) || opcode_writes_register(op, eq_rd))
}

fn opcode_reads_register(op: &Opcode, reg: Register) -> bool {
    let mut tainted = [false; 256];
    tainted[reg as usize] = true;
    opcode_reads_tainted(op, &tainted)
}

fn opcode_writes_register(op: &Opcode, reg: Register) -> bool {
    op.dest_register() == Some(reg) || op.binding_register() == Some(reg)
}

fn range_reads_tainted(tainted: &[bool; 256], start: Register, count: u8) -> bool {
    (0..count).any(|offset| tainted[start.saturating_add(offset) as usize])
}

fn opcode_reads_tainted(op: &Opcode, tainted: &[bool; 256]) -> bool {
    match op {
        Opcode::LoadImm { .. }
        | Opcode::LoadBool { .. }
        | Opcode::LoadConst { .. }
        | Opcode::LoadVar { .. }
        | Opcode::LoadPrime { .. }
        | Opcode::Jump { .. }
        | Opcode::SetPrimeMode { .. }
        | Opcode::Nop
        | Opcode::Halt
        | Opcode::Unchanged { .. } => false,
        Opcode::StoreVar { rs, .. }
        | Opcode::Move { rs, .. }
        | Opcode::NegInt { rs, .. }
        | Opcode::Not { rs, .. }
        | Opcode::Powerset { rs, .. }
        | Opcode::BigUnion { rs, .. }
        | Opcode::Domain { rs, .. }
        | Opcode::RecordGet { rs, .. }
        | Opcode::TupleGet { rs, .. }
        | Opcode::Tuple2SelfEq { value: rs, .. }
        | Opcode::Tuple2SelfSubseteq { value: rs, .. }
        | Opcode::Ret { rs }
        | Opcode::JumpTrue { rs, .. }
        | Opcode::JumpFalse { rs, .. } => tainted[*rs as usize],
        Opcode::AddInt { r1, r2, .. }
        | Opcode::SubInt { r1, r2, .. }
        | Opcode::MulInt { r1, r2, .. }
        | Opcode::DivInt { r1, r2, .. }
        | Opcode::IntDiv { r1, r2, .. }
        | Opcode::ModInt { r1, r2, .. }
        | Opcode::PowInt { r1, r2, .. }
        | Opcode::Eq { r1, r2, .. }
        | Opcode::Neq { r1, r2, .. }
        | Opcode::LtInt { r1, r2, .. }
        | Opcode::LeInt { r1, r2, .. }
        | Opcode::GtInt { r1, r2, .. }
        | Opcode::GeInt { r1, r2, .. }
        | Opcode::And { r1, r2, .. }
        | Opcode::Or { r1, r2, .. }
        | Opcode::Implies { r1, r2, .. }
        | Opcode::Equiv { r1, r2, .. }
        | Opcode::SetUnion { r1, r2, .. }
        | Opcode::SetIntersect { r1, r2, .. }
        | Opcode::SetDiff { r1, r2, .. }
        | Opcode::Subseteq { r1, r2, .. }
        | Opcode::StrConcat { r1, r2, .. }
        | Opcode::Concat { r1, r2, .. } => tainted[*r1 as usize] || tainted[*r2 as usize],
        Opcode::Range { lo, hi, .. } => tainted[*lo as usize] || tainted[*hi as usize],
        Opcode::KSubset { base, k, .. } => tainted[*base as usize] || tainted[*k as usize],
        Opcode::SetIn { elem, set, .. } => tainted[*elem as usize] || tainted[*set as usize],
        Opcode::Tuple2SetIn {
            first, second, set, ..
        } => tainted[*first as usize] || tainted[*second as usize] || tainted[*set as usize],
        Opcode::SetEnumSubseteq {
            start, count, set, ..
        } => tainted[*set as usize] || range_reads_tainted(tainted, *start, *count),
        Opcode::RoundStepEq { child, parent, .. } => {
            tainted[*child as usize] || tainted[*parent as usize]
        }
        Opcode::EdgeFilter {
            first, arg, domain, ..
        } => tainted[*first as usize] || tainted[*arg as usize] || tainted[*domain as usize],
        Opcode::FuncApply { func, arg, .. } => tainted[*func as usize] || tainted[*arg as usize],
        Opcode::FuncSet { domain, range, .. } => {
            tainted[*domain as usize] || tainted[*range as usize]
        }
        Opcode::FuncExcept {
            func, path, val, ..
        } => tainted[*func as usize] || tainted[*path as usize] || tainted[*val as usize],
        Opcode::EqFuncExcept {
            lhs,
            func,
            path,
            val,
            ..
        } => {
            tainted[*lhs as usize]
                || tainted[*func as usize]
                || tainted[*path as usize]
                || tainted[*val as usize]
        }
        Opcode::EqRecordNew {
            lhs,
            values_start,
            count,
            ..
        } => tainted[*lhs as usize] || range_reads_tainted(tainted, *values_start, *count),
        Opcode::CondMove { cond, rs, .. } => tainted[*cond as usize] || tainted[*rs as usize],
        Opcode::SetEnum { start, count, .. }
        | Opcode::TupleNew { start, count, .. }
        | Opcode::SeqNew { start, count, .. }
        | Opcode::Times { start, count, .. } => range_reads_tainted(tainted, *start, *count),
        Opcode::RecordNew {
            values_start,
            count,
            ..
        }
        | Opcode::RecordSet {
            values_start,
            count,
            ..
        } => range_reads_tainted(tainted, *values_start, *count),
        Opcode::FuncDef {
            r_domain,
            r_binding,
            ..
        } => tainted[*r_domain as usize] || tainted[*r_binding as usize],
        Opcode::Call {
            args_start, argc, ..
        }
        | Opcode::CallExternal {
            args_start, argc, ..
        }
        | Opcode::CallBuiltin {
            args_start, argc, ..
        } => range_reads_tainted(tainted, *args_start, *argc),
        Opcode::ValueApply {
            func,
            args_start,
            argc,
            ..
        } => tainted[*func as usize] || range_reads_tainted(tainted, *args_start, *argc),
        Opcode::MakeClosure {
            captures_start,
            capture_count,
            ..
        } => range_reads_tainted(tainted, *captures_start, *capture_count),
        Opcode::ForallBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::ExistsBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::ChooseBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::SetFilterBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::SetBuilderBegin {
            r_binding,
            r_domain,
            ..
        }
        | Opcode::FuncDefBegin {
            r_binding,
            r_domain,
            ..
        } => tainted[*r_binding as usize] || tainted[*r_domain as usize],
        Opcode::ForallNext {
            r_binding, r_body, ..
        }
        | Opcode::ExistsNext {
            r_binding, r_body, ..
        }
        | Opcode::ChooseNext {
            r_binding, r_body, ..
        }
        | Opcode::LoopNext {
            r_binding, r_body, ..
        } => tainted[*r_binding as usize] || tainted[*r_body as usize],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tla_value::Rp;

    fn model_value(name: &str) -> Value {
        Value::ModelValue(Rp::from(name))
    }

    fn model_scalar(name: &str) -> ActionScalarValue {
        ActionScalarValue::ModelValue(intern_name(name))
    }

    fn domain_scalar(name: &str) -> ActionDomainElement {
        ActionDomainElement::Scalar(model_scalar(name))
    }

    #[test]
    fn round_step_eq_is_rejected_by_action_transform() {
        let instructions = [Opcode::RoundStepEq {
            rd: 2,
            child: 0,
            parent: 1,
        }];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("VM-only RoundStepEq must not enter action/native transformation");
        };
        assert_eq!(
            reason,
            "RoundStepEq is VM-only and cannot enter action/native transformation"
        );
    }

    #[test]
    fn test_action_binding_specs_preserve_model_value_set_domain() {
        let bindings = vec![(
            Arc::from("procs"),
            Value::set([model_value("p1"), model_value("p2")]),
        )];

        let specs = action_binding_specs_from_values(&bindings)
            .expect("finite model-value set binding should preserve domain shape");

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name.as_ref(), "procs");
        let ActionBindingValue::FiniteDomain(domain) = &specs[0].value else {
            panic!("model-value set binding must not collapse into scalar metadata");
        };
        assert_eq!(domain.source, ActionDomainSource::BindingValue);
        assert_eq!(domain.elements.len(), 2);
        assert!(domain.elements.contains(&domain_scalar("p1")));
        assert!(domain.elements.contains(&domain_scalar("p2")));
    }

    #[test]
    fn test_action_binding_specs_fail_closed_for_unbounded_binding_domain() {
        let bindings = vec![(Arc::from("s"), Value::StringSet)];

        let err = action_binding_specs_from_values(&bindings)
            .expect_err("unbounded bindings must not get BindingSpec metadata");

        assert!(matches!(
            err,
            ActionDomainShapeError::UnsupportedBinding {
                ref name,
                value_kind: "StringSet"
            } if name == "s"
        ));
    }

    #[test]
    fn test_residual_exists_preserves_constant_model_value_domain() {
        let mut constants = ConstantPool::new();
        let procs = constants.add_value(Value::set([
            model_value("p1"),
            model_value("p2"),
            model_value("p3"),
        ]));
        let instructions = vec![
            Opcode::LoadConst { rd: 0, idx: procs },
            Opcode::ExistsBegin {
                rd: 1,
                r_binding: 2,
                r_domain: 0,
                loop_end: 2,
            },
            Opcode::Ret { rs: 1 },
        ];

        let domains = collect_residual_quantifier_domains(&instructions, Some(&constants))
            .expect("constant process set should be a finite residual EXISTS domain");

        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].begin_pc, 1);
        assert_eq!(domains[0].kind, ResidualQuantifierKind::Exists);
        assert_eq!(
            domains[0].domain.source,
            ActionDomainSource::ConstantPool { idx: procs }
        );
        assert_eq!(domains[0].domain.elements.len(), 3);
        assert!(domains[0].domain.elements.contains(&domain_scalar("p1")));
        assert!(domains[0].domain.elements.contains(&domain_scalar("p2")));
        assert!(domains[0].domain.elements.contains(&domain_scalar("p3")));
    }

    #[test]
    fn test_residual_forall_preserves_setdiff_process_domain_source() {
        let mut constants = ConstantPool::new();
        let procs = constants.add_value(Value::set([
            model_value("p1"),
            model_value("p2"),
            model_value("p3"),
        ]));
        let p1 = constants.add_value(model_value("p1"));
        let instructions = vec![
            Opcode::LoadConst { rd: 0, idx: procs },
            Opcode::LoadConst { rd: 1, idx: p1 },
            Opcode::SetEnum {
                rd: 2,
                start: 1,
                count: 1,
            },
            Opcode::SetDiff {
                rd: 3,
                r1: 0,
                r2: 2,
            },
            Opcode::ForallBegin {
                rd: 4,
                r_binding: 5,
                r_domain: 3,
                loop_end: 2,
            },
            Opcode::Ret { rs: 4 },
        ];

        let domains = collect_residual_quantifier_domains(&instructions, Some(&constants))
            .expect("finite process set difference should preserve FORALL domain shape");

        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].kind, ResidualQuantifierKind::Forall);
        assert_eq!(
            domains[0].domain.source,
            ActionDomainSource::SetDiff { pc: 3 }
        );
        assert_eq!(
            domains[0].domain.elements,
            vec![domain_scalar("p2"), domain_scalar("p3")]
        );
    }

    #[test]
    fn test_residual_exists_preserves_powerset_model_value_domain() {
        let mut constants = ConstantPool::new();
        let resources = constants.add_value(Value::set([model_value("r1"), model_value("r2")]));
        let instructions = vec![
            Opcode::LoadConst {
                rd: 0,
                idx: resources,
            },
            Opcode::Powerset { rd: 1, rs: 0 },
            Opcode::ExistsBegin {
                rd: 2,
                r_binding: 3,
                r_domain: 1,
                loop_end: 2,
            },
            Opcode::Ret { rs: 2 },
        ];

        let domains = collect_residual_quantifier_domains(&instructions, Some(&constants))
            .expect("SUBSET Resources should preserve exact finite set-valued domain elements");

        assert_eq!(domains.len(), 1);
        assert_eq!(
            domains[0].domain.source,
            ActionDomainSource::Powerset { pc: 1 }
        );
        assert_eq!(
            domains[0].domain.elements,
            vec![
                ActionDomainElement::Set(vec![]),
                ActionDomainElement::Set(vec![model_scalar("r1")]),
                ActionDomainElement::Set(vec![model_scalar("r2")]),
                ActionDomainElement::Set(vec![model_scalar("r1"), model_scalar("r2")]),
            ]
        );
    }

    #[test]
    fn test_residual_exists_preserves_ksubset_model_value_domain() {
        let mut constants = ConstantPool::new();
        let resources = constants.add_value(Value::set([
            model_value("r1"),
            model_value("r2"),
            model_value("r3"),
        ]));
        let instructions = vec![
            Opcode::LoadConst {
                rd: 0,
                idx: resources,
            },
            Opcode::LoadImm { rd: 1, value: 2 },
            Opcode::KSubset {
                rd: 2,
                base: 0,
                k: 1,
            },
            Opcode::ExistsBegin {
                rd: 3,
                r_binding: 4,
                r_domain: 2,
                loop_end: 2,
            },
            Opcode::Ret { rs: 3 },
        ];

        let domains = collect_residual_quantifier_domains(&instructions, Some(&constants)).expect(
            "KSubset(Resources, 2) should preserve exact finite set-valued domain elements",
        );

        assert_eq!(domains.len(), 1);
        assert_eq!(
            domains[0].domain.source,
            ActionDomainSource::KSubset { pc: 2 }
        );
        assert_eq!(
            domains[0].domain.elements,
            vec![
                ActionDomainElement::Set(vec![model_scalar("r1"), model_scalar("r2")]),
                ActionDomainElement::Set(vec![model_scalar("r1"), model_scalar("r3")]),
                ActionDomainElement::Set(vec![model_scalar("r2"), model_scalar("r3")]),
            ]
        );
    }

    #[test]
    fn test_residual_exists_accepts_explicit_compact_domain_hint() {
        let mut constants = ConstantPool::new();
        let p1 = constants.add_value(model_value("p1"));
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::LoadConst { rd: 1, idx: p1 },
            Opcode::FuncApply {
                rd: 2,
                func: 0,
                arg: 1,
            },
            Opcode::ExistsBegin {
                rd: 3,
                r_binding: 4,
                r_domain: 2,
                loop_end: 2,
            },
            Opcode::Ret { rs: 3 },
        ];
        let hint_domain = ActionFiniteDomain {
            elements: vec![domain_scalar("p1"), domain_scalar("p2")],
            source: ActionDomainSource::CompactProof {
                description: "temp[self] SetBitmask(Proc)".to_string(),
            },
        };
        let hints = [ActionDomainHint {
            before_pc: 3,
            domain_reg: 2,
            domain: hint_domain.clone(),
        }];

        let err = collect_residual_quantifier_domains(&instructions, Some(&constants))
            .expect_err("dynamic compact domain without source proof must fail closed");
        assert!(matches!(
            err,
            ActionDomainShapeError::UnknownQuantifierDomain {
                begin_pc: 3,
                kind: ResidualQuantifierKind::Exists,
                r_domain: 2,
            }
        ));

        let domains =
            collect_residual_quantifier_domains_with_hints(&instructions, Some(&constants), &hints)
                .expect("explicit compact proof should preserve residual EXISTS domain shape");
        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].domain, hint_domain);
    }

    #[test]
    fn test_simple_assignment_rewrite() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::AddInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::LoadPrime { rd: 3, var_idx: 0 },
            Opcode::Eq {
                rd: 4,
                r1: 3,
                r2: 2,
            },
            Opcode::Ret { rs: 4 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("should rewrite simple assignment");
        };

        assert_eq!(transformed[3], Opcode::LoadBool { rd: 4, value: true });
        assert_eq!(transformed[4], Opcode::StoreVar { var_idx: 0, rs: 2 });
    }

    #[test]
    fn test_reversed_eq_operands() {
        let instructions = vec![
            Opcode::LoadImm { rd: 0, value: 42 },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("should rewrite reversed assignment");
        };

        assert_eq!(transformed[1], Opcode::LoadBool { rd: 2, value: true });
        assert_eq!(transformed[2], Opcode::StoreVar { var_idx: 0, rs: 0 });
    }

    #[test]
    fn test_no_prime_returns_no_rewrite() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 5 },
            Opcode::LtInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ];

        assert_eq!(
            transform_action_to_next_state(&instructions),
            ActionTransformOutcome::NoRewrite
        );
    }

    #[test]
    fn test_unchanged_only_action_is_executable_next_state() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Unchanged {
                rd: 3,
                start: 0,
                count: 1,
            },
            Opcode::And {
                rd: 4,
                r1: 2,
                r2: 3,
            },
            Opcode::Ret { rs: 4 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("UNCHANGED-only action should stay executable");
        };

        assert_eq!(
            transformed, instructions,
            "UNCHANGED-only actions rely on caller-seeded state_out"
        );
    }

    #[test]
    fn test_unchanged_preserved() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::AddInt {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::LoadPrime { rd: 3, var_idx: 0 },
            Opcode::Eq {
                rd: 4,
                r1: 3,
                r2: 2,
            },
            Opcode::Unchanged {
                rd: 5,
                start: 0,
                count: 1,
            },
            Opcode::Ret { rs: 5 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("should rewrite assignment with unchanged");
        };

        assert_eq!(transformed[3], Opcode::LoadBool { rd: 4, value: true });
        assert_eq!(transformed[4], Opcode::StoreVar { var_idx: 0, rs: 2 });
        assert!(matches!(
            transformed[5],
            Opcode::Unchanged {
                rd: 5,
                start: 0,
                count: 1
            }
        ));
    }

    #[test]
    fn test_prime_first_self_copy_rewrite() {
        let instructions = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadVar { rd: 1, var_idx: 0 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::Ret { rs: 2 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("should rewrite prime-first self-copy");
        };

        assert_eq!(transformed[0], Opcode::LoadBool { rd: 2, value: true });
        assert_eq!(transformed[2], Opcode::StoreVar { var_idx: 0, rs: 1 });
    }

    #[test]
    fn test_prime_first_rhs_computation_rewrite() {
        let instructions = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadVar { rd: 1, var_idx: 0 },
            Opcode::LoadImm { rd: 2, value: 1 },
            Opcode::AddInt {
                rd: 3,
                r1: 1,
                r2: 2,
            },
            Opcode::Eq {
                rd: 4,
                r1: 0,
                r2: 3,
            },
            Opcode::Ret { rs: 4 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("should rewrite prime-first rhs computation");
        };

        assert_eq!(transformed[0], Opcode::LoadBool { rd: 4, value: true });
        assert_eq!(transformed[4], Opcode::StoreVar { var_idx: 0, rs: 3 });
    }

    #[test]
    fn test_unproven_residual_prime_rhs_is_rejected() {
        let instructions = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 1 },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Eq {
                rd: 2,
                r1: 1,
                r2: 0,
            },
            Opcode::Ret { rs: 2 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("cross-prime assignment must be rejected");
        };
        assert!(reason.contains("residual LoadPrime for primed var 1 at pc 0"));
        assert!(reason.contains("no definite prior StoreVar/UNCHANGED proof"));
    }

    #[test]
    fn test_residual_prime_after_store_rewrite_is_allowed() {
        let instructions = vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Eq {
                rd: 2,
                r1: 1,
                r2: 0,
            },
            Opcode::LoadPrime { rd: 3, var_idx: 0 },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::AddInt {
                rd: 5,
                r1: 3,
                r2: 4,
            },
            Opcode::LoadPrime { rd: 6, var_idx: 1 },
            Opcode::Eq {
                rd: 7,
                r1: 6,
                r2: 5,
            },
            Opcode::Ret { rs: 7 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("residual read after StoreVar proof should be rewritten");
        };

        assert_eq!(transformed[1], Opcode::LoadBool { rd: 2, value: true });
        assert_eq!(transformed[2], Opcode::StoreVar { var_idx: 0, rs: 0 });
        assert_eq!(transformed[3], Opcode::LoadPrime { rd: 3, var_idx: 0 });
        assert_eq!(transformed[6], Opcode::LoadBool { rd: 7, value: true });
        assert_eq!(transformed[7], Opcode::StoreVar { var_idx: 1, rs: 5 });
    }

    #[test]
    fn test_branch_refinement_proves_residual_prime_after_guarded_store() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 2 },
            Opcode::JumpFalse { rs: 0, offset: 6 },
            Opcode::LoadImm { rd: 1, value: 10 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 1,
            },
            Opcode::Move { rd: 4, rs: 3 },
            Opcode::Jump { offset: 2 },
            Opcode::Move { rd: 4, rs: 0 },
            Opcode::JumpFalse { rs: 4, offset: 4 },
            Opcode::LoadPrime { rd: 5, var_idx: 0 },
            Opcode::LoadPrime { rd: 6, var_idx: 1 },
            Opcode::Eq {
                rd: 7,
                r1: 6,
                r2: 5,
            },
            Opcode::Ret { rs: 4 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("branch-refined must-write proof should admit residual prime read");
        };

        assert_eq!(transformed[3], Opcode::LoadBool { rd: 3, value: true });
        assert_eq!(transformed[4], Opcode::StoreVar { var_idx: 0, rs: 1 });
        assert_eq!(transformed[9], Opcode::LoadPrime { rd: 5, var_idx: 0 });
        assert_eq!(transformed[10], Opcode::LoadBool { rd: 7, value: true });
        assert_eq!(transformed[11], Opcode::StoreVar { var_idx: 1, rs: 5 });
    }

    #[test]
    fn test_nested_short_circuit_store_dominates_later_prime_read() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 13 },
            Opcode::LoadVar { rd: 1, var_idx: 1 },
            Opcode::Move { rd: 2, rs: 1 },
            Opcode::JumpFalse { rs: 2, offset: 10 },
            Opcode::LoadImm { rd: 3, value: 10 },
            Opcode::LoadPrime { rd: 4, var_idx: 2 },
            Opcode::Eq {
                rd: 5,
                r1: 4,
                r2: 3,
            },
            Opcode::Move { rd: 6, rs: 5 },
            Opcode::JumpFalse { rs: 6, offset: 5 },
            Opcode::LoadPrime { rd: 7, var_idx: 2 },
            Opcode::LoadPrime { rd: 8, var_idx: 3 },
            Opcode::Eq {
                rd: 9,
                r1: 8,
                r2: 7,
            },
            Opcode::Ret { rs: 9 },
            Opcode::Ret { rs: 0 },
        ];

        let outcome = transform_action_to_next_state(&instructions);
        let ActionTransformOutcome::Transformed(transformed) = outcome else {
            panic!("nested short-circuit store should dominate later prime read: {outcome:?}");
        };

        assert_eq!(transformed[6], Opcode::LoadBool { rd: 5, value: true });
        assert_eq!(transformed[7], Opcode::StoreVar { var_idx: 2, rs: 3 });
        assert_eq!(transformed[10], Opcode::LoadPrime { rd: 7, var_idx: 2 });
        assert_eq!(transformed[11], Opcode::LoadBool { rd: 9, value: true });
        assert_eq!(transformed[12], Opcode::StoreVar { var_idx: 3, rs: 7 });
    }

    #[test]
    fn test_residual_prime_after_unchanged_proof_is_allowed() {
        let mut constants = ConstantPool::new();
        let start = constants.add_value(Value::SmallInt(1));
        let instructions = vec![
            Opcode::Unchanged {
                rd: 0,
                start,
                count: 1,
            },
            Opcode::LoadPrime { rd: 1, var_idx: 1 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 1,
            },
            Opcode::Ret { rs: 3 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state_with_constants(&instructions, &constants)
        else {
            panic!("residual read after UNCHANGED proof should be rewritten");
        };

        assert_eq!(transformed[1], Opcode::LoadPrime { rd: 1, var_idx: 1 });
        assert_eq!(transformed[2], Opcode::LoadBool { rd: 3, value: true });
        assert_eq!(transformed[3], Opcode::StoreVar { var_idx: 0, rs: 1 });
    }

    #[test]
    fn test_primed_membership_element_operand_stays_strictly_rejected() {
        let instructions = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 2, value: 10 },
            Opcode::LoadImm { rd: 3, value: 20 },
            Opcode::SetEnum {
                rd: 1,
                start: 2,
                count: 2,
            },
            Opcode::SetIn {
                rd: 4,
                elem: 0,
                set: 1,
            },
            Opcode::Ret { rs: 4 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("primed SetIn element must remain a strict residual LoadPrime rejection");
        };
        assert!(reason.contains("residual LoadPrime for primed var 0 at pc 0"));
    }

    #[test]
    fn test_primed_membership_in_negated_context_stays_strictly_rejected() {
        let instructions = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 2, value: 10 },
            Opcode::SetEnum {
                rd: 1,
                start: 2,
                count: 1,
            },
            Opcode::SetIn {
                rd: 4,
                elem: 0,
                set: 1,
            },
            Opcode::Not { rd: 5, rs: 4 },
            Opcode::Ret { rs: 5 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("primed SetIn under NOT must remain a strict residual LoadPrime rejection");
        };
        assert!(reason.contains("residual LoadPrime for primed var 0 at pc 0"));
    }

    #[test]
    fn test_primed_membership_set_operand_stays_strictly_rejected() {
        let instructions = vec![
            Opcode::LoadImm { rd: 0, value: 10 },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::SetIn {
                rd: 2,
                elem: 0,
                set: 1,
            },
            Opcode::Ret { rs: 2 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("primed SetIn domain must remain a strict residual LoadPrime rejection");
        };
        assert!(reason.contains("residual LoadPrime for primed var 0 at pc 1"));
    }

    #[test]
    fn test_duplicate_store_without_must_write_on_all_paths_is_rejected() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::JumpFalse { rs: 0, offset: 4 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 1,
            },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::LoadPrime { rd: 5, var_idx: 0 },
            Opcode::Eq {
                rd: 6,
                r1: 5,
                r2: 4,
            },
            Opcode::Ret { rs: 6 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("same-path duplicate stores must be rejected");
        };
        assert!(reason.contains("duplicate writes"));
    }

    #[test]
    fn test_set_prime_mode_remnant_is_rejected() {
        let instructions = vec![
            Opcode::LoadImm { rd: 0, value: 1 },
            Opcode::SetPrimeMode { enable: true },
            Opcode::LoadPrime { rd: 1, var_idx: 0 },
            Opcode::Eq {
                rd: 2,
                r1: 1,
                r2: 0,
            },
            Opcode::Ret { rs: 2 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("SetPrimeMode remnants must be rejected by the transform");
        };
        assert!(reason.contains("SetPrimeMode remains after action rewrite"));
    }

    #[test]
    fn test_branch_exclusive_duplicate_store_is_rewritten() {
        let instructions = vec![
            Opcode::LoadBool { rd: 0, value: true },
            Opcode::JumpFalse { rs: 0, offset: 5 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 1,
            },
            Opcode::Jump { offset: 4 },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::LoadPrime { rd: 5, var_idx: 0 },
            Opcode::Eq {
                rd: 6,
                r1: 5,
                r2: 4,
            },
            Opcode::Ret { rs: 0 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("branch-exclusive stores should be rewritten");
        };

        assert_eq!(transformed[3], Opcode::LoadBool { rd: 3, value: true });
        assert_eq!(transformed[4], Opcode::StoreVar { var_idx: 0, rs: 1 });
        assert_eq!(transformed[7], Opcode::LoadBool { rd: 6, value: true });
        assert_eq!(transformed[8], Opcode::StoreVar { var_idx: 0, rs: 4 });
    }

    #[test]
    fn test_exists_loop_duplicate_store_with_true_body_is_rewritten() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::ExistsBegin {
                rd: 10,
                r_binding: 11,
                r_domain: 12,
                loop_end: 12,
            },
            Opcode::JumpFalse { rs: 0, offset: 6 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 1,
            },
            Opcode::Move { rd: 9, rs: 3 },
            Opcode::Jump { offset: 5 },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::LoadPrime { rd: 5, var_idx: 0 },
            Opcode::Eq {
                rd: 6,
                r1: 5,
                r2: 4,
            },
            Opcode::Move { rd: 9, rs: 6 },
            Opcode::ExistsNext {
                rd: 10,
                r_binding: 11,
                r_body: 9,
                loop_begin: -10,
            },
            Opcode::Ret { rs: 10 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("duplicate stores behind a true EXISTS body should be rewritten");
        };

        assert_eq!(transformed[4], Opcode::LoadBool { rd: 3, value: true });
        assert_eq!(transformed[5], Opcode::StoreVar { var_idx: 0, rs: 1 });
        assert_eq!(transformed[9], Opcode::LoadBool { rd: 6, value: true });
        assert_eq!(transformed[10], Opcode::StoreVar { var_idx: 0, rs: 4 });
    }

    #[test]
    fn test_exists_loop_duplicate_store_with_false_body_is_rejected() {
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 0 },
            Opcode::ExistsBegin {
                rd: 10,
                r_binding: 11,
                r_domain: 12,
                loop_end: 12,
            },
            Opcode::JumpFalse { rs: 0, offset: 6 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 1,
            },
            Opcode::LoadBool {
                rd: 9,
                value: false,
            },
            Opcode::Jump { offset: 5 },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::LoadPrime { rd: 5, var_idx: 0 },
            Opcode::Eq {
                rd: 6,
                r1: 5,
                r2: 4,
            },
            Opcode::LoadBool {
                rd: 9,
                value: false,
            },
            Opcode::ExistsNext {
                rd: 10,
                r_binding: 11,
                r_body: 9,
                loop_begin: -10,
            },
            Opcode::Ret { rs: 10 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("duplicate stores behind a false EXISTS body must be rejected");
        };
        assert!(reason.contains("duplicate writes"));
    }

    #[test]
    fn test_exists_body_conjoined_prime_eq_witness_is_rewritten() {
        // Realistic witness shape: `\E x \in S : (v' = x /\ P(x))`. The prime
        // equality's Eq result is combined into the body truth value through an
        // `And` rather than flowing in via a bare `Move`. The Eq result still
        // reaches the quantifier body register, so it IS the binder witness and
        // the store rewrite must be admitted -- the soundness tightening must
        // not over-reject this common pattern.
        let instructions = vec![
            Opcode::LoadVar { rd: 12, var_idx: 1 }, // domain S
            Opcode::ExistsBegin {
                rd: 10,
                r_binding: 11,
                r_domain: 12,
                loop_end: 6,
            },
            Opcode::LoadPrime { rd: 2, var_idx: 0 },
            Opcode::Eq {
                rd: 3,
                r1: 2,
                r2: 11,
            }, // v' = x
            Opcode::LoadVar { rd: 4, var_idx: 2 },
            Opcode::Eq {
                rd: 5,
                r1: 11,
                r2: 4,
            }, // P(x): x = w
            Opcode::And {
                rd: 9,
                r1: 3,
                r2: 5,
            }, // body = (v' = x) /\ P(x)
            Opcode::ExistsNext {
                rd: 10,
                r_binding: 11,
                r_body: 9,
                loop_begin: -6,
            },
            Opcode::Ret { rs: 10 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("prime-eq conjoined into the EXISTS body must be rewritten");
        };
        assert_eq!(transformed[2], Opcode::LoadBool { rd: 3, value: true });
        assert_eq!(transformed[3], Opcode::StoreVar { var_idx: 0, rs: 11 });
    }

    #[test]
    fn test_complementary_guards_in_separate_registers_are_rewritten() {
        // Models `(c /\ x' = 1) \/ (~c /\ x' = 2)` where the two guards live in
        // DIFFERENT registers: `c` in r0 and `~c` in r1 (computed via `Not`).
        // The first guarded block falls through into the second block's guard
        // (no skip-jump), so per-register analysis -- which sees r0 and r1 as
        // unrelated -- believes the second store can fire on the same path that
        // already took the first store. Symbolic provenance proves r1 == NOT r0,
        // so on the path that stored under `c`, r1 (=~c) is known false and the
        // second guard's enabled branch is pruned: the two stores are NOT
        // co-reachable and the action is rewritten to native bytecode.
        //
        //  0: LoadVar  r0 = c
        //  1: Not      r1 = ~r0
        //  2: JumpFalse r0 -> 6     (skip x'=1 when !c)
        //  3..5: x' = 1             (Store_a, falls through to 6)
        //  6: JumpFalse r1 -> 10    (skip x'=2 when !(~c))
        //  7..9: x' = 2             (Store_b)
        // 10: Ret
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 1 },  // c
            Opcode::Not { rd: 1, rs: 0 },           // ~c (complement of r0)
            Opcode::JumpFalse { rs: 0, offset: 4 }, // if !c -> pc 6
            Opcode::LoadImm { rd: 2, value: 1 },
            Opcode::LoadPrime { rd: 3, var_idx: 0 },
            Opcode::Eq {
                rd: 4,
                r1: 3,
                r2: 2,
            }, // x' = 1  (falls through to pc 6)
            Opcode::JumpFalse { rs: 1, offset: 4 }, // if !(~c) -> pc 10
            Opcode::LoadImm { rd: 5, value: 2 },
            Opcode::LoadPrime { rd: 6, var_idx: 0 },
            Opcode::Eq {
                rd: 7,
                r1: 6,
                r2: 5,
            }, // x' = 2
            Opcode::Ret { rs: 0 },
        ];

        let ActionTransformOutcome::Transformed(transformed) =
            transform_action_to_next_state(&instructions)
        else {
            panic!("complementary guards in separate registers should be rewritten");
        };
        assert_eq!(transformed[5], Opcode::StoreVar { var_idx: 0, rs: 2 });
        assert_eq!(transformed[9], Opcode::StoreVar { var_idx: 0, rs: 5 });
    }

    #[test]
    fn test_unrelated_guards_in_separate_registers_stay_conservative() {
        // Same fall-through shape as the complementary case, but the second
        // guard (r1) is an INDEPENDENT predicate (`d`), not `Not r0`. The two
        // stores genuinely can both fire when `c` and `d` are both true, so the
        // conflict flag must be retained and the action must fall back.
        let instructions = vec![
            Opcode::LoadVar { rd: 0, var_idx: 1 }, // c
            Opcode::LoadVar { rd: 1, var_idx: 2 }, // d (unrelated to c)
            Opcode::JumpFalse { rs: 0, offset: 4 },
            Opcode::LoadImm { rd: 2, value: 1 },
            Opcode::LoadPrime { rd: 3, var_idx: 0 },
            Opcode::Eq {
                rd: 4,
                r1: 3,
                r2: 2,
            },
            Opcode::JumpFalse { rs: 1, offset: 4 },
            Opcode::LoadImm { rd: 5, value: 2 },
            Opcode::LoadPrime { rd: 6, var_idx: 0 },
            Opcode::Eq {
                rd: 7,
                r1: 6,
                r2: 5,
            },
            Opcode::Ret { rs: 0 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("genuinely co-satisfiable stores in separate registers must fall back");
        };
        assert!(reason.contains("duplicate writes"));
    }

    #[test]
    fn test_eq_result_register_live_before_rewrite_is_rejected() {
        let instructions = vec![
            Opcode::LoadImm { rd: 1, value: 42 },
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            // r1 is still live between LoadPrime and Eq, so moving the
            // Eq result write to pc=1 would clobber it before this Move.
            Opcode::Move { rd: 2, rs: 1 },
            Opcode::Eq {
                rd: 1,
                r1: 0,
                r2: 2,
            },
            Opcode::Ret { rs: 1 },
        ];

        let ActionTransformOutcome::Unsafe(reason) = transform_action_to_next_state(&instructions)
        else {
            panic!("live Eq destination must be rejected");
        };
        assert!(reason.contains("conflicts with live RHS"));
    }
}
