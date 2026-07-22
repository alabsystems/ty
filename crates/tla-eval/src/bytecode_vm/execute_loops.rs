// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Loop opcode handlers extracted from execute.rs (#3594).
//!
//! Owns the iterator-loop state machine: quantifier loops (Forall/Exists),
//! set builders, set filters, function-definition loops, and CHOOSE.

use tla_value::Rp;
use tla_value::error::EvalError;
use tla_value::{FuncValue, SortedSet, Value};

use super::execute::VmError;
use super::execute_helpers::as_bool;

// Concrete set values already own their normalized elements behind an Arc.
// Sharing that storage avoids cloning every domain element into a fresh Vec at
// each loop entry. The kill switch keeps the previous owned-Vec path available
// for differential testing.
feature_flag!(no_vm_shared_loop_domains, "TY_NO_VM_SHARED_LOOP_DOMAINS");

// Filtering a concrete set preserves its canonical sorted, duplicate-free
// order. Recording that fact on the result avoids constructing an
// `Unnormalized` set that must later copy and sort the same elements while
// retaining both arrays. The kill switch restores the previous lazy result
// constructor for differential testing.
feature_flag!(no_vm_sorted_filter_result, "TY_NO_VM_SORTED_FILTER_RESULT");

/// Action returned by loop helpers to tell the dispatch loop what to do next.
pub(super) enum LoopAction {
    /// Fall through to `pc += 1`.
    Continue,
    /// Jump to the given absolute `pc` (caller should `continue` without incrementing).
    Jump(usize),
}

/// Domain elements retained by a bytecode loop.
///
/// Concrete `Value::Set` domains share their existing set allocation. Other
/// enumerable set values retain the previous fully owned representation because
/// their iterators may synthesize values and borrow the source domain.
pub(super) enum LoopElements {
    Shared(Rp<SortedSet>),
    Owned(Vec<Value>),
}

impl LoopElements {
    #[inline]
    fn from_domain(domain: &Value) -> Option<Self> {
        Self::from_domain_with_sharing(domain, !no_vm_shared_loop_domains())
    }

    fn from_domain_with_sharing(domain: &Value, share_concrete_sets: bool) -> Option<Self> {
        if share_concrete_sets {
            if let Value::Set(set) = domain {
                return Some(Self::Shared(Rp::clone(set)));
            }
        }
        Some(Self::Owned(domain.iter_set()?.collect()))
    }

    /// Build a quantifier domain with the ordering required by the active VM
    /// mode. Ordinary expression execution retains its established iteration
    /// path. Action execution uses TLC-normalized order so short-circuiting and
    /// error selection match canonical successor enumeration.
    fn from_quantifier_domain(
        domain: &Value,
        action_execution: bool,
        expected: &'static str,
    ) -> Result<Self, VmError> {
        if !action_execution {
            return Self::from_domain(domain).ok_or_else(|| VmError::TypeError {
                expected,
                actual: format!("{domain:?}"),
            });
        }
        if !domain.is_set() {
            return Err(VmError::TypeError {
                expected,
                actual: format!("{domain:?}"),
            });
        }
        match domain.iter_set_tlc_normalized() {
            Ok(iter) => Ok(Self::Owned(iter.collect())),
            Err(EvalError::SetTooLarge { .. }) => Err(VmError::Unsupported(
                "action quantifier domain requires context-assisted tree iteration".to_string(),
            )),
            Err(err) => Err(VmError::Eval(err)),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Shared(set) => set.as_slice().len(),
            Self::Owned(elements) => elements.len(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        match self {
            Self::Shared(set) => set.is_empty(),
            Self::Owned(elements) => elements.is_empty(),
        }
    }

    #[inline]
    fn element(&self, index: usize) -> &Value {
        match self {
            Self::Shared(set) => &set.as_slice()[index],
            Self::Owned(elements) => &elements[index],
        }
    }

    /// Build the result of retaining a subsequence of this domain.
    ///
    /// A shared concrete-set domain is traversed through `SortedSet::as_slice`,
    /// so any retained subsequence remains strictly sorted and duplicate-free.
    /// Other enumerable domains keep the established fail-closed lazy
    /// constructor because their iterator ordering is not part of this proof.
    #[inline]
    fn filtered_set(&self, result: Vec<Value>) -> SortedSet {
        if !no_vm_sorted_filter_result() && matches!(self, Self::Shared(_)) {
            debug_assert!(result.windows(2).all(|pair| pair[0] < pair[1]));
            SortedSet::from_sorted_vec(result)
        } else {
            SortedSet::from_iter(result)
        }
    }
}

/// Iterator state for quantifier and loop opcodes.
pub(super) enum LoopState {
    Quantifier {
        elements: LoopElements,
        index: usize,
    },
    SetFilter {
        elements: LoopElements,
        index: usize,
        collected: Vec<Value>,
        rd: u8,
    },
    SetBuilder {
        elements: LoopElements,
        index: usize,
        collected: Vec<Value>,
        rd: u8,
    },
    FuncDef {
        elements: LoopElements,
        index: usize,
        entries: Vec<(Value, Value)>,
        rd: u8,
    },
    Choose {
        elements: Vec<Value>,
        index: usize,
        #[allow(dead_code)]
        rd: u8,
    },
}

pub(super) fn forall_begin(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
    loop_end: i32,
    action_execution: bool,
) -> Result<LoopAction, VmError> {
    let domain = &regs[r_domain as usize];
    let elements = LoopElements::from_quantifier_domain(
        domain,
        action_execution,
        "enumerable set for FORALL domain",
    )?;
    if elements.is_empty() {
        regs[rd as usize] = Value::Bool(true);
        return Ok(LoopAction::Jump(((pc as i64) + (loop_end as i64)) as usize));
    }
    regs[r_binding as usize] = elements.element(0).clone();
    regs[rd as usize] = Value::Bool(true);
    iter_stack.push(LoopState::Quantifier { elements, index: 0 });
    Ok(LoopAction::Continue)
}

pub(super) fn forall_next(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_body: u8,
    loop_begin: i32,
) -> Result<LoopAction, VmError> {
    let body_val = as_bool(&regs[r_body as usize])?;
    if !body_val {
        regs[rd as usize] = Value::Bool(false);
        iter_stack.pop();
    } else if let Some(LoopState::Quantifier { elements, index }) = iter_stack.last_mut() {
        *index += 1;
        if *index < elements.len() {
            regs[r_binding as usize] = elements.element(*index).clone();
            return Ok(LoopAction::Jump(
                ((pc as i64) + (loop_begin as i64)) as usize,
            ));
        }
        regs[rd as usize] = Value::Bool(true);
        iter_stack.pop();
    } else {
        return Err(VmError::Unsupported(
            "ForallNext without matching ForallBegin".to_string(),
        ));
    }
    Ok(LoopAction::Continue)
}

pub(super) fn exists_begin(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
    loop_end: i32,
    action_execution: bool,
) -> Result<LoopAction, VmError> {
    let domain = &regs[r_domain as usize];
    let elements = LoopElements::from_quantifier_domain(
        domain,
        action_execution,
        "enumerable set for EXISTS domain",
    )?;
    if elements.is_empty() {
        regs[rd as usize] = Value::Bool(false);
        return Ok(LoopAction::Jump(((pc as i64) + (loop_end as i64)) as usize));
    }
    regs[r_binding as usize] = elements.element(0).clone();
    regs[rd as usize] = Value::Bool(false);
    iter_stack.push(LoopState::Quantifier { elements, index: 0 });
    Ok(LoopAction::Continue)
}

pub(super) fn exists_next(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_body: u8,
    loop_begin: i32,
) -> Result<LoopAction, VmError> {
    let body_val = as_bool(&regs[r_body as usize])?;
    if body_val {
        regs[rd as usize] = Value::Bool(true);
        iter_stack.pop();
    } else if let Some(LoopState::Quantifier { elements, index }) = iter_stack.last_mut() {
        *index += 1;
        if *index < elements.len() {
            regs[r_binding as usize] = elements.element(*index).clone();
            return Ok(LoopAction::Jump(
                ((pc as i64) + (loop_begin as i64)) as usize,
            ));
        }
        regs[rd as usize] = Value::Bool(false);
        iter_stack.pop();
    } else {
        return Err(VmError::Unsupported(
            "ExistsNext without matching ExistsBegin".to_string(),
        ));
    }
    Ok(LoopAction::Continue)
}

pub(super) fn set_filter_begin(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
    loop_end: i32,
) -> Result<LoopAction, VmError> {
    let domain = &regs[r_domain as usize];
    let elements = LoopElements::from_domain(domain).ok_or_else(|| VmError::TypeError {
        expected: "enumerable set for set filter",
        actual: format!("{domain:?}"),
    })?;
    if elements.is_empty() {
        regs[rd as usize] = Value::empty_set();
        return Ok(LoopAction::Jump(((pc as i64) + (loop_end as i64)) as usize));
    }
    regs[r_binding as usize] = elements.element(0).clone();
    iter_stack.push(LoopState::SetFilter {
        elements,
        index: 0,
        collected: Vec::new(),
        rd,
    });
    Ok(LoopAction::Continue)
}

pub(super) fn set_builder_begin(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
    loop_end: i32,
) -> Result<LoopAction, VmError> {
    let domain = &regs[r_domain as usize];
    let elements = LoopElements::from_domain(domain).ok_or_else(|| VmError::TypeError {
        expected: "enumerable set for set builder",
        actual: format!("{domain:?}"),
    })?;
    if elements.is_empty() {
        regs[rd as usize] = Value::empty_set();
        return Ok(LoopAction::Jump(((pc as i64) + (loop_end as i64)) as usize));
    }
    regs[r_binding as usize] = elements.element(0).clone();
    iter_stack.push(LoopState::SetBuilder {
        elements,
        index: 0,
        collected: Vec::new(),
        rd,
    });
    Ok(LoopAction::Continue)
}

pub(super) fn func_def_begin(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
    loop_end: i32,
) -> Result<LoopAction, VmError> {
    let domain = &regs[r_domain as usize];
    let elements = LoopElements::from_domain(domain).ok_or_else(|| VmError::TypeError {
        expected: "enumerable set for function definition",
        actual: format!("{domain:?}"),
    })?;
    if elements.is_empty() {
        regs[rd as usize] = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![])));
        return Ok(LoopAction::Jump(((pc as i64) + (loop_end as i64)) as usize));
    }
    regs[r_binding as usize] = elements.element(0).clone();
    iter_stack.push(LoopState::FuncDef {
        elements,
        index: 0,
        entries: Vec::new(),
        rd,
    });
    Ok(LoopAction::Continue)
}

pub(super) fn loop_next(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    r_binding: u8,
    r_body: u8,
    loop_begin: i32,
) -> Result<LoopAction, VmError> {
    match iter_stack.last_mut() {
        Some(LoopState::SetFilter {
            elements,
            index,
            collected,
            rd,
        }) => {
            if as_bool(&regs[r_body as usize])? {
                collected.push(elements.element(*index).clone());
            }
            *index += 1;
            if *index < elements.len() {
                regs[r_binding as usize] = elements.element(*index).clone();
                return Ok(LoopAction::Jump(
                    ((pc as i64) + (loop_begin as i64)) as usize,
                ));
            } else {
                let rd_idx = *rd;
                let result = std::mem::take(collected);
                let result = elements.filtered_set(result);
                iter_stack.pop();
                regs[rd_idx as usize] = Value::Set(Rp::new(result));
            }
        }
        Some(LoopState::SetBuilder {
            elements,
            index,
            collected,
            rd,
        }) => {
            collected.push(regs[r_body as usize].clone());
            *index += 1;
            if *index < elements.len() {
                regs[r_binding as usize] = elements.element(*index).clone();
                return Ok(LoopAction::Jump(
                    ((pc as i64) + (loop_begin as i64)) as usize,
                ));
            } else {
                let rd_idx = *rd;
                let result = std::mem::take(collected);
                iter_stack.pop();
                regs[rd_idx as usize] = Value::Set(Rp::new(SortedSet::from_iter(result)));
            }
        }
        Some(LoopState::FuncDef {
            elements,
            index,
            entries,
            rd,
        }) => {
            let key = elements.element(*index).clone();
            let val = regs[r_body as usize].clone();
            entries.push((key, val));
            *index += 1;
            if *index < elements.len() {
                regs[r_binding as usize] = elements.element(*index).clone();
                return Ok(LoopAction::Jump(
                    ((pc as i64) + (loop_begin as i64)) as usize,
                ));
            } else {
                let rd_idx = *rd;
                let mut result = std::mem::take(entries);
                result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                iter_stack.pop();
                regs[rd_idx as usize] =
                    Value::Func(Rp::new(FuncValue::from_sorted_entries(result)));
            }
        }
        _ => {
            return Err(VmError::Unsupported(
                "LoopNext without matching loop begin".to_string(),
            ));
        }
    }
    Ok(LoopAction::Continue)
}

pub(super) fn choose_begin(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    _pc: usize,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
) -> Result<LoopAction, VmError> {
    let domain = &regs[r_domain as usize];
    if !domain.is_set() {
        return Err(VmError::TypeError {
            expected: "set for CHOOSE domain",
            actual: format!("{domain:?}"),
        });
    }
    let elements: Vec<Value> = match domain.iter_set_tlc_normalized() {
        Ok(iter) => iter.collect(),
        Err(EvalError::SetTooLarge { .. }) => {
            return Err(VmError::Unsupported(
                "CHOOSE domain requires tree-walk iteration".to_string(),
            ));
        }
        Err(err) => return Err(VmError::Eval(err)),
    };
    if elements.is_empty() {
        return Err(VmError::ChooseFailed);
    }
    regs[r_binding as usize] = elements[0].clone();
    regs[rd as usize] = Value::Bool(false);
    iter_stack.push(LoopState::Choose {
        elements,
        index: 0,
        rd,
    });
    Ok(LoopAction::Continue)
}

pub(super) fn choose_next(
    regs: &mut [Value],
    iter_stack: &mut Vec<LoopState>,
    pc: usize,
    rd: u8,
    r_binding: u8,
    r_body: u8,
    loop_begin: i32,
) -> Result<LoopAction, VmError> {
    let body_val = as_bool(&regs[r_body as usize])?;
    if body_val {
        regs[rd as usize] = regs[r_binding as usize].clone();
        iter_stack.pop();
    } else if let Some(LoopState::Choose {
        elements, index, ..
    }) = iter_stack.last_mut()
    {
        *index += 1;
        if *index < elements.len() {
            regs[r_binding as usize] = elements[*index].clone();
            return Ok(LoopAction::Jump(
                ((pc as i64) + (loop_begin as i64)) as usize,
            ));
        }
        iter_stack.pop();
        return Err(VmError::ChooseFailed);
    } else {
        return Err(VmError::Unsupported(
            "ChooseNext without matching ChooseBegin".to_string(),
        ));
    }
    Ok(LoopAction::Continue)
}

#[cfg(test)]
mod tests {
    use super::{LoopElements, VmError};
    use tla_value::Rp;
    use tla_value::{IntervalValue, SortedSet, Value};

    fn materialize(elements: &LoopElements) -> Vec<Value> {
        (0..elements.len())
            .map(|index| elements.element(index).clone())
            .collect()
    }

    #[test]
    fn concrete_domain_shares_set_arc() {
        let set = Rp::new(SortedSet::from_iter([
            Value::SmallInt(3),
            Value::SmallInt(1),
            Value::SmallInt(3),
        ]));
        let domain = Value::Set(Rp::clone(&set));

        let elements = LoopElements::from_domain_with_sharing(&domain, true)
            .expect("concrete set should be enumerable");

        let LoopElements::Shared(shared) = &elements else {
            panic!("concrete set should use shared loop elements");
        };
        assert!(Rp::ptr_eq(shared, &set));
        assert_eq!(
            materialize(&elements),
            vec![Value::SmallInt(1), Value::SmallInt(3)]
        );
    }

    #[test]
    fn sharing_disabled_uses_owned_fallback() {
        let domain = Value::Set(Rp::new(SortedSet::from_iter([
            Value::SmallInt(2),
            Value::SmallInt(1),
        ])));

        let elements = LoopElements::from_domain_with_sharing(&domain, false)
            .expect("concrete set should be enumerable");

        assert!(matches!(&elements, LoopElements::Owned(_)));
        assert_eq!(
            materialize(&elements),
            vec![Value::SmallInt(1), Value::SmallInt(2)]
        );
    }

    #[test]
    fn lazy_domain_uses_owned_fallback() {
        let domain = Value::Interval(Rp::new(IntervalValue::new(1.into(), 3.into())));

        let elements = LoopElements::from_domain_with_sharing(&domain, true)
            .expect("finite interval should be enumerable");

        assert!(matches!(&elements, LoopElements::Owned(_)));
        assert_eq!(
            materialize(&elements),
            vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]
        );
    }

    #[test]
    fn filtered_concrete_domain_retains_canonical_subsequence() {
        let domain = Value::Set(Rp::new(SortedSet::from_iter([
            Value::SmallInt(3),
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(1),
        ])));
        let elements = LoopElements::from_domain_with_sharing(&domain, true)
            .expect("concrete set should be enumerable");
        let retained = vec![elements.element(0).clone(), elements.element(2).clone()];

        let result = elements.filtered_set(retained);

        assert_eq!(result.as_slice(), &[Value::SmallInt(1), Value::SmallInt(3)]);
    }

    #[test]
    fn filtered_owned_domain_keeps_general_set_semantics() {
        // The owned fallback intentionally makes no ordering promise. Its
        // result constructor must continue sorting and deduplicating when the
        // set is observed.
        let elements = LoopElements::Owned(vec![
            Value::SmallInt(3),
            Value::SmallInt(1),
            Value::SmallInt(3),
        ]);

        let result = elements.filtered_set(materialize(&elements));

        assert_eq!(result.as_slice(), &[Value::SmallInt(1), Value::SmallInt(3)]);
    }

    #[test]
    fn action_quantifier_declines_context_assisted_domain_iteration() {
        let result = LoopElements::from_quantifier_domain(
            &Value::StringSet,
            true,
            "enumerable action domain",
        );

        assert!(matches!(
            result,
            Err(VmError::Unsupported(reason))
                if reason.contains("context-assisted tree iteration")
        ));
    }
}
