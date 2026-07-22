// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::ops::ControlFlow;

use rustc_hash::FxHashMap;
use thiserror::Error;

use super::resolve::ResolvedCtl;

use crate::explorer::fingerprint::fingerprint_marking;
use crate::explorer::{ExplorationConfig, ExplorationSetup};
use std::cell::RefCell;

use crate::marking::{pack_marking_config, unpack_marking_config, MarkingConfig};

thread_local! {
    /// Reused scratch for decoding a packed marking during atom evaluation.
    static LC_UNPACK_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::resolved_predicate::eval_predicate;

// The local fallback is recursive over both formula structure and successor
// paths. Very long acyclic paths can otherwise overflow the process stack
// before the normal deadline/state-budget guards fire.
const LOCAL_EVAL_RECURSION_LIMIT: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FormulaKey(*const ResolvedCtl);

impl FormulaKey {
    fn of(formula: &ResolvedCtl) -> Self {
        Self(std::ptr::from_ref(formula))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EvalKey {
    state_id: u32,
    formula: FormulaKey,
}

impl EvalKey {
    fn new(state_id: u32, formula: &ResolvedCtl) -> Self {
        Self {
            state_id,
            formula: FormulaKey::of(formula),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheEntry {
    Active { assume_on_cycle: bool },
    Ready(bool),
}

impl CacheEntry {
    fn value(self) -> bool {
        match self {
            Self::Active { assume_on_cycle } => assume_on_cycle,
            Self::Ready(value) => value,
        }
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalCheckAbort {
    #[error("local CTL checker exceeded state budget")]
    StateLimitReached,
    #[error("local CTL checker hit the deadline")]
    DeadlineExceeded,
    #[error("local CTL checker exceeded recursion depth")]
    RecursionDepthExceeded,
    /// Firing a transition would overflow a place's `u64` token count (#22).
    /// Treated as inconclusive (CANNOT_COMPUTE) downstream — never a verdict.
    #[error("local CTL checker hit a token-count overflow")]
    TokenOverflow,
    /// An operator the single-pass local checker cannot evaluate soundly
    /// reached it — the alternating fair-cycle `EGF`. The alternation-free
    /// routing gate (`ctl_is_alternation_free`) keeps `EGF` off this lane, so
    /// this is a defensive fail-closed abort (CANNOT_COMPUTE), never a verdict;
    /// `EGF` is decided by the GPU/CPU `CtlEngine` fair-cycle evaluators.
    #[error("local CTL checker cannot evaluate the alternating EGF operator")]
    UnsupportedOperator,
}

struct LocalStateSpace<'a> {
    net: &'a PetriNet,
    max_states: usize,
    pack_capacity: usize,
    marking_config: MarkingConfig,
    state_ids: FxHashMap<u128, u32>,
    /// PACKED marking per state (the dedup key) — decode via `marking_into`.
    /// Lossless, so state identity and verdicts are unchanged.
    markings: Vec<Box<[u8]>>,
    /// One adaptive probe covering BOTH the wall-clock deadline and the memory
    /// budget (`max_states` is `usize::MAX` on the auto-sized MCC path and the
    /// `(state, subformula)` memo is uncapped, so item caps do not bound
    /// bytes). Ticked per marking intern in [`Self::check_resources`].
    probe: tla_resource::MemoryProbe,
}

impl<'a> LocalStateSpace<'a> {
    fn new(net: &'a PetriNet, config: &ExplorationConfig) -> Self {
        let setup = ExplorationSetup::analyze(net);
        let mut state_ids = FxHashMap::default();
        state_ids.insert(fingerprint_marking(&setup.initial_packed), 0);

        Self {
            net,
            max_states: config.max_states(),
            pack_capacity: setup.pack_capacity,
            marking_config: setup.marking_config,
            state_ids,
            markings: vec![setup.initial_packed],
            probe: crate::memory::explorer_probe(config.deadline()),
        }
    }

    /// Decode state `state_id`'s marking into `out` (reused). Markings are stored
    /// PACKED, so reads go through the (lossless) codec.
    fn marking_into(&self, state_id: u32, out: &mut Vec<u64>) {
        unpack_marking_config(&self.markings[state_id as usize], &self.marking_config, out);
    }

    fn intern_marking(
        &mut self,
        marking: &[u64],
        pack_buf: &mut Vec<u8>,
    ) -> Result<u32, LocalCheckAbort> {
        self.check_resources()?;
        pack_marking_config(marking, &self.marking_config, pack_buf);
        let fingerprint = fingerprint_marking(pack_buf);
        if let Some(&existing) = self.state_ids.get(&fingerprint) {
            return Ok(existing);
        }
        if self.markings.len() >= self.max_states {
            return Err(LocalCheckAbort::StateLimitReached);
        }

        let state_id = self.markings.len() as u32;
        self.state_ids.insert(fingerprint, state_id);
        // Store the packed bytes already computed above — not the fat `Vec<u64>`.
        self.markings.push(pack_buf.as_slice().into());
        Ok(state_id)
    }

    fn check_resources(&mut self) -> Result<(), LocalCheckAbort> {
        // One adaptive probe, two distinct declines (both inconclusive).
        match self.probe.check() {
            Some(tla_resource::Trip::Deadline) => Err(LocalCheckAbort::DeadlineExceeded),
            Some(tla_resource::Trip::Memory) => Err(LocalCheckAbort::StateLimitReached),
            None => Ok(()),
        }
    }
}

struct SuccessorVisit<B> {
    saw_any: bool,
    break_value: Option<B>,
}

pub(super) struct LocalCtlChecker<'a> {
    state_space: LocalStateSpace<'a>,
    memo: FxHashMap<EvalKey, CacheEntry>,
    eval_depth: usize,
}

impl<'a> LocalCtlChecker<'a> {
    pub(super) fn new(net: &'a PetriNet, config: &ExplorationConfig) -> Self {
        Self {
            state_space: LocalStateSpace::new(net, config),
            memo: FxHashMap::default(),
            eval_depth: 0,
        }
    }

    pub(super) fn eval_root(&mut self, formula: &ResolvedCtl) -> Result<bool, LocalCheckAbort> {
        self.eval(0, formula)
    }

    fn eval(&mut self, state_id: u32, formula: &ResolvedCtl) -> Result<bool, LocalCheckAbort> {
        self.state_space.check_resources()?;
        if self.eval_depth >= LOCAL_EVAL_RECURSION_LIMIT {
            return Err(LocalCheckAbort::RecursionDepthExceeded);
        }

        self.eval_depth += 1;
        let result = self.eval_inner(state_id, formula);
        self.eval_depth -= 1;
        result
    }

    fn eval_inner(
        &mut self,
        state_id: u32,
        formula: &ResolvedCtl,
    ) -> Result<bool, LocalCheckAbort> {
        let key = EvalKey::new(state_id, formula);
        if let Some(entry) = self.memo.get(&key).copied() {
            return Ok(entry.value());
        }

        let value = match formula {
            ResolvedCtl::Atom(predicate) => LC_UNPACK_SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                self.state_space.marking_into(state_id, &mut scratch);
                eval_predicate(predicate, &scratch, self.state_space.net)
            }),
            ResolvedCtl::Not(inner) => !self.eval(state_id, inner)?,
            ResolvedCtl::And(children) => {
                let mut result = true;
                for child in children {
                    if !self.eval(state_id, child)? {
                        result = false;
                        break;
                    }
                }
                result
            }
            ResolvedCtl::Or(children) => {
                let mut result = false;
                for child in children {
                    if self.eval(state_id, child)? {
                        result = true;
                        break;
                    }
                }
                result
            }
            ResolvedCtl::EX(inner) => {
                let visit = self.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, inner)? {
                        Ok(ControlFlow::Break(()))
                    } else {
                        Ok(ControlFlow::Continue(()))
                    }
                })?;
                visit.break_value.is_some()
            }
            ResolvedCtl::AX(inner) => {
                let visit = self.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, inner)? {
                        Ok(ControlFlow::Continue(()))
                    } else {
                        Ok(ControlFlow::Break(()))
                    }
                })?;
                !visit.saw_any || visit.break_value.is_none()
            }
            ResolvedCtl::EF(inner) => self.eval_fixpoint(state_id, formula, false, |this| {
                if this.eval(state_id, inner)? {
                    return Ok(true);
                }
                let visit = this.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, formula)? {
                        Ok(ControlFlow::Break(()))
                    } else {
                        Ok(ControlFlow::Continue(()))
                    }
                })?;
                Ok(visit.break_value.is_some())
            })?,
            ResolvedCtl::AF(inner) => self.eval_fixpoint(state_id, formula, false, |this| {
                if this.eval(state_id, inner)? {
                    return Ok(true);
                }
                let visit = this.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, formula)? {
                        Ok(ControlFlow::Continue(()))
                    } else {
                        Ok(ControlFlow::Break(()))
                    }
                })?;
                Ok(visit.saw_any && visit.break_value.is_none())
            })?,
            ResolvedCtl::EG(inner) => self.eval_fixpoint(state_id, formula, true, |this| {
                if !this.eval(state_id, inner)? {
                    return Ok(false);
                }
                let visit = this.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, formula)? {
                        Ok(ControlFlow::Break(()))
                    } else {
                        Ok(ControlFlow::Continue(()))
                    }
                })?;
                Ok(!visit.saw_any || visit.break_value.is_some())
            })?,
            ResolvedCtl::AG(inner) => self.eval_fixpoint(state_id, formula, true, |this| {
                if !this.eval(state_id, inner)? {
                    return Ok(false);
                }
                let visit = this.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, formula)? {
                        Ok(ControlFlow::Continue(()))
                    } else {
                        Ok(ControlFlow::Break(()))
                    }
                })?;
                Ok(!visit.saw_any || visit.break_value.is_none())
            })?,
            ResolvedCtl::EU(phi, psi) => self.eval_fixpoint(state_id, formula, false, |this| {
                if this.eval(state_id, psi)? {
                    return Ok(true);
                }
                if !this.eval(state_id, phi)? {
                    return Ok(false);
                }
                let visit = this.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, formula)? {
                        Ok(ControlFlow::Break(()))
                    } else {
                        Ok(ControlFlow::Continue(()))
                    }
                })?;
                Ok(visit.break_value.is_some())
            })?,
            ResolvedCtl::AU(phi, psi) => self.eval_fixpoint(state_id, formula, false, |this| {
                if this.eval(state_id, psi)? {
                    return Ok(true);
                }
                if !this.eval(state_id, phi)? {
                    return Ok(false);
                }
                let visit = this.visit_successors(state_id, |this, successor_id| {
                    if this.eval(successor_id, formula)? {
                        Ok(ControlFlow::Continue(()))
                    } else {
                        Ok(ControlFlow::Break(()))
                    }
                })?;
                Ok(visit.saw_any && visit.break_value.is_none())
            })?,
            // EGF (fair cycle) is alternating (νμ) and is routed away from this
            // single-pass checker by `ctl_is_alternation_free`; if it ever
            // reaches here, fail closed to CANNOT_COMPUTE rather than risk an
            // unsound single-pass verdict.
            ResolvedCtl::EGF(_) => return Err(LocalCheckAbort::UnsupportedOperator),
        };

        self.memo.insert(key, CacheEntry::Ready(value));
        Ok(value)
    }

    fn eval_fixpoint(
        &mut self,
        state_id: u32,
        formula: &ResolvedCtl,
        assume_on_cycle: bool,
        compute: impl FnOnce(&mut Self) -> Result<bool, LocalCheckAbort>,
    ) -> Result<bool, LocalCheckAbort> {
        let key = EvalKey::new(state_id, formula);
        if let Some(entry) = self.memo.get(&key).copied() {
            return Ok(entry.value());
        }

        self.memo
            .insert(key, CacheEntry::Active { assume_on_cycle });
        let value = compute(self)?;
        self.memo.insert(key, CacheEntry::Ready(value));
        Ok(value)
    }

    fn visit_successors<B>(
        &mut self,
        state_id: u32,
        mut visitor: impl FnMut(&mut Self, u32) -> Result<ControlFlow<B>, LocalCheckAbort>,
    ) -> Result<SuccessorVisit<B>, LocalCheckAbort> {
        self.state_space.check_resources()?;

        let mut current = Vec::new();
        self.state_space.marking_into(state_id, &mut current);
        let mut pack_buf = Vec::with_capacity(self.state_space.pack_capacity);
        let mut saw_any = false;

        for tidx in 0..self.state_space.net.num_transitions() {
            let transition = TransitionIdx(tidx as u32);
            if !self.state_space.net.is_enabled(&current, transition) {
                continue;
            }

            saw_any = true;
            // Fail-closed (#22): token-count overflow leaves `current` partially
            // mutated, so do NOT undo — abort the local check as inconclusive.
            self.state_space
                .net
                .apply_delta(&mut current, transition)
                .map_err(|_| LocalCheckAbort::TokenOverflow)?;
            let successor_id = self.state_space.intern_marking(&current, &mut pack_buf)?;
            self.state_space.net.undo_delta(&mut current, transition);

            match visitor(self, successor_id)? {
                ControlFlow::Break(value) => {
                    return Ok(SuccessorVisit {
                        saw_any,
                        break_value: Some(value),
                    });
                }
                ControlFlow::Continue(()) => {}
            }
        }

        Ok(SuccessorVisit {
            saw_any,
            break_value: None,
        })
    }
}
