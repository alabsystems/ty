// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fail-closed compatibility surface for retired compiled backends.
//!
//! Part of #4400 / #4267: keep the old checker call sites compiling while
//! trust-codegen owns the active native execution path. These shims never report
//! compiled coverage, so production execution stays on the interpreter or
//! trust-codegen paths.

#![allow(dead_code)]

use rustc_hash::FxHashMap;
use std::fmt;
use std::time::Duration;
use tla_tir::bytecode::BytecodeChunk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompiledBackendUnavailable;

impl fmt::Display for CompiledBackendUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retired compiled backend is unavailable")
    }
}

impl std::error::Error for CompiledBackendUnavailable {}

#[derive(Debug, Default)]
pub(crate) struct JitInvariantCache {
    required_vars: Vec<u16>,
}

impl JitInvariantCache {
    pub(crate) fn build(
        _chunk: &BytecodeChunk,
        _op_indices: &FxHashMap<String, u16>,
    ) -> Result<Self, CompiledBackendUnavailable> {
        Ok(Self::default())
    }

    pub(crate) fn build_with_layout(
        _chunk: &BytecodeChunk,
        _op_indices: &FxHashMap<String, u16>,
        _state_layout: &tla_jit_abi::StateLayout,
    ) -> Result<Self, CompiledBackendUnavailable> {
        Ok(Self::default())
    }

    pub(crate) fn is_empty(&self) -> bool {
        true
    }

    pub(crate) fn len(&self) -> usize {
        0
    }

    pub(crate) fn required_vars(&self) -> &[u16] {
        &self.required_vars
    }

    pub(crate) fn covers_all(&self, _invariants: &[String]) -> bool {
        false
    }

    pub(crate) fn resolve_ordered(
        &self,
        _invariants: &[String],
    ) -> Option<Vec<tla_jit_abi::JitInvariantFn>> {
        None
    }

    pub(crate) fn check_invariant(
        &self,
        _name: &str,
        _state: &[i64],
    ) -> Option<Result<bool, CompiledBackendUnavailable>> {
        None
    }

    pub(crate) fn check_all_resolved<'a>(
        _invariants: &'a [String],
        _resolved_fns: &[tla_jit_abi::JitInvariantFn],
        _state: &[i64],
    ) -> (Result<Option<&'a str>, CompiledBackendUnavailable>, bool) {
        (Ok(None), true)
    }

    pub(crate) fn check_all_compiled<'a>(
        &self,
        _invariants: &'a [String],
        _state: &[i64],
    ) -> (Result<Option<&'a str>, CompiledBackendUnavailable>, bool) {
        (Ok(None), true)
    }

    pub(crate) fn check_all<'a>(
        &self,
        invariants: &'a [String],
        _state: &[i64],
        unchecked: &mut Vec<&'a str>,
    ) -> Result<Option<&'a str>, CompiledBackendUnavailable> {
        unchecked.extend(invariants.iter().map(String::as_str));
        Ok(None)
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ActionMeta {
    pub(crate) read_vars: Vec<u16>,
    pub(crate) write_vars: Vec<u16>,
}

#[derive(Debug, Default)]
pub(crate) struct JitNextStateCache {
    state_var_count: usize,
}

impl JitNextStateCache {
    pub(crate) fn build(
        _chunk: &BytecodeChunk,
        _action_indices: &FxHashMap<String, u16>,
        state_var_count: usize,
    ) -> Result<Self, CompiledBackendUnavailable> {
        Ok(Self { state_var_count })
    }

    pub(crate) fn build_with_stats_and_layout(
        _chunk: &BytecodeChunk,
        _action_indices: &FxHashMap<String, u16>,
        state_var_count: usize,
        _state_layout: Option<&tla_jit_abi::StateLayout>,
    ) -> Result<(Self, tla_jit_abi::CacheBuildStats), CompiledBackendUnavailable> {
        Ok((
            Self { state_var_count },
            tla_jit_abi::CacheBuildStats::default(),
        ))
    }

    pub(crate) fn build_with_stats_and_specializations(
        _chunk: &BytecodeChunk,
        _action_indices: &FxHashMap<String, u16>,
        state_var_count: usize,
        _state_layout: Option<&tla_jit_abi::StateLayout>,
        _specializations: &[tla_jit_abi::BindingSpec],
    ) -> Result<(Self, tla_jit_abi::CacheBuildStats), CompiledBackendUnavailable> {
        Ok((
            Self { state_var_count },
            tla_jit_abi::CacheBuildStats::default(),
        ))
    }

    pub(crate) fn is_empty(&self) -> bool {
        true
    }

    pub(crate) fn len(&self) -> usize {
        0
    }

    pub(crate) fn compiled_action_keys(&self) -> Vec<String> {
        Vec::new()
    }

    pub(crate) fn inner_exists_expansion_keys(&self, _base_name: &str) -> Vec<String> {
        Vec::new()
    }

    pub(crate) fn state_var_count(&self) -> usize {
        self.state_var_count
    }

    pub(crate) fn required_read_vars(&self) -> &[u16] {
        &[]
    }

    pub(crate) fn required_write_vars(&self) -> &[u16] {
        &[]
    }

    pub(crate) fn contains_action(&self, _name: &str) -> bool {
        false
    }

    pub(crate) fn resolve_ordered(
        &self,
        _action_names: &[String],
    ) -> Option<Vec<tla_jit_abi::JitNextStateFn>> {
        None
    }

    pub(crate) fn action_metadata(&self, _name: &str) -> Option<&ActionMeta> {
        None
    }

    pub(crate) fn eval_action(
        &self,
        _name: &str,
        _state: &[i64],
    ) -> Option<Result<tla_jit_abi::JitActionResult, CompiledBackendUnavailable>> {
        None
    }

    pub(crate) fn eval_action_into(
        &self,
        _name: &str,
        _state: &[i64],
        _state_out: &mut [i64],
    ) -> Option<Result<bool, CompiledBackendUnavailable>> {
        None
    }
}

#[derive(Debug, Default)]
pub(crate) struct TypeProfile;

impl TypeProfile {
    pub(crate) fn monomorphic_types(&self) -> Vec<Option<tla_jit_abi::SpecType>> {
        Vec::new()
    }

    pub(crate) fn total_states(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct TierSummary {
    pub(crate) total: usize,
    pub(crate) eligible: usize,
    pub(crate) interpreter: usize,
    pub(crate) tier1: usize,
    pub(crate) tier2: usize,
}

impl fmt::Display for TierSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "actions={} eligible={} tier0={} tier1={} tier2={}",
            self.total, self.eligible, self.interpreter, self.tier1, self.tier2
        )
    }
}

#[derive(Debug)]
pub(crate) struct TierManager {
    config: tla_jit_abi::TierConfig,
    action_count: usize,
}

impl TierManager {
    pub(crate) fn new(action_count: usize) -> Self {
        Self {
            config: tla_jit_abi::TierConfig::from_env(),
            action_count,
        }
    }

    pub(crate) fn with_config(action_count: usize, config: tla_jit_abi::TierConfig) -> Self {
        Self {
            config,
            action_count,
        }
    }

    pub(crate) fn set_eligible(&mut self, _action_id: usize) {}

    pub(crate) fn current_tier(&self, _action_id: usize) -> tla_jit_abi::CompilationTier {
        tla_jit_abi::CompilationTier::Interpreter
    }

    pub(crate) fn config(&self) -> &tla_jit_abi::TierConfig {
        &self.config
    }

    pub(crate) fn action_count(&self) -> usize {
        self.action_count
    }

    pub(crate) fn promotion_check(
        &mut self,
        _profiles: &[tla_jit_abi::ActionProfile],
    ) -> Vec<tla_jit_abi::TierPromotion> {
        Vec::new()
    }

    pub(crate) fn promote_all_actions(
        &mut self,
        _target_tier: tla_jit_abi::CompilationTier,
        _aggregate_evals: u64,
        _aggregate_branching_factor: f64,
    ) -> Vec<tla_jit_abi::TierPromotion> {
        Vec::new()
    }

    pub(crate) fn enable_type_profiling(&mut self, _var_count: usize) {}

    pub(crate) fn observe_state_types(&mut self, _types: &[tla_jit_abi::SpecType]) -> bool {
        false
    }

    pub(crate) fn type_profile_stable(&self) -> bool {
        true
    }

    pub(crate) fn type_profile(&self) -> Option<&TypeProfile> {
        None
    }

    pub(crate) fn tier_summary(&self) -> TierSummary {
        TierSummary {
            total: self.action_count,
            eligible: 0,
            interpreter: self.action_count,
            tier1: 0,
            tier2: 0,
        }
    }
}

pub(crate) struct RecompilationResult {
    pub(crate) cache: JitNextStateCache,
    pub(crate) stats: tla_jit_abi::CacheBuildStats,
    pub(crate) plan: tla_jit_abi::SpecializationPlan,
    pub(crate) total_time: Duration,
}

#[derive(Debug, Default)]
pub(crate) struct RecompilationController {
    attempted: bool,
}

impl RecompilationController {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn has_pending(&self) -> bool {
        false
    }

    pub(crate) fn already_attempted(&self) -> bool {
        self.attempted
    }

    pub(crate) fn trigger_recompilation(
        &mut self,
        _plan: tla_jit_abi::SpecializationPlan,
        _chunk: BytecodeChunk,
        _op_indices: FxHashMap<String, u16>,
        _state_var_count: usize,
        _state_layout: Option<tla_jit_abi::StateLayout>,
        _specializations: Vec<tla_jit_abi::BindingSpec>,
    ) -> Result<(), CompiledBackendUnavailable> {
        self.attempted = true;
        Err(CompiledBackendUnavailable)
    }

    pub(crate) fn poll_completion(&mut self) -> Option<Result<RecompilationResult, String>> {
        None
    }
}

#[derive(Debug, Default)]
pub(crate) struct CompiledBfsStep {
    state_len: usize,
}

impl CompiledBfsStep {
    pub(crate) fn build(
        _spec: &tla_jit_abi::BfsStepSpec,
        _action_fns: Vec<tla_jit_abi::CompiledActionFn>,
        _invariant_fns: Vec<tla_jit_abi::CompiledInvariantFn>,
        _expected_states: usize,
    ) -> Result<Self, CompiledBackendUnavailable> {
        Err(CompiledBackendUnavailable)
    }

    pub(crate) fn state_len(&self) -> usize {
        self.state_len
    }

    pub(crate) fn step_flat(
        &self,
        _state: &[i64],
    ) -> Result<tla_jit_abi::FlatBfsStepOutput, tla_jit_abi::BfsStepError> {
        Err(tla_jit_abi::BfsStepError::RuntimeError)
    }
}

#[derive(Debug, Default)]
pub(crate) struct CompiledBfsLevel {
    state_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledLevelResult {
    pub(crate) successor_arena: Vec<i64>,
    pub(crate) successor_parent_indices: Option<Vec<usize>>,
    pub(crate) successor_count: usize,
    pub(crate) state_len: usize,
    pub(crate) parents_processed: usize,
    pub(crate) total_generated: u64,
    pub(crate) total_new: u64,
    pub(crate) invariant_ok: bool,
    pub(crate) failed_parent_idx: Option<usize>,
    pub(crate) failed_invariant_idx: Option<u32>,
    pub(crate) failed_successor: Option<Vec<i64>>,
    pub(crate) regular_invariants_checked_by_backend: bool,
}

impl CompiledLevelResult {
    pub(crate) fn successor_count(&self) -> usize {
        self.successor_count
    }
}

impl CompiledBfsLevel {
    pub(crate) fn build_fused(
        _spec: &tla_jit_abi::BfsStepSpec,
        _action_fns: Vec<tla_jit_abi::CompiledActionFn>,
        _invariant_fns: Vec<tla_jit_abi::CompiledInvariantFn>,
        _expected_states: usize,
    ) -> Result<Self, CompiledBackendUnavailable> {
        Err(CompiledBackendUnavailable)
    }

    pub(crate) fn has_fused_level(&self) -> bool {
        false
    }

    pub(crate) fn state_len(&self) -> usize {
        self.state_len
    }

    pub(crate) fn run_level_fused_arena(
        &self,
        _arena: &[i64],
        _parent_count: usize,
    ) -> Option<Result<CompiledLevelResult, tla_jit_abi::BfsStepError>> {
        None
    }
}
