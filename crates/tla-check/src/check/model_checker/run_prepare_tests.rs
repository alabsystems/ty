// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for `precompute_constant_operators` — the setup-time constant-folding
//! pass that mirrors TLC's `SpecProcessor.processConstantDefns()`.
//!
//! Part of #3125: replaced shallow `body_references_state_var` tests with
//! semantic level classification tests against the actual precompute pass.

use super::super::bfs::compiled_step_trait::{
    BfsStepError, CompiledBfsLevel, CompiledBfsStep, FlatBfsStepOutput,
};
use super::super::bfs::flat_frontier::FlatBfsFrontier;
use tla_value::Rp;
use super::super::bfs::storage_modes::NoTraceQueueEntry;
use super::super::frontier::BfsFrontier;
use super::super::mc_struct::ActionInstanceMeta;
use super::super::CheckResult;
use super::ModelChecker;
use crate::check::model_checker::precompute::{
    precompute_constant_operators, promote_env_constants_to_precomputed,
};
use crate::config::{Config, ConstantValue};
use crate::constants::bind_constants_from_config;
use crate::eval::EvalCtx;
use crate::state::{
    ArrayState, FlatValueLayout, SequenceBoundEvidence, SlotType, StateLayout, VarLayoutKind,
};
use crate::test_support::parse_module;
use crate::value::{FuncValue, IntIntervalFunc, RecordValue, SeqValue, SortedSet, Value};
use crate::var_index::VarRegistry;
use std::sync::Arc;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::name_intern::intern_name;
use tla_core::span::Spanned;
use tla_tir::bytecode::CompileError;

fn make_op(name: &str, body: Expr) -> OperatorDef {
    OperatorDef {
        name: Spanned::dummy(name.to_string()),
        params: vec![],
        body: Spanned::dummy(body),
        local: false,
        contains_prime: false,
        guards_depend_on_prime: false,
        has_primed_param: false,
        is_recursive: false,
        self_call_count: 0,
    }
}

fn seed_split_action_meta(checker: &mut ModelChecker<'_>, names: &[&str]) {
    checker.compiled.split_action_meta = Some(
        names
            .iter()
            .map(|name| ActionInstanceMeta {
                name: Some((*name).to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            })
            .collect(),
    );
}

struct TestCompiledBfsStep {
    state_len: usize,
}

impl CompiledBfsStep for TestCompiledBfsStep {
    fn state_len(&self) -> usize {
        self.state_len
    }

    fn step_flat(&self, _state: &[i64]) -> Result<FlatBfsStepOutput, BfsStepError> {
        Err(BfsStepError::RuntimeError)
    }
}

struct TestCompiledBfsLevel;

impl CompiledBfsLevel for TestCompiledBfsLevel {
    fn has_fused_level(&self) -> bool {
        true
    }

    fn run_level_fused_arena(
        &self,
        _arena: &[i64],
        _parent_count: usize,
    ) -> Option<Result<super::super::bfs::compiled_step_trait::CompiledLevelResult, BfsStepError>>
    {
        Some(Err(BfsStepError::RuntimeError))
    }
}

struct TestNativeFusedBfsLevel;

impl CompiledBfsLevel for TestNativeFusedBfsLevel {
    fn has_fused_level(&self) -> bool {
        true
    }

    fn has_native_fused_level(&self) -> bool {
        true
    }

    fn run_level_fused_arena(
        &self,
        _arena: &[i64],
        _parent_count: usize,
    ) -> Option<Result<super::super::bfs::compiled_step_trait::CompiledLevelResult, BfsStepError>>
    {
        Some(Err(BfsStepError::RuntimeError))
    }
}

struct TestNativeFusedInvariantBfsLevel {
    state_len: usize,
}

impl CompiledBfsLevel for TestNativeFusedInvariantBfsLevel {
    fn has_fused_level(&self) -> bool {
        true
    }

    fn has_native_fused_level(&self) -> bool {
        true
    }

    fn fused_level_state_len(&self) -> Option<usize> {
        Some(self.state_len)
    }

    fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
        true
    }

    fn run_level_fused_arena(
        &self,
        _arena: &[i64],
        _parent_count: usize,
    ) -> Option<Result<super::super::bfs::compiled_step_trait::CompiledLevelResult, BfsStepError>>
    {
        Some(Err(BfsStepError::RuntimeError))
    }
}

struct TestNativeFusedConstrainedBfsLevel {
    state_len: usize,
    state_constraint_count: usize,
}

impl CompiledBfsLevel for TestNativeFusedConstrainedBfsLevel {
    fn has_fused_level(&self) -> bool {
        true
    }

    fn has_native_fused_level(&self) -> bool {
        true
    }

    fn fused_level_state_len(&self) -> Option<usize> {
        Some(self.state_len)
    }

    fn native_fused_state_constraint_count(&self) -> usize {
        self.state_constraint_count
    }

    fn native_fused_state_constraints_checked_by_backend(&self, expected_count: usize) -> bool {
        self.state_constraint_count == expected_count
    }

    fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
        true
    }

    fn run_level_fused_arena(
        &self,
        _arena: &[i64],
        _parent_count: usize,
    ) -> Option<Result<super::super::bfs::compiled_step_trait::CompiledLevelResult, BfsStepError>>
    {
        Some(Err(BfsStepError::RuntimeError))
    }
}

fn unsupported_failure_message(
    compiled: &tla_eval::bytecode_vm::CompiledBytecode,
    action_name: &str,
) -> String {
    match compiled
        .failed
        .iter()
        .find(|(name, _)| name == action_name)
        .map(|(_, err)| err)
    {
        Some(CompileError::Unsupported(message)) => message.clone(),
        Some(other) => panic!("expected Unsupported failure for {action_name}, got {other:?}"),
        None => panic!("missing failed compile entry for {action_name}"),
    }
}

fn assert_failed_message_contains(
    compiled: &tla_eval::bytecode_vm::CompiledBytecode,
    action_name: &str,
    expected_fragment: &str,
) {
    let message = unsupported_failure_message(compiled, action_name);
    assert!(
        message.contains(expected_fragment),
        "{action_name} should report {expected_fragment:?}, got: {message}",
    );
}

fn scalar_init_state() -> ArrayState {
    ArrayState::from_values(vec![Value::SmallInt(1), Value::Bool(true)])
}

fn fixed_record_init_state() -> ArrayState {
    // Entries must be sorted by field name (canonical record order).
    let rec = RecordValue::from_sorted_str_entries(vec![
        (Arc::from("count"), Value::SmallInt(7)),
        (Arc::from("flag"), Value::Bool(true)),
    ]);
    ArrayState::from_values(vec![Value::Record(rec)])
}

fn fixed_array_init_state(values: [i64; 3]) -> ArrayState {
    let func = IntIntervalFunc::new(0, 2, values.into_iter().map(Value::SmallInt).collect());
    ArrayState::from_values(vec![Value::IntFunc(Rp::new(func))])
}

fn model_value_keyed_function_state() -> ArrayState {
    let func = FuncValue::from_sorted_entries(vec![
        (
            Value::ModelValue(Rp::from("p1")),
            Value::ModelValue(Rp::from("defaultInitValue")),
        ),
        (
            Value::ModelValue(Rp::from("p2")),
            Value::ModelValue(Rp::from("defaultInitValue")),
        ),
    ]);
    ArrayState::from_values(vec![Value::Func(Rp::new(func))])
}

fn model_value_keyed_empty_set_function_state() -> ArrayState {
    let empty_set = || Value::Set(Rp::new(SortedSet::from_sorted_vec(Vec::<Value>::new())));
    let func = FuncValue::from_sorted_entries(vec![
        (Value::ModelValue(Rp::from("p1")), empty_set()),
        (Value::ModelValue(Rp::from("p2")), empty_set()),
    ]);
    ArrayState::from_values(vec![Value::Func(Rp::new(func))])
}

fn sequence_init_state() -> ArrayState {
    ArrayState::from_values(vec![Value::Seq(Rp::new(SeqValue::from_vec(vec![
        Value::SmallInt(1),
    ])))])
}

fn observed_network_value() -> Value {
    let msg = Value::Record(RecordValue::from_sorted_str_entries(vec![
        (Arc::from("clock"), Value::SmallInt(1)),
        (Arc::from("type"), Value::String(Rp::from("req"))),
    ]));
    let nonempty = Value::Seq(Rp::new(SeqValue::from_vec(vec![msg])));
    let empty = || Value::Seq(Rp::new(SeqValue::from_vec(vec![])));
    let row1 = Value::IntFunc(Rp::new(IntIntervalFunc::new(
        1,
        2,
        vec![empty(), nonempty],
    )));
    let row2 = Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![empty(), empty()])));
    Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![row1, row2])))
}

fn empty_network_value() -> Value {
    let empty = || Value::Seq(Rp::new(SeqValue::from_vec(vec![])));
    let empty_row = || Value::IntFunc(Rp::new(IntIntervalFunc::new(1, 2, vec![empty(), empty()])));
    Value::IntFunc(Rp::new(IntIntervalFunc::new(
        1,
        2,
        vec![empty_row(), empty_row()],
    )))
}

fn array_record_function_value() -> Value {
    let nonempty = Value::Seq(Rp::new(SeqValue::from_vec(vec![Value::SmallInt(1)])));
    let empty = Value::Seq(Rp::new(SeqValue::from_vec(Vec::new())));
    let record = |elems| {
        Value::Record(RecordValue::from_sorted_str_entries(vec![(
            Arc::from("elems"),
            elems,
        )]))
    };
    Value::IntFunc(Rp::new(IntIntervalFunc::new(
        1,
        2,
        vec![record(nonempty), record(empty)],
    )))
}

fn empty_sequence_network_value() -> Value {
    let empty_channel = || Value::Seq(Rp::new(SeqValue::from_vec(Vec::new())));
    let empty_row = || {
        Value::Seq(Rp::new(SeqValue::from_vec(vec![
            empty_channel(),
            empty_channel(),
        ])))
    };
    Value::Seq(Rp::new(SeqValue::from_vec(vec![empty_row(), empty_row()])))
}

fn full_mcl_sequence_init_state(checker: &ModelChecker<'_>) -> ArrayState {
    fn seq(values: Vec<Value>) -> Value {
        Value::Seq(Rp::new(SeqValue::from_vec(values)))
    }

    let empty_proc_set = || Value::Set(Rp::new(SortedSet::from_sorted_vec(vec![])));
    let mut values = vec![Value::SmallInt(0); checker.ctx.var_registry().len()];
    let mut set_var = |name: &str, value: Value| {
        let idx = checker
            .ctx
            .var_registry()
            .get(name)
            .unwrap_or_else(|| panic!("missing variable {name}"))
            .as_usize();
        values[idx] = value;
    };

    set_var(
        "clock",
        seq(vec![
            Value::SmallInt(1),
            Value::SmallInt(1),
            Value::SmallInt(1),
        ]),
    );
    let req_row = || {
        seq(vec![
            Value::SmallInt(0),
            Value::SmallInt(0),
            Value::SmallInt(0),
        ])
    };
    set_var("req", seq(vec![req_row(), req_row(), req_row()]));
    set_var(
        "ack",
        seq(vec![empty_proc_set(), empty_proc_set(), empty_proc_set()]),
    );
    let empty_channel = || Value::tuple(Vec::<Value>::new());
    let network_row = || seq(vec![empty_channel(), empty_channel(), empty_channel()]);
    set_var(
        "network",
        seq(vec![network_row(), network_row(), network_row()]),
    );
    set_var("crit", empty_proc_set());
    ArrayState::from_values(values)
}

fn assert_network_channel_bound_observed(layout: &crate::state::StateLayout, message: &str) {
    assert!(
        !layout.supports_flat_primary(),
        "{message}: invalid proof must not make network primary-safe"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(*bound, SequenceBoundEvidence::Observed, "{message}");
                    assert_eq!(*max_len, 1, "{message}");
                }
                other => {
                    panic!("{message}: expected network channel sequence layout, got {other:?}")
                }
            },
            other => panic!("{message}: expected nested network function layout, got {other:?}"),
        },
        other => panic!("{message}: expected recursive network layout, got {other:?}"),
    }
}

fn assert_array_elems_sequence_bound(
    layout: &crate::state::StateLayout,
    expected_bound: SequenceBoundEvidence,
    expected_max_len: usize,
    message: &str,
) {
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::Record {
                field_names,
                field_layouts,
            } => {
                let elems_pos = field_names
                    .iter()
                    .position(|name| name.as_ref() == "elems")
                    .unwrap_or_else(|| panic!("{message}: missing elems field"));
                match &field_layouts[elems_pos] {
                    FlatValueLayout::Sequence { bound, max_len, .. } => {
                        assert_eq!(*bound, expected_bound, "{message}");
                        assert_eq!(*max_len, expected_max_len, "{message}");
                    }
                    other => panic!("{message}: expected elems sequence layout, got {other:?}"),
                }
            }
            other => panic!("{message}: expected array record layout, got {other:?}"),
        },
        other => panic!("{message}: expected recursive array layout, got {other:?}"),
    }
}

fn assert_message_record_layout(layout: &FlatValueLayout, message: &str) {
    let FlatValueLayout::Record {
        field_names,
        field_layouts,
    } = layout
    else {
        panic!("{message}: expected proven message record layout, got {layout:?}");
    };
    assert_eq!(
        field_layouts.len(),
        field_names.len(),
        "{message}: message field layout count"
    );
    assert_eq!(field_names.len(), 2, "{message}: message fields");
    let clock_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "clock")
        .unwrap_or_else(|| panic!("{message}: missing message clock field"));
    let type_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "type")
        .unwrap_or_else(|| panic!("{message}: missing message type field"));
    assert_eq!(
        field_layouts[clock_pos],
        FlatValueLayout::Scalar(SlotType::Int),
        "{message}: message clock field"
    );
    assert_eq!(
        field_layouts[type_pos],
        FlatValueLayout::Scalar(SlotType::String),
        "{message}: message type field"
    );
}

fn assert_undo_entry_record_layout(layout: &FlatValueLayout, message: &str) {
    let FlatValueLayout::Record {
        field_names,
        field_layouts,
    } = layout
    else {
        panic!("{message}: expected undo-entry record layout, got {layout:?}");
    };
    assert_eq!(field_names.len(), 3, "{message}: undo-entry fields");
    let kind_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "kind")
        .unwrap_or_else(|| panic!("{message}: missing kind field"));
    let line_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "cursorLine")
        .unwrap_or_else(|| panic!("{message}: missing cursorLine field"));
    let col_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "cursorCol")
        .unwrap_or_else(|| panic!("{message}: missing cursorCol field"));
    assert_eq!(
        field_layouts[kind_pos],
        FlatValueLayout::Scalar(SlotType::String),
        "{message}: kind field"
    );
    assert_eq!(
        field_layouts[line_pos],
        FlatValueLayout::Scalar(SlotType::Int),
        "{message}: cursorLine field"
    );
    assert_eq!(
        field_layouts[col_pos],
        FlatValueLayout::Scalar(SlotType::Int),
        "{message}: cursorCol field"
    );
}

fn assert_top_level_sequence_element_layout(
    layout: &crate::state::StateLayout,
    registry: &VarRegistry,
    var_name: &str,
    expected_bound: SequenceBoundEvidence,
    expected_max_len: usize,
    message: &str,
) {
    let idx = registry
        .get(var_name)
        .unwrap_or_else(|| panic!("{message}: missing var {var_name}"))
        .as_usize();
    match &layout.var_layout(idx).unwrap().kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                },
        } => {
            assert_eq!(*bound, expected_bound, "{message}: {var_name} bound");
            assert_eq!(*max_len, expected_max_len, "{message}: {var_name} capacity");
            assert_undo_entry_record_layout(element_layout, message);
        }
        other => panic!("{message}: expected {var_name} sequence layout, got {other:?}"),
    }
}

fn assert_network_channel_capacity_only_at(
    layout: &crate::state::StateLayout,
    network_idx: usize,
    message: &str,
) {
    match &layout.var_layout(network_idx).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariant {
                            invariant: Arc::from("BoundedNetwork"),
                        },
                        "{message}: invalid TypeOK must leave only capacity evidence"
                    );
                    assert_eq!(*max_len, 3, "{message}: BoundedNetwork capacity");
                }
                other => {
                    panic!("{message}: expected network channel sequence layout, got {other:?}")
                }
            },
            other => panic!("{message}: expected nested network function layout, got {other:?}"),
        },
        other => panic!("{message}: expected recursive network layout, got {other:?}"),
    }
}

fn assert_sequence_network_parent_bounds_observed(
    layout: &crate::state::StateLayout,
    message: &str,
) {
    assert!(
        !layout.supports_flat_primary(),
        "{message}: element-only proof must not make network primary-safe"
    );
    let VarLayoutKind::Recursive {
        layout:
            FlatValueLayout::Sequence {
                bound: network_bound,
                element_layout: row_layout,
                ..
            },
    } = &layout.var_layout(0).unwrap().kind
    else {
        panic!("{message}: expected outer sequence network layout");
    };
    assert_eq!(
        *network_bound,
        SequenceBoundEvidence::Observed,
        "{message}: parent network sequence domain was not proven"
    );
    let FlatValueLayout::Sequence {
        bound: row_bound,
        element_layout: channel_layout,
        ..
    } = row_layout.as_ref()
    else {
        panic!("{message}: expected network row sequence layout, got {row_layout:?}");
    };
    assert_eq!(
        *row_bound,
        SequenceBoundEvidence::Observed,
        "{message}: parent network row sequence domain was not proven"
    );
    let FlatValueLayout::Sequence {
        bound: channel_bound,
        max_len: channel_len,
        element_layout: message_layout,
    } = channel_layout.as_ref()
    else {
        panic!("{message}: expected channel sequence layout, got {channel_layout:?}");
    };
    assert_eq!(
        *channel_bound,
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("BoundedNetwork"),
            element_invariant: Arc::from("TypeOK"),
        },
        "{message}: channel should keep capacity plus element-layout proof"
    );
    assert_eq!(*channel_len, 3, "{message}: channel capacity");
    assert_message_record_layout(message_layout, message);
}

fn assert_single_sequence_bound_observed(layout: &crate::state::StateLayout, message: &str) {
    assert!(
        !layout.supports_flat_primary(),
        "{message}: invalid fixed-domain TypeOK must not activate flat-primary"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    element_layout,
                    ..
                },
        } => {
            assert_eq!(
                *bound,
                SequenceBoundEvidence::Observed,
                "{message}: fixed-domain evidence should not attach"
            );
            assert_eq!(
                element_layout.as_ref(),
                &FlatValueLayout::Scalar(SlotType::Int),
                "{message}: observed integer element layout should remain"
            );
        }
        other => panic!("{message}: expected recursive sequence layout, got {other:?}"),
    }
}

fn assert_fixed_int_sequence_layout(
    layout: &crate::state::StateLayout,
    max_len: usize,
    message: &str,
) {
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    max_len: actual_len,
                    element_layout,
                },
        } => {
            assert_eq!(
                *actual_len, max_len,
                "{message}: fixed-domain sequence length"
            );
            assert_eq!(
                *bound,
                SequenceBoundEvidence::FixedDomainTypeLayout {
                    invariant: Arc::from("TypeOK")
                },
                "{message}: fixed-domain sequence bound"
            );
            assert_eq!(
                element_layout.as_ref(),
                &FlatValueLayout::Scalar(SlotType::Int),
                "{message}: fixed-domain sequence element layout"
            );
        }
        other => panic!("{message}: expected recursive sequence layout, got {other:?}"),
    }
}

fn assert_fixed_nested_int_sequence_layout(
    layout: &crate::state::StateLayout,
    outer_len: usize,
    row_len: usize,
    message: &str,
) {
    let VarLayoutKind::Recursive { layout: req } = &layout.var_layout(0).unwrap().kind else {
        panic!("{message}: expected recursive nested sequence layout");
    };
    let FlatValueLayout::Sequence {
        bound: outer_bound,
        max_len: actual_outer_len,
        element_layout: row_layout,
    } = req
    else {
        panic!("{message}: expected outer sequence layout, got {req:?}");
    };
    assert_eq!(
        *outer_bound,
        SequenceBoundEvidence::FixedDomainTypeLayout {
            invariant: Arc::from("TypeOK")
        },
        "{message}: outer fixed-domain bound"
    );
    assert_eq!(
        *actual_outer_len, outer_len,
        "{message}: outer fixed-domain length"
    );
    let FlatValueLayout::Sequence {
        bound: row_bound,
        max_len: actual_row_len,
        element_layout: cell_layout,
    } = row_layout.as_ref()
    else {
        panic!("{message}: expected row sequence layout, got {row_layout:?}");
    };
    assert_eq!(
        *row_bound,
        SequenceBoundEvidence::FixedDomainTypeLayout {
            invariant: Arc::from("TypeOK")
        },
        "{message}: row fixed-domain bound"
    );
    assert_eq!(
        *actual_row_len, row_len,
        "{message}: row fixed-domain length"
    );
    assert_eq!(
        cell_layout.as_ref(),
        &FlatValueLayout::Scalar(SlotType::Int),
        "{message}: row cell layout"
    );
}

fn assert_sequence_network_type_layout_not_proven(
    layout: &crate::state::StateLayout,
    message: &str,
) {
    assert!(
        !layout.supports_flat_primary(),
        "{message}: invalid type alias must not make network primary-safe"
    );
    let VarLayoutKind::Recursive {
        layout:
            FlatValueLayout::Sequence {
                bound: network_bound,
                element_layout: row_layout,
                ..
            },
    } = &layout.var_layout(0).unwrap().kind
    else {
        panic!("{message}: expected outer sequence network layout");
    };
    assert_eq!(
        *network_bound,
        SequenceBoundEvidence::Observed,
        "{message}: invalid alias must not prove outer network domain"
    );
    let FlatValueLayout::Sequence {
        bound: row_bound,
        element_layout: channel_layout,
        ..
    } = row_layout.as_ref()
    else {
        panic!("{message}: expected network row sequence layout, got {row_layout:?}");
    };
    assert_eq!(
        *row_bound,
        SequenceBoundEvidence::Observed,
        "{message}: invalid alias must not prove row domain"
    );
    let FlatValueLayout::Sequence { bound, max_len, .. } = channel_layout.as_ref() else {
        panic!("{message}: expected channel sequence layout, got {channel_layout:?}");
    };
    assert_eq!(
        *bound,
        SequenceBoundEvidence::ProvenInvariant {
            invariant: Arc::from("BoundedNetwork")
        },
        "{message}: channel should retain capacity-only evidence"
    );
    assert_eq!(*max_len, 3, "{message}: channel capacity");
}

#[test]
fn test_flat_state_primary_still_promotes_scalar_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryScalarActivation ----
VARIABLES x, ok
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let init = scalar_init_state();

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for scalar state");
    assert!(layout.is_all_scalar());
    assert!(layout.is_fully_flat());
    assert!(
        checker.is_flat_state_primary(),
        "verified all-scalar layouts remain flat_state_primary"
    );
}

#[test]
fn test_trace_invariants_disable_flat_bfs_and_primary_storage() {
    let module = parse_module(
        r#"
---- MODULE FlatBfsTraceInvariantGuard ----
VARIABLES x, ok
====
"#,
    );
    let config = Config {
        use_flat_state: Some(true),
        trace_invariants: vec!["HistLenInv".to_string()],
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    let init = scalar_init_state();

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for scalar state");
    assert!(layout.supports_flat_bfs_auto_admission());
    assert!(
        !checker.should_use_flat_bfs(),
        "trace invariants require full parent-chain provenance even when flat state is forced"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "trace invariants must keep full-state storage as the primary trace provenance domain"
    );
}

#[test]
fn test_flat_state_primary_promotes_fixed_record_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryRecordActivation ----
VARIABLE rec
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let init = fixed_record_init_state();

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for record state");
    assert!(
        layout.is_fully_flat(),
        "fixed scalar-field records are complete flat layouts"
    );
    assert!(
        !layout.is_all_scalar(),
        "record layout must exercise the fully-flat non-scalar path"
    );
    assert!(
        checker.is_flat_state_primary(),
        "verified fully-flat records should become flat_state_primary"
    );
}

#[test]
fn test_flat_state_primary_rejects_record_when_flat_bfs_auto_rejects() {
    let registry = VarRegistry::from_names(["rec"]);
    let layout = StateLayout::new(
        &registry,
        vec![VarLayoutKind::Record {
            field_range_proofs: None,
            field_names: vec![Arc::from("count"), Arc::from("label")],
            field_is_bool: vec![false, false],
            field_types: vec![SlotType::Int, SlotType::String],
        }],
    );

    assert!(layout.is_fully_flat());
    assert!(
        !layout.supports_flat_primary(),
        "init-sampled string record slots are not primary-safe without a stronger type proof"
    );
    assert!(
        !layout.supports_flat_bfs_auto_admission(),
        "string-field records stay outside default flat-BFS auto admission"
    );
    assert!(
        !super::flat_state_primary_storage_admitted(
            false,
            false,
            false,
            false,
            true,
            layout.supports_flat_primary(),
        ),
        "primary storage must reject sampled string-field records"
    );
    assert!(
        !super::flat_state_primary_storage_admitted(
            false,
            false,
            false,
            false,
            false,
            layout.supports_flat_primary(),
        ),
        "primary storage still fails closed without a verified fully-flat adapter"
    );
    assert!(
        !super::flat_state_primary_storage_admitted(
            true,
            false,
            false,
            false,
            true,
            layout.supports_flat_primary(),
        ),
        "explicit flat-state disable blocks primary promotion"
    );
}

#[test]
fn test_flat_state_primary_rejects_apalache_variant_record_layout() {
    let registry = VarRegistry::from_names(["result"]);
    let layout = StateLayout::new(
        &registry,
        vec![VarLayoutKind::Record {
            field_range_proofs: None,
            field_names: vec![Arc::from("tag"), Arc::from("value")],
            field_is_bool: vec![false, false],
            field_types: vec![SlotType::String, SlotType::String],
        }],
    );

    assert!(layout.is_fully_flat());
    assert!(
        !layout.supports_flat_primary(),
        "Apalache Variant records can change payload type across tags and must stay out of flat-primary storage"
    );
}

#[test]
fn test_flat_state_primary_promotion_clears_stale_compiled_bfs_artifacts() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryRecordClearsStaleCompiledBfs ----
VARIABLE rec
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    checker.compiled_bfs_step = Some(Box::new(TestCompiledBfsStep { state_len: 1 }));
    checker.compiled_bfs_level = Some(Box::new(TestCompiledBfsLevel));
    {
        checker.trust_cg_build_stats = Some(Default::default());
    }

    checker.infer_flat_state_layout(&fixed_record_init_state());

    assert!(checker.is_flat_state_primary());
    assert!(
        checker.compiled_bfs_step.is_none(),
        "multi-slot flat-primary promotion must drop stale logical-width step"
    );
    assert!(
        checker.compiled_bfs_level.is_none(),
        "multi-slot flat-primary promotion must drop stale logical-width fused level"
    );
    assert!(
        checker.trust_cg_build_stats.is_none(),
        "trust-cg build stats must be cleared with stale layout-sensitive artifacts"
    );
}

#[test]
fn test_fused_level_deferral_requires_flat_primary_step_and_no_constraints() {
    let module = parse_module(
        r#"
---- MODULE FusedLevelDeferralGate ----
VARIABLE rec
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);

    // No flat-primary layout and no step: never defer.
    assert!(
        !checker.should_defer_fused_level_build(),
        "deferral must require a flat-primary layout"
    );

    checker.infer_flat_state_layout(&fixed_record_init_state());
    assert!(checker.is_flat_state_primary());
    assert!(
        !checker.should_defer_fused_level_build(),
        "deferral must require a built per-parent compiled step to drive the loop"
    );

    let flat_slots = checker
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    checker.compiled_bfs_step = Some(Box::new(TestCompiledBfsStep {
        state_len: flat_slots,
    }));
    assert!(
        checker.should_defer_fused_level_build(),
        "flat-primary + step + no constraints/implied actions is the deferral population"
    );

    // Interpreter-evaluated implied actions exclude deferral (the fused level
    // is fenced for them and the post-layout rebuild keys on the level).
    checker
        .compiled
        .eval_implied_actions
        .push(crate::checker_ops::EvalImpliedActionTerm {
            name: "Prop".to_string(),
            expr: Spanned::dummy(tla_core::ast::Expr::Bool(true)),
            truth_if_unchanged: Default::default(),
            vm: None,
        });
    assert!(
        !checker.should_defer_fused_level_build(),
        "implied-action runs must keep the eager fused-level build"
    );

    // State constraints inspect the installed level at setup
    // (state_constrained_native_fused_admission_active): keep the eager build.
    let constrained_config = Config {
        constraints: vec!["Constraint".to_string()],
        ..Default::default()
    };
    let mut constrained = ModelChecker::new(&module, &constrained_config);
    constrained.infer_flat_state_layout(&fixed_record_init_state());
    assert!(constrained.is_flat_state_primary());
    let constrained_slots = constrained
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    constrained.compiled_bfs_step = Some(Box::new(TestCompiledBfsStep {
        state_len: constrained_slots,
    }));
    assert!(
        !constrained.should_defer_fused_level_build(),
        "state-constrained runs must keep the eager fused-level build"
    );
}

#[test]
fn test_compiled_bfs_activation_rejects_stale_flat_width() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryRecordRejectsStaleCompiledWidth ----
VARIABLE rec
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    checker.infer_flat_state_layout(&fixed_record_init_state());
    assert!(checker.is_flat_state_primary());
    let flat_slots = checker
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    assert_ne!(flat_slots, checker.test_vars().len());

    checker.compiled_bfs_step = Some(Box::new(TestCompiledBfsStep {
        state_len: checker.test_vars().len(),
    }));
    assert!(
        !checker.should_use_compiled_bfs(),
        "stale logical-width compiled step must not activate on a flat frontier"
    );

    checker.compiled_bfs_step = Some(Box::new(TestCompiledBfsStep {
        state_len: flat_slots,
    }));
    assert!(
        checker.should_use_compiled_bfs(),
        "matching flat-slot compiled step remains eligible"
    );
}

#[test]
fn test_compiled_bfs_activation_rejects_eval_implied_actions() {
    let module = parse_module(
        r#"
---- MODULE CompiledBfsRejectsEvalImpliedActions ----
VARIABLES x, ok
====
"#,
    );
    let config = Config {
        use_compiled_bfs: Some(true),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.infer_flat_state_layout(&scalar_init_state());
    assert!(checker.is_flat_state_primary());

    let flat_slots = checker
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    checker.compiled_bfs_step = Some(Box::new(TestCompiledBfsStep {
        state_len: flat_slots,
    }));
    assert!(
        checker.should_use_compiled_bfs(),
        "forced compiled BFS should be eligible before eval implied actions are present"
    );

    checker
        .compiled
        .eval_implied_actions
        .push(crate::checker_ops::EvalImpliedActionTerm {
            name: "IA".to_string(),
            expr: Spanned::dummy(Expr::Bool(true)),
            truth_if_unchanged: smallvec::SmallVec::new(),
            vm: None,
        });
    assert!(
        !checker.should_use_compiled_bfs(),
        "eval implied actions require interpreter checks and must fail compiled BFS closed"
    );
}

#[test]
fn test_flat_state_primary_promotes_fixed_array_wavefront_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryArrayActivation ----
VARIABLE arr
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let states = vec![
        fixed_array_init_state([1, 2, 3]),
        fixed_array_init_state([4, 5, 6]),
    ];

    checker.infer_flat_state_layout_from_wavefront(&states);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for fixed-array wavefront");
    assert!(
        layout.is_fully_flat(),
        "fixed integer-indexed arrays are complete flat layouts"
    );
    assert!(
        !layout.is_all_scalar(),
        "array layout must exercise the fully-flat non-scalar path"
    );
    assert!(
        checker.is_flat_state_primary(),
        "verified fully-flat arrays should become flat_state_primary"
    );
}

#[test]
fn test_flat_bfs_auto_admission_rejects_model_value_keyed_function_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatBfsModelValueKeyedFunctionGuard ----
VARIABLE temp
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&model_value_keyed_function_state());

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for model-value keyed function");
    assert!(
        layout.is_fully_flat(),
        "the fixed function layout is representable for the sampled init state"
    );
    assert!(
        !layout.supports_flat_bfs_auto_admission(),
        "init-only model-value keyed function layouts must not auto-admit flat BFS"
    );
    assert!(
        !layout.supports_flat_primary(),
        "init-only model-value keyed function layouts must not become primary flat storage"
    );
    assert!(
        !checker.should_use_flat_bfs(),
        "default flat BFS admission must reject this layout"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "flat-state-primary must not activate without flat-BFS admission"
    );
}

#[test]
fn test_flat_state_primary_rejects_model_value_keyed_function_layout_when_forced() {
    let module = parse_module(
        r#"
---- MODULE FlatBfsForcedModelValueKeyedFunctionGuard ----
VARIABLE temp
====
"#,
    );
    let config = Config {
        use_flat_state: Some(true),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&model_value_keyed_function_state());

    assert!(
        checker.should_use_flat_bfs(),
        "config.use_flat_state=true should still enable the adapter sandwich"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "force-enabled flat BFS must not promote init-only model-value keyed functions to primary storage"
    );
}

#[test]
fn test_should_use_flat_bfs_force_rejects_observed_recursive_sequence() {
    // Regression for the force-flat (`use_flat_state=Some(true)`) dedup
    // over-count. The
    // `network` variable is a recursive `Dom -> Seq` whose sequence capacity is
    // only sampled (`SequenceBoundEvidence::Observed`), so a successor longer
    // than the inferred capacity corrupts the fixed flat buffer and silently
    // breaks dedup, inflating the reported state count. Force-enable must fail
    // closed here even though roundtrip verification of the (short) init state
    // succeeds.
    let module = parse_module(
        r#"
---- MODULE FlatBfsForcedObservedNetworkGuard ----
VARIABLE network
====
"#,
    );
    let config = Config {
        use_flat_state: Some(true),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);

    let init = ArrayState::from_values(vec![observed_network_value()]);
    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for observed network");
    assert!(layout.is_fully_flat());
    assert!(
        !layout.supports_forced_flat_bfs(),
        "sampled-capacity sequence must be refused by the force-enable growth-safety floor"
    );

    // Prove the rejection comes from the growth-safety floor, not a roundtrip
    // failure: the short init state still roundtrips through the flat layout.
    assert!(
        checker
            .flat_bfs_adapter
            .as_ref()
            .expect("flat adapter should be installed")
            .roundtrip_verified(),
        "short init state roundtrips; rejection must come from supports_forced_flat_bfs"
    );
    assert!(
        !checker.should_use_flat_bfs(),
        "force-enabled flat BFS must refuse an observed-capacity recursive sequence"
    );
}

#[test]
fn test_native_fused_admission_rejects_config_only_without_strict_mode() {
    let module = parse_module(
        r#"
---- MODULE NativeFusedFlatFrontierAdmission ----
VARIABLE temp
====
"#,
    );
    let config = Config {
        use_flat_state: Some(true),
        use_compiled_bfs: Some(true),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&model_value_keyed_function_state());

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for model-value keyed function");
    assert!(layout.is_fully_flat());
    assert!(
        !checker.is_flat_state_primary(),
        "model-value keyed function slots remain unsafe as global primary storage"
    );
    assert!(
        !checker.native_fused_flat_frontier_admission_candidate(),
        "config-only compiled+flat BFS must not admit non-primary flat frontiers; \
         non-strict mode can fall back to unsafe Rust paths"
    );
    assert!(
        !checker.should_use_compiled_bfs(),
        "non-strict config-only admission must fail closed"
    );

    checker.compiled_bfs_level = Some(Box::new(TestCompiledBfsLevel));
    assert!(
        !checker.should_use_compiled_bfs(),
        "prototype fused levels must not admit non-primary flat frontiers"
    );

    checker.compiled_bfs_level = Some(Box::new(TestNativeFusedBfsLevel));
    assert!(
        !checker.should_use_compiled_bfs(),
        "even native fused levels must wait for strict native-fused mode before \
         consuming non-primary flat frontiers"
    );
}

#[test]
fn test_native_install_gate_summary_mapping_tracks_exact_pins_and_clean_ancestry() {
    const REQUESTED_AY_REV: &str = "0adeaab4d66b1414a95ab5cee4ec64078c9dbd97";
    const REQUESTED_CLEAN_REV: &str = "659a6eeb15b29f7d739ecca852a77483fcfd88ea";
    const STANDALONE_CLEAN_FALLBACK_REV: &str =
        "659a6eeb15b29f7d739ecca852a77483fcfd88ea";
    const REQUESTED_TRUST_IR_REV: &str = "9de13453d69f84f24556bd75636bf020206f33c9";
    const REQUESTED_TRUST_CG_REV: &str = "7005df3c00a3e1b4042cc49a6608feb1aaa1bfec";
    const AUDITED_CLEAN_AY_PATH_PACKAGES: &[&str] = &[
        "ay",
        "ay-allsat",
        "ay-arrays",
        "ay-bv",
        "ay-chc",
        "ay-core",
        "ay-count",
        "ay-diff-logic",
        "ay-dispatch",
        "ay-dpll",
        "ay-drat-check",
        "ay-dt",
        "ay-euf",
        "ay-fp",
        "ay-frontend",
        "ay-intsat",
        "ay-jit",
        "ay-lia",
        "ay-lra",
        "ay-map",
        "ay-milp",
        "ay-model-check",
        "ay-multiset",
        "ay-nia",
        "ay-nonlinear-common",
        "ay-nra",
        "ay-prefetch",
        "ay-proof",
        "ay-proof-common",
        "ay-sat",
        "ay-sat-congruence-core",
        "ay-seq",
        "ay-set",
        "ay-strings",
        "ay-sys",
        "ay-translate",
    ];
    let root_cargo = include_str!("../../../../../Cargo.toml");
    let root_lock = include_str!("../../../../../Cargo.lock");
    let tla_check_cargo = include_str!("../../../Cargo.toml");
    let dockerfile = include_str!("../../../../../mcc/Dockerfile.mcc");
    let benchkit = include_str!("../../../../../mcc/BenchKit_head.sh");
    let lock_preflight = include_str!("../../../../../mcc/first_party_lock_preflight.sh");
    let root_manifest: toml::Value =
        toml::from_str(root_cargo).expect("TY root Cargo.toml must parse");
    let workspace_dependencies = root_manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(toml::Value::as_table)
        .expect("TY root must retain [workspace.dependencies]");
    let assert_exact_git_dependency = |dep_name: &str, repo: &str, rev: &str| {
        let dependency = workspace_dependencies
            .get(dep_name)
            .and_then(toml::Value::as_table)
            .unwrap_or_else(|| panic!("TY dependency `{dep_name}` must be a table"));
        let expected_git = format!("https://github.com/alabsystems/{repo}.git");
        assert_eq!(
            dependency.get("git").and_then(toml::Value::as_str),
            Some(expected_git.as_str()),
            "TY dependency `{dep_name}` must use the canonical `{repo}` repository"
        );
        assert_eq!(
            dependency.get("rev").and_then(toml::Value::as_str),
            Some(rev),
            "TY dependency `{dep_name}` must use the audited exact `{repo}` rev"
        );
        assert!(
            !dependency.contains_key("path")
                && !dependency.contains_key("branch")
                && !dependency.contains_key("tag"),
            "TY dependency `{dep_name}` must not mix its exact Git source with path/branch/tag selectors"
        );
    };
    for dep_name in [
        "ay",
        "ay-dpll",
        "ay-core",
        "ay-proof",
        "ay-allsat",
        "ay-chc",
        "ay-sat",
        "ay-lrat-check",
        "ay-frontend",
        "ay-encode",
    ] {
        assert_exact_git_dependency(dep_name, "ay", REQUESTED_AY_REV);
    }
    for dep_name in ["trust-ir", "trust-ir-build"] {
        assert_exact_git_dependency(dep_name, "trust-ir", REQUESTED_TRUST_IR_REV);
    }
    for dep_name in [
        "trust-cg-codegen",
        "trust-cg-ir",
        "trust-cg-lower",
        "trust-cg-opt",
        "trust-cg-jit-matrix",
    ] {
        assert_exact_git_dependency(dep_name, "trust-cg", REQUESTED_TRUST_CG_REV);
    }
    let clean_patch = root_manifest
        .get("patch")
        .and_then(|patch| patch.get("https://github.com/alabsystems/clean.git"))
        .and_then(toml::Value::as_table)
        .expect("TY root must patch the complete canonical Clean source");
    for dep_name in [
        "clean-ck0",
        "clean-elab",
        "clean-kernel",
        "clean-mathverse",
        "clean-olean",
        "clean-parser",
    ] {
        assert_exact_git_dependency(dep_name, "clean", STANDALONE_CLEAN_FALLBACK_REV);
        let patched_dependency = clean_patch
            .get(dep_name)
            .and_then(toml::Value::as_table)
            .unwrap_or_else(|| panic!("TY-root patch must include Clean package `{dep_name}`"));
        let expected_path = format!("../clean/crates/{dep_name}");
        assert_eq!(
            patched_dependency.get("path").and_then(toml::Value::as_str),
            Some(expected_path.as_str()),
            "TY-root patch must redirect Clean package `{dep_name}` to the audited sibling"
        );
        assert!(
            patched_dependency.get("git").is_none()
                && patched_dependency.get("rev").is_none()
                && patched_dependency.get("branch").is_none()
                && patched_dependency.get("tag").is_none(),
            "TY-root Clean patch `{dep_name}` must contain only its local path authority"
        );
    }
    assert!(
        !tla_check_cargo.contains("../../../clean"),
        "tla-check must not retain an escaping Clean path that breaks Cargo Git consumers"
    );
    let root_lock_document: toml::Value =
        toml::from_str(root_lock).expect("TY root Cargo.lock must parse");
    let locked_packages = root_lock_document
        .get("package")
        .and_then(toml::Value::as_array)
        .expect("TY root Cargo.lock must contain packages");
    for package_name in [
        "clean-ck0",
        "clean-elab",
        "clean-kernel",
        "clean-mathverse",
        "clean-olean",
        "clean-parser",
    ] {
        let matches = locked_packages
            .iter()
            .filter_map(toml::Value::as_table)
            .filter(|package| {
                package.get("name").and_then(toml::Value::as_str) == Some(package_name)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "TY-root lock must contain exactly one Clean package `{package_name}`"
        );
        assert!(
            matches[0].get("source").is_none(),
            "TY-root Clean package `{package_name}` must resolve only through the audited local patch"
        );
    }
    for package in locked_packages.iter().filter_map(toml::Value::as_table) {
        let Some(package_name) = package.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        if package_name == "clean" || package_name.starts_with("clean-") {
            assert!(
                package.get("source").is_none(),
                "TY-root Clean-family package `{package_name}` must resolve only through the audited local checkout, observed source {:?}",
                package.get("source").and_then(toml::Value::as_str)
            );
        }
        if let Some(source) = package.get("source").and_then(toml::Value::as_str) {
            assert!(
                !source.contains("alabsystems/clean"),
                "TY-root lock must not retain any package from a Clean Git source: `{package_name}` uses `{source}`"
            );
        }
    }
    for (repo, rev) in [
        ("ay", REQUESTED_AY_REV),
        ("trust-ir", REQUESTED_TRUST_IR_REV),
        ("trust-cg", REQUESTED_TRUST_CG_REV),
    ] {
        let expected_source = format!(
            "git+https://github.com/alabsystems/{repo}.git?rev={rev}#{rev}"
        );
        let is_repo_package = |name: &str| match repo {
            "ay" => name == "ay" || name.starts_with("ay-"),
            "trust-ir" => name == "trust-ir" || name.starts_with("trust-ir-"),
            "trust-cg" => name == "trust-cg" || name.starts_with("trust-cg-"),
            _ => false,
        };
        let observed_packages = locked_packages
            .iter()
            .filter_map(toml::Value::as_table)
            .filter(|package| {
                package
                    .get("name")
                    .and_then(toml::Value::as_str)
                    .is_some_and(|name| is_repo_package(name))
            })
            .collect::<Vec<_>>();
        assert!(
            !observed_packages.is_empty(),
            "Cargo.lock must contain packages from the exact `{repo}` source identity"
        );
        let mut exact_source_count = 0;
        let mut path_ay_packages = Vec::new();
        for package in observed_packages {
            let package_name = package
                .get("name")
                .and_then(toml::Value::as_str)
                .expect("locked package name must be a string");
            let source = package.get("source").and_then(toml::Value::as_str);
            if source == Some(expected_source.as_str()) {
                exact_source_count += 1;
                continue;
            }
            if repo == "ay" && source.is_none() {
                assert_eq!(
                    package.get("version").and_then(toml::Value::as_str),
                    Some("0.1.0"),
                    "audited Clean cycle-boundary AY package `{package_name}` must retain version 0.1.0"
                );
                path_ay_packages.push(package_name);
                continue;
            }
            assert_eq!(
                source,
                Some(expected_source.as_str()),
                "Cargo.lock `{repo}` package `{package_name}` must have the one exact canonical source"
            );
        }
        assert!(
            exact_source_count > 0,
            "Cargo.lock must retain at least one package from the exact `{repo}` Git source"
        );
        if repo == "ay" {
            path_ay_packages.sort_unstable();
            assert_eq!(
                path_ay_packages, AUDITED_CLEAN_AY_PATH_PACKAGES,
                "TY-root source-less AY packages must exactly equal the audited Clean cycle boundary"
            );
        }
    }
    for (arg, rev) in [
        ("AY_REV", REQUESTED_AY_REV),
        ("CLEAN_REV", REQUESTED_CLEAN_REV),
        ("CLEAN_FALLBACK_REV", STANDALONE_CLEAN_FALLBACK_REV),
        ("TRUST_IR_REV", REQUESTED_TRUST_IR_REV),
        ("TRUST_CG_REV", REQUESTED_TRUST_CG_REV),
    ] {
        assert!(
            dockerfile.contains(&format!("ARG {arg}={rev}")),
            "MCC Dockerfile `{arg}` must match the audited workspace pin"
        );
    }
    assert!(
        benchkit.contains(&format!("TY_MCC_PACKAGED_AY_REV:={REQUESTED_AY_REV}")),
        "BenchKit packaged AY fallback must match the audited workspace pin"
    );
    assert!(
        dockerfile.contains(&format!(
            "test \"$CLEAN_REV\" = \"{REQUESTED_CLEAN_REV}\""
        )),
        "MCC Dockerfile must reject a Clean build-arg override that differs from the committed authority"
    );
    assert!(
        dockerfile.contains(&format!(
            "test \"$CLEAN_FALLBACK_REV\" = \"{STANDALONE_CLEAN_FALLBACK_REV}\""
        )),
        "MCC Dockerfile must reject an override of the immutable standalone Clean cycle-cut"
    );
    assert!(
        dockerfile.contains("--bin ty-mcc-ay-pin-validate")
            && dockerfile.contains("ty-mcc-ay-pin-validate --repo-root /src"),
        "MCC image build must execute the structural AY manifest/lock/Docker pin validator"
    );
    let early_preflight = dockerfile
        .find("mcc/first_party_lock_preflight.sh Cargo.lock")
        .expect("MCC image must run the dependency-free lock preflight");
    let cargo_build = dockerfile
        .find("cargo build --profile agent")
        .expect("MCC image must retain its workspace build");
    assert!(
        early_preflight < cargo_build,
        "dependency-free source validation must run before Cargo compiles dependency code"
    );
    for family_pattern in [
        "'^ay($|-)'",
        "'^trust-ir($|-)'",
        "'^trust-cg($|-)'",
        "'^clean($|-)'",
    ] {
        assert!(
            lock_preflight.contains(family_pattern),
            "early lock preflight must classify the complete `{family_pattern}` package family"
        );
    }
    let audited_path_assignment = format!(
        "audited_clean_ay_path_packages='{}'",
        AUDITED_CLEAN_AY_PATH_PACKAGES.join(",")
    );
    assert!(
        lock_preflight.contains(&audited_path_assignment),
        "Rust and shell guards must carry the same exact Clean-to-AY cycle-boundary allowlist"
    );
    assert!(
        lock_preflight.contains("    '0.1.0'"),
        "early lock preflight must enforce the audited AY cycle-boundary version"
    );

    assert!(
        REQUESTED_CLEAN_REV.len() == 40
            && REQUESTED_CLEAN_REV
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "audited Clean revision must be lowercase 40-hex"
    );
    let clean_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../clean")
        .canonicalize()
        .expect("effective sibling Clean checkout must exist");
    let git_clean = |args: &[&str]| -> String {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(&clean_root)
            .args(args)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to inspect effective Clean checkout {}: {error}",
                    clean_root.display()
                )
            });
        assert!(
            output.status.success(),
            "git -C {} {} failed: {}",
            clean_root.display(),
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("Clean git metadata must be UTF-8")
            .trim()
            .to_string()
    };
    let clean_head = git_clean(&["rev-parse", "HEAD"]);
    // The standalone MCC image and package graph retain the immutable Clean
    // cycle-cut above, while an umbrella checkout follows the current public
    // Clean main and records that exact descendant in its own gitlink. Requiring
    // the frozen content pin to remain public main made this test inevitably
    // stale as soon as Clean advanced. Preserve both authorities explicitly:
    // current sibling/public-main identity below, and frozen ancestry here.
    git_clean(&[
        "merge-base",
        "--is-ancestor",
        REQUESTED_CLEAN_REV,
        clean_head.as_str(),
    ]);
    let clean_status = git_clean(&["status", "--porcelain=v1", "--untracked-files=normal"]);
    assert!(
        clean_status.is_empty(),
        "effective sibling Clean worktree must be clean, observed:\n{clean_status}"
    );
    let clean_origin = git_clean(&["remote", "get-url", "origin"]);
    let clean_git_root = std::path::PathBuf::from(git_clean(&["rev-parse", "--show-toplevel"]))
        .canonicalize()
        .expect("Clean git toplevel must resolve");
    assert_eq!(
        clean_git_root, clean_root,
        "effective sibling Clean path must be the root of its own checkout"
    );
    assert!(
        matches!(
            clean_origin.trim_end_matches('/'),
            "https://github.com/alabsystems/clean"
                | "https://github.com/alabsystems/clean.git"
                | "ssh://git@github.com/alabsystems/clean"
                | "ssh://git@github.com/alabsystems/clean.git"
                | "git@github.com:alabsystems/clean"
                | "git@github.com:alabsystems/clean.git"
        ),
        "effective sibling Clean origin must be a secure canonical alabsystems/clean URL, observed `{clean_origin}`"
    );
    let clean_remote_main = git_clean(&["rev-parse", "refs/remotes/origin/main"]);
    assert_eq!(
        clean_head, clean_remote_main,
        "effective sibling Clean checkout must be bidirectionally aligned with fetched origin/main; the standalone content pin is validated separately as an ancestor"
    );

    struct AdmissionRowMapping {
        row_code: &'static str,
        summary_fields: &'static [&'static str],
        consumer_evidence_fields: &'static [&'static str],
    }

    let required_rows = [
        AdmissionRowMapping {
            row_code: "artifact_digest",
            summary_fields: &[
                "packet_hash",
                "persisted_packet_hash",
                "artifact_id",
                "manifest_checksum",
                "source_sha256",
                "trust_ir_sha256",
                "native_payload_sha256",
                "target_checksum",
            ],
            consumer_evidence_fields: &[],
        },
        AdmissionRowMapping {
            row_code: "abi_layout",
            summary_fields: &[
                "abi_checksum",
                "layout_checksum",
                "proof_policy_checksum",
                "invalidation_checksum",
            ],
            consumer_evidence_fields: &[
                "target_checksum",
                "proof_policy_checksum",
                "layout_checksum",
                "invalidation_checksum",
                "runtime_generation",
            ],
        },
        AdmissionRowMapping {
            row_code: "replay",
            summary_fields: &[
                "replay_root_sha256",
                "install_consumer_verdict_sha256",
                "telemetry_event_id",
                "telemetry_record_sha256",
                "admission_evidence_sha256",
            ],
            consumer_evidence_fields: &[
                "telemetry_event_id",
                "telemetry_record_sha256",
                "replay_root_sha256",
                "install_consumer_verdict_sha256",
                "evidence_sha256",
            ],
        },
        AdmissionRowMapping {
            row_code: "rollback",
            summary_fields: &[
                "disposition",
                "reason_code",
                "requested_authority",
                "install_authority",
                "actions",
                "useful_native_delta",
            ],
            consumer_evidence_fields: &["rollback_ready", "status_ready", "deopt_ready"],
        },
    ];

    assert_eq!(
        required_rows
            .iter()
            .map(|row| row.row_code)
            .collect::<Vec<_>>(),
        ["artifact_digest", "abi_layout", "replay", "rollback"],
        "native/JIT promotion must stay blocked until every required row is present"
    );
    assert!(
        required_rows
            .iter()
            .all(|row| !row.summary_fields.is_empty()),
        "each native install admission row must map to summary fields"
    );
    assert!(
        required_rows.iter().any(|row| {
            row.row_code == "rollback"
                && row.summary_fields.contains(&"reason_code")
                && row.consumer_evidence_fields.contains(&"rollback_ready")
                && row.consumer_evidence_fields.contains(&"status_ready")
                && row.consumer_evidence_fields.contains(&"deopt_ready")
        }),
        "TY activation must consume trust-codegen reason-code helpers and block without rollback/status/deopt readiness"
    );

    {
        fn assert_summary_type<T>() {}
        assert_summary_type::<tla_trust_cg::NativeInstallGateAdmissionSummary>();
        assert_eq!(
            tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA,
            "trust-cg.phase6.native_install_gate.admission_summary.v1"
        );
        assert_eq!(
            tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA_VERSION,
            1
        );
    }
}

#[test]
fn test_native_fused_strict_admission_rejects_model_value_keyed_function_layout() {
    let module = parse_module(
        r#"
---- MODULE NativeFusedStrictModelValueKeyedFunctionGuard ----
VARIABLE temp
====
"#,
    );
    let config = Config {
        use_flat_state: Some(true),
        use_compiled_bfs: Some(true),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&model_value_keyed_function_state());
    checker.compiled_bfs_level = Some(Box::new(TestNativeFusedBfsLevel));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for model-value keyed function");
    assert!(layout.is_fully_flat());
    assert!(
        !layout.supports_flat_bfs_auto_admission(),
        "init-only model-value keyed function layouts can later store non-scalar range values"
    );
    assert!(
        !checker.native_fused_flat_frontier_admission_candidate_for_strict(true),
        "strict native-fused mode must not bypass the flat-admission guard for \
         Dijkstra-shaped model-value function slots"
    );
}

#[test]
fn test_flat_state_primary_promotes_typeok_tagged_scalar_set_layout() {
    let module = parse_module(
        r#"
---- MODULE NativeFusedStrictTaggedScalarSetAdmission ----
CONSTANTS Proc, NoOwner
VARIABLE owner
TypeOK == owner \in [Proc -> ({NoOwner} \union SUBSET Proc)]
====
"#,
    );
    let mut config = Config {
        use_flat_state: Some(true),
        ..Default::default()
    };
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );
    config.add_constant("NoOwner".to_string(), ConstantValue::ModelValue);
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![Value::Func(Rp::new(
        FuncValue::from_sorted_entries(vec![
            (
                Value::ModelValue(Rp::from("p1")),
                Value::ModelValue(Rp::from("NoOwner")),
            ),
            (
                Value::ModelValue(Rp::from("p2")),
                Value::ModelValue(Rp::from("NoOwner")),
            ),
        ]),
    ))]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for tagged scalar/set function");
    assert!(layout.has_model_value_keyed_tagged_scalar_set_range());
    assert!(layout.supports_flat_bfs_auto_admission());
    assert!(
        layout.supports_flat_primary(),
        "tagged scalar/set proof metadata should make the fixed function primary-safe"
    );
    assert!(
        checker.is_flat_state_primary(),
        "roundtrip-verified tagged scalar/set layouts should activate flat_state_primary"
    );
    assert!(
        checker.compiled_bfs_flat_frontier_admitted(),
        "compiled BFS may consume the verified tagged scalar/set layout through primary flat storage"
    );
}

#[test]
fn test_action_producer_promotes_dijkstra_temp_tagged_scalar_set_layout() {
    let module = parse_module(
        r#"
---- MODULE ActionProducerTaggedScalarSetAdmission ----
CONSTANTS Proc, defaultInitValue
VARIABLES b, c, k, pc, temp

ProcAlias == Proc

Li0(self) ==
    /\ pc[self] = "Li0"
    /\ b' = [b EXCEPT ![self] = FALSE]
    /\ pc' = [pc EXCEPT ![self] = "Li1"]
    /\ UNCHANGED << c, k, temp >>

Li1(self) ==
    /\ pc[self] = "Li1"
    /\ IF k # self
          THEN pc' = [pc EXCEPT ![self] = "Li2"]
          ELSE pc' = [pc EXCEPT ![self] = "Li4a"]
    /\ UNCHANGED << b, c, k, temp >>

Li2(self) ==
    /\ pc[self] = "Li2"
    /\ c' = [c EXCEPT ![self] = TRUE]
    /\ pc' = [pc EXCEPT ![self] = "Li3a"]
    /\ UNCHANGED << b, k, temp >>

Li3a(self) ==
    /\ pc[self] = "Li3a"
    /\ temp' = [temp EXCEPT ![self] = k]
    /\ pc' = [pc EXCEPT ![self] = "Li3b"]
    /\ UNCHANGED << b, c, k >>

Li3b(self) ==
    /\ pc[self] = "Li3b"
    /\ IF b[temp[self]]
          THEN pc' = [pc EXCEPT ![self] = "Li3c"]
          ELSE pc' = [pc EXCEPT ![self] = "Li3d"]
    /\ UNCHANGED << b, c, k, temp >>

Li3c(self) ==
    /\ pc[self] = "Li3c"
    /\ k' = self
    /\ pc' = [pc EXCEPT ![self] = "Li3d"]
    /\ UNCHANGED << b, c, temp >>

Li3d(self) ==
    /\ pc[self] = "Li3d"
    /\ pc' = [pc EXCEPT ![self] = "Li1"]
    /\ UNCHANGED << b, c, k, temp >>

Li4a(self) ==
    /\ pc[self] = "Li4a"
    /\ c' = [c EXCEPT ![self] = FALSE]
    /\ temp' = [temp EXCEPT ![self] = ProcAlias \ {self}]
    /\ pc' = [pc EXCEPT ![self] = "Li4b"]
    /\ UNCHANGED << b, k >>

Li4b(self) ==
    /\ pc[self] = "Li4b"
    /\ IF temp[self] # {}
          THEN \E j \in temp[self]:
                   /\ temp' = [temp EXCEPT ![self] = temp[self] \ {j}]
                   /\ IF ~c[j]
                         THEN pc' = [pc EXCEPT ![self] = "Li1"]
                         ELSE pc' = [pc EXCEPT ![self] = "Li4b"]
          ELSE /\ pc' = [pc EXCEPT ![self] = "cs"]
               /\ temp' = temp
    /\ UNCHANGED << b, c, k >>

cs(self) ==
    /\ pc[self] = "cs"
    /\ TRUE
    /\ pc' = [pc EXCEPT ![self] = "Li5"]
    /\ UNCHANGED << b, c, k, temp >>

Li5(self) ==
    /\ pc[self] = "Li5"
    /\ c' = [c EXCEPT ![self] = TRUE]
    /\ pc' = [pc EXCEPT ![self] = "Li6"]
    /\ UNCHANGED << b, k, temp >>

Li6(self) ==
    /\ pc[self] = "Li6"
    /\ b' = [b EXCEPT ![self] = TRUE]
    /\ pc' = [pc EXCEPT ![self] = "ncs"]
    /\ UNCHANGED << c, k, temp >>

ncs(self) ==
    /\ pc[self] = "ncs"
    /\ TRUE
    /\ pc' = [pc EXCEPT ![self] = "Li0"]
    /\ UNCHANGED << b, c, k, temp >>

TypeOK == TRUE
====
"#,
    );
    let mut config = Config {
        use_flat_state: Some(true),
        ..Default::default()
    };
    config.invariants.push("TypeOK".to_string());
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::Value(r#"{"p1", "p2"}"#.to_string()),
    );
    config.add_constant(
        "defaultInitValue".to_string(),
        ConstantValue::Value(r#""defaultInitValue""#.to_string()),
    );
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);
    seed_split_action_meta(
        &mut checker,
        &[
            "Li0", "Li1", "Li2", "Li3a", "Li3b", "Li3c", "Li3d", "Li4a", "Li4b", "cs", "Li5",
            "Li6", "ncs",
        ],
    );
    checker.compile_action_bytecode();

    let proc_entries = |value: Value| {
        Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::String(Rp::from("p1")), value.clone()),
            (Value::String(Rp::from("p2")), value),
        ])))
    };
    let init = ArrayState::from_values(vec![
        proc_entries(Value::Bool(true)),
        proc_entries(Value::Bool(true)),
        Value::String(Rp::from("p1")),
        proc_entries(Value::String(Rp::from("Li0"))),
        proc_entries(Value::String(Rp::from("defaultInitValue"))),
    ]);

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for Dijkstra-shaped temp");
    let temp_idx = checker
        .ctx
        .var_registry()
        .get("temp")
        .expect("temp variable index")
        .as_usize();
    let temp_proof = layout
        .var_layout(temp_idx)
        .and_then(|var| var.kind.tagged_scalar_set_range_proof())
        .expect("temp should carry an action-producer tagged scalar/set proof");
    assert!(
        layout.has_model_value_keyed_tagged_scalar_set_range(),
        "action producer proof should promote temp from generic scalar slots"
    );
    assert_eq!(temp_proof.scalar_type(), SlotType::String);
    assert_eq!(temp_proof.set_universe().len(), 2);
    assert!(
        temp_proof.source().starts_with("action-producer:"),
        "proof should be tied to action-local producer evidence"
    );
    let jit_layout = crate::state::check_layout_to_jit_layout(layout);
    match jit_layout.var_layout(temp_idx) {
        Some(tla_jit_abi::VarLayout::Compound(tla_jit_abi::CompoundLayout::Function {
            value_layout,
            ..
        })) => match value_layout.as_ref() {
            tla_jit_abi::CompoundLayout::TaggedScalarOrSet {
                scalar_kind,
                set_universe,
                ..
            } => {
                assert_eq!(*scalar_kind, tla_jit_abi::ScalarSlotKind::String);
                assert_eq!(set_universe.len(), 2);
            }
            other => panic!("temp range should bridge to TaggedScalarOrSet, got {other:?}"),
        },
        other => panic!("temp should bridge as a compound function, got {other:?}"),
    }
    assert!(
        !checker.is_flat_state_primary(),
        "the mixed Dijkstra layout remains non-primary because pc is a legacy scalar function"
    );
    assert!(
        checker.native_fused_flat_frontier_admission_candidate_for_strict(true),
        "strict native-fused build may request the flat slot width once the tagged temp proof is present"
    );

    let flat_slots = checker
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    checker.compiled_bfs_level = Some(Box::new(TestNativeFusedBfsLevel));
    assert!(
        !checker.native_fused_flat_frontier_admission_active_for_strict(true),
        "non-primary compact frontiers must not admit a native level that falls back to Rust invariants"
    );

    checker.compiled_bfs_level = Some(Box::new(TestNativeFusedInvariantBfsLevel {
        state_len: flat_slots - 1,
    }));
    assert!(
        !checker.native_fused_flat_frontier_admission_active_for_strict(true),
        "native-fused admission must prove the exact flat slot width"
    );

    checker.compiled_bfs_level = Some(Box::new(TestNativeFusedInvariantBfsLevel {
        state_len: flat_slots,
    }));
    assert!(
        !checker.native_fused_flat_frontier_admission_active_for_strict(true),
        "non-primary invariant-only flat-frontier native-fused activation stays fail-closed until #4433 proves parent-loop parity"
    );
}

#[test]
fn test_native_fused_strict_admits_dijkstra_model_value_pc_with_fixed_scalar_proof() {
    let module = parse_module(
        r#"
---- MODULE NativeFusedStrictDijkstraPcAdmission ----
CONSTANTS Proc, defaultInitValue
VARIABLES b, c, k, pc, temp

PcLabels == {"Li0", "Li1", "Li4a", "Li4b", "cs"}

TempOnlyTypeOK ==
    /\ b \in [Proc -> {TRUE, FALSE}]
    /\ c \in [Proc -> {TRUE, FALSE}]
    /\ k \in Proc
    /\ temp \in [Proc -> ({defaultInitValue} \union SUBSET Proc)]

TypeOK ==
    /\ TempOnlyTypeOK
    /\ pc \in [Proc -> PcLabels]
====
"#,
    );

    let dijkstra_init = || {
        let proc_entries = |value: Value| {
            Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
                (Value::ModelValue(Rp::from("p1")), value.clone()),
                (Value::ModelValue(Rp::from("p2")), value),
            ])))
        };
        ArrayState::from_values(vec![
            proc_entries(Value::Bool(true)),
            proc_entries(Value::Bool(true)),
            Value::ModelValue(Rp::from("p1")),
            proc_entries(Value::String(Rp::from("Li0"))),
            proc_entries(Value::ModelValue(Rp::from("defaultInitValue"))),
        ])
    };

    let make_config = |invariant: &str| {
        let mut config = Config {
            use_flat_state: Some(true),
            use_compiled_bfs: Some(true),
            ..Default::default()
        };
        config.add_constant(
            "Proc".to_string(),
            ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
        );
        config.add_constant("defaultInitValue".to_string(), ConstantValue::ModelValue);
        config.invariants.push(invariant.to_string());
        config
    };

    let unproven_config = make_config("TempOnlyTypeOK");
    let mut unproven_pc = ModelChecker::new(&module, &unproven_config);
    bind_constants_from_config(&mut unproven_pc.ctx, &unproven_config)
        .expect("config constants bind");
    precompute_constant_operators(&mut unproven_pc.ctx);
    promote_env_constants_to_precomputed(&mut unproven_pc.ctx);
    unproven_pc.infer_flat_state_layout(&dijkstra_init());
    assert!(
        !unproven_pc.native_fused_flat_frontier_admission_candidate_for_strict(true),
        "model-value-keyed scalar pc must stay rejected when only temp is proven"
    );

    let proven_config = make_config("TypeOK");
    let mut proven_pc = ModelChecker::new(&module, &proven_config);
    bind_constants_from_config(&mut proven_pc.ctx, &proven_config).expect("config constants bind");
    precompute_constant_operators(&mut proven_pc.ctx);
    promote_env_constants_to_precomputed(&mut proven_pc.ctx);
    proven_pc.infer_flat_state_layout(&dijkstra_init());
    let layout = proven_pc
        .flat_state_layout()
        .expect("Dijkstra-shaped layout should be inferred");
    let pc_idx = proven_pc
        .ctx
        .var_registry()
        .get("pc")
        .expect("pc variable index")
        .as_usize();
    let temp_idx = proven_pc
        .ctx
        .var_registry()
        .get("temp")
        .expect("temp variable index")
        .as_usize();
    let pc_proof = layout
        .var_layout(pc_idx)
        .and_then(|var| var.kind.fixed_scalar_range_proof())
        .expect("pc should carry a fixed scalar string range proof");
    assert_eq!(pc_proof.scalar_type(), SlotType::String);
    assert_eq!(pc_proof.scalar_universe().len(), 5);
    assert!(
        layout
            .var_layout(temp_idx)
            .and_then(|var| var.kind.tagged_scalar_set_range_proof())
            .is_some(),
        "temp should remain backed by scalar/set range proof"
    );
    // b298fe51 widened `fixed_scalar_range_primary_proof` to admit Bool-ranged fixed-scalar
    // function layouts (Dijkstra's `b`/`c` are `[Proc -> {TRUE, FALSE}]`). With every variable
    // proven, the whole layout is now flat-state PRIMARY — not a non-primary strict
    // flat-frontier admission proof. The pc/temp proofs above still hold; only the layout's
    // classification (and hence the frontier-admission gates) moves to the primary semantics.
    assert!(
        proven_pc.is_flat_state_primary(),
        "with pc/temp and the Bool-ranged b/c proven, the Dijkstra layout is flat-state primary"
    );
    assert!(
        !proven_pc.native_fused_flat_frontier_admission_candidate_for_strict(true),
        "a flat-primary layout is rejected by the non-primary strict flat-frontier admission gate"
    );
    assert!(
        proven_pc.compiled_bfs_step_intermediate_artifact_needed_for_strict(true),
        "a flat-primary strict native-fused run builds the per-parent intermediate step"
    );

    let flat_slots = proven_pc
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    proven_pc.compiled_bfs_level = Some(Box::new(TestNativeFusedInvariantBfsLevel {
        state_len: flat_slots,
    }));
    assert!(
        !proven_pc.native_fused_flat_frontier_admission_active_for_strict(true),
        "a flat-primary layout is not admitted through the non-primary strict flat-frontier path"
    );
    assert_eq!(
        proven_pc.native_fused_non_primary_flat_frontier_parent_loop_rejection_for_strict(true),
        None,
        "a flat-primary layout short-circuits the non-primary parent-loop rejection to None"
    );

    let mut constrained_config = make_config("TypeOK");
    constrained_config
        .constraints
        .push("StateConstraint".to_string());
    let mut constrained_pc = ModelChecker::new(&module, &constrained_config);
    bind_constants_from_config(&mut constrained_pc.ctx, &constrained_config)
        .expect("config constants bind");
    precompute_constant_operators(&mut constrained_pc.ctx);
    promote_env_constants_to_precomputed(&mut constrained_pc.ctx);
    constrained_pc.infer_flat_state_layout(&dijkstra_init());
    assert!(
        !constrained_pc.compiled_bfs_step_intermediate_artifact_needed_for_strict(true),
        "state-constrained native-fused runs must build the backend-validated level directly"
    );
    let constrained_flat_slots = constrained_pc
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    constrained_pc.compiled_bfs_level = Some(Box::new(TestNativeFusedConstrainedBfsLevel {
        state_len: constrained_flat_slots,
        state_constraint_count: 1,
    }));
    assert_eq!(
        constrained_pc.native_fused_non_primary_flat_frontier_parent_loop_rejection_for_strict(
            true
        ),
        None,
        "a flat-primary layout short-circuits the non-primary parent-loop rejection to None even with a state constraint"
    );
    assert!(
        !constrained_pc.native_fused_flat_frontier_admission_active_for_strict(true),
        "a flat-primary layout is not admitted through the non-primary strict flat-frontier path (the state-constrained non-primary path is bypassed)"
    );
}

#[test]
fn test_native_fused_strict_discovers_dijkstra_fixed_scalar_ranges_without_typeok() {
    let module = parse_module(
        r#"
---- MODULE NativeFusedStrictDijkstraConfigShape ----
CONSTANTS Proc, defaultInitValue
VARIABLES b, c, k, pc, temp

vars == << b, c, k, pc, temp >>
ProcSet == Proc

Init ==
    /\ b = [i \in Proc |-> TRUE]
    /\ c = [i \in Proc |-> TRUE]
    /\ k \in Proc
    /\ temp = [self \in Proc |-> defaultInitValue]
    /\ pc = [self \in ProcSet |-> "Li0"]

Li0(self) ==
    /\ pc[self] = "Li0"
    /\ b' = [b EXCEPT ![self] = FALSE]
    /\ pc' = [pc EXCEPT ![self] = "Li1"]
    /\ UNCHANGED << c, k, temp >>

Li1(self) ==
    /\ pc[self] = "Li1"
    /\ IF k # self
          THEN pc' = [pc EXCEPT ![self] = "Li2"]
          ELSE pc' = [pc EXCEPT ![self] = "Li4a"]
    /\ UNCHANGED << b, c, k, temp >>

Li2(self) ==
    /\ pc[self] = "Li2"
    /\ c' = [c EXCEPT ![self] = TRUE]
    /\ pc' = [pc EXCEPT ![self] = "Li3a"]
    /\ UNCHANGED << b, k, temp >>

Li3a(self) ==
    /\ pc[self] = "Li3a"
    /\ temp' = [temp EXCEPT ![self] = k]
    /\ pc' = [pc EXCEPT ![self] = "Li3b"]
    /\ UNCHANGED << b, c, k >>

Li3b(self) ==
    /\ pc[self] = "Li3b"
    /\ IF b[temp[self]]
          THEN pc' = [pc EXCEPT ![self] = "Li3c"]
          ELSE pc' = [pc EXCEPT ![self] = "Li3d"]
    /\ UNCHANGED << b, c, k, temp >>

Li3c(self) ==
    /\ pc[self] = "Li3c"
    /\ k' = self
    /\ pc' = [pc EXCEPT ![self] = "Li3d"]
    /\ UNCHANGED << b, c, temp >>

Li3d(self) ==
    /\ pc[self] = "Li3d"
    /\ pc' = [pc EXCEPT ![self] = "Li1"]
    /\ UNCHANGED << b, c, k, temp >>

Li4a(self) ==
    /\ pc[self] = "Li4a"
    /\ c' = [c EXCEPT ![self] = FALSE]
    /\ temp' = [temp EXCEPT ![self] = Proc \ {self}]
    /\ pc' = [pc EXCEPT ![self] = "Li4b"]
    /\ UNCHANGED << b, k >>

Li4b(self) ==
    /\ pc[self] = "Li4b"
    /\ IF temp[self] # {}
          THEN \E j \in temp[self]:
                   /\ temp' = [temp EXCEPT ![self] = temp[self] \ {j}]
                   /\ IF ~c[j]
                         THEN pc' = [pc EXCEPT ![self] = "Li1"]
                         ELSE pc' = [pc EXCEPT ![self] = "Li4b"]
          ELSE /\ pc' = [pc EXCEPT ![self] = "cs"]
               /\ temp' = temp
    /\ UNCHANGED << b, c, k >>

cs(self) ==
    /\ pc[self] = "cs"
    /\ pc' = [pc EXCEPT ![self] = "Li5"]
    /\ UNCHANGED << b, c, k, temp >>

Li5(self) ==
    /\ pc[self] = "Li5"
    /\ c' = [c EXCEPT ![self] = TRUE]
    /\ pc' = [pc EXCEPT ![self] = "Li6"]
    /\ UNCHANGED << b, k, temp >>

Li6(self) ==
    /\ pc[self] = "Li6"
    /\ b' = [b EXCEPT ![self] = TRUE]
    /\ pc' = [pc EXCEPT ![self] = "ncs"]
    /\ UNCHANGED << c, k, temp >>

ncs(self) ==
    /\ pc[self] = "ncs"
    /\ pc' = [pc EXCEPT ![self] = "Li0"]
    /\ UNCHANGED << b, c, k, temp >>

P(self) ==
    Li0(self) \/ Li1(self) \/ Li2(self) \/ Li3a(self) \/ Li3b(self)
    \/ Li3c(self) \/ Li3d(self) \/ Li4a(self) \/ Li4b(self)
    \/ cs(self) \/ Li5(self) \/ Li6(self) \/ ncs(self)

Next == \E self \in Proc : P(self)
Spec == /\ Init /\ [][Next]_vars
MutualExclusion == \A i, j \in Proc :
    (i # j) => ~ /\ pc[i] = "cs"
                 /\ pc[j] = "cs"
====
"#,
    );

    let mut config = Config {
        use_flat_state: Some(true),
        use_compiled_bfs: Some(true),
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.invariants.push("MutualExclusion".to_string());
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );
    config.add_constant("defaultInitValue".to_string(), ConstantValue::ModelValue);

    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);

    let proc_entries = |value: Value| {
        Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::ModelValue(Rp::from("p1")), value.clone()),
            (Value::ModelValue(Rp::from("p2")), value),
        ])))
    };
    let init = ArrayState::from_values(vec![
        proc_entries(Value::Bool(true)),
        proc_entries(Value::Bool(true)),
        Value::ModelValue(Rp::from("p1")),
        proc_entries(Value::String(Rp::from("Li0"))),
        proc_entries(Value::ModelValue(Rp::from("defaultInitValue"))),
    ]);

    checker.infer_flat_state_layout(&init);
    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for Dijkstra config shape");

    for (name, scalar_type, universe_len) in [
        ("b", SlotType::Bool, 2),
        ("c", SlotType::Bool, 2),
        ("pc", SlotType::String, 13),
    ] {
        let idx = checker
            .ctx
            .var_registry()
            .get(name)
            .unwrap_or_else(|| panic!("{name} variable index"))
            .as_usize();
        let proof = layout
            .var_layout(idx)
            .and_then(|var| var.kind.fixed_scalar_range_proof())
            .unwrap_or_else(|| panic!("{name} should carry an Init/Next fixed scalar proof"));
        assert_eq!(proof.scalar_type(), scalar_type, "{name} scalar type");
        assert_eq!(
            proof.scalar_universe().len(),
            universe_len,
            "{name} finite scalar universe"
        );
        assert_eq!(proof.source().as_ref(), "Init/Next writer proof");
    }

    let temp_idx = checker
        .ctx
        .var_registry()
        .get("temp")
        .expect("temp variable index")
        .as_usize();
    assert!(
        layout
            .var_layout(temp_idx)
            .and_then(|var| var.kind.tagged_scalar_set_range_proof())
            .is_some(),
        "real Dijkstra config shape should still discover the temp tagged range proof from Init/Next"
    );
    // b298fe51: the discovered Bool-ranged b/c plus the fixed-scalar pc/temp proofs make the
    // real Dijkstra config-shape layout flat-state PRIMARY without any configured TypeOK.
    assert!(
        checker.is_flat_state_primary(),
        "the discovered fixed-scalar ranges (incl. Bool-ranged b/c) make the Dijkstra config-shape layout flat-state primary"
    );
    assert!(
        !checker.native_fused_flat_frontier_admission_candidate_for_strict(true),
        "a flat-primary layout is rejected by the non-primary strict flat-frontier admission gate"
    );
    assert_eq!(
        checker.native_fused_non_primary_flat_frontier_parent_loop_rejection_for_strict(true),
        None,
        "a flat-primary layout short-circuits the non-primary parent-loop rejection to None"
    );
}

fn string_temp_tagged_range_candidate() -> super::ActionTaggedRangeCandidate {
    super::ActionTaggedRangeCandidate {
        var_idx: 0,
        domain: Arc::from(
            vec![
                Value::String(Rp::from("p1")),
                Value::String(Rp::from("p2")),
            ]
            .into_boxed_slice(),
        ),
        scalar_type: SlotType::String,
        set_universe: vec![
            crate::state::FlatScalarValue::String(std::sync::Arc::from("p1")),
            crate::state::FlatScalarValue::String(std::sync::Arc::from("p2")),
        ],
    }
}

fn model_value_temp_tagged_range_candidate() -> super::ActionTaggedRangeCandidate {
    super::ActionTaggedRangeCandidate {
        var_idx: 0,
        domain: Arc::from(
            vec![
                Value::ModelValue(Rp::from("p1")),
                Value::ModelValue(Rp::from("p2")),
            ]
            .into_boxed_slice(),
        ),
        scalar_type: SlotType::ModelValue,
        set_universe: vec![
            crate::state::FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
            crate::state::FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
        ],
    }
}

#[test]
fn test_action_producer_scanner_allows_tagged_read_quantifier_setdiff_domains() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    for forall in [false, true] {
        let mut chunk = BytecodeChunk::new();
        let p1 = chunk.constants.add_value(Value::String(Rp::from("p1")));
        let mut func = BytecodeFunction::new(
            if forall {
                "TaggedReadForallSetDiff".to_string()
            } else {
                "TaggedReadExistsSetDiff".to_string()
            },
            0,
        );
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
        func.emit(Opcode::LoadConst { rd: 1, idx: p1 });
        func.emit(Opcode::FuncApply {
            rd: 2,
            func: 0,
            arg: 1,
        });
        if forall {
            func.emit(Opcode::ForallBegin {
                rd: 5,
                r_binding: 3,
                r_domain: 2,
                loop_end: 0,
            });
        } else {
            func.emit(Opcode::ExistsBegin {
                rd: 5,
                r_binding: 3,
                r_domain: 2,
                loop_end: 0,
            });
        }
        func.emit(Opcode::LoadVar { rd: 4, var_idx: 0 });
        func.emit(Opcode::FuncApply {
            rd: 6,
            func: 4,
            arg: 1,
        });
        func.emit(Opcode::SetEnum {
            rd: 7,
            start: 3,
            count: 1,
        });
        func.emit(Opcode::SetDiff {
            rd: 8,
            r1: 6,
            r2: 7,
        });
        func.emit(Opcode::LoadVar { rd: 9, var_idx: 0 });
        func.emit(Opcode::FuncExcept {
            rd: 10,
            func: 9,
            path: 1,
            val: 8,
        });
        func.emit(Opcode::StoreVar { var_idx: 0, rs: 10 });

        let candidate = string_temp_tagged_range_candidate();
        let scan = super::action_function_supports_tagged_range_candidate(
            &func,
            &chunk,
            &candidate,
            &[None],
            &[],
        )
        .expect("tagged read quantifier SetDiff should be proof-backed");
        assert!(scan.saw_store);
        assert!(scan.saw_set_write);
    }
}

#[test]
fn test_action_producer_scanner_tagged_read_copy_does_not_count_as_set_write() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let p1 = chunk.constants.add_value(Value::String(Rp::from("p1")));
    let mut func = BytecodeFunction::new("TaggedReadSelfCopy".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadConst { rd: 1, idx: p1 });
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    func.emit(Opcode::LoadVar { rd: 3, var_idx: 0 });
    func.emit(Opcode::FuncExcept {
        rd: 4,
        func: 3,
        path: 1,
        val: 2,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 4 });

    let candidate = string_temp_tagged_range_candidate();
    let scan = super::action_function_supports_tagged_range_candidate(
        &func,
        &chunk,
        &candidate,
        &[None],
        &[],
    )
    .expect("tagged self-copy should preserve the candidate function shape");
    assert!(scan.saw_store);
    assert!(
        !scan.saw_set_write,
        "reading a tagged scalar/set value is not enough to prove a set-producing action"
    );
}

#[test]
fn test_action_producer_scanner_rejects_bare_loadimm_nameid_for_string_universe() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let p1_name_id = i64::from(intern_name("p1").0);
    let candidate = string_temp_tagged_range_candidate();
    assert!(
        super::action_shape_for_load_imm(p1_name_id, &candidate, &[]).is_none(),
        "bare LoadImm values do not carry string/model-value provenance"
    );

    let chunk = BytecodeChunk::new();
    let mut func = BytecodeFunction::new("BareLoadImmNameIdSetWriter".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadImm {
        rd: 1,
        value: p1_name_id,
    });
    func.emit(Opcode::SetEnum {
        rd: 2,
        start: 1,
        count: 1,
    });
    func.emit(Opcode::FuncExcept {
        rd: 3,
        func: 0,
        path: 1,
        val: 2,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 3 });

    assert!(
        super::action_function_supports_tagged_range_candidate(
            &func,
            &chunk,
            &candidate,
            &[None],
            &[],
        )
        .is_none(),
        "bare LoadImm NameIds must not prove string/model-value scalar or set writes"
    );
}

#[test]
fn test_action_producer_scanner_allows_dijkstra_model_value_loadimm_setdiff() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let p1 = i64::from(intern_name("p1").0);
    let action_name =
        tla_jit_abi::binding_key_for_values("Li4a", &[Value::ModelValue(Rp::from("p1"))])
            .expect("model-value scalar binding key");
    let split_meta = vec![ActionInstanceMeta {
        name: Some("Li4a".to_string()),
        bindings: vec![(Arc::from("self"), Value::ModelValue(Rp::from("p1")))],
        formal_bindings: vec![(Arc::from("self"), Value::ModelValue(Rp::from("p1")))],
        expr: None,
    }];
    let typed_by_action = super::action_typed_load_imms_by_split_action(Some(&split_meta));
    let typed_load_imms = typed_by_action
        .get(&action_name)
        .expect("split model-value action should expose typed LoadImm provenance");
    let candidate = model_value_temp_tagged_range_candidate();
    assert_eq!(
        super::action_shape_for_load_imm(p1, &candidate, typed_load_imms),
        Some(super::ActionTaggedRangeShape::Scalar),
        "split-action model-value binding metadata should preserve LoadImm provenance"
    );
    assert!(
        super::action_shape_for_load_imm(p1, &candidate, &[]).is_none(),
        "the same raw NameId remains rejected without split-action provenance"
    );

    let mut func = BytecodeFunction::new(action_name.clone(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadImm { rd: 1, value: p1 });
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    func.emit(Opcode::LoadImm { rd: 3, value: p1 });
    func.emit(Opcode::SetEnum {
        rd: 4,
        start: 3,
        count: 1,
    });
    func.emit(Opcode::SetDiff {
        rd: 5,
        r1: 2,
        r2: 4,
    });
    func.emit(Opcode::LoadVar { rd: 6, var_idx: 0 });
    func.emit(Opcode::FuncExcept {
        rd: 7,
        func: 6,
        path: 1,
        val: 5,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 7 });

    let mut chunk = BytecodeChunk::new();
    let idx = chunk.add_function(func);
    let mut op_indices = rustc_hash::FxHashMap::default();
    op_indices.insert(action_name.clone(), idx);
    let bytecode = tla_eval::bytecode_vm::CompiledBytecode {
        chunk,
        op_indices,
        failed: Vec::new(),
    };

    assert!(
        super::action_bytecode_supports_tagged_range_candidate(
            &bytecode,
            &candidate,
            &[None],
            &typed_by_action,
        ),
        "Dijkstra-shaped model-value temp[self] \\ {{self}} writes should prove the tagged range"
    );
    assert!(
        !super::action_bytecode_supports_tagged_range_candidate(
            &bytecode,
            &candidate,
            &[None],
            &std::collections::BTreeMap::new(),
        ),
        "raw model-value NameIds remain fail-closed without typed split-action metadata"
    );
}

#[test]
fn test_action_producer_candidate_rejects_partial_action_bytecode_coverage() {
    use tla_eval::bytecode_vm::CompiledBytecode;
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let p1 = chunk.constants.add_value(Value::String(Rp::from("p1")));
    let mut func = BytecodeFunction::new("CompiledSetWriter".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadConst { rd: 1, idx: p1 });
    func.emit(Opcode::SetEnum {
        rd: 2,
        start: 1,
        count: 1,
    });
    func.emit(Opcode::FuncExcept {
        rd: 3,
        func: 0,
        path: 1,
        val: 2,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 3 });
    let func_idx = chunk.add_function(func);

    let mut op_indices = rustc_hash::FxHashMap::default();
    op_indices.insert("CompiledSetWriter".to_string(), func_idx);
    let bytecode = CompiledBytecode {
        chunk,
        op_indices,
        failed: vec![(
            "UncompiledWriter".to_string(),
            CompileError::Unsupported("intentional partial coverage".to_string()),
        )],
    };

    assert!(
        !super::action_bytecode_supports_tagged_range_candidate(
            &bytecode,
            &string_temp_tagged_range_candidate(),
            &[None],
            &std::collections::BTreeMap::new(),
        ),
        "partial action coverage must not mint tagged scalar/set range proofs"
    );
}

#[test]
fn test_action_producer_rejects_scalar_only_model_value_function_layout() {
    let module = parse_module(
        r#"
---- MODULE ActionProducerRejectsScalarOnlyFunction ----
CONSTANTS Proc, NoOwner
VARIABLE owner

KeepScalar(self) ==
    owner' = [owner EXCEPT ![self] = NoOwner]
====
"#,
    );
    let mut config = Config {
        use_flat_state: Some(true),
        ..Default::default()
    };
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::ModelValueSet(vec!["p1".to_string(), "p2".to_string()]),
    );
    config.add_constant("NoOwner".to_string(), ConstantValue::ModelValue);
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);
    seed_split_action_meta(&mut checker, &["KeepScalar"]);
    checker.compile_action_bytecode();

    let owner = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (
            Value::ModelValue(Rp::from("p1")),
            Value::ModelValue(Rp::from("NoOwner")),
        ),
        (
            Value::ModelValue(Rp::from("p2")),
            Value::ModelValue(Rp::from("NoOwner")),
        ),
    ])));
    checker.infer_flat_state_layout(&ArrayState::from_values(vec![owner]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for scalar-only function");
    let owner_idx = checker
        .ctx
        .var_registry()
        .get("owner")
        .expect("owner variable index")
        .as_usize();
    assert!(
        layout
            .var_layout(owner_idx)
            .and_then(|var| var.kind.tagged_scalar_set_range_proof())
            .is_none(),
        "scalar-only producers must not manufacture tagged scalar/set proof"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "legacy scalar-slot function remains fail-closed for flat primary"
    );
}

#[test]
fn test_flat_bfs_auto_admission_rejects_model_value_keyed_function_set_slots() {
    let module = parse_module(
        r#"
---- MODULE FlatBfsFunctionSetSlotGuard ----
VARIABLE temp
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&model_value_keyed_empty_set_function_state());

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for model-value keyed set function");
    assert!(
        !layout.is_fully_flat(),
        "unproven empty set function slots must stay dynamic without a range proof"
    );
    assert!(
        !layout.supports_flat_bfs_auto_admission(),
        "function range slots that can hold sets must not auto-admit flat BFS"
    );
    assert!(
        !layout.supports_flat_primary(),
        "function range slots that can hold sets must not become primary flat storage"
    );
    assert!(
        !checker.should_use_flat_bfs(),
        "default flat BFS admission must reject set-valued function slots"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "flat-state-primary must not activate for set-valued function slots"
    );
}

#[test]
fn test_flat_state_primary_rejects_model_value_keyed_function_set_slots_when_forced() {
    let module = parse_module(
        r#"
---- MODULE FlatBfsForcedFunctionSetSlotGuard ----
VARIABLE temp
====
"#,
    );
    let config = Config {
        use_flat_state: Some(true),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&model_value_keyed_empty_set_function_state());

    assert!(
        checker.should_use_flat_bfs(),
        "config.use_flat_state=true should still enable the adapter sandwich"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "force-enabled flat BFS must not promote set-valued function slots to primary storage"
    );
}

#[test]
fn test_flat_state_primary_rejects_recursive_sequence_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimarySequenceGuard ----
VARIABLE queue
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&sequence_init_state());

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for sequence state");
    assert!(
        layout.is_fully_flat(),
        "fixed-capacity sequences still use a complete slot layout for fitting states"
    );
    assert!(
        !layout.supports_flat_primary(),
        "sequence capacity inferred from one state must not activate flat-primary BFS"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "recursive sequence layouts must stay on ArrayState-primary BFS"
    );
}

#[test]
fn test_flat_state_primary_rejects_observed_recursive_network_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryObservedNetworkGuard ----
VARIABLE network
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    let init = ArrayState::from_values(vec![observed_network_value()]);

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for observed network");
    assert!(layout.is_fully_flat());
    assert!(
        !layout.supports_flat_primary(),
        "observed recursive sequence capacity must not be primary-safe"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "observed-only recursive network must stay ArrayState-primary"
    );
}

#[test]
fn test_bounded_network_proof_marks_matching_path_capacity_only() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryBoundedNetworkProof ----
EXTENDS Naturals, Sequences
VARIABLES log, network
Proc == {1, 2}
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    let log_value = Value::Seq(Rp::new(SeqValue::from_vec(vec![Value::SmallInt(1)])));
    let init = ArrayState::from_values(vec![log_value, observed_network_value()]);

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for proven network");
    assert!(
        !layout.supports_flat_primary(),
        "non-matching observed log sequence must still block primary-safe layout"
    );
    assert!(!checker.is_flat_state_primary());

    match &layout.var_layout(1).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(*max_len, 3);
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariant {
                            invariant: Arc::from("BoundedNetwork")
                        }
                    );
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }

    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::Sequence { bound, .. },
        } => assert_eq!(*bound, SequenceBoundEvidence::Observed),
        other => panic!("expected recursive log sequence layout, got {other:?}"),
    }

    let network_only_module = parse_module(
        r#"
---- MODULE FlatPrimaryBoundedNetworkOnlyProof ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {1, 2}
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut network_only_checker = ModelChecker::new(&network_only_module, &config);
    network_only_checker
        .infer_flat_state_layout(&ArrayState::from_values(vec![observed_network_value()]));
    assert!(
        !network_only_checker
            .flat_state_layout()
            .expect("layout should be inferred for network-only state")
            .supports_flat_primary(),
        "a length proof alone must not make observed element shape primary-safe"
    );
    assert!(
        !network_only_checker.is_flat_state_primary(),
        "proven length without proven element shape must stay ArrayState-primary"
    );
}

/// A `Len(s) <= k` predicate used as a CONSTRAINT (not an INVARIANT) must
/// still derive a `ProvenInvariant` capacity proof. TLC enforces a CONSTRAINT
/// by successor pruning, so `Len(s) <= k` bounds the length of `s` across every
/// explored state by construction — exactly the soundness an INVARIANT bound
/// provides — and the canonical `qConstraint == Len(q) \leq qLen` idiom (FIFO,
/// AlternatingBit, …) lives in the CONSTRAINT slot, not INVARIANTS.
#[test]
fn test_state_constraint_len_bound_proves_sequence_capacity() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryConstraintLenCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLES log, network
Proc == {1, 2}
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    // The SAME predicate, but registered as a CONSTRAINT rather than an
    // INVARIANT, must produce the identical capacity proof.
    let mut config = Config::default();
    config.constraints.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    let log_value = Value::Seq(Rp::new(SeqValue::from_vec(vec![Value::SmallInt(1)])));
    let init = ArrayState::from_values(vec![log_value, observed_network_value()]);

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for constraint-bounded network");

    match &layout.var_layout(1).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(*max_len, 3, "constraint capacity bound");
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariant {
                            invariant: Arc::from("BoundedNetwork")
                        },
                        "a Len(s) <= k CONSTRAINT must prove the sequence capacity"
                    );
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }

    // Capacity-only proof: the element shape is still observed, so flat-primary
    // stays fail-closed (mirrors the INVARIANT-sourced capacity-only case).
    assert!(
        !layout.supports_flat_primary(),
        "constraint capacity proof alone must not promote primary flat state"
    );
    assert!(!checker.is_flat_state_primary());

    // The unbounded `log` sequence (no constraint covers it) stays Observed.
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::Sequence { bound, .. },
        } => assert_eq!(*bound, SequenceBoundEvidence::Observed),
        other => panic!("expected recursive log sequence layout, got {other:?}"),
    }
}

#[test]
fn test_operator_wrapped_len_proves_record_sequence_capacity() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedLenCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
ArrayLen(a) == Len(a.elems)
BoundedArray == \A i \in Idx : ArrayLen(array[i]) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BoundedArray".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for operator-wrapped Len proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::ProvenInvariant {
            invariant: Arc::from("BoundedArray"),
        },
        2,
        "operator-wrapped Len proof",
    );
    assert!(
        layout.is_fully_flat(),
        "operator-wrapped Len proof should keep a fixed flat layout"
    );
    assert!(
        !layout.supports_flat_primary(),
        "operator-wrapped Len proof proves capacity only; element shape still needs separate proof"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "capacity-only wrapper proof must not promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_path_proves_record_sequence_capacity() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedPathCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
Elems(a) == a.elems
BoundedArray == \A i \in Idx : Len(Elems(array[i])) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BoundedArray".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for operator-wrapped path proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::ProvenInvariant {
            invariant: Arc::from("BoundedArray"),
        },
        2,
        "operator-wrapped path proof",
    );
    assert!(
        layout.is_fully_flat(),
        "operator-wrapped path proof should keep a fixed flat layout"
    );
    assert!(
        !layout.supports_flat_primary(),
        "operator-wrapped path proof proves capacity only; element shape still needs separate proof"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "capacity-only wrapper proof must not promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_index_path_proves_record_sequence_capacity() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedIndexPathCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
ElemsAt(a, idx) == a[idx].elems
BoundedArray == \A i \in Idx : Len(ElemsAt(array, i)) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BoundedArray".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for operator-wrapped index path proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::ProvenInvariant {
            invariant: Arc::from("BoundedArray"),
        },
        2,
        "operator-wrapped index path proof",
    );
    assert!(
        layout.is_fully_flat(),
        "operator-wrapped index path proof should keep a fixed flat layout"
    );
    assert!(
        !layout.supports_flat_primary(),
        "operator-wrapped index path proof proves capacity only; element shape still needs separate proof"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "capacity-only wrapper proof must not promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_capacity_with_typeok_element_layout_promotes_record_sequence_primary_safe()
{
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedCapacityTypeOkProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
ArrayLen(a) == Len(a.elems)
TypeOK == \A i \in Idx : array[i].elems \in Seq(Nat)
BoundedArray == \A i \in Idx : ArrayLen(array[i]) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedArray".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for wrapped capacity plus TypeOK proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("BoundedArray"),
            element_invariant: Arc::from("TypeOK"),
        },
        2,
        "operator-wrapped capacity plus TypeOK proof",
    );
    assert!(
        layout.supports_flat_primary(),
        "capacity plus element-layout proof should make the record sequence primary-safe"
    );
    assert!(
        checker.is_flat_state_primary(),
        "capacity plus element-layout proof should promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_typeok_element_layout_promotes_record_sequence_primary_safe() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedTypeOkElementProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
Elems(a) == a.elems
NatSeq(T) == Seq(T)
TypeOK == \A i \in Idx : Elems(array[i]) \in NatSeq(Nat)
BoundedArray == \A i \in Idx : Len(Elems(array[i])) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedArray".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for wrapped TypeOK proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("BoundedArray"),
            element_invariant: Arc::from("TypeOK"),
        },
        2,
        "operator-wrapped TypeOK element proof",
    );
    assert!(
        layout.supports_flat_primary(),
        "lowered capacity plus element-layout wrappers should make the record sequence primary-safe"
    );
    assert!(
        checker.is_flat_state_primary(),
        "lowered capacity plus element-layout wrappers should promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_quantifier_domain_promotes_record_sequence_primary_safe() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedQuantifierDomainProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
Dom(S) == S
Elems(a) == a.elems
TypeOK == \A i \in Dom(Idx) : Elems(array[i]) \in Seq(Nat)
BoundedArray == \A i \in Dom(Idx) : Len(Elems(array[i])) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedArray".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for wrapped quantifier-domain proofs");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("BoundedArray"),
            element_invariant: Arc::from("TypeOK"),
        },
        2,
        "operator-wrapped quantifier-domain proof",
    );
    assert!(
        layout.supports_flat_primary(),
        "parameterized quantifier-domain wrappers should preserve primary-safe sequence proofs"
    );
    assert!(
        checker.is_flat_state_primary(),
        "parameterized quantifier-domain wrappers should promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_concrete_path_capacity_fails_closed() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedConcretePathCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Elems(a) == a.elems
\* A concrete function index does not prove the homogeneous range array[i].
BadBound == Len(Elems(array[1])) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BadBound".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for failed wrapped concrete proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::Observed,
        1,
        "operator-wrapped concrete path proof",
    );
    assert!(
        !layout.supports_flat_primary(),
        "operator-wrapped concrete path proof must not activate flat-primary layout"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "operator-wrapped concrete path proof must not promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_capture_collision_capacity_proof_fails_closed() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedCaptureCollisionCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
ElemsAt(i) == array[i].elems
\* The wrapper formal `i` must not capture the caller quantifier when the
\* actual is a concrete index.
BadBound == \A i \in Idx : Len(ElemsAt(1)) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BadBound".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for failed wrapped capture-collision proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::Observed,
        1,
        "operator-wrapped capture-collision concrete index proof",
    );
    assert!(
        !layout.supports_flat_primary(),
        "operator-wrapped capture-collision proof must not activate flat-primary layout"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "operator-wrapped capture-collision proof must not promote primary flat state"
    );
}

#[test]
fn test_operator_wrapped_free_body_name_capture_capacity_proof_fails_closed() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryWrappedFreeBodyNameCaptureCapacityProof ----
EXTENDS Naturals, Sequences
VARIABLE array
Idx == {1, 2}
Elems == array[i].elems
\* The free `i` in Elems must not be captured by the caller quantifier.
BadBound == \A i \in Idx : Len(Elems) <= 2
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BadBound".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(
        vec![array_record_function_value()],
    ));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for failed free-body-name capture proof");
    assert_array_elems_sequence_bound(
        layout,
        SequenceBoundEvidence::Observed,
        1,
        "operator-wrapped free-body-name capture proof",
    );
    assert!(
        !layout.supports_flat_primary(),
        "operator-wrapped free-body-name capture proof must not activate flat-primary layout"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "operator-wrapped free-body-name capture proof must not promote primary flat state"
    );
}

#[test]
fn test_init_next_writer_proves_undo_redo_record_sequence_element_layouts() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryUndoRedoWriterElementProof ----
EXTENDS Naturals, Sequences
VARIABLES undoStack, redoStack, groupChanges, cursorLine, cursorCol
UndoEntries == [kind: {"Insert", "Delete"}, cursorLine: 1..2, cursorCol: 0..3]
RecordGroupChange(entry) ==
    IF groupChanges = <<>> THEN <<entry>> ELSE groupChanges
TypeOK ==
    /\ Len(undoStack) <= 2
    /\ Len(redoStack) <= 2
    /\ groupChanges \in Seq(UndoEntries)
    /\ Len(groupChanges) <= 1
Init ==
    /\ undoStack = <<>>
    /\ redoStack = <<>>
    /\ groupChanges = <<>>
    /\ cursorLine = 1
    /\ cursorCol = 0
Push ==
    LET entry == [kind |-> "Insert", cursorLine |-> cursorLine, cursorCol |-> cursorCol]
    IN /\ undoStack' = Append(undoStack, entry)
       /\ redoStack' = <<>>
       /\ groupChanges' = RecordGroupChange(entry)
       /\ UNCHANGED <<cursorLine, cursorCol>>
Undo ==
    /\ undoStack # <<>>
    /\ LET entry == Head(undoStack)
       IN /\ redoStack' = Append(redoStack, entry)
          /\ cursorLine' = entry.cursorLine
          /\ cursorCol' = entry.cursorCol
    /\ undoStack' = Tail(undoStack)
    /\ UNCHANGED groupChanges
EndGroup ==
    /\ groupChanges # <<>>
    /\ LET firstChange == Head(groupChanges)
           entry == [kind |-> firstChange.kind,
                     cursorLine |-> firstChange.cursorLine,
                     cursorCol |-> firstChange.cursorCol]
       IN undoStack' = Append(undoStack, entry)
    /\ groupChanges' = <<>>
    /\ UNCHANGED <<redoStack, cursorLine, cursorCol>>
Next == Push \/ Undo \/ EndGroup
====
"#,
    );
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    let mut values = vec![Value::SmallInt(0); checker.ctx.var_registry().len()];
    let mut set_var = |name: &str, value: Value| {
        let idx = checker
            .ctx
            .var_registry()
            .get(name)
            .unwrap_or_else(|| panic!("missing variable {name}"))
            .as_usize();
        values[idx] = value;
    };
    let empty_seq = || Value::Seq(Rp::new(SeqValue::from_vec(Vec::new())));
    set_var("undoStack", empty_seq());
    set_var("redoStack", empty_seq());
    set_var("groupChanges", empty_seq());
    set_var("cursorLine", Value::SmallInt(1));
    set_var("cursorCol", Value::SmallInt(0));
    let init = ArrayState::from_values(values);
    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for undo/redo writer proof");
    assert!(
        layout.supports_flat_primary(),
        "capacity proofs plus Init/Next writer element proofs should be primary-safe"
    );
    assert_top_level_sequence_element_layout(
        layout,
        checker.ctx.var_registry(),
        "undoStack",
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("TypeOK"),
            element_invariant: Arc::from("Init/Next sequence writer proof"),
        },
        2,
        "undoStack writer proof",
    );
    assert_top_level_sequence_element_layout(
        layout,
        checker.ctx.var_registry(),
        "redoStack",
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("TypeOK"),
            element_invariant: Arc::from("Init/Next sequence writer proof"),
        },
        2,
        "redoStack writer proof",
    );
    assert_top_level_sequence_element_layout(
        layout,
        checker.ctx.var_registry(),
        "groupChanges",
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("TypeOK"),
            element_invariant: Arc::from("TypeOK"),
        },
        1,
        "groupChanges TypeOK proof",
    );
}

#[test]
fn test_init_next_writer_proves_trimmed_sequence_capacity_without_type_invariant() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryUndoRedoWriterNoTypeInvariant ----
EXTENDS Naturals, Sequences
CONSTANT MaxUndoDepth
VARIABLES undoStack, redoStack, groupChanges, groupDepth, cursorLine, cursorCol
UndoEntries == [kind: {"Insert", "Delete"}, cursorLine: 1..2, cursorCol: 0..3]
RecordGroupChange(entry) ==
    IF groupChanges = <<>> THEN <<entry>> ELSE groupChanges
Init ==
    /\ undoStack = <<>>
    /\ redoStack = <<>>
    /\ groupChanges = <<>>
    /\ groupDepth = 0
    /\ cursorLine = 1
    /\ cursorCol = 0
Push ==
    LET entry == [kind |-> "Insert", cursorLine |-> cursorLine, cursorCol |-> cursorCol]
        willTrim == Len(undoStack) >= MaxUndoDepth
    IN /\ undoStack' = IF willTrim
                       THEN Tail(undoStack) \o <<entry>>
                       ELSE Append(undoStack, entry)
       /\ redoStack' = <<>>
       /\ groupChanges' = RecordGroupChange(entry)
       /\ UNCHANGED <<groupDepth, cursorLine, cursorCol>>
Undo ==
    /\ undoStack # <<>>
    /\ LET entry == Head(undoStack)
           willTrim == Len(redoStack) >= MaxUndoDepth
       IN /\ redoStack' = IF willTrim
                          THEN Tail(redoStack) \o <<entry>>
                          ELSE Append(redoStack, entry)
          /\ cursorLine' = entry.cursorLine
          /\ cursorCol' = entry.cursorCol
    /\ undoStack' = Tail(undoStack)
    /\ UNCHANGED <<groupChanges, groupDepth>>
Redo ==
    /\ redoStack # <<>>
    /\ LET entry == Head(redoStack)
           willTrim == Len(undoStack) >= MaxUndoDepth
       IN /\ undoStack' = IF willTrim
                          THEN Tail(undoStack) \o <<entry>>
                          ELSE Append(undoStack, entry)
          /\ cursorLine' = entry.cursorLine
          /\ cursorCol' = entry.cursorCol
    /\ redoStack' = Tail(redoStack)
    /\ UNCHANGED <<groupChanges, groupDepth>>
EndGroup ==
    /\ groupChanges # <<>>
    /\ LET firstChange == Head(groupChanges)
           entry == [kind |-> firstChange.kind,
                     cursorLine |-> firstChange.cursorLine,
                     cursorCol |-> firstChange.cursorCol]
           willTrim == Len(undoStack) >= MaxUndoDepth
       IN undoStack' = IF willTrim
                       THEN Tail(undoStack) \o <<entry>>
                       ELSE Append(undoStack, entry)
    /\ groupChanges' = <<>>
    /\ UNCHANGED <<redoStack, groupDepth, cursorLine, cursorCol>>
Next == Push \/ Undo \/ Redo \/ EndGroup
====
"#,
    );
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.add_constant(
        "MaxUndoDepth".to_string(),
        ConstantValue::Value("1".to_string()),
    );
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);

    let mut values = vec![Value::SmallInt(0); checker.ctx.var_registry().len()];
    let mut set_var = |name: &str, value: Value| {
        let idx = checker
            .ctx
            .var_registry()
            .get(name)
            .unwrap_or_else(|| panic!("missing variable {name}"))
            .as_usize();
        values[idx] = value;
    };
    let empty_seq = || Value::Seq(Rp::new(SeqValue::from_vec(Vec::new())));
    set_var("undoStack", empty_seq());
    set_var("redoStack", empty_seq());
    set_var("groupChanges", empty_seq());
    set_var("groupDepth", Value::SmallInt(0));
    set_var("cursorLine", Value::SmallInt(1));
    set_var("cursorCol", Value::SmallInt(0));
    let init = ArrayState::from_values(values);
    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred from writer relation without TypeInvariant");
    assert!(
        layout.supports_flat_primary(),
        "writer capacity plus writer element layout should make trimmed record sequences primary-safe"
    );
    let expected = SequenceBoundEvidence::ProvenInvariantWithElementLayout {
        invariant: Arc::from("Init/Next sequence writer proof"),
        element_invariant: Arc::from("Init/Next sequence writer proof"),
    };
    assert_top_level_sequence_element_layout(
        layout,
        checker.ctx.var_registry(),
        "undoStack",
        expected.clone(),
        1,
        "undoStack writer-only proof",
    );
    assert_top_level_sequence_element_layout(
        layout,
        checker.ctx.var_registry(),
        "redoStack",
        expected.clone(),
        1,
        "redoStack writer-only proof",
    );
    assert_top_level_sequence_element_layout(
        layout,
        checker.ctx.var_registry(),
        "groupChanges",
        expected,
        1,
        "groupChanges writer-only proof",
    );
}

#[test]
fn test_init_next_writer_uses_len_guard_for_sequence_capacity_without_type_invariant() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryGuardedLineLengthsWriterNoTypeInvariant ----
EXTENDS Naturals, Sequences
CONSTANT MaxLines
VARIABLE lineLengths
LineCount == Len(lineLengths)
Init == lineLengths = <<0>>
InsertNewline ==
    /\ LineCount < MaxLines
    /\ lineLengths' = [i \in 1..(LineCount + 1) |->
           IF i = LineCount + 1 THEN 0 ELSE lineLengths[i]]
Next == InsertNewline
====
"#,
    );
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.add_constant(
        "MaxLines".to_string(),
        ConstantValue::Value("2".to_string()),
    );
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);

    let init = ArrayState::from_values(vec![Value::Seq(Rp::new(SeqValue::from_vec(vec![
        Value::SmallInt(0),
    ])))]);
    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred from guarded writer relation");
    assert!(
        layout.supports_flat_primary(),
        "guarded LineCount writer should prove bounded integer sequence layout"
    );
    let idx = checker
        .ctx
        .var_registry()
        .get("lineLengths")
        .expect("lineLengths variable")
        .as_usize();
    match &layout.var_layout(idx).unwrap().kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                },
        } => {
            assert_eq!(
                *bound,
                SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                    invariant: Arc::from("Init/Next sequence writer proof"),
                    element_invariant: Arc::from("Init/Next sequence writer proof"),
                }
            );
            assert_eq!(*max_len, 2);
            assert_eq!(
                **element_layout,
                FlatValueLayout::Scalar(SlotType::Int),
                "lineLengths element layout"
            );
        }
        other => panic!("expected lineLengths sequence layout, got {other:?}"),
    }
}

#[test]
fn test_writer_element_layout_without_capacity_fails_closed() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryUndoWriterNoCapacity ----
EXTENDS Naturals, Sequences
VARIABLES undoStack, cursorLine, cursorCol
Init == /\ undoStack = <<>> /\ cursorLine = 1 /\ cursorCol = 0
Next ==
    LET entry == [kind |-> "Insert", cursorLine |-> cursorLine, cursorCol |-> cursorCol]
    IN /\ undoStack' = Append(undoStack, entry)
       /\ UNCHANGED <<cursorLine, cursorCol>>
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    let mut values = vec![Value::SmallInt(0); checker.ctx.var_registry().len()];
    let mut set_var = |name: &str, value: Value| {
        let idx = checker
            .ctx
            .var_registry()
            .get(name)
            .unwrap_or_else(|| panic!("missing variable {name}"))
            .as_usize();
        values[idx] = value;
    };
    set_var(
        "undoStack",
        Value::Seq(Rp::new(SeqValue::from_vec(Vec::new()))),
    );
    set_var("cursorLine", Value::SmallInt(1));
    set_var("cursorCol", Value::SmallInt(0));
    checker.infer_flat_state_layout(&ArrayState::from_values(values));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for capacity-free writer proof");
    assert!(
        !layout.supports_flat_primary(),
        "writer element layout alone must not admit strict native flat-primary storage"
    );
}

#[test]
fn test_typeok_and_bounded_network_make_empty_mcl_channels_primary_safe() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryMclTypeOkBoundedNetwork ----
EXTENDS Naturals, Sequences
VARIABLES network
Proc == {1, 2}
Message == {
    [type |-> "req", clock |-> 1],
    [type |-> "ack", clock |-> 0],
    [type |-> "rel", clock |-> 0]
}
TypeOK == network \in [Proc -> [Proc -> Seq(Message)]]
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![empty_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for MCL network");
    assert!(
        layout.supports_flat_primary(),
        "TypeOK proves channel element shape and BoundedNetwork proves capacity"
    );
    assert!(
        checker.is_flat_state_primary(),
        "empty MCL channels with TypeOK+BoundedNetwork should qualify for flat_state_primary"
    );

    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                } => {
                    assert_eq!(*max_len, 3);
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                            invariant: Arc::from("BoundedNetwork"),
                            element_invariant: Arc::from("TypeOK"),
                        }
                    );
                    match element_layout.as_ref() {
                        FlatValueLayout::Record {
                            field_names,
                            field_layouts,
                        } => {
                            assert!(field_names.contains(&Arc::from("type")));
                            assert!(field_names.contains(&Arc::from("clock")));
                            assert!(
                                field_layouts.contains(&FlatValueLayout::Scalar(SlotType::String))
                            );
                            assert!(field_layouts.contains(&FlatValueLayout::Scalar(SlotType::Int)));
                        }
                        other => panic!("expected proven message record layout, got {other:?}"),
                    }
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }
}

#[test]
fn test_typeok_real_mcl_message_operator_proves_empty_channel_element_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryRealMclMessageTypeOk ----
EXTENDS Naturals, Sequences
VARIABLES network
Proc == {1, 2}
Clock == Nat \ {0}
ReqMessage(c) == [type |-> "req", clock |-> c]
AckMessage == [type |-> "ack", clock |-> 0]
RelMessage == [type |-> "rel", clock |-> 0]
Message == {AckMessage, RelMessage} \cup {ReqMessage(c) : c \in Clock}
TypeOK == network \in [Proc -> [Proc -> Seq(Message)]]
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![empty_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for real MCL-shaped network");
    assert!(
        layout.supports_flat_primary(),
        "operator-defined Message should prove empty channel element layout"
    );
    assert!(
        checker.is_flat_state_primary(),
        "real MCL-shaped TypeOK+BoundedNetwork should activate flat_state_primary"
    );

    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence {
                    bound,
                    element_layout,
                    ..
                } => {
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                            invariant: Arc::from("BoundedNetwork"),
                            element_invariant: Arc::from("TypeOK"),
                        }
                    );
                    match element_layout.as_ref() {
                        FlatValueLayout::Record {
                            field_names,
                            field_layouts,
                        } => {
                            assert!(field_names.contains(&Arc::from("type")));
                            assert!(field_names.contains(&Arc::from("clock")));
                            assert!(
                                field_layouts.contains(&FlatValueLayout::Scalar(SlotType::String))
                            );
                            assert!(field_layouts.contains(&FlatValueLayout::Scalar(SlotType::Int)));
                        }
                        other => panic!("expected Message record layout, got {other:?}"),
                    }
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }
}

#[test]
fn test_typeok_mcl_range_alias_proc_proves_empty_channel_element_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryMclRangeAliasProc ----
EXTENDS Naturals, Sequences
VARIABLES network
Proc == 1..2
Clock == 1..7
ReqMessage(c) == [type |-> "req", clock |-> c]
AckMessage == [type |-> "ack", clock |-> 0]
RelMessage == [type |-> "rel", clock |-> 0]
Message == {AckMessage, RelMessage} \cup {ReqMessage(c) : c \in Clock}
TypeOK == network \in [Proc -> [Proc -> Seq(Message)]]
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![empty_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for range-alias MCL network");
    assert!(
        layout.supports_flat_primary(),
        "range aliases like Proc == 1..N must prove the same recursive layout as set aliases"
    );
    assert!(
        checker.is_flat_state_primary(),
        "range-alias MCL TypeOK+BoundedNetwork should activate flat_state_primary"
    );
}

#[test]
fn test_typeok_mcl_cfg_nat_replacement_proves_empty_channel_element_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryMclCfgNatReplacement ----
EXTENDS Naturals, Sequences
CONSTANTS N, Zero, MaxNat, MaxChannel
VARIABLES network
ZeroOverride == 0
MaxChannelOverride == 3
NatOverride == Zero..MaxNat
Proc == 1..N
Clock == Nat \ {Zero}
ReqMessage(c) == [type |-> "req", clock |-> c]
AckMessage == [type |-> "ack", clock |-> 0]
RelMessage == [type |-> "rel", clock |-> 0]
Message == {AckMessage, RelMessage} \union {ReqMessage(c) : c \in Clock}
TypeOK == network \in [Proc -> [Proc -> Seq(Message)]]
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= MaxChannel
====
"#,
    );
    let mut config = Config::default();
    config.add_constant("N".to_string(), ConstantValue::Value("2".to_string()));
    config.add_constant("MaxNat".to_string(), ConstantValue::Value("7".to_string()));
    config.add_constant(
        "Zero".to_string(),
        ConstantValue::Replacement("ZeroOverride".to_string()),
    );
    config.add_constant(
        "MaxChannel".to_string(),
        ConstantValue::Replacement("MaxChannelOverride".to_string()),
    );
    config.add_constant(
        "Nat".to_string(),
        ConstantValue::Replacement("NatOverride".to_string()),
    );
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![empty_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for cfg-replaced MCL network");
    assert!(
        layout.supports_flat_primary(),
        "cfg replacement Nat <- NatOverride must still prove MCL sequence capacity and element layout"
    );
    assert!(
        checker.is_flat_state_primary(),
        "MCL-shaped TypeOK+BoundedNetwork with cfg Nat replacement should activate flat_state_primary"
    );
}

#[test]
fn test_typeok_mcl_multi_hop_cfg_replacements_prove_empty_channel_element_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryMclMultiHopCfgReplacement ----
EXTENDS Naturals, Sequences
CONSTANTS N, Zero, MaxNat, MaxChannel
VARIABLES network
ZeroOverride == 0
MaxChannelOverride == 3
NatOverride == Zero..MaxNat
Proc == {1}
ProcOverride == 1..N
Clock == Nat \ {Zero}
ReqMessage(c) == [type |-> "req", clock |-> c]
AckMessage == [type |-> "ack", clock |-> Zero]
RelMessage == [type |-> "rel", clock |-> Zero]
Message == {AckMessage, RelMessage} \union {ReqMessage(c) : c \in Clock}
TypeOK == network \in [Proc \ {1} -> [Proc -> Seq(Message)]]
MCLTypeOK == network \in [Proc -> [Proc -> Seq(Message)]]
BoundedNetwork == Len(network[1][1]) <= 1
MCBoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= MaxChannel
====
"#,
    );
    let mut config = Config::default();
    config.add_constant("N".to_string(), ConstantValue::Value("2".to_string()));
    config.add_constant("MaxNat".to_string(), ConstantValue::Value("7".to_string()));
    config.add_constant(
        "Zero".to_string(),
        ConstantValue::Replacement("ZeroAlias".to_string()),
    );
    config.add_constant(
        "ZeroAlias".to_string(),
        ConstantValue::Replacement("ZeroOverride".to_string()),
    );
    config.add_constant(
        "MaxChannel".to_string(),
        ConstantValue::Replacement("MaxChannelAlias".to_string()),
    );
    config.add_constant(
        "MaxChannelAlias".to_string(),
        ConstantValue::Replacement("MaxChannelOverride".to_string()),
    );
    config.add_constant(
        "Nat".to_string(),
        ConstantValue::Replacement("NatAlias".to_string()),
    );
    config.add_constant(
        "NatAlias".to_string(),
        ConstantValue::Replacement("NatOverride".to_string()),
    );
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::Replacement("ProcAlias".to_string()),
    );
    config.add_constant(
        "ProcAlias".to_string(),
        ConstantValue::Replacement("ProcOverride".to_string()),
    );
    config.add_constant(
        "TypeOK".to_string(),
        ConstantValue::Replacement("TypeOKAlias".to_string()),
    );
    config.add_constant(
        "TypeOKAlias".to_string(),
        ConstantValue::Replacement("MCLTypeOK".to_string()),
    );
    config.add_constant(
        "BoundedNetwork".to_string(),
        ConstantValue::Replacement("BoundedNetworkAlias".to_string()),
    );
    config.add_constant(
        "BoundedNetworkAlias".to_string(),
        ConstantValue::Replacement("MCBoundedNetwork".to_string()),
    );
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![empty_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for multi-hop cfg-replaced MCL network");
    assert!(
        layout.supports_flat_primary(),
        "multi-hop cfg replacements must prove MCL sequence capacity and element layout"
    );
    assert!(
        checker.is_flat_state_primary(),
        "MCL-shaped TypeOK+BoundedNetwork with multi-hop cfg replacements should activate flat_state_primary"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                } => {
                    assert_eq!(*max_len, 3);
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                            invariant: Arc::from("BoundedNetwork"),
                            element_invariant: Arc::from("TypeOK"),
                        }
                    );
                    assert_message_record_layout(element_layout, "multi-hop cfg replacements");
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }
}

#[test]
fn test_replacement_cycle_does_not_fall_back_to_original_proof_domain() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryReplacementCycleProofDomain ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {1, 2}
Message == {
    [type |-> "req", clock |-> 1],
    [type |-> "ack", clock |-> 0],
    [type |-> "rel", clock |-> 0]
}
TypeOK == network \in [Proc -> [Proc -> Seq(Message)]]
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::Replacement("ProcAlias".to_string()),
    );
    config.add_constant(
        "ProcAlias".to_string(),
        ConstantValue::Replacement("Proc".to_string()),
    );
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![observed_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred with replacement cycle rejected");
    assert_network_channel_bound_observed(layout, "replacement-cycle proof domain");
    assert!(
        !checker.is_flat_state_primary(),
        "cyclic replacement provenance must not activate flat_state_primary"
    );
}

#[test]
fn test_full_mcl_sequence_init_typeok_proves_flat_primary_safe() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryFullMclSequenceTypeOk ----
EXTENDS Naturals, Sequences
CONSTANTS N, Zero, MaxNat, MaxChannel
VARIABLES clock, req, ack, network, crit
ZeroOverride == 0
MaxChannelOverride == 3
NatOverride == Zero..MaxNat
Proc == 1..N
Clock == Nat \ {Zero}
ReqMessage(c) == [type |-> "req", clock |-> c]
AckMessage == [type |-> "ack", clock |-> 0]
RelMessage == [type |-> "rel", clock |-> 0]
Message == {AckMessage, RelMessage} \union {ReqMessage(c) : c \in Clock}
TypeOK ==
  /\ clock \in [Proc -> Clock]
  /\ req \in [Proc -> [Proc -> Nat]]
  /\ ack \in [Proc -> SUBSET Proc]
  /\ network \in [Proc -> [Proc -> Seq(Message)]]
  /\ crit \in SUBSET Proc
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= MaxChannel
====
"#,
    );
    let mut config = Config::default();
    config.add_constant("N".to_string(), ConstantValue::Value("3".to_string()));
    config.add_constant("MaxNat".to_string(), ConstantValue::Value("7".to_string()));
    config.add_constant(
        "Zero".to_string(),
        ConstantValue::Replacement("ZeroOverride".to_string()),
    );
    config.add_constant(
        "MaxChannel".to_string(),
        ConstantValue::Replacement("MaxChannelOverride".to_string()),
    );
    config.add_constant(
        "Nat".to_string(),
        ConstantValue::Replacement("NatOverride".to_string()),
    );
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);
    let init = full_mcl_sequence_init_state(&checker);

    checker.infer_flat_state_layout(&init);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for full MCL sequence-shaped init");
    assert_eq!(layout.total_slots(), 89);
    assert!(layout.is_fully_flat());
    assert!(
        layout.supports_flat_primary(),
        "full MCL TypeOK plus BoundedNetwork should make every recursive sequence primary-safe: {:?}",
        layout.var_layout(
            checker
                .ctx
                .var_registry()
                .get("network")
                .expect("network var")
                .as_usize()
        )
        .unwrap()
        .kind
    );
    assert!(
        checker.is_flat_state_primary(),
        "real sequence-shaped MCL init should activate flat_state_primary"
    );
    let mut adapter = checker
        .flat_bfs_adapter()
        .expect("flat_state_primary should install a FlatBfsAdapter");
    let flat_init = adapter
        .try_array_to_flat_lossless(&init)
        .expect("full MCL init must flatten losslessly for flat-primary BFS");
    let mut frontier = FlatBfsFrontier::with_capacity(Arc::clone(adapter.layout()), 1);
    let fp = flat_init.fingerprint_compiled();
    frontier.push((
        NoTraceQueueEntry::Flat {
            flat: flat_init,
            fp,
        },
        0,
        0,
    ));
    assert_eq!(frontier.total_pushed(), 1);
    assert_eq!(frontier.flat_pushed(), 1);
    assert_eq!(frontier.remaining_flat_count(), 1);
    assert!(
        !frontier.has_fallback_entries(),
        "full MCL init must enter the flat frontier arena, not the fallback queue"
    );

    let clock_idx = checker
        .ctx
        .var_registry()
        .get("clock")
        .expect("clock var")
        .as_usize();
    let req_idx = checker
        .ctx
        .var_registry()
        .get("req")
        .expect("req var")
        .as_usize();
    let ack_idx = checker
        .ctx
        .var_registry()
        .get("ack")
        .expect("ack var")
        .as_usize();
    let crit_idx = checker
        .ctx
        .var_registry()
        .get("crit")
        .expect("crit var")
        .as_usize();
    let network_idx = checker
        .ctx
        .var_registry()
        .get("network")
        .expect("network var")
        .as_usize();
    let clock_layout = layout.var_layout(clock_idx).expect("clock layout");
    assert_eq!(clock_layout.slot_count, 4);
    match &clock_layout.kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                },
        } => {
            assert_eq!(*max_len, 3);
            assert_eq!(
                *bound,
                SequenceBoundEvidence::FixedDomainTypeLayout {
                    invariant: Arc::from("TypeOK")
                }
            );
            assert_eq!(
                element_layout.as_ref(),
                &FlatValueLayout::Scalar(SlotType::Int)
            );
        }
        other => panic!("expected recursive clock sequence layout, got {other:?}"),
    }
    let req_layout = layout.var_layout(req_idx).expect("req layout");
    assert_eq!(req_layout.slot_count, 13);
    let VarLayoutKind::Recursive { layout: req } = &req_layout.kind else {
        panic!("expected recursive req layout, got {:?}", req_layout.kind);
    };
    let FlatValueLayout::Sequence {
        bound: req_bound,
        max_len: req_len,
        element_layout: req_row_layout,
    } = req
    else {
        panic!("expected req as sequence layout, got {req:?}");
    };
    assert_eq!(
        *req_bound,
        SequenceBoundEvidence::FixedDomainTypeLayout {
            invariant: Arc::from("TypeOK")
        }
    );
    assert_eq!(*req_len, 3);
    let FlatValueLayout::Sequence {
        bound: req_row_bound,
        max_len: req_row_len,
        element_layout: req_cell_layout,
    } = req_row_layout.as_ref()
    else {
        panic!("expected req rows as sequence layout, got {req_row_layout:?}");
    };
    assert_eq!(
        *req_row_bound,
        SequenceBoundEvidence::FixedDomainTypeLayout {
            invariant: Arc::from("TypeOK")
        }
    );
    assert_eq!(*req_row_len, 3);
    assert_eq!(
        req_cell_layout.as_ref(),
        &FlatValueLayout::Scalar(SlotType::Int)
    );
    match &layout.var_layout(ack_idx).unwrap().kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                },
        } => {
            assert_eq!(*max_len, 3);
            assert_eq!(
                *bound,
                SequenceBoundEvidence::FixedDomainTypeLayout {
                    invariant: Arc::from("TypeOK")
                }
            );
            match element_layout.as_ref() {
                FlatValueLayout::SetBitmask { universe, .. } => assert_eq!(universe.len(), 3),
                other => panic!("expected ack sequence values as SetBitmask, got {other:?}"),
            }
        }
        other => panic!("expected recursive ack sequence layout, got {other:?}"),
    }
    let network_layout = layout.var_layout(network_idx).expect("network layout");
    assert_eq!(network_layout.slot_count, 67);
    let VarLayoutKind::Recursive { layout: network } = &network_layout.kind else {
        panic!(
            "expected recursive network layout, got {:?}",
            network_layout.kind
        );
    };
    let FlatValueLayout::Sequence {
        bound: network_bound,
        max_len: network_len,
        element_layout: network_row_layout,
    } = network
    else {
        panic!("expected network as sequence layout, got {network:?}");
    };
    assert_eq!(
        *network_bound,
        SequenceBoundEvidence::FixedDomainTypeLayout {
            invariant: Arc::from("TypeOK")
        }
    );
    assert_eq!(*network_len, 3);
    let FlatValueLayout::Sequence {
        bound: row_bound,
        max_len: row_len,
        element_layout: channel_layout,
    } = network_row_layout.as_ref()
    else {
        panic!("expected network rows as sequence layout, got {network_row_layout:?}");
    };
    assert_eq!(
        *row_bound,
        SequenceBoundEvidence::FixedDomainTypeLayout {
            invariant: Arc::from("TypeOK")
        }
    );
    assert_eq!(*row_len, 3);
    let FlatValueLayout::Sequence {
        bound: channel_bound,
        max_len: channel_len,
        element_layout: message_layout,
    } = channel_layout.as_ref()
    else {
        panic!("expected network channels as sequence layout, got {channel_layout:?}");
    };
    assert_eq!(
        *channel_bound,
        SequenceBoundEvidence::ProvenInvariantWithElementLayout {
            invariant: Arc::from("BoundedNetwork"),
            element_invariant: Arc::from("TypeOK"),
        }
    );
    assert_eq!(*channel_len, 3);
    let FlatValueLayout::Record {
        field_names,
        field_layouts,
    } = message_layout.as_ref()
    else {
        panic!("expected message element record layout, got {message_layout:?}");
    };
    assert_eq!(field_names.len(), 2);
    let clock_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "clock")
        .expect("message clock field");
    let type_pos = field_names
        .iter()
        .position(|name| name.as_ref() == "type")
        .expect("message type field");
    assert_eq!(
        field_layouts[clock_pos],
        FlatValueLayout::Scalar(SlotType::Int)
    );
    assert_eq!(
        field_layouts[type_pos],
        FlatValueLayout::Scalar(SlotType::String)
    );
    match &layout.var_layout(crit_idx).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::SetBitmask { universe, .. },
        } => assert_eq!(universe.len(), 3),
        other => panic!("expected recursive crit bitmask, got {other:?}"),
    }
}

#[test]
fn test_malformed_network_bound_does_not_mark_sequence_proven() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryMalformedNetworkProof ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {1, 2}
\* This is only one concrete channel, not a universally quantified path.
BadNetworkBound == Len(network[1][1]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BadNetworkBound".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![observed_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for malformed bound");
    assert!(
        !layout.supports_flat_primary(),
        "non-universal channel bound must not prove all network[p][q] capacities"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(*bound, SequenceBoundEvidence::Observed);
                    assert_eq!(*max_len, 1);
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }
}

#[test]
fn test_subset_domain_network_bound_does_not_mark_sequence_proven() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimarySubsetDomainNetworkProof ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {1, 2}
\* This ranges over SUBSET Proc, so it must not prove every homogeneous network row.
BadSubsetDomainBound == \A s \in SUBSET Proc, q \in Proc : Len(network[s][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BadSubsetDomainBound".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![observed_network_value()]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for subset-domain bound");
    assert!(
        !layout.supports_flat_primary(),
        "SUBSET-domain proof must not make the recursive sequence primary-safe"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(*bound, SequenceBoundEvidence::Observed);
                    assert_eq!(*max_len, 1);
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }
}

#[test]
fn test_invalid_capacity_domains_do_not_mark_sequence_proven() {
    let cases = [
        (
            "FlatPrimaryPartialLiteralDomainProof",
            r#"BadBound == \A p \in {1}, q \in Proc : Len(network[p][q]) <= 3"#,
            "partial literal domain",
        ),
        (
            "FlatPrimaryProperSubsetAliasDomainProof",
            r#"SomeProc == {1}
BadBound == \A p \in SomeProc, q \in Proc : Len(network[p][q]) <= 3"#,
            "proper-subset alias domain",
        ),
        (
            "FlatPrimarySetMinusUnknownAliasDomainProof",
            r#"SomeProc == Proc \ Unknown
BadBound == \A p \in SomeProc, q \in Proc : Len(network[p][q]) <= 3"#,
            "set-minus alias with unresolved RHS",
        ),
        (
            "FlatPrimaryLiteralDomainProof",
            r#"BadBound == \A p \in {1, 2}, q \in Proc : Len(network[p][q]) <= 3"#,
            "literal domain",
        ),
        (
            "FlatPrimaryRangeDomainProof",
            r#"BadBound == \A p \in 1..2, q \in Proc : Len(network[p][q]) <= 3"#,
            "range domain",
        ),
        (
            "FlatPrimaryUnknownDomainProof",
            r#"BadBound == \A p \in Unknown, q \in Proc : Len(network[p][q]) <= 3"#,
            "arbitrary identifier domain",
        ),
        (
            "FlatPrimaryProperSubsetDomainProof",
            r#"BadBound == \A p \in Proc \ {1}, q \in Proc : Len(network[p][q]) <= 3"#,
            "proper-subset domain",
        ),
        (
            "FlatPrimaryDiagonalDomainProof",
            r#"BadBound == \A p \in Proc : Len(network[p][p]) <= 3"#,
            "diagonal-only domain",
        ),
        (
            "FlatPrimaryShadowedProcDomainProof",
            r#"BadBound == \A Proc \in {1, 2} : \A p, q \in Proc : Len(network[p][q]) <= 3"#,
            "shadowed domain operator",
        ),
        (
            "FlatPrimaryShadowedNetworkDomainProof",
            r#"BadBound == \A network \in Proc, p \in Proc, q \in Proc : Len(network[p][q]) <= 3"#,
            "bound variable shadows state variable",
        ),
        (
            "FlatPrimaryNestedShadowedDomainProof",
            r#"BadBound == \A p \in Proc : \A p \in SUBSET Proc, q \in Proc : Len(network[p][q]) <= 3"#,
            "nested shadowed bound variable",
        ),
    ];

    for (module_name, bad_bound, message) in cases {
        let source = format!(
            r#"
---- MODULE {module_name} ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {{1, 2}}
{bad_bound}
====
"#
        );
        let module = parse_module(&source);
        let mut config = Config::default();
        config.invariants.push("BadBound".to_string());
        let mut checker = ModelChecker::new(&module, &config);

        checker.infer_flat_state_layout(&ArrayState::from_values(vec![observed_network_value()]));

        let layout = checker
            .flat_state_layout()
            .expect("layout should be inferred for invalid capacity proof");
        assert_network_channel_bound_observed(layout, message);
    }
}

#[test]
fn test_state_variable_capacity_domain_does_not_mark_sequence_proven() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryStateVarDomainProof ----
EXTENDS Naturals, Sequences
VARIABLES network, active
Proc == {1, 2}
BadBound == \A p \in active, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BadBound".to_string());
    let mut checker = ModelChecker::new(&module, &config);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![
        observed_network_value(),
        Value::SmallInt(1),
    ]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for state-domain capacity proof");
    assert_network_channel_bound_observed(layout, "state-variable domain");
}

#[test]
fn test_typeok_direct_domains_prove_replacement_aware_mcl_element_layout() {
    let cases = [
        (
            "FlatPrimaryLiteralTypeDomainProof",
            r#"MCTypeOK == network \in [{1, 2} -> [{1, 2} -> Seq(Message)]]"#,
            "literal TypeOK domains",
        ),
        (
            "FlatPrimaryRangeTypeDomainProof",
            r#"MCTypeOK == network \in [1..2 -> [1..2 -> Seq(Message)]]"#,
            "range TypeOK domains",
        ),
    ];

    for (module_name, replacement_type_ok, message) in cases {
        let source = format!(
            r#"
---- MODULE {module_name} ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {{1, 2}}
Message == {{
    [type |-> "req", clock |-> 1],
    [type |-> "ack", clock |-> 0],
    [type |-> "rel", clock |-> 0]
}}
TypeOK == network \in [Proc \ {{1}} -> [Proc -> Seq(Message)]]
{replacement_type_ok}
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#
        );
        let module = parse_module(&source);
        let mut config = Config::default();
        config.add_constant(
            "TypeOK".to_string(),
            ConstantValue::Replacement("MCTypeOK".to_string()),
        );
        config.invariants.push("TypeOK".to_string());
        config.invariants.push("BoundedNetwork".to_string());
        let mut checker = ModelChecker::new(&module, &config);
        bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
        precompute_constant_operators(&mut checker.ctx);

        checker.infer_flat_state_layout(&ArrayState::from_values(vec![empty_network_value()]));

        let layout = checker
            .flat_state_layout()
            .expect("layout should be inferred for direct-domain TypeOK proof");
        assert!(
            layout.supports_flat_primary(),
            "{message}: replacement-routed direct domains should prove sequence element layout"
        );
        assert!(
            checker.is_flat_state_primary(),
            "{message}: direct-domain TypeOK plus BoundedNetwork should activate flat_state_primary"
        );
        match &layout.var_layout(0).unwrap().kind {
            VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction { value_layout, .. },
            } => match value_layout.as_ref() {
                FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                    FlatValueLayout::Sequence {
                        bound,
                        max_len,
                        element_layout,
                    } => {
                        assert_eq!(*max_len, 3, "{message}");
                        assert_eq!(
                            *bound,
                            SequenceBoundEvidence::ProvenInvariantWithElementLayout {
                                invariant: Arc::from("BoundedNetwork"),
                                element_invariant: Arc::from("TypeOK"),
                            },
                            "{message}"
                        );
                        assert_message_record_layout(element_layout, message);
                    }
                    other => {
                        panic!("{message}: expected network channel sequence layout, got {other:?}")
                    }
                },
                other => {
                    panic!("{message}: expected nested network function layout, got {other:?}")
                }
            },
            other => panic!("{message}: expected recursive network layout, got {other:?}"),
        }
    }
}

#[test]
fn test_invalid_typeok_domains_do_not_prove_mcl_element_layout() {
    let cases = [
        (
            "FlatPrimaryUnknownTypeDomainProof",
            r#"TypeOK == network \in [Unknown -> [Proc -> Seq(Message)]]"#,
            "arbitrary identifier TypeOK domain",
        ),
        (
            "FlatPrimaryStateVarTypeDomainProof",
            r#"TypeOK == network \in [active -> [Proc -> Seq(Message)]]"#,
            "state-variable TypeOK domain",
        ),
        (
            "FlatPrimaryProperSubsetAliasTypeDomainProof",
            r#"SomeProc == {1}
TypeOK == network \in [SomeProc -> [Proc -> Seq(Message)]]"#,
            "proper-subset alias TypeOK domain",
        ),
        (
            "FlatPrimarySetMinusUnknownAliasTypeDomainProof",
            r#"SomeProc == Proc \ Unknown
TypeOK == network \in [SomeProc -> [Proc -> Seq(Message)]]"#,
            "set-minus alias TypeOK domain with unresolved RHS",
        ),
        (
            "FlatPrimaryProperSubsetTypeDomainProof",
            r#"TypeOK == network \in [Proc \ {1} -> [Proc -> Seq(Message)]]"#,
            "proper-subset TypeOK domain",
        ),
    ];

    for (module_name, type_ok, message) in cases {
        let source = format!(
            r#"
---- MODULE {module_name} ----
EXTENDS Naturals, Sequences
VARIABLES network, active
Proc == {{1, 2}}
Message == {{
    [type |-> "req", clock |-> 1],
    [type |-> "ack", clock |-> 0],
    [type |-> "rel", clock |-> 0]
}}
{type_ok}
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#
        );
        let module = parse_module(&source);
        let mut config = Config::default();
        config.invariants.push("TypeOK".to_string());
        config.invariants.push("BoundedNetwork".to_string());
        let mut checker = ModelChecker::new(&module, &config);
        precompute_constant_operators(&mut checker.ctx);

        let mut init_values = vec![Value::SmallInt(0); checker.ctx.var_registry().len()];
        let network_idx = checker
            .ctx
            .var_registry()
            .get("network")
            .expect("network var")
            .as_usize();
        let active_idx = checker
            .ctx
            .var_registry()
            .get("active")
            .expect("active var")
            .as_usize();
        init_values[network_idx] = observed_network_value();
        init_values[active_idx] = Value::SmallInt(1);

        checker.infer_flat_state_layout(&ArrayState::from_values(init_values));

        let layout = checker
            .flat_state_layout()
            .expect("layout should be inferred for invalid TypeOK proof");
        assert!(
            !layout.supports_flat_primary(),
            "{message}: invalid TypeOK domain must not prove sequence element layout"
        );
        assert!(
            !checker.is_flat_state_primary(),
            "{message}: invalid TypeOK domain must not activate flat_state_primary"
        );
        assert_network_channel_capacity_only_at(layout, network_idx, message);
    }
}

#[test]
fn test_quantified_element_only_typeok_does_not_prove_parent_sequence_domains() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryQuantifiedElementOnlyTypeOK ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {1, 2}
Message == {
    [type |-> "req", clock |-> 1],
    [type |-> "ack", clock |-> 0],
    [type |-> "rel", clock |-> 0]
}
TypeOK == \A p, q \in Proc : network[p][q] \in Seq(Message)
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![
        empty_sequence_network_value(),
    ]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for element-only TypeOK");
    assert_sequence_network_parent_bounds_observed(layout, "element-only quantified TypeOK");
    assert!(
        !checker.is_flat_state_primary(),
        "element-only quantified TypeOK must not activate flat_state_primary"
    );
}

#[test]
fn test_replacement_routed_fixed_domain_typeok_promotes_sequence_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryReplacementFixedDomainSequence ----
EXTENDS Naturals, Sequences
CONSTANT Proc
VARIABLE clock
MCProc == {1, 2, 3}
TypeOK == clock \in [Proc -> Nat]
====
"#,
    );
    let mut config = Config::default();
    config.add_constant(
        "Proc".to_string(),
        ConstantValue::Replacement("MCProc".to_string()),
    );
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![Value::Seq(Rp::new(
        SeqValue::from_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(1),
            Value::SmallInt(1),
        ]),
    ))]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for replacement-routed fixed-domain TypeOK");
    assert_fixed_int_sequence_layout(layout, 3, "replacement-routed fixed-domain TypeOK");
    assert!(
        checker.is_flat_state_primary(),
        "replacement-routed fixed-domain sequence layout should activate flat_state_primary"
    );
}

#[test]
fn test_operator_wrapped_fixed_domain_sequence_enables_native_fused_non_primary_admission() {
    let module = parse_module(
        r#"
---- MODULE NativeFusedWrappedFixedDomainSequence ----
EXTENDS Naturals, Sequences
VARIABLES clock, pc
Idx == 1..2
Clock == clock
ClockType(d, r) == [d -> r]
TypeOK ==
    /\ Clock \in ClockType(Idx, Nat)
    /\ pc \in [{"p1", "p2"} -> {0, 1}]
====
"#,
    );
    let mut config = Config {
        use_flat_state: Some(true),
        use_compiled_bfs: Some(true),
        ..Default::default()
    };
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    let pc = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (Value::String(Rp::from("p1")), Value::SmallInt(0)),
        (Value::String(Rp::from("p2")), Value::SmallInt(0)),
    ])));

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![
        Value::Seq(Rp::new(SeqValue::from_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(1),
        ]))),
        pc,
    ]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for wrapped fixed-domain TypeOK");
    assert_fixed_int_sequence_layout(layout, 2, "operator-wrapped fixed-domain TypeOK");
    assert!(
        !layout.supports_flat_primary(),
        "string-keyed i64-range pc carries no scalar-interning range proof, so it keeps the whole layout out of flat-primary storage"
    );
    assert!(
        !layout.supports_flat_bfs_auto_admission(),
        "string-keyed i64-range pc keeps default flat-BFS auto admission fail-closed"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "non-primary mixed layout must not flip global flat-primary storage"
    );
    assert!(
        checker.native_fused_flat_frontier_admission_candidate_for_strict(true),
        "strict native-fused admission should accept the non-primary layout once the wrapped sequence proof lowers"
    );

    let flat_slots = checker
        .flat_bfs_adapter
        .as_ref()
        .expect("flat adapter should be installed")
        .num_slots();
    checker.compiled_bfs_level = Some(Box::new(TestNativeFusedInvariantBfsLevel {
        state_len: flat_slots,
    }));
    assert!(
        !checker.native_fused_flat_frontier_admission_active_for_strict(true),
        "non-primary invariant-only flat-frontier native-fused activation stays fail-closed until #4433 proves parent-loop parity"
    );
}

/// Regression: a finite string-enum function range whose *domain* is written in
/// terms of model-constant arithmetic (e.g. EWD998's `Node == 0 .. N-1`) must
/// still collect the `TypeOK` range proof so the IntArray/Record string vars
/// auto-admit to flat BFS without forced flags. Before the constant-arithmetic
/// fold in `const_int_value_with_replacements`, the inlined `0 .. N-1` bound
/// failed to resolve, the range proof was dropped, and the spec fell back to the
/// interpreter.
#[test]
fn test_constant_arithmetic_function_domain_enables_string_enum_auto_admission() {
    let module = parse_module(
        r#"
---- MODULE ConstArithDomainEnum ----
EXTENDS Integers, FiniteSets
CONSTANT N
Node == 0 .. N-1
Color == {"white", "black"}
Token == [pos : Node, q : Int, color : Color]
VARIABLES active, color, counter, pending, token
TypeOK ==
  /\ active \in [Node -> BOOLEAN]
  /\ color \in [Node -> Color]
  /\ counter \in [Node -> Int]
  /\ pending \in [Node -> Nat]
  /\ token \in Token
====
"#,
    );
    let mut config = Config::default();
    config.add_constant("N".to_string(), ConstantValue::Value("3".to_string()));
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    bind_constants_from_config(&mut checker.ctx, &config).expect("config constants bind");
    precompute_constant_operators(&mut checker.ctx);
    promote_env_constants_to_precomputed(&mut checker.ctx);

    let active = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (Value::SmallInt(0), Value::Bool(true)),
        (Value::SmallInt(1), Value::Bool(true)),
        (Value::SmallInt(2), Value::Bool(true)),
    ])));
    let color_v = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (Value::SmallInt(0), Value::String(Rp::from("white"))),
        (Value::SmallInt(1), Value::String(Rp::from("white"))),
        (Value::SmallInt(2), Value::String(Rp::from("white"))),
    ])));
    let counter = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
        (Value::SmallInt(0), Value::SmallInt(0)),
        (Value::SmallInt(1), Value::SmallInt(0)),
        (Value::SmallInt(2), Value::SmallInt(0)),
    ])));
    let pending = counter.clone();
    let token = Value::Record(RecordValue::from_sorted_str_entries(vec![
        (Arc::from("color"), Value::String(Rp::from("black"))),
        (Arc::from("pos"), Value::SmallInt(0)),
        (Arc::from("q"), Value::SmallInt(0)),
    ]));

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![
        active, color_v, counter, pending, token,
    ]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred");
    assert!(
        layout.supports_flat_bfs_auto_admission(),
        "string-enum function range over a constant-arithmetic domain (0 .. N-1) \
         must auto-admit to flat BFS"
    );
}

#[test]
fn test_empty_fixed_domain_typeok_does_not_promote_sequence_layout() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryEmptyFixedDomainSequence ----
EXTENDS Naturals, Sequences
VARIABLE clock
Empty == {}
TypeOK == clock \in [Empty -> Nat]
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    precompute_constant_operators(&mut checker.ctx);

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![Value::Seq(Rp::new(
        SeqValue::from_vec(Vec::new()),
    ))]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for empty fixed-domain TypeOK");
    assert!(
        !layout.supports_flat_primary(),
        "empty fixed-domain TypeOK must not make a sequence primary-safe"
    );
    assert!(
        !checker.is_flat_state_primary(),
        "empty fixed-domain TypeOK must not activate flat_state_primary"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::Sequence { bound, .. },
        } => assert_eq!(
            *bound,
            SequenceBoundEvidence::Observed,
            "empty fixed-domain evidence should be rejected"
        ),
        other => panic!("expected recursive sequence layout, got {other:?}"),
    }
}

#[test]
fn test_fixed_domain_typeok_range_must_prove_observed_element_layout() {
    let cases = [
        (
            "FlatPrimaryUnknownFixedDomainRange",
            r#"TypeOK == clock \in [Proc -> Unknown]"#,
            "unknown fixed-domain range",
        ),
        (
            "FlatPrimaryMismatchedBooleanFixedDomainRange",
            r#"TypeOK == clock \in [Proc -> BOOLEAN]"#,
            "mismatched BOOLEAN fixed-domain range",
        ),
    ];

    for (module_name, type_ok, message) in cases {
        let source = format!(
            r#"
---- MODULE {module_name} ----
EXTENDS Naturals, Sequences
VARIABLE clock
Proc == {{1, 2}}
{type_ok}
====
"#
        );
        let module = parse_module(&source);
        let mut config = Config::default();
        config.invariants.push("TypeOK".to_string());
        let mut checker = ModelChecker::new(&module, &config);
        precompute_constant_operators(&mut checker.ctx);

        checker.infer_flat_state_layout(&ArrayState::from_values(vec![Value::Seq(Rp::new(
            SeqValue::from_vec(vec![Value::SmallInt(1), Value::SmallInt(1)]),
        ))]));

        let layout = checker
            .flat_state_layout()
            .expect("layout should be inferred for fixed-domain range mismatch");
        assert_single_sequence_bound_observed(layout, message);
        assert!(
            !checker.is_flat_state_primary(),
            "{message}: fixed-domain range mismatch must not activate flat_state_primary"
        );
    }
}

#[test]
fn test_broad_fixed_domain_typeok_range_proves_sequence_primary_safe() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryBroadFixedDomainRange ----
EXTENDS Naturals, Sequences
VARIABLE clock
TypeOK == clock \in [1..64 -> Nat]
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("TypeOK".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    let values = (1..=64).map(|_| Value::SmallInt(1)).collect();

    checker.infer_flat_state_layout(&ArrayState::from_values(vec![Value::Seq(Rp::new(
        SeqValue::from_vec(values),
    ))]));

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred for broad fixed-domain TypeOK");
    assert!(
        layout.supports_flat_primary(),
        "broad fixed-domain TypeOK with scalar range should remain primary-safe"
    );
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout:
                FlatValueLayout::Sequence {
                    bound,
                    max_len,
                    element_layout,
                },
        } => {
            assert_eq!(*max_len, 64);
            assert_eq!(
                *bound,
                SequenceBoundEvidence::FixedDomainTypeLayout {
                    invariant: Arc::from("TypeOK")
                }
            );
            assert_eq!(
                element_layout.as_ref(),
                &FlatValueLayout::Scalar(SlotType::Int)
            );
        }
        other => panic!("expected broad recursive sequence layout, got {other:?}"),
    }
    assert!(
        checker.is_flat_state_primary(),
        "broad fixed-domain TypeOK should activate flat_state_primary"
    );
}

#[test]
fn test_empty_network_wavefront_channels_inherit_proven_bound_but_not_primary_safe() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryEmptyBoundedNetworkProof ----
EXTENDS Naturals, Sequences
VARIABLE network
Proc == {1, 2}
BoundedNetwork == \A p, q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let mut config = Config::default();
    config.invariants.push("BoundedNetwork".to_string());
    let mut checker = ModelChecker::new(&module, &config);
    let states = vec![
        ArrayState::from_values(vec![empty_network_value()]),
        ArrayState::from_values(vec![observed_network_value()]),
    ];

    checker.infer_flat_state_layout_from_wavefront(&states);

    let layout = checker
        .flat_state_layout()
        .expect("layout should be inferred from network wavefront");
    assert!(
        !layout.supports_flat_primary(),
        "empty channels can inherit proven capacity, but not primary safety for observed element shape"
    );
    assert!(!checker.is_flat_state_primary());
    match &layout.var_layout(0).unwrap().kind {
        VarLayoutKind::Recursive {
            layout: FlatValueLayout::IntFunction { value_layout, .. },
        } => match value_layout.as_ref() {
            FlatValueLayout::IntFunction { value_layout, .. } => match value_layout.as_ref() {
                FlatValueLayout::Sequence { bound, max_len, .. } => {
                    assert_eq!(*max_len, 3);
                    assert_eq!(
                        *bound,
                        SequenceBoundEvidence::ProvenInvariant {
                            invariant: Arc::from("BoundedNetwork")
                        }
                    );
                }
                other => panic!("expected network channel sequence layout, got {other:?}"),
            },
            other => panic!("expected nested network function layout, got {other:?}"),
        },
        other => panic!("expected recursive network layout, got {other:?}"),
    }
}

#[test]
fn test_flat_state_primary_rejects_fixed_layout_when_full_state_storage_active() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryFullStateGuard ----
VARIABLE rec
====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    checker.set_store_states(true);

    checker.infer_flat_state_layout(&fixed_record_init_state());

    assert!(
        !checker.is_flat_state_primary(),
        "full-state storage must keep BFS out of the flat primary fingerprint domain"
    );
}

#[test]
fn test_flat_state_primary_rejects_fixed_layout_with_view_or_symmetry() {
    let module = parse_module(
        r#"
---- MODULE FlatPrimaryViewSymmetryGuard ----
VARIABLE rec
====
"#,
    );
    let config = Config::default();

    let mut view_checker = ModelChecker::new(&module, &config);
    view_checker.compiled.cached_view_name = Some("View".to_string());
    view_checker.infer_flat_state_layout(&fixed_record_init_state());
    assert!(
        !view_checker.is_flat_state_primary(),
        "VIEW runs must stay out of flat_state_primary"
    );

    let mut symmetry_checker = ModelChecker::new(&module, &config);
    symmetry_checker
        .symmetry
        .perms
        .push(FuncValue::from_sorted_entries(Vec::<(Value, Value)>::new()));
    symmetry_checker.infer_flat_state_layout(&fixed_record_init_state());
    assert!(
        !symmetry_checker.is_flat_state_primary(),
        "SYMMETRY runs must stay out of flat_state_primary"
    );
}

/// #3125 regression: a zero-arg wrapper (`JsonInv`) that transitively references
/// a state-level operator (`Inv`) must NOT be precomputed. The old shallow gate
/// missed this because it only checked for direct `Expr::StateVar` nodes.
#[test]
fn test_precompute_skips_transitive_state_wrapper() {
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");

    // Inv == x = 0  (state-level: references state variable x)
    let inv_body = Expr::Eq(
        Box::new(Spanned::dummy(Expr::StateVar(
            "x".to_string(),
            0,
            intern_name("x"),
        ))),
        Box::new(Spanned::dummy(Expr::Int(0.into()))),
    );
    ctx.define_op("Inv".to_string(), make_op("Inv", inv_body));

    // JsonInv == Inv  (wrapper — body is just Ident("Inv"))
    let json_inv_body = Expr::Ident("Inv".to_string(), intern_name("Inv"));
    ctx.define_op("JsonInv".to_string(), make_op("JsonInv", json_inv_body));

    precompute_constant_operators(&mut ctx);

    let name_id = intern_name("JsonInv");
    assert!(
        ctx.shared().precomputed_constants.get(&name_id).is_none(),
        "JsonInv transitively references state var x — must NOT be precomputed"
    );

    let inv_id = intern_name("Inv");
    assert!(
        ctx.shared().precomputed_constants.get(&inv_id).is_none(),
        "Inv directly references state var x — must NOT be precomputed"
    );
}

/// Genuine constant operators (no state dependency at any level) must still
/// be precomputed for O(1) lookup during BFS.
#[test]
fn test_precompute_keeps_true_constants() {
    let mut ctx = EvalCtx::new();

    // N == 3  (constant: pure integer literal)
    let n_body = Expr::Int(3.into());
    ctx.define_op("N".to_string(), make_op("N", n_body));

    precompute_constant_operators(&mut ctx);

    let name_id = intern_name("N");
    assert!(
        ctx.shared().precomputed_constants.get(&name_id).is_some(),
        "N is a pure constant — must be precomputed"
    );
}

/// A constant wrapper over a constant operator should still be precomputed.
#[test]
fn test_precompute_keeps_transitive_constant_wrapper() {
    let mut ctx = EvalCtx::new();

    // Base == 42
    let base_body = Expr::Int(42.into());
    ctx.define_op("Base".to_string(), make_op("Base", base_body));

    // Wrapper == Base  (transitively constant)
    let wrapper_body = Expr::Ident("Base".to_string(), intern_name("Base"));
    ctx.define_op("Wrapper".to_string(), make_op("Wrapper", wrapper_body));

    precompute_constant_operators(&mut ctx);

    let base_id = intern_name("Base");
    assert!(
        ctx.shared().precomputed_constants.get(&base_id).is_some(),
        "Base is constant — must be precomputed"
    );

    let wrapper_id = intern_name("Wrapper");
    assert!(
        ctx.shared()
            .precomputed_constants
            .get(&wrapper_id)
            .is_some(),
        "Wrapper transitively references only constants — must be precomputed"
    );
}

#[test]
fn test_compile_action_bytecode_prunes_unsafe_and_unrewriteable_actions() {
    let module = parse_module(
        r#"
---- MODULE PrepareActionBytecodePrune ----
EXTENDS Naturals

VARIABLES x, y

Good ==
    /\ x' = x + 1
    /\ y' = y

UnsafeCrossPrime ==
    /\ x' = y'
    /\ y' = y + 1

ValidatorOverlapForward ==
    /\ UNCHANGED x
    /\ x' = x + 1
    /\ y' = y

ValidatorOverlapBackward ==
    /\ x' = x + 1
    /\ UNCHANGED x
    /\ y' = y

NoRewriteGuard ==
    /\ x < 10
    /\ y < 10

====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    seed_split_action_meta(
        &mut checker,
        &[
            "Good",
            "UnsafeCrossPrime",
            "ValidatorOverlapBackward",
            "NoRewriteGuard",
        ],
    );

    checker.compile_action_bytecode();

    let compiled = checker
        .action_bytecode
        .as_ref()
        .expect("Good should remain in action_bytecode after pruning invalid actions");

    assert!(
        compiled.op_indices.contains_key("Good"),
        "safe actions should remain available for next-state JIT compilation",
    );
    assert!(
        !compiled.op_indices.contains_key("UnsafeCrossPrime"),
        "unsafe actions must be pruned from action_bytecode",
    );
    assert!(
        !compiled.op_indices.contains_key("NoRewriteGuard"),
        "actions with no rewrite must be pruned from action_bytecode",
    );
    assert!(
        !compiled.op_indices.contains_key("ValidatorOverlapBackward"),
        "validator-rejected actions must be pruned from action_bytecode",
    );
    assert_eq!(
        compiled.failed.len(),
        3,
        "unsafe, validator-rejected, and no-rewrite actions should be surfaced as failures",
    );

    assert_failed_message_contains(
        compiled,
        "UnsafeCrossPrime",
        "unsafe next-state transform: residual LoadPrime",
    );
    assert_failed_message_contains(
        compiled,
        "ValidatorOverlapBackward",
        "unsafe next-state transform: primed var 0 is both written and UNCHANGED",
    );
    assert_failed_message_contains(
        compiled,
        "NoRewriteGuard",
        "no safe next-state rewrite found",
    );
}

#[test]
fn test_if_shaped_next_produces_safe_action_bytecode_and_preserves_state_count() {
    let module = parse_module(
        r#"
---- MODULE PrepareIfShapedNextBytecode ----
EXTENDS Naturals

VARIABLE x

Init == x = 0
Next == IF x < 2 THEN x' = x + 1 ELSE UNCHANGED x
Inv == x <= 2

====
"#,
    );
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    config.invariants.push("Inv".to_string());

    let mut checker = ModelChecker::new(&module, &config);
    match checker.check() {
        CheckResult::Success(stats) => {
            assert_eq!(stats.states_found, 3);
        }
        other => panic!("IF-shaped action model should preserve BFS parity, got {other:?}"),
    }

    let meta = checker
        .compiled
        .split_action_meta
        .as_ref()
        .expect("IF-shaped Next should still produce split metadata");
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].name.as_deref(), Some("Next"));

    if checker.action_bytecode.is_none() {
        checker.compile_action_bytecode();
    }
    let compiled = checker
        .action_bytecode
        .as_ref()
        .expect("IF-shaped Next should produce safe next-state action bytecode");
    assert!(
        compiled.op_indices.contains_key("Next"),
        "safe action bytecode should retain the monolithic IF-shaped Next entry"
    );
    assert!(
        !compiled.failed.iter().any(|(name, _)| name == "Next"),
        "IF-shaped Next should not be reported as an unsafe action: {:?}",
        compiled.failed
    );
}

#[test]
fn test_compile_action_bytecode_prunes_transitive_prime_helpers() {
    let module = parse_module(
        r#"
---- MODULE PrepareActionBytecodeTransitiveCallees ----
EXTENDS Naturals

VARIABLES x, y

SafeValue ==
    x + 1

PrimeValue ==
    x'

PrimeModeCheck ==
    UNCHANGED (x + 1)

SafeWrapped ==
    /\ x' = SafeValue
    /\ y' = y

HiddenPrimeWrapped ==
    /\ x' = PrimeValue
    /\ y' = y

HiddenPrimeModeWrapped ==
    /\ x' = x + 1
    /\ y' = y
    /\ PrimeModeCheck

====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    seed_split_action_meta(
        &mut checker,
        &[
            "SafeWrapped",
            "HiddenPrimeWrapped",
            "HiddenPrimeModeWrapped",
        ],
    );

    checker.compile_action_bytecode();

    let compiled = checker
        .action_bytecode
        .as_ref()
        .expect("SafeWrapped should remain in action_bytecode after transitive pruning");

    assert!(
        compiled.op_indices.contains_key("SafeWrapped"),
        "actions that only reach pure helpers should remain eligible",
    );
    assert!(
        !compiled.op_indices.contains_key("HiddenPrimeWrapped"),
        "actions reaching helper callees with LoadPrime must be pruned",
    );
    assert!(
        !compiled.op_indices.contains_key("HiddenPrimeModeWrapped"),
        "actions reaching helper callees with SetPrimeMode must be pruned",
    );
    assert_eq!(
        compiled.failed.len(),
        2,
        "only the transitive helper violations should surface as failures",
    );

    assert_failed_message_contains(
        compiled,
        "HiddenPrimeWrapped",
        "unsafe next-state transform: reachable callee",
    );
    assert_failed_message_contains(compiled, "HiddenPrimeWrapped", "contains LoadPrime");
    assert_failed_message_contains(
        compiled,
        "HiddenPrimeModeWrapped",
        "unsafe next-state transform: reachable callee",
    );
    assert_failed_message_contains(compiled, "HiddenPrimeModeWrapped", "contains SetPrimeMode");
}

#[test]
fn test_compile_action_bytecode_prunes_multi_hop_prime_helpers() {
    let module = parse_module(
        r#"
---- MODULE PrepareActionBytecodeMultiHopCallees ----
EXTENDS Naturals

VARIABLES x, y

SafeLeaf ==
    x + 1

SafeMid ==
    SafeLeaf

PrimeLeaf ==
    x'

PrimeMid ==
    PrimeLeaf

PrimeModeLeaf ==
    UNCHANGED (x + 1)

PrimeModeMid ==
    PrimeModeLeaf

SafeWrapped ==
    /\ x' = SafeMid
    /\ y' = y

HiddenPrimeWrapped ==
    /\ x' = PrimeMid
    /\ y' = y

HiddenPrimeModeWrapped ==
    /\ x' = x + 1
    /\ y' = y
    /\ PrimeModeMid

====
"#,
    );
    let config = Config::default();
    let mut checker = ModelChecker::new(&module, &config);
    seed_split_action_meta(
        &mut checker,
        &[
            "SafeWrapped",
            "HiddenPrimeWrapped",
            "HiddenPrimeModeWrapped",
        ],
    );

    checker.compile_action_bytecode();

    let compiled = checker
        .action_bytecode
        .as_ref()
        .expect("SafeWrapped should remain in action_bytecode after multi-hop pruning");

    assert!(
        compiled.op_indices.contains_key("SafeWrapped"),
        "actions that only reach pure helper chains should remain eligible",
    );
    assert!(
        !compiled.op_indices.contains_key("HiddenPrimeWrapped"),
        "actions reaching multi-hop helper chains with LoadPrime must be pruned",
    );
    assert!(
        !compiled.op_indices.contains_key("HiddenPrimeModeWrapped"),
        "actions reaching multi-hop helper chains with SetPrimeMode must be pruned",
    );
    assert_eq!(
        compiled.failed.len(),
        2,
        "only the multi-hop helper violations should surface as failures",
    );

    assert_failed_message_contains(
        compiled,
        "HiddenPrimeWrapped",
        "unsafe next-state transform: reachable callee",
    );
    assert_failed_message_contains(compiled, "HiddenPrimeWrapped", "contains LoadPrime");
    assert_failed_message_contains(
        compiled,
        "HiddenPrimeModeWrapped",
        "unsafe next-state transform: reachable callee",
    );
    assert_failed_message_contains(compiled, "HiddenPrimeModeWrapped", "contains SetPrimeMode");
}
