// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Interactive symbolic exploration via JSON-RPC over TCP.
//!
//! Provides a server that allows external tools to explore a TLA+ state space
//! step by step. Clients connect via TCP and issue JSON-RPC 2.0 requests:
//!
//! - `init()` — compute initial states from the Init predicate
//! - `step(state_id)` — compute successor states via the Next relation
//! - `eval(state_id, expr)` — evaluate a TLA+ expression in a given state
//! - `check_invariant(state_id, inv)` — check whether an invariant holds in a state
//!
//! This is modeled after Apalache's interactive mode (Gap 3) but uses the
//! concrete-state TY evaluation engine rather than symbolic SMT encoding.
//!
//! Part of #3751: Interactive symbolic exploration API.

use rustc_hash::FxHashMap;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::constants::bind_constants_from_config;
use crate::enumerate::{
    enumerate_states_from_constraint_branches, enumerate_successors, extract_init_constraints,
};
use crate::eval::{eval, EvalCtx};
use crate::json_output::JsonValue;
use crate::state::State;
use crate::value::Value;
use tla_core::ast::{Module, OperatorDef, Unit};

#[cfg(feature = "ay")]
use crate::symbolic_explore::SymbolicExplorer;
#[cfg(feature = "ay")]
use tla_mc_core::{ProblemKind, NO_REASON_CODE};

/// Default TCP port for the interactive server.
pub const DEFAULT_PORT: u16 = 8765;

/// Default maximum symbolic exploration depth.
pub const DEFAULT_MAX_SYMBOLIC_DEPTH: usize = 20;

// ---------------------------------------------------------------------------
// Exploration mode
// ---------------------------------------------------------------------------

/// Whether the interactive server uses concrete or symbolic exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServerExploreMode {
    /// Concrete-state exploration (default).
    Concrete,
    /// Symbolic SMT-based exploration (requires ay feature).
    Symbolic,
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
    id: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
    id: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

impl RpcResponse {
    fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: serde_json::Value, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            result: None,
            error: Some(RpcError { code, message }),
            id,
        }
    }
}

// JSON-RPC standard error codes
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

// ---------------------------------------------------------------------------
// State snapshot for JSON serialization
// ---------------------------------------------------------------------------

/// A state representation suitable for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Opaque identifier for this state within the server session.
    pub state_id: u64,
    /// Variable-name to JSON-value mapping.
    pub variables: FxHashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Module helpers
// ---------------------------------------------------------------------------

/// Extract variable names from a TLA+ module.
fn extract_var_names(module: &Module) -> Vec<Arc<str>> {
    let mut vars = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(var_decls) = &unit.node {
            for decl in var_decls {
                vars.push(Arc::<str>::from(decl.node.as_str()));
            }
        }
    }
    vars
}

/// Find an operator definition by name in a TLA+ module.
fn find_operator<'a>(module: &'a Module, name: &str) -> Option<&'a OperatorDef> {
    module.units.iter().find_map(|unit| {
        if let Unit::Operator(op_def) = &unit.node {
            if op_def.name.node == name {
                return Some(op_def);
            }
        }
        None
    })
}

// ---------------------------------------------------------------------------
// Interactive server
// ---------------------------------------------------------------------------

/// Holds the loaded spec and accumulated states for interactive exploration.
#[allow(dead_code)] // max_symbolic_depth is only read behind #[cfg(feature = "ay")]
pub struct InteractiveServer {
    /// The loaded TLA+ module.
    module: Arc<Module>,
    /// Parsed configuration (Init, Next, invariants, constants, ...).
    config: Config,
    /// Monotonic counter for state IDs.
    next_id: u64,
    /// All states discovered so far, indexed by their assigned ID.
    states: FxHashMap<u64, State>,
    /// Cached variable names extracted from the module.
    var_names: Vec<Arc<str>>,
    /// Current exploration mode (concrete or symbolic).
    mode: ServerExploreMode,
    /// Maximum symbolic depth (only used in symbolic mode).
    max_symbolic_depth: usize,
    /// Symbolic exploration engine (lazily initialized when mode is Symbolic).
    #[cfg(feature = "ay")]
    symbolic: Option<SymbolicExplorer>,
}

impl InteractiveServer {
    /// Create a new server for the given module and configuration.
    pub fn new(module: Arc<Module>, config: Config) -> Self {
        let var_names = extract_var_names(&module);
        Self {
            module,
            config,
            next_id: 0,
            states: FxHashMap::default(),
            var_names,
            mode: ServerExploreMode::Concrete,
            max_symbolic_depth: DEFAULT_MAX_SYMBOLIC_DEPTH,
            #[cfg(feature = "ay")]
            symbolic: None,
        }
    }

    /// Create a new server with a specific exploration mode and max symbolic depth.
    pub fn with_mode(
        module: Arc<Module>,
        config: Config,
        mode: ServerExploreMode,
        max_symbolic_depth: usize,
    ) -> Self {
        let var_names = extract_var_names(&module);
        Self {
            module,
            config,
            next_id: 0,
            states: FxHashMap::default(),
            var_names,
            mode,
            max_symbolic_depth,
            #[cfg(feature = "ay")]
            symbolic: None,
        }
    }

    // -- Helpers ----------------------------------------------------------

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn state_to_snapshot(&self, id: u64, state: &State) -> StateSnapshot {
        let variables = state
            .vars()
            .map(|(k, v)| (k.to_string(), value_to_serde(v)))
            .collect();
        StateSnapshot {
            state_id: id,
            variables,
        }
    }

    fn get_state(&self, state_id: u64) -> Result<&State, String> {
        self.states
            .get(&state_id)
            .ok_or_else(|| format!("unknown state_id: {state_id}"))
    }

    /// Build a fresh EvalCtx with the module loaded, variables registered,
    /// and constants bound.
    fn build_ctx(&self) -> Result<EvalCtx, String> {
        let mut ctx = EvalCtx::new();
        ctx.load_module(&self.module);
        // Register variables so they are known to the evaluator (needed for
        // init constraint extraction and successor enumeration).
        for name in &self.var_names {
            ctx.register_var(Arc::clone(name));
        }
        bind_constants_from_config(&mut ctx, &self.config).map_err(|e| format!("{e}"))?;
        Ok(ctx)
    }

    // -- RPC methods ------------------------------------------------------

    /// Compute initial states from the Init predicate.
    ///
    /// Returns a list of `StateSnapshot` objects.
    fn rpc_init(&mut self) -> Result<serde_json::Value, String> {
        let init_name = self
            .config
            .init
            .as_deref()
            .ok_or_else(|| "no INIT operator configured".to_string())?;

        let init_def = find_operator(&self.module, init_name)
            .ok_or_else(|| format!("Init operator not found: {init_name}"))?;

        let ctx = self.build_ctx()?;

        // Extract init constraints and enumerate states.
        let branches = extract_init_constraints(&ctx, &init_def.body, &self.var_names, None)
            .ok_or_else(|| "failed to extract Init constraints".to_string())?;

        let init_states =
            enumerate_states_from_constraint_branches(Some(&ctx), &self.var_names, &branches)
                .map_err(|e| format!("{e}"))?
                .ok_or_else(|| "Init constraint enumeration returned None".to_string())?;

        let mut snapshots = Vec::with_capacity(init_states.len());
        for state in init_states {
            let id = self.alloc_id();
            let snap = self.state_to_snapshot(id, &state);
            self.states.insert(id, state);
            snapshots.push(snap);
        }
        serde_json::to_value(&snapshots).map_err(|e| format!("serialization error: {e}"))
    }

    /// Compute successor states of a given state via the Next relation.
    fn rpc_step(&mut self, state_id: u64) -> Result<serde_json::Value, String> {
        let state = self.get_state(state_id)?.clone();

        let next_name = self
            .config
            .next
            .as_deref()
            .ok_or_else(|| "no NEXT operator configured".to_string())?;

        let next_def = find_operator(&self.module, next_name)
            .ok_or_else(|| format!("Next operator not found: {next_name}"))?
            .clone();

        let mut ctx = self.build_ctx()?;

        let successors = enumerate_successors(&mut ctx, &next_def, &state, &self.var_names)
            .map_err(|e| format!("{e}"))?;

        let mut snapshots = Vec::with_capacity(successors.len());
        for succ in successors {
            let id = self.alloc_id();
            let snap = self.state_to_snapshot(id, &succ);
            self.states.insert(id, succ);
            snapshots.push(snap);
        }
        serde_json::to_value(&snapshots).map_err(|e| format!("serialization error: {e}"))
    }

    /// Evaluate a TLA+ expression string in the context of a specific state.
    fn rpc_eval(&self, state_id: u64, expr_str: &str) -> Result<serde_json::Value, String> {
        // Each rpc_eval parses, evaluates, and DROPS an independent expression
        // AST. Some eval caches (e.g. the constant-set cache) are keyed by raw
        // AST-node pointer address; without a run-boundary reset the allocator
        // can reuse a freed node's address across REPL evals and the cache would
        // replay a STALE value from a prior, structurally-different expression.
        // Treat each REPL eval as a fresh run (run-boundary clear only — does
        // NOT touch interns or live bound state).
        tla_eval::clear_for_run_reset();
        let state = self.get_state(state_id)?;
        let mut ctx = self.build_ctx()?;

        // Bind state variables into the evaluation context.
        for (name, value) in state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        // Parse the expression as a tiny module with one operator.
        let src = format!("---- MODULE __interactive_eval ----\n__result == {expr_str}\n====");
        let tree = tla_core::parse_to_syntax_tree(&src);
        let lowered = tla_core::lower(tla_core::FileId(9999), &tree);
        let m = lowered
            .module
            .as_ref()
            .ok_or_else(|| "failed to parse expression".to_string())?;

        let result_def = find_operator(m, "__result")
            .ok_or_else(|| "internal: __result operator not found after parse".to_string())?;

        let result = eval(&ctx, &result_def.body).map_err(|e| format!("{e}"))?;
        Ok(value_to_serde(&result))
    }

    /// Generate a random execution trace of N steps starting from a random
    /// initial state.
    ///
    /// Returns a list of `StateSnapshot` objects representing the trace.
    /// The trace starts with one of the initial states (chosen at random)
    /// and follows randomly chosen successors for up to `steps` transitions.
    /// The trace ends early if a deadlock is reached (no successors).
    fn rpc_random_trace(&mut self, steps: usize) -> Result<serde_json::Value, String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;

        // Seed a simple RNG from system time.
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);

        let init_result = self.rpc_init()?;
        let init_snapshots: Vec<StateSnapshot> =
            serde_json::from_value(init_result).map_err(|e| format!("internal: {e}"))?;

        if init_snapshots.is_empty() {
            return Err("Init produced no states".to_string());
        }

        // Pick a random initial state.
        let pick = (seed as usize) % init_snapshots.len();
        let mut current_id = init_snapshots[pick].state_id;
        let mut trace = vec![init_snapshots[pick].clone()];

        let mut hasher_state = seed;
        for _ in 0..steps {
            let step_result = self.rpc_step(current_id)?;
            let successors: Vec<StateSnapshot> =
                serde_json::from_value(step_result).map_err(|e| format!("internal: {e}"))?;

            if successors.is_empty() {
                break; // Deadlock — no successors.
            }

            // Simple hash-based pseudo-random selection.
            let mut h = DefaultHasher::new();
            hasher_state.hash(&mut h);
            hasher_state = h.finish();
            let idx = (hasher_state as usize) % successors.len();
            current_id = successors[idx].state_id;
            trace.push(successors[idx].clone());
        }

        serde_json::to_value(&trace).map_err(|e| format!("serialization error: {e}"))
    }

    /// Check whether an invariant (by operator name) holds in a given state.
    fn rpc_check_invariant(
        &self,
        state_id: u64,
        inv_name: &str,
    ) -> Result<serde_json::Value, String> {
        let state = self.get_state(state_id)?;
        let mut ctx = self.build_ctx()?;

        for (name, value) in state.vars() {
            ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let inv_def = find_operator(&self.module, inv_name)
            .ok_or_else(|| format!("invariant operator not found: {inv_name}"))?;

        let result = eval(&ctx, &inv_def.body).map_err(|e| format!("{e}"))?;
        match result {
            Value::Bool(b) => {
                let resp = serde_json::json!({
                    "holds": b,
                    "invariant": inv_name,
                    "state_id": state_id,
                });
                Ok(resp)
            }
            other => Err(format!(
                "invariant {inv_name} returned non-boolean: {other:?}"
            )),
        }
    }

    // -- Symbolic RPC methods ---------------------------------------------

    /// Switch between concrete and symbolic exploration modes.
    fn rpc_set_mode(&mut self, mode_str: &str) -> Result<serde_json::Value, String> {
        match mode_str {
            "concrete" => {
                self.mode = ServerExploreMode::Concrete;
                Ok(serde_json::json!({"mode": "concrete"}))
            }
            "symbolic" => {
                #[cfg(feature = "ay")]
                {
                    self.mode = ServerExploreMode::Symbolic;
                    // Lazily initialize the symbolic explorer.
                    if self.symbolic.is_none() {
                        let ctx = self.build_ctx()?;
                        let explorer = SymbolicExplorer::new(
                            &self.module,
                            &self.config,
                            &ctx,
                            self.max_symbolic_depth,
                        )
                        .map_err(|e| format!("{e}"))?;
                        self.symbolic = Some(explorer);
                    }
                    Ok(serde_json::json!({"mode": "symbolic"}))
                }
                #[cfg(not(feature = "ay"))]
                {
                    Err("symbolic mode requires the ay feature".to_string())
                }
            }
            other => Err(format!(
                "unknown mode: {other}. Expected 'concrete' or 'symbolic'."
            )),
        }
    }

    /// Rollback the symbolic solver to the previous push point.
    fn rpc_rollback(&mut self) -> Result<serde_json::Value, String> {
        #[cfg(feature = "ay")]
        {
            let explorer = self.symbolic.as_mut().ok_or_else(|| {
                "symbolic explorer not initialized (call set_mode first)".to_string()
            })?;
            explorer.pop().map_err(|e| format!("{e}"))?;
            Ok(serde_json::json!({
                "depth": explorer.current_depth(),
                "status": "rolled_back"
            }))
        }
        #[cfg(not(feature = "ay"))]
        {
            Err("rollback requires the ay feature".to_string())
        }
    }

    /// Assume state constraints in symbolic mode.
    fn rpc_assume_state(
        &mut self,
        assignments: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        #[cfg(feature = "ay")]
        {
            use tla_ay::BmcValue;

            let explorer = self
                .symbolic
                .as_mut()
                .ok_or_else(|| "symbolic explorer not initialized".to_string())?;

            // Parse assignments from JSON: {"var_name": value, ...}
            let obj = assignments.as_object().ok_or_else(|| {
                "assume_state requires an object of variable assignments".to_string()
            })?;

            let mut parsed: Vec<(String, BmcValue)> = Vec::new();
            for (name, val) in obj {
                let bmc_val = json_to_bmc_value(val)?;
                parsed.push((name.clone(), bmc_val));
            }

            explorer.assume_state(&parsed).map_err(|e| format!("{e}"))?;
            Ok(serde_json::json!({"status": "assumed"}))
        }
        #[cfg(not(feature = "ay"))]
        {
            let _ = assignments;
            Err("assume_state requires the ay feature".to_string())
        }
    }

    /// Get the next satisfying model (blocking the current one).
    fn rpc_next_model(&mut self) -> Result<serde_json::Value, String> {
        #[cfg(feature = "ay")]
        {
            let explorer = self
                .symbolic
                .as_mut()
                .ok_or_else(|| "symbolic explorer not initialized".to_string())?;
            match explorer.next_model().map_err(|e| format!("{e}"))? {
                Some(state) => {
                    let snap = bmc_state_to_snapshot(state);
                    serde_json::to_value(&snap).map_err(|e| format!("serialization error: {e}"))
                }
                None => Ok(serde_json::json!(null)),
            }
        }
        #[cfg(not(feature = "ay"))]
        {
            Err("next_model requires the ay feature".to_string())
        }
    }

    /// Emit shared symbolic-execution evidence for the interactive AY lane.
    fn rpc_symbolic_evidence(&self) -> Result<serde_json::Value, String> {
        #[cfg(feature = "ay")]
        {
            let detection = SymbolicExplorer::model_enumeration_detection();
            let problem = ProblemKind::SymbolicExecution;
            let preferred_backend = detection.preferred_ay_backend(problem);
            let symbolic_evidence_report = self.symbolic.as_ref().map_or_else(
                || SymbolicExplorer::symbolic_evidence_report("TLA"),
                |explorer| explorer.current_symbolic_evidence_report("TLA"),
            );
            let solver_decision_profile = symbolic_evidence_report.solver_decision_profile();
            let solver_decision_profile_evidence =
                solver_decision_profile.evidence_row().to_string();
            let model_blocking_capability = symbolic_evidence_report.model_blocking_capability();
            let model_blocking_capability_evidence =
                model_blocking_capability.evidence_row().to_string();
            let model_blocking_capability_json = model_blocking_capability.to_json();
            let symbolic_route_admission_evidence = symbolic_evidence_report
                .route_admission_evidence_row()
                .to_string();
            let symbolic_route_admission = symbolic_evidence_report.route_admission_json();
            let symbolic_capability_route_readiness_evidence = symbolic_evidence_report
                .capability_route_readiness_evidence_row()
                .to_string();
            let symbolic_capability_route_readiness =
                symbolic_evidence_report.all_supported_capability_route_readiness_json();
            let model_blocking_capability_route_readiness =
                symbolic_evidence_report.model_blocking_capability_route_readiness_json();
            Ok(serde_json::json!({
                "symbolic_execution": {
                    "status": detection.status_name(),
                    "status_code": detection.status_code(),
                    "problem": problem.name(),
                    "reason": detection.reason_name_or_none(),
                    "reason_code": detection.normalized_reason_code(),
                    "preferred_backend": preferred_backend.map(|backend| backend.name()).unwrap_or("None"),
                    "preferred_backend_code": preferred_backend.map(|backend| backend.code()).unwrap_or(NO_REASON_CODE),
                    "evidence": SymbolicExplorer::model_enumeration_evidence("TLA"),
                    "solver_decision_profile_evidence": solver_decision_profile_evidence,
                    "solver_decision_profile": solver_decision_profile.to_json(),
                    "model_blocking_capability_evidence": model_blocking_capability_evidence.clone(),
                    "model_blocking_capability": model_blocking_capability_json,
                    "symbolic_route_admission_evidence": symbolic_route_admission_evidence,
                    "symbolic_route_admission": symbolic_route_admission,
                    "symbolic_capability_route_readiness_evidence": symbolic_capability_route_readiness_evidence,
                    "symbolic_capability_route_readiness": symbolic_capability_route_readiness,
                    "model_blocking_capability_route_readiness": model_blocking_capability_route_readiness,
                }
            }))
        }
        #[cfg(not(feature = "ay"))]
        {
            Err("symbolic evidence requires the ay feature".to_string())
        }
    }

    /// Compact the symbolic solver state.
    fn rpc_compact(&mut self) -> Result<serde_json::Value, String> {
        #[cfg(feature = "ay")]
        {
            let explorer = self
                .symbolic
                .as_mut()
                .ok_or_else(|| "symbolic explorer not initialized".to_string())?;
            let state = explorer.compact().map_err(|e| format!("{e}"))?;
            let snap = bmc_state_to_snapshot(state);
            serde_json::to_value(&snap).map_err(|e| format!("serialization error: {e}"))
        }
        #[cfg(not(feature = "ay"))]
        {
            Err("compact requires the ay feature".to_string())
        }
    }

    /// Apply transitions in order (symbolic mode).
    fn rpc_apply_in_order(
        &mut self,
        actions: &[String],
        steps: usize,
    ) -> Result<serde_json::Value, String> {
        #[cfg(feature = "ay")]
        {
            let explorer = self
                .symbolic
                .as_mut()
                .ok_or_else(|| "symbolic explorer not initialized".to_string())?;
            let trace = explorer
                .apply_in_order(actions, steps)
                .map_err(|e| format!("{e}"))?;
            let snapshots: Vec<_> = trace.into_iter().map(bmc_state_to_snapshot).collect();
            serde_json::to_value(&snapshots).map_err(|e| format!("serialization error: {e}"))
        }
        #[cfg(not(feature = "ay"))]
        {
            let _ = (actions, steps);
            Err("apply_in_order requires the ay feature".to_string())
        }
    }

    /// Dispose the spec and server state.
    fn rpc_dispose(&mut self) -> Result<serde_json::Value, String> {
        self.states.clear();
        self.next_id = 0;
        #[cfg(feature = "ay")]
        {
            self.symbolic = None;
        }
        self.mode = ServerExploreMode::Concrete;
        Ok(serde_json::json!({"status": "disposed"}))
    }

    // -- Dispatch ---------------------------------------------------------

    fn dispatch(&mut self, req: &RpcRequest) -> RpcResponse {
        if req.jsonrpc != "2.0" {
            return RpcResponse::error(
                req.id.clone(),
                INVALID_REQUEST,
                "expected jsonrpc: \"2.0\"".into(),
            );
        }

        let result = match req.method.as_str() {
            "init" => self.rpc_init(),
            "step" => {
                let state_id = match req.params.get("state_id").and_then(|v| v.as_u64()) {
                    Some(id) => id,
                    None => {
                        return RpcResponse::error(
                            req.id.clone(),
                            INVALID_PARAMS,
                            "step requires params.state_id (u64)".into(),
                        );
                    }
                };
                self.rpc_step(state_id)
            }
            "eval" => {
                let state_id = match req.params.get("state_id").and_then(|v| v.as_u64()) {
                    Some(id) => id,
                    None => {
                        return RpcResponse::error(
                            req.id.clone(),
                            INVALID_PARAMS,
                            "eval requires params.state_id (u64)".into(),
                        );
                    }
                };
                let expr = match req.params.get("expr").and_then(|v| v.as_str()) {
                    Some(e) => e.to_owned(),
                    None => {
                        return RpcResponse::error(
                            req.id.clone(),
                            INVALID_PARAMS,
                            "eval requires params.expr (string)".into(),
                        );
                    }
                };
                self.rpc_eval(state_id, &expr)
            }
            "check_invariant" => {
                let state_id = match req.params.get("state_id").and_then(|v| v.as_u64()) {
                    Some(id) => id,
                    None => {
                        return RpcResponse::error(
                            req.id.clone(),
                            INVALID_PARAMS,
                            "check_invariant requires params.state_id (u64)".into(),
                        );
                    }
                };
                let inv = match req.params.get("inv").and_then(|v| v.as_str()) {
                    Some(i) => i.to_owned(),
                    None => {
                        return RpcResponse::error(
                            req.id.clone(),
                            INVALID_PARAMS,
                            "check_invariant requires params.inv (string)".into(),
                        );
                    }
                };
                self.rpc_check_invariant(state_id, &inv)
            }
            "random_trace" => {
                let steps = req
                    .params
                    .get("steps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                self.rpc_random_trace(steps)
            }

            // -- Symbolic exploration methods --
            "set_mode" => {
                let mode = match req.params.get("mode").and_then(|v| v.as_str()) {
                    Some(m) => m.to_owned(),
                    None => {
                        return RpcResponse::error(
                            req.id.clone(),
                            INVALID_PARAMS,
                            "set_mode requires params.mode (string: 'concrete' or 'symbolic')"
                                .into(),
                        );
                    }
                };
                self.rpc_set_mode(&mode)
            }
            "rollback" => self.rpc_rollback(),
            "assume_state" => {
                let assignments = req
                    .params
                    .get("assignments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                self.rpc_assume_state(&assignments)
            }
            "assume_transition" => {
                // In the current implementation, assume_transition is a no-op
                // placeholder — action-level filtering requires disjunct
                // decomposition of the Next relation which is not yet implemented.
                let _action = req.params.get("action").and_then(|v| v.as_str());
                Ok(
                    serde_json::json!({"status": "acknowledged", "note": "action-level filtering not yet implemented"}),
                )
            }
            "next_model" => self.rpc_next_model(),
            "symbolic_evidence" => self.rpc_symbolic_evidence(),
            "compact" => self.rpc_compact(),
            "apply_in_order" => {
                let actions: Vec<String> = req
                    .params
                    .get("actions")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let steps = req
                    .params
                    .get("steps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(actions.len() as u64) as usize;
                self.rpc_apply_in_order(&actions, steps)
            }
            "dispose" => self.rpc_dispose(),

            _ => {
                return RpcResponse::error(
                    req.id.clone(),
                    METHOD_NOT_FOUND,
                    format!("unknown method: {}", req.method),
                );
            }
        };

        match result {
            Ok(val) => RpcResponse::success(req.id.clone(), val),
            Err(msg) => RpcResponse::error(req.id.clone(), INTERNAL_ERROR, msg),
        }
    }

    // -- TCP listener -----------------------------------------------------

    /// Dispatch an HTTP-style request by method name and JSON parameters.
    ///
    /// This is the public entry point used by the HTTP explore server
    /// (`cmd_explore`). It constructs a synthetic JSON-RPC request and
    /// delegates to the internal dispatch logic.
    ///
    /// Returns `Ok(json)` on success or `Err(message)` on failure.
    pub fn dispatch_http(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let req = RpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: serde_json::Value::Number(serde_json::Number::from(0)),
        };
        let resp = self.dispatch(&req);
        if let Some(err) = resp.error {
            Err(err.message)
        } else {
            Ok(resp.result.unwrap_or(serde_json::Value::Null))
        }
    }

    /// Run the interactive server, listening on the given port.
    ///
    /// The server accepts one connection at a time and processes newline-delimited
    /// JSON-RPC requests until the client disconnects, then waits for the next
    /// connection. This loop runs until the process is terminated.
    pub fn listen(&mut self, port: u16) -> Result<(), InteractiveServerError> {
        let addr = format!("127.0.0.1:{port}");
        let listener = TcpListener::bind(&addr).map_err(|e| InteractiveServerError::Bind {
            addr: addr.clone(),
            source: e,
        })?;
        eprintln!("Interactive server listening on {addr}");

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    if let Err(e) = self.handle_connection(stream) {
                        eprintln!("Connection error: {e}");
                    }
                }
                Err(e) => {
                    eprintln!("Accept error: {e}");
                }
            }
        }
        Ok(())
    }

    fn handle_connection(&mut self, stream: TcpStream) -> Result<(), std::io::Error> {
        let peer = stream.peer_addr()?;
        eprintln!("Client connected: {peer}");

        let reader = BufReader::new(stream.try_clone()?);
        let mut writer = stream;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<RpcRequest>(&line) {
                Ok(req) => self.dispatch(&req),
                Err(e) => RpcResponse {
                    jsonrpc: "2.0",
                    result: None,
                    error: Some(RpcError {
                        code: PARSE_ERROR,
                        message: format!("JSON parse error: {e}"),
                    }),
                    id: serde_json::Value::Null,
                },
            };

            let mut resp_bytes =
                serde_json::to_vec(&response).expect("response serialization infallible");
            resp_bytes.push(b'\n');
            writer.write_all(&resp_bytes)?;
            writer.flush()?;
        }

        eprintln!("Client disconnected: {peer}");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the interactive server infrastructure (not RPC-level errors).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InteractiveServerError {
    /// The server could not bind its TCP listener.
    #[error("failed to bind TCP listener on {addr}: {source}")]
    Bind {
        /// The address that failed to bind.
        addr: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
}

// ---------------------------------------------------------------------------
// Value conversion helper
// ---------------------------------------------------------------------------

/// Convert a TLA+ `Value` to a plain `serde_json::Value` for JSON-RPC responses.
///
/// Delegates to the existing `value_to_json` in `json_codec` for the core
/// conversion, then maps our internal tagged `JsonValue` enum to plain JSON.
fn value_to_serde(value: &Value) -> serde_json::Value {
    let jv = crate::json_codec::value_to_json(value);
    json_value_to_serde(&jv)
}

/// Recursively convert the crate-internal `JsonValue` (tagged-enum representation)
/// to a plain `serde_json::Value`.
fn json_value_to_serde(jv: &JsonValue) -> serde_json::Value {
    match jv {
        JsonValue::Bool(b) => serde_json::Value::Bool(*b),
        JsonValue::Int(n) => serde_json::json!(n),
        JsonValue::BigInt(s) => serde_json::Value::String(s.clone()),
        JsonValue::String(s) => serde_json::Value::String(s.clone()),
        JsonValue::Set(elems) | JsonValue::Seq(elems) | JsonValue::Tuple(elems) => {
            serde_json::Value::Array(elems.iter().map(json_value_to_serde).collect())
        }
        JsonValue::Record(fields) => {
            let obj: serde_json::Map<String, serde_json::Value> = fields
                .iter()
                .map(|(k, v)| (k.clone(), json_value_to_serde(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        JsonValue::Function { mapping, .. } => {
            // Represent functions as an array of [key, value] pairs.
            let pairs: Vec<serde_json::Value> = mapping
                .iter()
                .map(|(k, v)| serde_json::json!([json_value_to_serde(k), json_value_to_serde(v)]))
                .collect();
            serde_json::Value::Array(pairs)
        }
        JsonValue::ModelValue(s) => serde_json::Value::String(s.clone()),
        JsonValue::Interval { lo, hi } => {
            serde_json::json!({"lo": lo, "hi": hi})
        }
        JsonValue::Undefined => serde_json::Value::Null,
    }
}

// ---------------------------------------------------------------------------
// BMC state conversion helpers (used by symbolic mode)
// ---------------------------------------------------------------------------

/// Convert a JSON value to a `BmcValue` for symbolic state assertions.
#[cfg(feature = "ay")]
fn json_to_bmc_value(val: &serde_json::Value) -> Result<tla_ay::BmcValue, String> {
    match val {
        serde_json::Value::Bool(b) => Ok(tla_ay::BmcValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(tla_ay::BmcValue::Int(i))
            } else {
                Err(format!("number {n} is not a valid integer"))
            }
        }
        _ => Err(format!(
            "unsupported JSON value type for symbolic state: {val}"
        )),
    }
}

/// Convert a `BmcState` to a `StateSnapshot`-like JSON structure.
#[cfg(feature = "ay")]
fn bmc_state_to_snapshot(state: tla_ay::BmcState) -> StateSnapshot {
    let variables: FxHashMap<String, serde_json::Value> = state
        .assignments
        .into_iter()
        .map(|(k, v)| (k, bmc_value_to_json(&v)))
        .collect();
    StateSnapshot {
        state_id: state.step as u64,
        variables,
    }
}

/// Convert a `BmcValue` to a JSON value.
#[cfg(feature = "ay")]
fn bmc_value_to_json(val: &tla_ay::BmcValue) -> serde_json::Value {
    match val {
        tla_ay::BmcValue::Bool(b) => serde_json::Value::Bool(*b),
        tla_ay::BmcValue::Int(i) => serde_json::json!(i),
        tla_ay::BmcValue::BigInt(n) => serde_json::Value::String(n.to_string()),
        tla_ay::BmcValue::String(s) => serde_json::Value::String(s.clone()),
        tla_ay::BmcValue::Set(members) => {
            serde_json::Value::Array(members.iter().map(bmc_value_to_json).collect())
        }
        tla_ay::BmcValue::Sequence(elems) => {
            serde_json::Value::Array(elems.iter().map(bmc_value_to_json).collect())
        }
        tla_ay::BmcValue::Function(entries) => {
            let pairs: Vec<serde_json::Value> = entries
                .iter()
                .map(|(k, v)| serde_json::json!([k, bmc_value_to_json(v)]))
                .collect();
            serde_json::Value::Array(pairs)
        }
        tla_ay::BmcValue::Record(fields) => {
            let obj: serde_json::Map<String, serde_json::Value> = fields
                .iter()
                .map(|(name, val)| (name.clone(), bmc_value_to_json(val)))
                .collect();
            serde_json::Value::Object(obj)
        }
        tla_ay::BmcValue::Tuple(elems) => {
            serde_json::Value::Array(elems.iter().map(bmc_value_to_json).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_response_success_serialization() {
        let resp = RpcResponse::success(serde_json::json!(1), serde_json::json!({"ok": true}));
        let s = serde_json::to_string(&resp).expect("serialize");
        assert!(s.contains("\"jsonrpc\":\"2.0\""));
        assert!(s.contains("\"result\""));
        assert!(!s.contains("\"error\""));
    }

    #[test]
    fn test_rpc_response_error_serialization() {
        let resp = RpcResponse::error(serde_json::json!(2), METHOD_NOT_FOUND, "nope".into());
        let s = serde_json::to_string(&resp).expect("serialize");
        assert!(s.contains("\"error\""));
        assert!(s.contains("-32601"));
        assert!(!s.contains("\"result\""));
    }

    #[test]
    fn test_rpc_request_deserialization() {
        let json = r#"{"jsonrpc":"2.0","method":"init","params":{},"id":1}"#;
        let req: RpcRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.method, "init");
        assert_eq!(req.jsonrpc, "2.0");
    }

    #[test]
    fn test_value_to_serde_bool() {
        let v = Value::Bool(true);
        let s = value_to_serde(&v);
        assert_eq!(s, serde_json::Value::Bool(true));
    }

    #[test]
    fn test_value_to_serde_int() {
        let v = Value::SmallInt(42);
        let s = value_to_serde(&v);
        assert_eq!(s, serde_json::json!(42));
    }

    #[cfg(feature = "ay")]
    fn ay_model_blocking_contract() -> tla_ay::SolverCapabilityContract {
        tla_ay::model_blocking_symbolic_execution_contract()
    }

    #[cfg(feature = "ay")]
    fn assert_evidence_contains_ay_manifest_pair(
        blocking_evidence: &str,
        key: &str,
        value: impl ToString,
    ) {
        let value = value.to_string();
        let value = if value.is_empty() {
            NO_REASON_CODE.to_string()
        } else {
            value
        };
        assert!(
            blocking_evidence.contains(&format!("{key}={value}")),
            "RPC model-blocking evidence should forward {key} from AY manifest: {blocking_evidence}"
        );
    }

    #[cfg(feature = "ay")]
    fn assert_rpc_model_blocking_evidence_matches_ay_manifest(
        blocking_evidence: &str,
        blocking_capability: &serde_json::Value,
    ) {
        let manifest = tla_ay::symbolic_execution_contract_manifest();
        let manifest_pairs = tla_ay::symbolic_execution_contract_manifest_key_value_pairs();
        let health = tla_ay::validate_symbolic_execution_contract_manifest_round_trip(
            &manifest,
            &manifest_pairs,
        );
        let diagnostic_summary =
            tla_ay::symbolic_execution_contract_manifest_diagnostic_summary_for_round_trip(
                &manifest,
                &manifest_pairs,
            );
        let route_admission =
            tla_ay::symbolic_execution_route_admission_decision_for_summary(&diagnostic_summary);
        let model_blocking_contract = ay_model_blocking_contract();
        let incremental_contract = tla_ay::incremental_assumptions_symbolic_execution_contract();
        let all_sat_contract = tla_ay::all_sat_enumeration_symbolic_execution_contract();
        let model_blocking_readiness =
            tla_ay::symbolic_execution_capability_route_readiness_for_decision(
                tla_ay::SolverCapabilityCode::ModelBlocking,
                &route_admission,
            );
        let all_supported_readiness =
            tla_ay::symbolic_execution_all_supported_capability_route_readiness_for_decision(
                &route_admission,
            );

        assert!(blocking_evidence.contains("TLA ay_symbolic_execution_contract_manifest"));
        for (key, value) in manifest_pairs {
            assert_evidence_contains_ay_manifest_pair(blocking_evidence, key, value);
        }
        for (key, value) in health.to_key_value_pairs() {
            assert_evidence_contains_ay_manifest_pair(
                blocking_evidence,
                &format!("health_{key}"),
                value,
            );
        }
        for (key, value) in diagnostic_summary.to_key_value_rows() {
            assert_evidence_contains_ay_manifest_pair(
                blocking_evidence,
                &format!("diagnostic_summary_{key}"),
                value,
            );
        }
        for (key, value) in route_admission.to_key_value_rows() {
            assert_evidence_contains_ay_manifest_pair(
                blocking_evidence,
                &format!("route_admission_{key}"),
                value,
            );
        }
        for (key, value) in model_blocking_contract.to_key_value_pairs() {
            assert_evidence_contains_ay_manifest_pair(
                blocking_evidence,
                &format!("model_blocking_contract_{key}"),
                value,
            );
        }
        assert_evidence_contains_ay_manifest_pair(blocking_evidence, "model_blocking_ready", true);
        assert_evidence_contains_ay_manifest_pair(
            blocking_evidence,
            "incremental_assumptions_ready",
            true,
        );
        assert_evidence_contains_ay_manifest_pair(
            blocking_evidence,
            "all_sat_enumeration_ready",
            true,
        );
        assert_evidence_contains_ay_manifest_pair(
            blocking_evidence,
            "capability",
            model_blocking_contract.capability_code,
        );
        assert_evidence_contains_ay_manifest_pair(
            blocking_evidence,
            "status_code",
            model_blocking_readiness.status_code,
        );
        assert_evidence_contains_ay_manifest_pair(
            blocking_evidence,
            "reason_code",
            model_blocking_readiness.reason_code,
        );
        assert_evidence_contains_ay_manifest_pair(blocking_evidence, "typed_consumer", true);
        assert_evidence_contains_ay_manifest_pair(
            blocking_evidence,
            "fail_closed",
            model_blocking_readiness.fail_closed,
        );
        assert_evidence_contains_ay_manifest_pair(blocking_evidence, "manifest_valid", true);

        assert_eq!(blocking_capability["manifest_schema"], manifest.schema);
        assert_eq!(
            blocking_capability["manifest_schema_version"],
            manifest.schema_version
        );
        assert_eq!(blocking_capability["solver"], manifest.solver);
        assert_eq!(
            blocking_capability["contract_count"],
            manifest.contracts.len()
        );
        assert_eq!(
            blocking_capability["capability"],
            model_blocking_contract.capability_code
        );
        assert_eq!(
            blocking_capability["status_code"],
            model_blocking_readiness.status_code
        );
        assert_eq!(
            blocking_capability["reason_code"],
            model_blocking_readiness.reason_code
        );
        assert_eq!(blocking_capability["typed_consumer"], true);
        assert_eq!(blocking_capability["manifest_valid"], true);
        assert_eq!(
            blocking_capability["all_contracts_fail_closed"],
            manifest.all_contracts_fail_closed
        );
        assert_eq!(
            blocking_capability["fail_closed"],
            model_blocking_readiness.fail_closed
        );
        assert_eq!(
            blocking_capability["symbolic_execution_contract_manifest"],
            manifest.to_json_value()
        );
        assert_eq!(
            blocking_capability["symbolic_execution_contract_manifest_health"],
            health.to_json_value()
        );
        assert_eq!(
            blocking_capability["symbolic_execution_contract_diagnostic_summary"],
            diagnostic_summary.to_json_value()
        );
        assert_eq!(
            blocking_capability["symbolic_execution_route_admission"],
            route_admission.to_json_value()
        );
        assert_eq!(
            blocking_capability["health_status_code"],
            health.status_code
        );
        assert_eq!(
            blocking_capability["health_reason_code"],
            health.reason_code
        );
        assert_eq!(
            blocking_capability["health_diagnostic_code"],
            health.diagnostic_code()
        );
        assert_eq!(
            blocking_capability["accepted_for_consumer"],
            health.accepted_for_consumer
        );
        assert_eq!(
            blocking_capability["route_admission_status_code"],
            route_admission.status_code
        );
        assert_eq!(
            blocking_capability["route_admission_reason_code"],
            route_admission.reason_code
        );
        assert_eq!(
            blocking_capability["route_admission_accepted_for_consumer"],
            route_admission.accepted_for_consumer
        );
        assert_eq!(
            blocking_capability["route_admission_fail_closed"],
            route_admission.fail_closed
        );
        assert_eq!(blocking_capability["model_blocking_ready"], true);
        assert_eq!(blocking_capability["incremental_assumptions_ready"], true);
        assert_eq!(blocking_capability["all_sat_enumeration_ready"], true);
        assert_eq!(
            blocking_capability["symbolic_execution_all_supported_capability_route_readiness"],
            serde_json::json!(all_supported_readiness
                .iter()
                .map(|readiness| readiness.to_json_value())
                .collect::<Vec<_>>())
        );
        assert_eq!(
            blocking_capability["model_blocking_capability_route_readiness"],
            model_blocking_readiness.to_json_value()
        );
        assert_eq!(
            blocking_capability["all_supported_capability_route_readiness_ready"],
            true
        );
        let route_readiness_evidence = blocking_capability
            ["symbolic_execution_all_supported_capability_route_readiness_evidence"]
            .as_str()
            .expect("all-supported route readiness evidence should be present");
        assert!(route_readiness_evidence
            .contains("TLA ay_symbolic_execution_all_supported_capability_route_readiness"));
        assert!(route_readiness_evidence.contains("model_blocking_status=ready"));
        assert!(route_readiness_evidence
            .contains("model_blocking_reason=ay_authoritative_capability_route"));
        assert!(route_readiness_evidence.contains("incremental_assumptions_status=ready"));
        assert!(route_readiness_evidence.contains("all_sat_enumeration_status=ready"));
        assert_eq!(
            blocking_capability["capability_manifest"],
            manifest.to_json_value()
        );

        let contract_capabilities = blocking_capability["contract_capabilities"]
            .as_array()
            .expect("contract_capabilities array");
        for contract in [
            model_blocking_contract,
            incremental_contract,
            all_sat_contract,
        ] {
            assert!(
                contract_capabilities
                    .iter()
                    .any(|capability| capability == contract.capability_code),
                "missing symbolic contract capability {}",
                contract.capability_code
            );
        }

        let api_symbols = blocking_capability["api_symbols"]
            .as_array()
            .expect("api_symbols array");
        for expected in model_blocking_readiness.api_symbols {
            assert!(
                api_symbols.iter().any(|symbol| symbol == *expected),
                "missing AY readiness API symbol {expected}"
            );
        }
        let evidence_schemas = blocking_capability["evidence_schemas"]
            .as_array()
            .expect("evidence_schemas array");
        for expected in model_blocking_readiness.evidence_schemas {
            assert!(
                evidence_schemas.iter().any(|schema| schema == *expected),
                "missing AY readiness evidence schema {expected}"
            );
        }
        let consumer_responsibilities = blocking_capability["consumer_responsibilities"]
            .as_array()
            .expect("consumer_responsibilities array");
        for expected in model_blocking_readiness.consumer_responsibilities {
            assert!(
                consumer_responsibilities
                    .iter()
                    .any(|responsibility| responsibility == *expected),
                "missing AY readiness consumer responsibility {expected}"
            );
        }

        let readiness_rows = blocking_capability["symbolic_execution_contracts"]
            .as_array()
            .expect("symbolic_execution_contracts array");
        for contract in [
            model_blocking_contract,
            incremental_contract,
            all_sat_contract,
        ] {
            let row = readiness_rows
                .iter()
                .find(|row| row["capability"] == contract.capability_code)
                .expect("contract readiness row");
            let expected_readiness = all_supported_readiness
                .iter()
                .find(|readiness| readiness.capability_code == contract.capability_code)
                .expect("AY all-supported readiness should include contract capability");
            assert_eq!(row, &expected_readiness.to_json_value());
        }
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_symbolic_evidence_rpc_uses_shared_detection_helpers() {
        let src = r#"
---- MODULE InteractiveSymbolicEvidence ----
VARIABLE x
Init == x \in {0, 1}
Next == x' = x
====
"#;
        let module = Arc::new(crate::test_support::parse_module(src));
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut server = InteractiveServer::new(module, config);
        let response = server.dispatch(&RpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "symbolic_evidence".to_string(),
            params: serde_json::json!({}),
            id: serde_json::json!(7),
        });

        assert!(response.error.is_none(), "unexpected RPC error");
        let result = response.result.expect("expected RPC result");
        let evidence = &result["symbolic_execution"];
        assert_eq!(evidence["status_code"], "ay_required");
        assert_eq!(evidence["reason_code"], "model_enumeration");
        assert_eq!(evidence["preferred_backend_code"], "ay_smt");
        assert_eq!(
            evidence["evidence"],
            "TLA symbolic_execution domain=tla status=AYRequired status_code=ay_required problem=SymbolicExecution reason=ModelEnumeration reason_code=model_enumeration preferred_backend=AYSmt preferred_backend_code=ay_smt"
        );
        let solver_evidence = evidence["solver_decision_profile_evidence"]
            .as_str()
            .expect("solver decision/profile evidence should be a string");
        assert!(solver_evidence.contains("TLA ay_solver_decision_profile_summary"));
        assert!(solver_evidence.contains("status_code=missing_typed_summary"));
        assert!(solver_evidence.contains("typed_consumer=false"));
        assert!(solver_evidence.contains("production_selected=false"));
        assert!(solver_evidence.contains("fail_closed=true"));
        let solver_profile = &evidence["solver_decision_profile"];
        assert_eq!(solver_profile["status_code"], "missing_typed_summary");
        assert_eq!(solver_profile["accepted_for_consumer"], false);
        assert_eq!(
            solver_profile["consumer_rejection_code"],
            "missing_typed_summary"
        );
        assert_eq!(solver_profile["model_validated"], false);
        assert_eq!(solver_profile["typed_consumer"], false);
        assert_eq!(solver_profile["fail_closed"], true);

        let blocking_evidence = evidence["model_blocking_capability_evidence"]
            .as_str()
            .expect("model-blocking capability evidence should be a string");
        let blocking_capability = &evidence["model_blocking_capability"];
        let shared_symbolic_report = SymbolicExplorer::symbolic_evidence_report("TLA");
        assert_eq!(
            solver_profile,
            &shared_symbolic_report.solver_decision_profile().to_json()
        );
        assert_eq!(
            solver_profile["model_blocking_capability_evidence"].as_str(),
            Some(blocking_evidence)
        );
        assert_eq!(
            &solver_profile["model_blocking_capability"],
            blocking_capability
        );
        assert_eq!(
            blocking_evidence,
            shared_symbolic_report
                .model_blocking_capability()
                .evidence_row()
        );
        assert_eq!(
            blocking_capability,
            &shared_symbolic_report.model_blocking_capability().to_json()
        );
        assert_eq!(
            evidence["symbolic_route_admission_evidence"].as_str(),
            Some(shared_symbolic_report.route_admission_evidence_row())
        );
        assert!(shared_symbolic_report
            .route_admission_evidence_row()
            .contains("TLA ay_symbolic_execution_route_admission"));
        assert_eq!(
            &evidence["symbolic_route_admission"],
            &blocking_capability["symbolic_execution_route_admission"]
        );
        let readiness_evidence = evidence["symbolic_capability_route_readiness_evidence"]
            .as_str()
            .expect("readiness evidence should be a string");
        assert_eq!(
            readiness_evidence,
            shared_symbolic_report
                .model_blocking_capability()
                .capability_route_readiness_evidence_row()
        );
        assert!(readiness_evidence.contains("model_blocking_status=ready"));
        assert!(readiness_evidence.contains("incremental_assumptions_status=ready"));
        assert!(readiness_evidence.contains("all_sat_enumeration_status=ready"));
        assert_eq!(
            &evidence["symbolic_capability_route_readiness"],
            &blocking_capability["symbolic_execution_all_supported_capability_route_readiness"]
        );
        assert_eq!(
            &evidence["model_blocking_capability_route_readiness"],
            &blocking_capability["model_blocking_capability_route_readiness"]
        );
        assert_rpc_model_blocking_evidence_matches_ay_manifest(
            blocking_evidence,
            blocking_capability,
        );
    }
}
