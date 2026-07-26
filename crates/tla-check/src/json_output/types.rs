// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Type definitions for the JSON output format.
//!
//! All struct/enum types used in the structured JSON output for AI agents
//! and automated tooling.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub use super::trace_types::{
    ActionRef, CounterexampleInfo, JsonValue, StateDiff, StateInfo, ValueChange,
};

/// Version of the JSON output format
pub const OUTPUT_VERSION: &str = "1.3";

/// Soundness classification for the run path (engine parity / experimental status).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SoundnessMode {
    /// Selected engine path is intended to satisfy the checker soundness gate.
    Sound,
    /// Intended to be sound, but not yet proven / validated for parity.
    Experimental,
    /// Heuristic / incomplete engine (e.g., abstraction, subset support).
    Heuristic,
}

/// Soundness provenance record for CLI outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SoundnessProvenance {
    /// Overall soundness classification for the engine path that produced this result.
    pub mode: SoundnessMode,
    /// Engine features that were active for this run (e.g. symmetry reduction, POR).
    pub features_used: Vec<String>,
    /// Ways the run deviated from the fully-sound reference semantics, if any.
    pub deviations: Vec<String>,
    /// Assumptions the result relies on (e.g. constant bounds, finite-model premises).
    pub assumptions: Vec<String>,
}

impl SoundnessProvenance {
    /// Returns a provenance record asserting a fully [`Sound`](SoundnessMode::Sound)
    /// run with no features, deviations, or assumptions recorded.
    pub fn sound() -> Self {
        Self {
            mode: SoundnessMode::Sound,
            features_used: Vec::new(),
            deviations: Vec::new(),
            assumptions: Vec::new(),
        }
    }
}

/// Search completeness classification (exhaustive vs user-configured bounds).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SearchCompleteness {
    /// The full reachable state space was explored; no user bound was hit.
    Exhaustive,
    /// Search was cut off at a maximum BFS depth.
    BoundedDepth {
        /// The depth bound that limited the search.
        max_depth: usize,
    },
    /// Search was cut off at a maximum number of distinct states.
    BoundedStates {
        /// The state-count bound that limited the search.
        max_states: usize,
    },
    /// Search was cut off by both a depth and a state-count bound.
    Bounded {
        /// The depth bound that limited the search.
        max_depth: usize,
        /// The state-count bound that limited the search.
        max_states: usize,
    },
}

impl SearchCompleteness {
    /// Classifies completeness from the configured bounds, where `0` means "no bound".
    ///
    /// `(0, 0)` yields [`Exhaustive`](Self::Exhaustive); a single non-zero bound yields
    /// the matching single-bound variant; two non-zero bounds yield [`Bounded`](Self::Bounded).
    pub fn from_bounds(max_states: usize, max_depth: usize) -> Self {
        match (max_states, max_depth) {
            (0, 0) => SearchCompleteness::Exhaustive,
            (0, d) => SearchCompleteness::BoundedDepth { max_depth: d },
            (s, 0) => SearchCompleteness::BoundedStates { max_states: s },
            (s, d) => SearchCompleteness::Bounded {
                max_depth: d,
                max_states: s,
            },
        }
    }

    /// Returns `true` only for the [`Exhaustive`](Self::Exhaustive) variant.
    pub fn is_exhaustive(self) -> bool {
        matches!(self, SearchCompleteness::Exhaustive)
    }

    /// Renders a short human-readable description (e.g. `"bounded (max_depth=10)"`).
    pub fn format_human(self) -> String {
        match self {
            SearchCompleteness::Exhaustive => "exhaustive".to_string(),
            SearchCompleteness::BoundedDepth { max_depth } => {
                format!("bounded (max_depth={})", max_depth)
            }
            SearchCompleteness::BoundedStates { max_states } => {
                format!("bounded (max_states={})", max_states)
            }
            SearchCompleteness::Bounded {
                max_depth,
                max_states,
            } => format!(
                "bounded (max_states={}, max_depth={})",
                max_states, max_depth
            ),
        }
    }
}

/// Complete JSON output for model checking results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JsonOutput {
    /// Schema version
    pub version: String,
    /// Tool identifier
    pub tool: String,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Input files and configuration
    pub input: InputInfo,
    /// Specification details
    pub specification: SpecInfo,
    /// Soundness provenance
    pub soundness: SoundnessProvenance,
    /// Search completeness (exhaustive vs bounded)
    pub completeness: SearchCompleteness,
    /// Model checking result
    pub result: ResultInfo,
    /// Counterexample trace (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<CounterexampleInfo>,
    /// Statistics
    pub statistics: StatisticsInfo,
    /// Diagnostic messages
    pub diagnostics: DiagnosticsInfo,
    /// Backend capability/admission evidence consumed by capability summarizers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_capability_report: Option<serde_json::Value>,
    /// Engine-provenance record: the execution tier that actually ran this
    /// check (plus value-action VM engagement). Benchmark harnesses read this
    /// to attribute every measured row to its engine.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub engine_provenance: Option<serde_json::Value>,
    /// Action coverage information
    #[serde(
        skip_serializing_if = "Vec::is_empty",
        default,
        deserialize_with = "deserialize_actions_detected"
    )]
    pub actions_detected: Vec<ActionInfo>,
}

/// Input file information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InputInfo {
    /// Path to the TLA+ spec file that was checked.
    pub spec_file: String,
    /// Path to the `.cfg` configuration file, if one was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_file: Option<String>,
    /// Name of the root TLA+ module.
    pub module: String,
    /// Number of parallel worker threads used for exploration.
    pub workers: usize,
}

/// Specification structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpecInfo {
    /// Name of the resolved `INIT` predicate, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<String>,
    /// Name of the resolved `NEXT` action, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    /// Names of the invariants checked during the run.
    pub invariants: Vec<String>,
    /// Names of the temporal properties checked during the run.
    pub properties: Vec<String>,
    /// CONSTANT bindings resolved from the config, keyed by constant name.
    pub constants: HashMap<String, JsonValue>,
    /// Declared state variables of the spec.
    pub variables: Vec<String>,
}

/// Model checking result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ResultInfo {
    /// Status: "ok", "error", "timeout", "interrupted", "limit_reached"
    pub status: String,
    /// Error type if status is "error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    /// Structured error code for programmatic handling
    ///
    /// Error codes follow a prefix convention:
    /// - `TLC_` - Model checker errors (deadlock, invariant violation, etc.)
    /// - `CFG_` - Configuration file parsing errors
    /// - `TLA_` - TLA+ source parsing errors
    /// - `SYS_` - System/runtime errors
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// Human-readable error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Details about violated property
    #[serde(skip_serializing_if = "Option::is_none")]
    pub violated_property: Option<ViolatedProperty>,
    /// Actionable suggestion for fixing the error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<ErrorSuggestion>,
}

/// Actionable suggestion for fixing an error
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorSuggestion {
    /// Brief description of the suggested action
    pub action: String,
    /// Example code or configuration fix
    #[serde(skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
    /// Alternative options if applicable
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub options: Vec<String>,
}

impl ErrorSuggestion {
    /// Creates a suggestion with the given action text and no example or options.
    pub fn new(action: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            example: None,
            options: Vec::new(),
        }
    }

    /// Builder: attaches an example code/config snippet to the suggestion.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }

    /// Builder: attaches a list of alternative options to the suggestion.
    pub fn with_options(mut self, options: Vec<String>) -> Self {
        self.options = options;
        self
    }
}

/// Information about a violated property
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ViolatedProperty {
    /// Name of the violated property as declared in the spec/config.
    pub name: String,
    /// Type: "invariant", "liveness", "assertion"
    #[serde(rename = "type")]
    pub prop_type: String,
    /// Source location of the property definition, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    /// Textual form of the property expression, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
}

/// Source code location
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SourceLocation {
    /// Source file path.
    pub file: String,
    /// 1-based start line.
    pub line: usize,
    /// 1-based start column.
    pub column: usize,
    /// 1-based end line of the span, if it covers a range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    /// 1-based end column of the span, if it covers a range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
}

/// Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StatisticsInfo {
    /// Number of distinct states found.
    pub states_found: usize,
    /// Number of distinct initial states.
    pub states_initial: usize,
    /// Initial states generated before state constraints and deduplication.
    #[serde(default)]
    pub raw_initial_states_generated: usize,
    /// Successors generated before state/action constraints and reductions.
    #[serde(default)]
    pub raw_successors_generated: usize,
    /// Total number of states generated, including initial states and duplicate
    /// successors, using TLC's pre-constraint accounting boundary.
    #[serde(default)]
    pub states_generated: usize,
    /// Number of distinct states stored, if tracked separately from `states_found`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states_distinct: Option<usize>,
    /// Number of state transitions explored.
    pub transitions: usize,
    /// Count of guard-evaluation errors that were suppressed during exploration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suppressed_guard_errors: Option<usize>,
    /// Maximum BFS depth reached.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    /// Maximum size the BFS work queue reached.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_queue_depth: Option<usize>,
    /// Wall-clock duration of the run, in seconds.
    pub time_seconds: f64,
    /// Throughput in states per second, if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub states_per_second: Option<f64>,
    /// Peak memory usage, in megabytes, if measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<f64>,
    /// Fingerprint storage backend counters. Present when disk-tier activity or
    /// reserved-memory usage is worth surfacing to downstream tooling.
    /// Part of #2665.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageStatsInfo>,
    /// PROPERTY-check counters and timing for promoted BFS-time safety checks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub property_check: Option<PropertyCheckStatsInfo>,
}

/// PROPERTY-check statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PropertyCheckStatsInfo {
    /// Number of transition-level checks performed for implied-action PROPERTYs.
    pub implied_action_transition_checks: u64,
    /// Number of term evaluations performed while checking implied-action PROPERTYs.
    pub implied_action_term_evals: u64,
    /// Wall-clock time spent on implied-action PROPERTY checking, in seconds.
    pub implied_action_time_seconds: f64,
}

/// Fingerprint storage backend statistics.
///
/// Provides visibility into the two-tier (memory + disk) fingerprint storage
/// system's behavior during model checking. Part of #2665.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StorageStatsInfo {
    /// Number of fingerprints held in the in-memory tier.
    pub memory_count: usize,
    /// Number of fingerprints spilled to the on-disk tier.
    pub disk_count: usize,
    /// Approximate bytes consumed by the in-memory tier, if measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<usize>,
    /// Number of lookups that consulted the disk tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_lookups: Option<usize>,
    /// Number of disk lookups that found a match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_hits: Option<usize>,
    /// Number of entries evicted from memory to disk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eviction_count: Option<usize>,
}

/// Diagnostic messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DiagnosticsInfo {
    /// Warning-level diagnostics emitted during the run.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<DiagnosticMessage>,
    /// Informational diagnostics emitted during the run.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub info: Vec<DiagnosticMessage>,
    /// Captured output of `Print`/`PrintT` statements.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub print_outputs: Vec<PrintOutput>,
}

/// A diagnostic message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DiagnosticMessage {
    /// Stable machine-readable diagnostic code (see [`error_codes`](super::error_codes)).
    pub code: String,
    /// Human-readable diagnostic text.
    pub message: String,
    /// Source location the diagnostic refers to, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
    /// Optional remediation hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
    /// Optional structured payload carrying diagnostic-specific extra data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// Print statement output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PrintOutput {
    /// Rendered value produced by the `Print`/`PrintT` statement.
    pub value: String,
    /// Source location of the originating print statement, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Action coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ActionInfo {
    /// Stable, unique identifier for the action (canonicalized to be collision-free).
    #[serde(default)]
    pub id: String,
    /// Display name of the action.
    pub name: String,
    /// Number of times the action fired during exploration.
    pub occurrences: usize,
    /// Share of total transitions attributed to this action, if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub percentage: Option<f64>,
}

pub(super) fn canonicalize_action_ids(actions: &mut [ActionInfo]) {
    let base_ids: Vec<String> = actions
        .iter()
        .enumerate()
        .map(|(idx, action)| {
            if action.id.is_empty() {
                format!("detected:{idx}")
            } else {
                action.id.clone()
            }
        })
        .collect();

    let mut counts = HashMap::<String, usize>::new();
    for id in &base_ids {
        *counts.entry(id.clone()).or_insert(0) += 1;
    }

    let mut used = HashSet::<String>::new();
    for (idx, action) in actions.iter_mut().enumerate() {
        let base = &base_ids[idx];
        let mut candidate = if counts[base] == 1 {
            base.clone()
        } else {
            // Repair duplicate recorded ids deterministically. This keeps the
            // original span-backed id visible while making the JSON payload
            // usable by downstream consumers that require unique action ids.
            format!("{base}#dup{idx}")
        };
        while !used.insert(candidate.clone()) {
            candidate.push('#');
        }
        action.id = candidate;
    }
}

fn deserialize_actions_detected<'de, D>(deserializer: D) -> Result<Vec<ActionInfo>, D::Error>
where
    D: Deserializer<'de>,
{
    let mut actions = Vec::<ActionInfo>::deserialize(deserializer)?;
    canonicalize_action_ids(&mut actions);
    Ok(actions)
}

impl JsonOutput {
    /// Create a new JSON output structure
    pub fn new(
        spec_file: &Path,
        config_file: Option<&Path>,
        module_name: &str,
        workers: usize,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            version: OUTPUT_VERSION.to_string(),
            tool: "ty".to_string(),
            timestamp: now.to_rfc3339(),
            input: InputInfo {
                spec_file: spec_file.display().to_string(),
                config_file: config_file.map(|p| p.display().to_string()),
                module: module_name.to_string(),
                workers,
            },
            specification: SpecInfo {
                init: None,
                next: None,
                invariants: Vec::new(),
                properties: Vec::new(),
                constants: HashMap::new(),
                variables: Vec::new(),
            },
            soundness: SoundnessProvenance::sound(),
            completeness: SearchCompleteness::Exhaustive,
            result: ResultInfo {
                status: "ok".to_string(),
                error_type: None,
                error_code: None,
                error_message: None,
                violated_property: None,
                suggestion: None,
            },
            counterexample: None,
            statistics: StatisticsInfo {
                states_found: 0,
                states_initial: 0,
                raw_initial_states_generated: 0,
                raw_successors_generated: 0,
                states_generated: 0,
                states_distinct: None,
                transitions: 0,
                suppressed_guard_errors: None,
                max_depth: None,
                max_queue_depth: None,
                time_seconds: 0.0,
                states_per_second: None,
                memory_mb: None,
                storage: None,
                property_check: None,
            },
            diagnostics: DiagnosticsInfo {
                warnings: Vec::new(),
                info: Vec::new(),
                print_outputs: Vec::new(),
            },
            backend_capability_report: None,
            engine_provenance: None,
            actions_detected: Vec::new(),
        }
    }

    /// Add an info diagnostic
    pub fn add_info(&mut self, code: &str, message: &str) {
        self.diagnostics.info.push(DiagnosticMessage {
            code: code.to_string(),
            message: message.to_string(),
            location: None,
            suggestion: None,
            payload: None,
        });
    }

    /// Add a warning diagnostic
    pub fn add_warning(&mut self, code: &str, message: &str) {
        self.diagnostics.warnings.push(DiagnosticMessage {
            code: code.to_string(),
            message: message.to_string(),
            location: None,
            suggestion: None,
            payload: None,
        });
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON string
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}
