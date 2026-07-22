// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared backend capability and solver-discovery primitives.
//!
//! Domain frontends should keep semantic lowering local, but solver/native
//! availability, unsupported reasons, and evidence strings should be common.
//! This module is deliberately lightweight: it does not depend on AY crates or
//! any domain crate, so MCC, TLA, AIGER, BTOR2, trust-ir, and trust-codegen adapters can all
//! use it without introducing dependency cycles.

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// High-level subsystem asking for a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendDomain {
    /// Shared infrastructure or a caller that is not domain-specific.
    Shared,
    /// TLA+ explicit-state or symbolic checking.
    Tla,
    /// MCC/PNML/Petri-net checking.
    PetriMcc,
    /// AIGER hardware checking.
    Aiger,
    /// BTOR2 hardware checking.
    Btor2,
    /// trust-ir lowering infrastructure.
    TrustIr,
    /// trust_cg code generation.
    TrustCg,
    /// AY solver infrastructure.
    AY,
}

/// Backend family used by a domain adapter or portfolio lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Explicit-state search using Rust/domain successor generation.
    ExplicitState,
    /// Native/JIT successor or predicate kernel.
    NativeKernel,
    /// Domain-local symbolic execution lane.
    ///
    /// Production use should record why the preferred AY lane was not selected.
    LocalSymbolicExecution,
    /// External AY binary invoked as a process.
    ExternalAYBinary,
    /// In-process AY SMT APIs.
    AYSmt,
    /// In-process ay-sat CDCL APIs.
    AYSat,
    /// In-process ay-chc/PDR APIs.
    AYChc,
    /// AIGER hardware portfolio backend.
    AigerPortfolio,
    /// BTOR2 hardware portfolio backend.
    Btor2Portfolio,
}

/// Problem class a backend is being considered for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemKind {
    /// Generic explicit-state reachability.
    ExplicitReachability,
    /// Safety property checking.
    Safety,
    /// Liveness or temporal property checking.
    Liveness,
    /// Deadlock detection.
    Deadlock,
    /// State-space cardinality or enumeration.
    StateSpace,
    /// Symbolic execution or symbolic model enumeration.
    SymbolicExecution,
    /// Invariant proof or checking.
    Invariant,
    /// Bounded model checking.
    Bmc,
    /// K-induction.
    KInduction,
    /// CHC/PDR solving.
    Chc,
    /// Raw SAT solving.
    Sat,
    /// SMT solving.
    Smt,
    /// Native/JIT successor generation.
    NativeSuccessor,
}

/// Capability facet exposed by a solver/backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverFacet {
    /// Runs as an external process.
    ExternalProcess,
    /// Runs in-process.
    InProcess,
    /// Solves SAT.
    Sat,
    /// Solves SMT.
    Smt,
    /// Performs symbolic execution.
    SymbolicExecution,
    /// Solves constrained Horn clauses (CHC).
    Chc,
    /// Performs bounded model checking.
    Bmc,
    /// Performs k-induction.
    KInduction,
    /// Performs property-directed reachability (PDR/IC3).
    Pdr,
    /// Enumerates all satisfying models (AllSAT).
    AllSat,
    /// Supports incremental solving.
    Incremental,
    /// Supports solving under assumptions.
    Assumptions,
    /// Produces unsat cores.
    UnsatCore,
    /// Returns model values.
    ModelValues,
    /// Supports cancellation of in-flight work.
    Cancellation,
    /// Emits proofs.
    Proof,
    /// Emits witnesses / counterexamples.
    Witness,
    /// Supports bit-vector reasoning.
    BitVector,
    /// Supports linear integer arithmetic.
    LinearIntegerArithmetic,
    /// Generates native code.
    NativeCodegen,
}

/// Optional routing limits known before backend selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SolverLimits {
    /// Wall-clock time budget, if bounded.
    pub time_budget: Option<Duration>,
    /// Maximum exploration depth, if bounded.
    pub max_depth: Option<u32>,
    /// Maximum number of states, if bounded.
    pub max_states: Option<u64>,
    /// Maximum memory in bytes, if bounded.
    pub max_memory_bytes: Option<u64>,
}

/// Availability state for a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityStatus {
    /// Backend is available and eligible.
    Available,
    /// Backend was not found in the current environment.
    Unavailable,
    /// Backend exists but this model/property is outside its supported fragment.
    Unsupported,
    /// Backend is deliberately disabled by policy, feature flag, or env.
    Disabled,
    /// Backend is present but may only run behind parity/evidence gates.
    Experimental,
}

/// Policy role for a backend lane in a portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityRole {
    /// Eligible to produce the reported answer.
    Production,
    /// Used after preferred production lanes decline or time out.
    Fallback,
    /// Cross-checks another lane but should not be the sole source of truth.
    Validation,
    /// Unit-test or development-only backend.
    TestOnly,
}

/// Production-routing policy classification for a capability report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProductionRoutingStatus {
    /// A AY-backed production lane is available and selected.
    AYFirst,
    /// A local production lane is selected after a recorded AY rejection.
    JustifiedLocalFallback,
    /// A local production lane is selected without a recorded AY rejection.
    UnjustifiedLocalFallback,
    /// A production lane that does not require a AY rejection is selected.
    OtherProduction,
    /// No available production lane is selected.
    NoProductionSelection,
}

/// Shared status for detecting symbolic-execution needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolicExecutionStatus {
    /// No symbolic-execution need was detected.
    NotDetected,
    /// Symbolic execution is useful and should prefer a AY-backed lane.
    AYPreferred,
    /// Symbolic execution is required; local production needs an explicit AY rejection.
    AYRequired,
    /// A local lane is being used only after a recorded AY rejection.
    LocalFallbackAfterAYRejection,
}

/// Domain-neutral reason a caller detected symbolic execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolicExecutionReason {
    /// Initial-state constraints need solver-backed model enumeration.
    SymbolicInitialState,
    /// Transition/action relation needs symbolic solving.
    SymbolicTransitionRelation,
    /// Explicit enumeration is expected to explode.
    StateSpaceExplosion,
    /// A local/native lane cannot handle the required fragment.
    UnsupportedLocalFragment,
    /// A bit-vector SAT/SMT formula should be handled symbolically.
    BitVectorFormula,
    /// Alternate models require solver-level AllSAT or model blocking.
    ModelEnumeration,
    /// Native/JIT kernel generation declined this shape.
    NativeKernelUnsupported,
}

/// One symbolic-execution detection result for shared evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolicExecutionDetection {
    /// Detected symbolic-execution status.
    pub status: SymbolicExecutionStatus,
    /// Reason for the detection, when one applies.
    pub reason: Option<SymbolicExecutionReason>,
}

impl SymbolicExecutionDetection {
    /// A "no symbolic execution detected" result (status `NotDetected`).
    pub fn not_detected() -> Self {
        Self {
            status: SymbolicExecutionStatus::NotDetected,
            reason: None,
        }
    }

    /// A detection that prefers a AY-backed symbolic lane.
    pub fn ay_preferred(reason: SymbolicExecutionReason) -> Self {
        Self {
            status: SymbolicExecutionStatus::AYPreferred,
            reason: Some(reason),
        }
    }

    /// A detection that requires a AY-backed symbolic lane.
    pub fn ay_required(reason: SymbolicExecutionReason) -> Self {
        Self {
            status: SymbolicExecutionStatus::AYRequired,
            reason: Some(reason),
        }
    }

    /// A detection recording a local lane chosen only after a AY rejection.
    pub fn local_fallback_after_ay_rejection(reason: SymbolicExecutionReason) -> Self {
        Self {
            status: SymbolicExecutionStatus::LocalFallbackAfterAYRejection,
            reason: Some(reason),
        }
    }

    /// Whether this detection prefers a AY lane (`AYPreferred` or `AYRequired`).
    pub fn prefers_ay(self) -> bool {
        matches!(
            self.status,
            SymbolicExecutionStatus::AYPreferred | SymbolicExecutionStatus::AYRequired
        )
    }

    /// Whether this detection requires a AY lane (`AYRequired`).
    pub fn requires_ay(self) -> bool {
        matches!(self.status, SymbolicExecutionStatus::AYRequired)
    }

    /// The preferred AY backend for `problem` when this detection prefers AY,
    /// otherwise `None`.
    pub fn preferred_ay_backend(self, problem: ProblemKind) -> Option<BackendKind> {
        if self.prefers_ay() {
            Some(preferred_ay_backend_for_symbolic_execution(problem))
        } else {
            None
        }
    }
}

/// Structured reason for declining a backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnsupportedReason {
    /// A required external binary was not found (binary name carried).
    MissingBinary(&'static str),
    /// A required build feature is disabled (feature name carried).
    MissingFeature(&'static str),
    /// The model uses a fragment the backend does not support (detail carried).
    UnsupportedFragment(&'static str),
    /// The model uses a sort the backend does not support (sort name carried).
    UnsupportedSort(&'static str),
    /// The model needs nonlinear arithmetic the backend cannot handle.
    NonlinearArithmetic,
    /// The model has an unbounded domain the backend cannot enumerate.
    UnboundedDomain,
    /// The problem exceeds a size limit (limit description carried).
    TooLarge(&'static str),
    /// A deadline/time policy declined the backend (policy carried).
    DeadlinePolicy(&'static str),
    /// No native/JIT kernel is available.
    NativeKernelUnavailable,
    /// The backend is disabled by policy/flag/env (policy carried).
    DisabledByPolicy(&'static str),
    /// The backend is declined due to a known solver-bug risk (detail carried).
    SolverBugRisk(&'static str),
    /// A AY lane is preferred for this symbolic-execution problem.
    AYPreferredForSymbolicExecution,
    /// Any other reason (free-form detail carried).
    Other(&'static str),
}

impl UnsupportedReason {
    /// Stable machine-readable reason category.
    ///
    /// The returned value is intentionally independent of display text and
    /// variant payloads, so evidence consumers can group unsupported reasons
    /// without parsing human-readable messages.
    pub fn code(&self) -> &'static str {
        match self {
            Self::MissingBinary(_) => "missing_binary",
            Self::MissingFeature(_) => "missing_feature",
            Self::UnsupportedFragment(_) => "unsupported_fragment",
            Self::UnsupportedSort(_) => "unsupported_sort",
            Self::NonlinearArithmetic => "nonlinear_arithmetic",
            Self::UnboundedDomain => "unbounded_domain",
            Self::TooLarge(_) => "too_large",
            Self::DeadlinePolicy(_) => "deadline_policy",
            Self::NativeKernelUnavailable => "native_kernel_unavailable",
            Self::DisabledByPolicy(_) => "disabled_by_policy",
            Self::SolverBugRisk(_) => "solver_bug_risk",
            Self::AYPreferredForSymbolicExecution => "ay_preferred_for_symbolic_execution",
            Self::Other(_) => "other",
        }
    }
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBinary(name) => write!(f, "missing binary: {name}"),
            Self::MissingFeature(feature) => write!(f, "missing feature: {feature}"),
            Self::UnsupportedFragment(fragment) => write!(f, "unsupported fragment: {fragment}"),
            Self::UnsupportedSort(sort) => write!(f, "unsupported sort: {sort}"),
            Self::NonlinearArithmetic => write!(f, "nonlinear arithmetic"),
            Self::UnboundedDomain => write!(f, "unbounded domain"),
            Self::TooLarge(limit) => write!(f, "too large: {limit}"),
            Self::DeadlinePolicy(policy) => write!(f, "deadline policy: {policy}"),
            Self::NativeKernelUnavailable => write!(f, "native kernel unavailable"),
            Self::DisabledByPolicy(policy) => write!(f, "disabled by policy: {policy}"),
            Self::SolverBugRisk(risk) => write!(f, "solver bug risk: {risk}"),
            Self::AYPreferredForSymbolicExecution => {
                write!(f, "AY preferred for symbolic execution")
            }
            Self::Other(reason) => f.write_str(reason),
        }
    }
}

/// A normalized capability record that can be emitted as evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapability {
    /// Subsystem the backend serves.
    pub domain: BackendDomain,
    /// Backend family this record describes.
    pub backend: BackendKind,
    /// Problem class the record applies to, when scoped.
    pub problem: Option<ProblemKind>,
    /// Capability facets the backend exposes.
    pub facets: Vec<SolverFacet>,
    /// Policy role of the backend lane.
    pub role: CapabilityRole,
    /// Availability status.
    pub status: CapabilityStatus,
    /// Reason the backend is unavailable/unsupported/disabled, when applicable.
    pub reason: Option<UnsupportedReason>,
    /// Human-readable detail string.
    pub detail: Option<String>,
}

impl BackendCapability {
    /// Build an `Available` production capability with a detail string.
    pub fn available(
        domain: BackendDomain,
        backend: BackendKind,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            domain,
            backend,
            problem: None,
            facets: Vec::new(),
            role: CapabilityRole::Production,
            status: CapabilityStatus::Available,
            reason: None,
            detail: Some(detail.into()),
        }
    }

    /// Build an `Unavailable` production capability with a decline reason.
    pub fn unavailable(
        domain: BackendDomain,
        backend: BackendKind,
        reason: UnsupportedReason,
    ) -> Self {
        Self {
            domain,
            backend,
            problem: None,
            facets: Vec::new(),
            role: CapabilityRole::Production,
            status: CapabilityStatus::Unavailable,
            reason: Some(reason),
            detail: None,
        }
    }

    /// Build an `Unsupported` production capability with a decline reason.
    pub fn unsupported(
        domain: BackendDomain,
        backend: BackendKind,
        reason: UnsupportedReason,
    ) -> Self {
        Self {
            domain,
            backend,
            problem: None,
            facets: Vec::new(),
            role: CapabilityRole::Production,
            status: CapabilityStatus::Unsupported,
            reason: Some(reason),
            detail: None,
        }
    }

    /// Build a `Disabled` production capability with a decline reason.
    pub fn disabled(
        domain: BackendDomain,
        backend: BackendKind,
        reason: UnsupportedReason,
    ) -> Self {
        Self {
            domain,
            backend,
            problem: None,
            facets: Vec::new(),
            role: CapabilityRole::Production,
            status: CapabilityStatus::Disabled,
            reason: Some(reason),
            detail: None,
        }
    }

    /// Scope this capability to a specific [`problem`](Self::problem).
    pub fn for_problem(mut self, problem: ProblemKind) -> Self {
        self.problem = Some(problem);
        self
    }

    /// Add capability facets.
    pub fn with_facets(mut self, facets: impl IntoIterator<Item = SolverFacet>) -> Self {
        self.facets.extend(facets);
        self
    }

    /// Add the [`SolverFacet::SymbolicExecution`] facet (de-duplicated).
    pub fn with_symbolic_execution(mut self) -> Self {
        push_unique_facet(&mut self.facets, SolverFacet::SymbolicExecution);
        self
    }

    /// Set the policy [`role`](Self::role).
    pub fn with_role(mut self, role: CapabilityRole) -> Self {
        self.role = role;
        self
    }

    /// Set the [`detail`](Self::detail) string.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Stable reason code of the decline [`reason`](Self::reason), when any.
    pub fn reason_code(&self) -> Option<&'static str> {
        self.reason.as_ref().map(UnsupportedReason::code)
    }
}

impl BackendKind {
    /// Whether this backend is one of the AY in-process or external lanes.
    pub fn is_ay(self) -> bool {
        matches!(
            self,
            Self::ExternalAYBinary | Self::AYSmt | Self::AYSat | Self::AYChc
        )
    }

    /// Whether selecting this backend for production requires a recorded AY
    /// rejection first (true for non-AY, non-hardware-portfolio backends).
    pub fn requires_ay_rejection_for_production(self) -> bool {
        !self.is_ay() && !matches!(self, Self::AigerPortfolio | Self::Btor2Portfolio)
    }
}

/// Where a solver obligation is delegated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverDelegationTarget {
    /// Delegate the solver obligation to a AY-backed lane.
    AY,
    /// Use a local/domain-specific solver lane.
    Local,
}

/// A solver delegation decision paired with the chosen backend's capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverDelegation {
    /// Delegation target chosen.
    pub target: SolverDelegationTarget,
    /// Capability record of the chosen backend.
    pub capability: BackendCapability,
}

impl SolverDelegation {
    /// Build a delegation targeting a AY lane.
    pub fn ay(capability: BackendCapability) -> Self {
        Self {
            target: SolverDelegationTarget::AY,
            capability,
        }
    }

    /// Build a delegation targeting a local lane.
    pub fn local(capability: BackendCapability) -> Self {
        Self {
            target: SolverDelegationTarget::Local,
            capability,
        }
    }
}

/// Backend routing evidence for one problem or model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityReport {
    /// Problem class the report is for, when scoped.
    pub problem: Option<ProblemKind>,
    /// Routing limits known before selection.
    pub limits: SolverLimits,
    /// Backends selected to run.
    pub selected: Vec<BackendCapability>,
    /// Backends considered and rejected.
    pub rejected: Vec<BackendCapability>,
    /// Free-form evidence lines accumulated during routing.
    pub evidence: Vec<String>,
}

impl CapabilityReport {
    /// Create an empty report scoped to `problem`.
    pub fn new(problem: ProblemKind) -> Self {
        Self {
            problem: Some(problem),
            ..Self::default()
        }
    }

    /// Set the routing [`limits`](Self::limits).
    pub fn with_limits(mut self, limits: SolverLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Record `capability` as selected.
    pub fn select(&mut self, capability: BackendCapability) {
        self.selected.push(capability);
    }

    /// Record `capability` as rejected.
    pub fn reject(&mut self, capability: BackendCapability) {
        self.rejected.push(capability);
    }

    /// Append an evidence line.
    pub fn add_evidence(&mut self, evidence: impl Into<String>) {
        self.evidence.push(evidence.into());
    }

    /// Whether `backend` is among the selected backends.
    pub fn has_selected(&self, backend: BackendKind) -> bool {
        self.selected.iter().any(|cap| cap.backend == backend)
    }

    /// The decline reason recorded for a rejected `backend`, if any.
    pub fn rejection_reason(&self, backend: BackendKind) -> Option<&UnsupportedReason> {
        self.rejected
            .iter()
            .find(|cap| cap.backend == backend)
            .and_then(|cap| cap.reason.as_ref())
    }

    /// Stable reason code for a rejected `backend`, if any.
    pub fn rejection_reason_code(&self, backend: BackendKind) -> Option<&'static str> {
        self.rejection_reason(backend).map(UnsupportedReason::code)
    }

    /// Whether an available AY production lane was selected.
    pub fn ay_selected_for_production(&self) -> bool {
        self.selected.iter().any(|capability| {
            capability.backend.is_ay()
                && capability.role == CapabilityRole::Production
                && capability.status == CapabilityStatus::Available
        })
    }

    /// Whether a AY lane was rejected with a recorded reason.
    pub fn ay_rejected(&self) -> bool {
        self.rejected
            .iter()
            .any(|capability| capability.backend.is_ay() && capability.reason.is_some())
    }

    /// Whether selecting a local production lane is allowed (a AY lane was rejected).
    pub fn local_production_allowed(&self) -> bool {
        self.ay_rejected()
    }

    /// Classify the production-routing outcome of this report.
    ///
    /// Returns [`ProductionRoutingStatus::AYFirst`] when a AY production lane is
    /// selected; otherwise distinguishes justified vs unjustified local
    /// fallback, other production selections, and no production selection.
    pub fn production_routing_status(&self) -> ProductionRoutingStatus {
        if self.ay_selected_for_production() {
            return ProductionRoutingStatus::AYFirst;
        }

        let has_local_production_requiring_ay_rejection = self.selected.iter().any(|capability| {
            capability.backend.requires_ay_rejection_for_production()
                && capability.role == CapabilityRole::Production
                && capability.status == CapabilityStatus::Available
        });
        if has_local_production_requiring_ay_rejection {
            if self.local_production_allowed() {
                ProductionRoutingStatus::JustifiedLocalFallback
            } else {
                ProductionRoutingStatus::UnjustifiedLocalFallback
            }
        } else if self.selected.iter().any(|capability| {
            capability.role == CapabilityRole::Production
                && capability.status == CapabilityStatus::Available
        }) {
            ProductionRoutingStatus::OtherProduction
        } else {
            ProductionRoutingStatus::NoProductionSelection
        }
    }

    /// Whether a local production lane requiring a AY rejection was selected
    /// without that rejection being recorded.
    pub fn has_unjustified_local_production(&self) -> bool {
        self.selected.iter().any(|capability| {
            capability.backend.requires_ay_rejection_for_production()
                && capability.role == CapabilityRole::Production
                && capability.status == CapabilityStatus::Available
        }) && !self.local_production_allowed()
    }
}

/// Build an available in-process AY SMT capability for `problem`.
pub fn ay_smt_capability(domain: BackendDomain, problem: ProblemKind) -> BackendCapability {
    let mut facets = vec![
        SolverFacet::InProcess,
        SolverFacet::Smt,
        SolverFacet::ModelValues,
    ];
    push_problem_facet(&mut facets, problem);
    BackendCapability::available(domain, BackendKind::AYSmt, "in-process AY SMT")
        .for_problem(problem)
        .with_facets(facets)
}

/// Build an available in-process AY CHC/PDR capability for `problem`.
pub fn ay_chc_capability(domain: BackendDomain, problem: ProblemKind) -> BackendCapability {
    let mut facets = vec![SolverFacet::InProcess, SolverFacet::Chc, SolverFacet::Pdr];
    push_problem_facet(&mut facets, problem);
    BackendCapability::available(domain, BackendKind::AYChc, "in-process AY CHC/PDR")
        .for_problem(problem)
        .with_facets(facets)
}

/// Build an available in-process ay-sat CDCL capability for `problem`.
pub fn ay_sat_capability(domain: BackendDomain, problem: ProblemKind) -> BackendCapability {
    let mut facets = vec![
        SolverFacet::InProcess,
        SolverFacet::Sat,
        SolverFacet::Incremental,
        SolverFacet::Assumptions,
    ];
    push_problem_facet(&mut facets, problem);
    BackendCapability::available(domain, BackendKind::AYSat, "in-process ay-sat CDCL")
        .for_problem(problem)
        .with_facets(facets)
}

/// Build the preferred AY symbolic-execution capability for `problem`,
/// selecting the SAT/CHC/SMT lane that best fits the problem kind.
pub fn ay_symbolic_execution_capability(
    domain: BackendDomain,
    problem: ProblemKind,
) -> BackendCapability {
    let capability = match preferred_ay_backend_for_symbolic_execution(problem) {
        BackendKind::AYSat => ay_sat_capability(domain, problem),
        BackendKind::AYChc => ay_chc_capability(domain, problem),
        _ => ay_smt_capability(domain, problem),
    };
    capability
        .with_symbolic_execution()
        .with_detail("AY symbolic execution preferred over local solver reimplementation")
}

/// Build an available local symbolic-execution capability for `problem`.
///
/// Callers should record in `detail` why the preferred AY lane was not used.
pub fn local_symbolic_execution_capability(
    domain: BackendDomain,
    problem: ProblemKind,
    detail: impl Into<String>,
) -> BackendCapability {
    BackendCapability::available(domain, BackendKind::LocalSymbolicExecution, detail)
        .for_problem(problem)
        .with_symbolic_execution()
}

/// The preferred AY backend for a symbolic-execution `problem`
/// (`AYSat` for SAT, `AYChc` for CHC, otherwise `AYSmt`).
pub fn preferred_ay_backend_for_symbolic_execution(problem: ProblemKind) -> BackendKind {
    match problem {
        ProblemKind::Sat => BackendKind::AYSat,
        ProblemKind::Chc => BackendKind::AYChc,
        _ => BackendKind::AYSmt,
    }
}

fn push_unique_facet(facets: &mut Vec<SolverFacet>, facet: SolverFacet) {
    if !facets.contains(&facet) {
        facets.push(facet);
    }
}

fn push_problem_facet(facets: &mut Vec<SolverFacet>, problem: ProblemKind) {
    let facet = match problem {
        ProblemKind::Bmc => Some(SolverFacet::Bmc),
        ProblemKind::KInduction => Some(SolverFacet::KInduction),
        ProblemKind::Chc => Some(SolverFacet::Chc),
        ProblemKind::Sat => Some(SolverFacet::Sat),
        ProblemKind::Smt => Some(SolverFacet::Smt),
        ProblemKind::SymbolicExecution => Some(SolverFacet::SymbolicExecution),
        _ => None,
    };
    if let Some(facet) = facet {
        push_unique_facet(facets, facet);
    }
}

/// Where an external AY binary was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AYBinarySource {
    /// Explicit `AY_PATH`.
    EnvAYPath,
    /// Conventional local checkout build at `~/ay/target/release/ay`.
    HomeBuild,
    /// A `ay` executable found on `PATH`.
    Path,
}

/// External AY binary availability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AYBinaryAvailability {
    /// Path to the discovered AY executable.
    pub path: PathBuf,
    /// How the executable was located.
    pub source: AYBinarySource,
}

impl AYBinaryAvailability {
    /// Build an available external-AY-binary capability for `domain` that names
    /// the discovered path and source.
    pub fn capability(&self, domain: BackendDomain) -> BackendCapability {
        BackendCapability::available(
            domain,
            BackendKind::ExternalAYBinary,
            format!("ay binary at {} via {:?}", self.path.display(), self.source),
        )
    }
}

/// Find an external AY binary using the shared TY policy.
///
/// Search order:
/// 1. `AY_PATH`
/// 2. `~/ay/target/release/ay`
/// 3. `PATH`
///
/// This intentionally avoids shelling out to `which`; callers can use it in MCC
/// deadline-sensitive paths without leaving helper processes behind.
pub fn find_ay_binary() -> Option<AYBinaryAvailability> {
    find_ay_binary_from(
        std::env::var_os("AY_PATH").as_deref(),
        std::env::var_os("HOME").as_deref().map(Path::new),
        std::env::var_os("PATH").as_deref(),
    )
}

/// Build a capability record for the external AY binary.
pub fn external_ay_binary_capability(domain: BackendDomain) -> BackendCapability {
    match find_ay_binary() {
        Some(found) => found.capability(domain),
        None => BackendCapability::unavailable(
            domain,
            BackendKind::ExternalAYBinary,
            UnsupportedReason::MissingBinary("ay"),
        ),
    }
}

fn find_ay_binary_from(
    ay_path: Option<&OsStr>,
    home: Option<&Path>,
    path_env: Option<&OsStr>,
) -> Option<AYBinaryAvailability> {
    if let Some(path) = ay_path {
        let path = PathBuf::from(path);
        if is_executable_file(&path) {
            return Some(AYBinaryAvailability {
                path,
                source: AYBinarySource::EnvAYPath,
            });
        }
    }

    if let Some(home) = home {
        let path = home.join("ay/target/release/ay");
        if is_executable_file(&path) {
            return Some(AYBinaryAvailability {
                path,
                source: AYBinarySource::HomeBuild,
            });
        }
    }

    let path_env = path_env?;
    for dir in std::env::split_paths(path_env) {
        let candidate = dir.join(executable_name("ay"));
        if is_executable_file(&candidate) {
            return Some(AYBinaryAvailability {
                path: candidate,
                source: AYBinarySource::Path,
            });
        }
    }

    None
}

fn executable_name(base: &str) -> String {
    #[cfg(windows)]
    {
        format!("{base}.exe")
    }
    #[cfg(not(windows))]
    {
        base.to_string()
    }
}

fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn ay_path_takes_precedence() {
        let temp = temp_dir("ay_path_takes_precedence");
        let env_ay = temp.join("env-ay");
        let home_ay = temp.join("home/ay/target/release/ay");
        let path_ay_dir = temp.join("bin");
        let path_ay = path_ay_dir.join(executable_name("ay"));
        fs::create_dir_all(home_ay.parent().unwrap()).unwrap();
        fs::create_dir_all(&path_ay_dir).unwrap();
        fs::write(&env_ay, b"").unwrap();
        fs::write(&home_ay, b"").unwrap();
        fs::write(&path_ay, b"").unwrap();

        let found = find_ay_binary_from(
            Some(env_ay.as_os_str()),
            Some(&temp.join("home")),
            Some(path_ay_dir.as_os_str()),
        )
        .unwrap();

        assert_eq!(found.path, env_ay);
        assert_eq!(found.source, AYBinarySource::EnvAYPath);
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn falls_back_to_home_build_before_path() {
        let temp = temp_dir("falls_back_to_home_build_before_path");
        let home_ay = temp.join("home/ay/target/release/ay");
        let path_ay_dir = temp.join("bin");
        let path_ay = path_ay_dir.join(executable_name("ay"));
        fs::create_dir_all(home_ay.parent().unwrap()).unwrap();
        fs::create_dir_all(&path_ay_dir).unwrap();
        fs::write(&home_ay, b"").unwrap();
        fs::write(&path_ay, b"").unwrap();

        let found = find_ay_binary_from(
            None,
            Some(&temp.join("home")),
            Some(path_ay_dir.as_os_str()),
        )
        .unwrap();

        assert_eq!(found.path, home_ay);
        assert_eq!(found.source, AYBinarySource::HomeBuild);
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn finds_ay_on_path() {
        let temp = temp_dir("finds_ay_on_path");
        let path_ay_dir = temp.join("bin");
        let path_ay = path_ay_dir.join(executable_name("ay"));
        fs::create_dir_all(&path_ay_dir).unwrap();
        fs::write(&path_ay, b"").unwrap();

        let found = find_ay_binary_from(None, None, Some(path_ay_dir.as_os_str())).unwrap();

        assert_eq!(found.path, path_ay);
        assert_eq!(found.source, AYBinarySource::Path);
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn capability_report_tracks_selection_and_rejection() {
        let mut report = CapabilityReport::new(ProblemKind::Bmc).with_limits(SolverLimits {
            time_budget: Some(Duration::from_secs(5)),
            max_depth: Some(8),
            max_states: None,
            max_memory_bytes: None,
        });

        report.select(
            BackendCapability::available(
                BackendDomain::PetriMcc,
                BackendKind::ExternalAYBinary,
                "ay binary found",
            )
            .for_problem(ProblemKind::Bmc)
            .with_facets([SolverFacet::ExternalProcess, SolverFacet::Smt])
            .with_role(CapabilityRole::Production),
        );
        report.reject(BackendCapability::unsupported(
            BackendDomain::PetriMcc,
            BackendKind::NativeKernel,
            UnsupportedReason::NativeKernelUnavailable,
        ));
        report.add_evidence("external ay selected for depth ladder");

        assert!(report.has_selected(BackendKind::ExternalAYBinary));
        assert_eq!(report.problem, Some(ProblemKind::Bmc));
        assert_eq!(report.selected[0].role, CapabilityRole::Production);
        assert_eq!(
            report.selected[0].facets,
            vec![SolverFacet::ExternalProcess, SolverFacet::Smt]
        );
        assert_eq!(report.limits.max_depth, Some(8));
        assert_eq!(
            report.rejection_reason(BackendKind::NativeKernel),
            Some(&UnsupportedReason::NativeKernelUnavailable)
        );
        assert_eq!(
            report.rejection_reason_code(BackendKind::NativeKernel),
            Some("native_kernel_unavailable")
        );
        assert_eq!(report.evidence.len(), 1);
    }

    #[test]
    fn unsupported_reason_codes_are_stable_categories() {
        assert_eq!(
            UnsupportedReason::MissingBinary("ay").code(),
            "missing_binary"
        );
        assert_eq!(
            UnsupportedReason::MissingFeature("ay-sat").code(),
            "missing_feature"
        );
        assert_eq!(
            UnsupportedReason::UnsupportedFragment("quantifier").code(),
            "unsupported_fragment"
        );
        assert_eq!(
            UnsupportedReason::UnsupportedSort("set").code(),
            "unsupported_sort"
        );
        assert_eq!(
            UnsupportedReason::NonlinearArithmetic.code(),
            "nonlinear_arithmetic"
        );
        assert_eq!(
            UnsupportedReason::UnboundedDomain.code(),
            "unbounded_domain"
        );
        assert_eq!(
            UnsupportedReason::TooLarge("state space").code(),
            "too_large"
        );
        assert_eq!(
            UnsupportedReason::DeadlinePolicy("no time left").code(),
            "deadline_policy"
        );
        assert_eq!(
            UnsupportedReason::NativeKernelUnavailable.code(),
            "native_kernel_unavailable"
        );
        assert_eq!(
            UnsupportedReason::DisabledByPolicy("env").code(),
            "disabled_by_policy"
        );
        assert_eq!(
            UnsupportedReason::SolverBugRisk("known issue").code(),
            "solver_bug_risk"
        );
        assert_eq!(
            UnsupportedReason::AYPreferredForSymbolicExecution.code(),
            "ay_preferred_for_symbolic_execution"
        );
        assert_eq!(UnsupportedReason::Other("legacy detail").code(), "other");
    }

    #[test]
    fn backend_capability_exposes_optional_reason_code() {
        let available = BackendCapability::available(
            BackendDomain::Shared,
            BackendKind::ExplicitState,
            "explicit state available",
        );
        assert_eq!(available.reason_code(), None);

        let unsupported = BackendCapability::unsupported(
            BackendDomain::Tla,
            BackendKind::AYSmt,
            UnsupportedReason::UnsupportedSort("sequence"),
        );
        assert_eq!(unsupported.reason_code(), Some("unsupported_sort"));
    }

    #[test]
    fn delegation_policy_flags_unjustified_local_production() {
        let mut report = CapabilityReport::new(ProblemKind::Sat);
        report.select(
            BackendCapability::available(
                BackendDomain::Aiger,
                BackendKind::ExplicitState,
                "local SAT fallback",
            )
            .for_problem(ProblemKind::Sat),
        );

        assert!(!report.ay_selected_for_production());
        assert!(!report.local_production_allowed());
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::UnjustifiedLocalFallback
        );
        assert!(report.has_unjustified_local_production());
    }

    #[test]
    fn ay_rejection_allows_local_production() {
        let mut report = CapabilityReport::new(ProblemKind::Sat);
        report.reject(
            BackendCapability::unavailable(
                BackendDomain::Aiger,
                BackendKind::AYSat,
                UnsupportedReason::MissingFeature("ay-sat"),
            )
            .for_problem(ProblemKind::Sat),
        );
        report.select(
            BackendCapability::available(
                BackendDomain::Aiger,
                BackendKind::ExplicitState,
                "local SAT fallback",
            )
            .for_problem(ProblemKind::Sat),
        );

        assert!(report.ay_rejected());
        assert!(report.local_production_allowed());
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::JustifiedLocalFallback
        );
        assert!(!report.has_unjustified_local_production());
    }

    #[test]
    fn ay_production_selection_is_detected() {
        let mut report = CapabilityReport::new(ProblemKind::Bmc);
        report.select(ay_smt_capability(BackendDomain::Tla, ProblemKind::Bmc));

        assert!(report.ay_selected_for_production());
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::AYFirst
        );
        assert!(!report.has_unjustified_local_production());
    }

    #[test]
    fn portfolio_wrappers_do_not_require_ay_rejection() {
        let mut report = CapabilityReport::new(ProblemKind::Safety);
        report.select(
            BackendCapability::available(
                BackendDomain::Btor2,
                BackendKind::Btor2Portfolio,
                "BTOR2 portfolio wrapper",
            )
            .for_problem(ProblemKind::Safety),
        );
        report.select(ay_chc_capability(BackendDomain::Btor2, ProblemKind::Chc));

        assert!(report.ay_selected_for_production());
        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::AYFirst
        );
        assert!(!report.has_unjustified_local_production());
    }

    #[test]
    fn production_routing_status_reports_other_production() {
        let mut report = CapabilityReport::new(ProblemKind::Safety);
        report.select(
            BackendCapability::available(
                BackendDomain::Aiger,
                BackendKind::AigerPortfolio,
                "AIGER portfolio wrapper",
            )
            .for_problem(ProblemKind::Safety),
        );

        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::OtherProduction
        );
        assert!(!report.has_unjustified_local_production());
    }

    #[test]
    fn production_routing_status_reports_no_production_selection() {
        let mut report = CapabilityReport::new(ProblemKind::Sat);
        report.select(
            BackendCapability::available(
                BackendDomain::Aiger,
                BackendKind::ExplicitState,
                "local validation lane",
            )
            .for_problem(ProblemKind::Sat)
            .with_role(CapabilityRole::Validation),
        );

        assert_eq!(
            report.production_routing_status(),
            ProductionRoutingStatus::NoProductionSelection
        );
        assert!(!report.has_unjustified_local_production());
    }

    #[test]
    fn ay_capability_constructors_attach_expected_facets() {
        let smt = ay_smt_capability(BackendDomain::Tla, ProblemKind::KInduction);
        assert_eq!(smt.backend, BackendKind::AYSmt);
        assert!(smt.facets.contains(&SolverFacet::InProcess));
        assert!(smt.facets.contains(&SolverFacet::Smt));
        assert!(smt.facets.contains(&SolverFacet::ModelValues));
        assert!(smt.facets.contains(&SolverFacet::KInduction));

        let chc = ay_chc_capability(BackendDomain::Btor2, ProblemKind::Chc);
        assert_eq!(chc.backend, BackendKind::AYChc);
        assert!(chc.facets.contains(&SolverFacet::Chc));
        assert!(chc.facets.contains(&SolverFacet::Pdr));

        let sat = ay_sat_capability(BackendDomain::Aiger, ProblemKind::Sat);
        assert_eq!(sat.backend, BackendKind::AYSat);
        assert!(sat.facets.contains(&SolverFacet::Sat));
        assert!(sat.facets.contains(&SolverFacet::Incremental));
        assert!(sat.facets.contains(&SolverFacet::Assumptions));
    }

    #[test]
    fn symbolic_execution_detection_prefers_problem_appropriate_ay_backend() {
        let detection =
            SymbolicExecutionDetection::ay_preferred(SymbolicExecutionReason::ModelEnumeration);

        assert!(detection.prefers_ay());
        assert!(!detection.requires_ay());
        assert_eq!(
            detection.preferred_ay_backend(ProblemKind::Sat),
            Some(BackendKind::AYSat)
        );
        assert_eq!(
            detection.preferred_ay_backend(ProblemKind::Chc),
            Some(BackendKind::AYChc)
        );
        assert_eq!(
            detection.preferred_ay_backend(ProblemKind::SymbolicExecution),
            Some(BackendKind::AYSmt)
        );
        assert_eq!(
            SymbolicExecutionDetection::not_detected().preferred_ay_backend(ProblemKind::Bmc),
            None
        );

        let required =
            SymbolicExecutionDetection::ay_required(SymbolicExecutionReason::StateSpaceExplosion);
        assert!(required.prefers_ay());
        assert!(required.requires_ay());
    }

    #[test]
    fn symbolic_execution_capability_helpers_mark_ay_as_preferred_path() {
        let ay_sat = ay_symbolic_execution_capability(BackendDomain::Aiger, ProblemKind::Sat);
        assert_eq!(ay_sat.backend, BackendKind::AYSat);
        assert!(ay_sat.facets.contains(&SolverFacet::SymbolicExecution));
        assert!(ay_sat.facets.contains(&SolverFacet::Sat));
        assert_eq!(
            ay_sat.detail.as_deref(),
            Some("AY symbolic execution preferred over local solver reimplementation")
        );

        let ay_smt =
            ay_symbolic_execution_capability(BackendDomain::Tla, ProblemKind::SymbolicExecution);
        assert_eq!(ay_smt.backend, BackendKind::AYSmt);
        assert_eq!(
            ay_smt
                .facets
                .iter()
                .filter(|facet| **facet == SolverFacet::SymbolicExecution)
                .count(),
            1
        );

        let local = local_symbolic_execution_capability(
            BackendDomain::Tla,
            ProblemKind::SymbolicExecution,
            "local symbolic prototype",
        );
        assert_eq!(local.backend, BackendKind::LocalSymbolicExecution);
        assert!(local.facets.contains(&SolverFacet::SymbolicExecution));
        assert!(local.backend.requires_ay_rejection_for_production());
    }

    #[test]
    fn local_symbolic_execution_production_requires_ay_rejection() {
        let mut unjustified = CapabilityReport::new(ProblemKind::SymbolicExecution);
        unjustified.select(local_symbolic_execution_capability(
            BackendDomain::Tla,
            ProblemKind::SymbolicExecution,
            "local symbolic prototype",
        ));
        assert_eq!(
            unjustified.production_routing_status(),
            ProductionRoutingStatus::UnjustifiedLocalFallback
        );
        assert!(unjustified.has_unjustified_local_production());

        let mut justified = CapabilityReport::new(ProblemKind::SymbolicExecution);
        justified.reject(
            BackendCapability::unsupported(
                BackendDomain::Tla,
                BackendKind::AYSmt,
                UnsupportedReason::UnsupportedFragment("quantifier"),
            )
            .for_problem(ProblemKind::SymbolicExecution)
            .with_symbolic_execution(),
        );
        justified.select(local_symbolic_execution_capability(
            BackendDomain::Tla,
            ProblemKind::SymbolicExecution,
            "local symbolic prototype",
        ));
        assert_eq!(
            justified.production_routing_status(),
            ProductionRoutingStatus::JustifiedLocalFallback
        );
        assert!(!justified.has_unjustified_local_production());
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("tla-mc-core-{name}-{nanos}"))
    }
}
