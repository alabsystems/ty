// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The typed request that flows through every layer, plus the engine identity and
//! the AUTO/Forced/Oracle selection contract. Replaces the boolean/env soup at
//! `crates/tla-cli/src/main.rs:548-575`.

// Shared capability vocabulary — reused, NOT re-minted. No TLC variant exists here
// (constraint 7 falls out for free).
pub use tla_mc_core::{BackendDomain, BackendKind, ProblemKind};

/// Stable identity for each selectable engine. There is deliberately **no** `Tlc`
/// variant — TLC is only an external testing oracle, never a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EngineId {
    /// Tree-walking interpreter — the permanent oracle and universal fallback.
    Interpreter,
    /// trust-cg native-compiled BFS (sequential today; O3 in production).
    TrustCgNative,
    /// AY symbolic lanes: BMC / PDR / k-induction / CHC.
    AySymbolic,
    /// MCC / Petri explorer (+ BDD + LP).
    PetriExplorer,
    /// AIGER / BTOR2 hardware portfolio.
    Hardware,
}

impl EngineId {
    /// Map onto the existing [`BackendDomain`]. (`Hardware` defaults to `Aiger`;
    /// BTOR2 vs AIGER is resolved per-spec by the hardware adapter.)
    #[must_use]
    pub fn domain(self) -> BackendDomain {
        match self {
            EngineId::Interpreter | EngineId::TrustCgNative => BackendDomain::Tla,
            EngineId::AySymbolic => BackendDomain::AY,
            EngineId::PetriExplorer => BackendDomain::PetriMcc,
            EngineId::Hardware => BackendDomain::Aiger,
        }
    }

    /// Map onto the existing [`BackendKind`].
    #[must_use]
    pub fn backend_kind(self) -> BackendKind {
        match self {
            EngineId::Interpreter | EngineId::PetriExplorer => BackendKind::ExplicitState,
            EngineId::TrustCgNative => BackendKind::NativeKernel,
            EngineId::AySymbolic => BackendKind::AYSmt,
            EngineId::Hardware => BackendKind::AigerPortfolio,
        }
    }

    /// The interpreter is the never-unavailable oracle.
    #[must_use]
    pub fn is_oracle(self) -> bool {
        matches!(self, EngineId::Interpreter)
    }
}

/// What the user asked for. The typed encoding of the structural-veto contract at
/// `main.rs:543-552`: `Auto` runs the structural veto + post-compile teardown;
/// `Forced(native)` does NOT (this is what the supremacy/oracle harnesses, which
/// pass `--backend trust-cg` explicitly, depend on); `Oracle` forces the interpreter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    /// No `--backend`: native default + structural auto-selection.
    Auto,
    /// `--backend <engine>`: forced, no structural veto.
    Forced(EngineId),
    /// `--backend interpreter`: forced oracle.
    Oracle,
}

/// One typed value that flows through every layer.
#[derive(Clone, Debug)]
pub struct EngineRequest {
    /// What the user asked for: AUTO selection, a forced engine, or the oracle.
    pub mode: SelectionMode,
    /// The verification problem class (safety, liveness, …) the run addresses.
    pub problem: ProblemKind,
}

impl EngineRequest {
    /// `ty check` request (safety problem class).
    #[must_use]
    pub fn for_check(mode: SelectionMode) -> Self {
        Self::for_problem(ProblemKind::Safety, mode)
    }

    /// A request for an arbitrary problem class.
    #[must_use]
    pub fn for_problem(problem: ProblemKind, mode: SelectionMode) -> Self {
        EngineRequest { mode, problem }
    }

    /// AUTO mode runs the structural veto + post-compile coverage teardown.
    #[must_use]
    pub fn auto_select_enabled(&self) -> bool {
        matches!(self.mode, SelectionMode::Auto)
    }

    /// Whether the native compiled path is requested: AUTO, or an explicit native
    /// `Forced`. `Oracle` and a non-native `Forced` are not native.
    #[must_use]
    pub fn wants_native(&self) -> bool {
        match self.mode {
            SelectionMode::Auto => true,
            SelectionMode::Forced(e) => matches!(e, EngineId::TrustCgNative),
            SelectionMode::Oracle => false,
        }
    }
}
