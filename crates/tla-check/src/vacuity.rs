// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Vacuity-gate verdict and warning types (design: TRUST_VACUITY_GATE §1.A).
//!
//! General vacuity is undecidable; this module ships only the *detectable*,
//! *sound* sub-cases:
//!
//! - **V1 — empty reachable set** (`VacuityReason::EmptyReachableSet`): a model
//!   that declares at least one of Init/Next/an invariant/a temporal property
//!   but admits zero reachable states. This is a hard `VACUOUS` verdict (exit
//!   code 3). A genuinely-degenerate `ASSUME`-only module is exempt because it
//!   declares nothing to check.
//! - **V2 — never-enabled (dead) action** (`VacuityWarning::DeadActions`): an
//!   action disjunct of `Next` that never fired in any reachable state.
//!   Default-on WARNING, promotable to exit 3 via `--strict-vacuity`.
//! - **V3 — vacuously-true invariant** (`VacuityWarning::VacuousInvariant`):
//!   the two SOUND special-cases only — a top-level implication `P => Q` whose
//!   antecedent `P` never holds, or an invariant that constant-folds to `TRUE`
//!   independent of state. Default-on WARNING, promotable via `--strict-vacuity`.
//!
//! The verdict (`VACUOUS`, exit 3) is distinct from `FAILED` (exit 1) so CI can
//! tell "your spec is wrong" apart from "your property is false." The escape
//! hatch `--allow-vacuous=<class>` downgrades a named class to WARNING; the
//! policy is applied at the CLI verdict/exit-mapping layer.

/// The shared, escape-hatch-addressable class of a vacuity finding.
///
/// Used by `--allow-vacuous=<class>[,...]` to downgrade a named class to a
/// recorded WARNING. The string forms are the stable, user-facing names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VacuityClass {
    /// V1 — empty reachable set (`--allow-vacuous=empty-init`).
    EmptyInit,
    /// V2 — never-enabled (dead) action (`--allow-vacuous=dead-action`).
    DeadAction,
    /// V3 — vacuously-true invariant (`--allow-vacuous=vacuous-invariant`).
    VacuousInvariant,
}

impl VacuityClass {
    /// Stable, user-facing class name as accepted by `--allow-vacuous`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            VacuityClass::EmptyInit => "empty-init",
            VacuityClass::DeadAction => "dead-action",
            VacuityClass::VacuousInvariant => "vacuous-invariant",
        }
    }

    /// Parse a `--allow-vacuous` class token. Accepts a couple of synonyms so
    /// the surface is forgiving (e.g. `empty-reachable-set`).
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "empty-init" | "empty-reachable-set" | "empty_init" => Some(VacuityClass::EmptyInit),
            "dead-action" | "dead-actions" | "dead_action" => Some(VacuityClass::DeadAction),
            "vacuous-invariant" | "vacuous-invariants" | "vacuous_invariant" => {
                Some(VacuityClass::VacuousInvariant)
            }
            _ => None,
        }
    }

    /// All recognized classes, for help text and validation.
    #[must_use]
    pub const fn all() -> [VacuityClass; 3] {
        [
            VacuityClass::EmptyInit,
            VacuityClass::DeadAction,
            VacuityClass::VacuousInvariant,
        ]
    }
}

/// Why a run is `VACUOUS` (the hard, exit-3 verdict). Today only V1 reaches a
/// hard verdict in the core; V2/V3 surface as [`VacuityWarning`]s and are only
/// promoted to a verdict at the CLI policy layer under `--strict-vacuity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VacuityReason {
    /// V1 — fixpoint reached with zero reachable states despite the module
    /// declaring at least one of Init/Next/an invariant/a temporal property.
    EmptyReachableSet,
    /// V2 — one or more anchored actions never fired (promoted from WARNING by
    /// `--strict-vacuity`).
    DeadActions(Vec<String>),
    /// V3 — one or more invariants are vacuously true (promoted from WARNING by
    /// `--strict-vacuity`).
    VacuousInvariants(Vec<String>),
}

impl VacuityReason {
    /// The escape-hatch class this reason belongs to.
    #[must_use]
    pub fn class(&self) -> VacuityClass {
        match self {
            VacuityReason::EmptyReachableSet => VacuityClass::EmptyInit,
            VacuityReason::DeadActions(_) => VacuityClass::DeadAction,
            VacuityReason::VacuousInvariants(_) => VacuityClass::VacuousInvariant,
        }
    }

    /// One-line, user-facing description of the verdict.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            VacuityReason::EmptyReachableSet => {
                "empty reachable set: the model admits no states, yet declares \
                 Init/Next/an invariant/a temporal property"
                    .to_string()
            }
            VacuityReason::DeadActions(names) => {
                format!(
                    "{} dead action(s) (never fired): {}",
                    names.len(),
                    names.join(", ")
                )
            }
            VacuityReason::VacuousInvariants(names) => {
                format!(
                    "{} vacuously-true invariant(s): {}",
                    names.len(),
                    names.join(", ")
                )
            }
        }
    }
}

/// A default-on, non-fatal vacuity WARNING (V2/V3). Promotable to a `VACUOUS`
/// verdict (exit 3) by `--strict-vacuity`; downgradable to nothing by
/// `--allow-vacuous=<class>` (it stays recorded so the relaxation is audited).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VacuityWarning {
    /// V2 — a `Next` disjunct that never fired in any reachable state.
    DeadActions(Vec<String>),
    /// V3 — a top-level `P => Q` invariant whose antecedent never held.
    AntecedentNeverHolds {
        /// The invariant name.
        invariant: String,
    },
    /// V3 — an invariant that constant-folds to `TRUE` independent of state.
    ConstantTrueInvariant {
        /// The invariant name.
        invariant: String,
    },
}

impl VacuityWarning {
    /// The escape-hatch class this warning belongs to.
    #[must_use]
    pub fn class(&self) -> VacuityClass {
        match self {
            VacuityWarning::DeadActions(_) => VacuityClass::DeadAction,
            VacuityWarning::AntecedentNeverHolds { .. }
            | VacuityWarning::ConstantTrueInvariant { .. } => VacuityClass::VacuousInvariant,
        }
    }

    /// One-line, user-facing warning text.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            VacuityWarning::DeadActions(names) => format!(
                "{} dead action(s) (never fired): {}",
                names.len(),
                names.join(", ")
            ),
            VacuityWarning::AntecedentNeverHolds { invariant } => {
                format!("invariant {invariant} is vacuously true: antecedent never holds")
            }
            VacuityWarning::ConstantTrueInvariant { invariant } => format!(
                "invariant {invariant} is vacuously true: constant-folds to TRUE \
                 independent of state"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_roundtrip() {
        for c in VacuityClass::all() {
            assert_eq!(VacuityClass::parse(c.as_str()), Some(c));
        }
    }

    #[test]
    fn class_synonyms_parse() {
        assert_eq!(
            VacuityClass::parse("empty-reachable-set"),
            Some(VacuityClass::EmptyInit)
        );
        assert_eq!(
            VacuityClass::parse("dead-actions"),
            Some(VacuityClass::DeadAction)
        );
        assert_eq!(VacuityClass::parse("nonsense"), None);
    }

    #[test]
    fn reason_class_mapping() {
        assert_eq!(
            VacuityReason::EmptyReachableSet.class(),
            VacuityClass::EmptyInit
        );
        assert_eq!(
            VacuityReason::DeadActions(vec!["A".into()]).class(),
            VacuityClass::DeadAction
        );
    }

    #[test]
    fn warning_messages_nonempty() {
        let w = VacuityWarning::DeadActions(vec!["A".into(), "B".into()]);
        assert!(w.message().contains("dead action"));
        let w = VacuityWarning::AntecedentNeverHolds {
            invariant: "Inv".into(),
        };
        assert!(w.message().contains("antecedent never holds"));
    }
}
