// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Typed work-equivalence evidence shared by supremacy claim surfaces.
//!
//! Schema v1 deliberately defines exactly one rule. It proves equivalent work
//! only for exhaustive, successful model-checking outcomes with exact distinct
//! state, raw initial-state generation, raw successor generation, and total
//! generated-state parity. An admitted/internal transition counter is not work
//! evidence. An early violation, deadlock, simulation, randomized workload, or
//! timeout cannot borrow this rule, and workloads with external I/O are
//! excluded before evidence binding.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(super) const WORK_EQUIVALENCE_SCHEMA_VERSION: u64 = 1;
pub(super) const EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID: &str =
    "exhaustive_generated_work_parity_v1";
pub(super) const REQUIRED_VERDICT_HOLDS: &str = "holds";

/// Baseline/compare evidence wire object.
///
/// No prose or extension fields are accepted. Callers must additionally use
/// [`WorkEquivalenceEvidence::is_exact_exhaustive_holds_rule`] before treating
/// a parsed value as qualifying evidence.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkEquivalenceEvidence {
    pub(super) schema_version: u64,
    pub(super) rule_id: String,
}

impl WorkEquivalenceEvidence {
    #[allow(dead_code)] // canonical constructor is also used by sibling-module fixtures
    pub(super) fn exhaustive_holds() -> Self {
        Self {
            schema_version: WORK_EQUIVALENCE_SCHEMA_VERSION,
            rule_id: EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID.to_string(),
        }
    }

    pub(super) fn parse_exact(value: &Value) -> Result<Self> {
        let evidence: Self = serde_json::from_value(value.clone())
            .context("parse typed work-equivalence evidence")?;
        if !evidence.is_exact_exhaustive_holds_rule() {
            bail!(
                "work-equivalence evidence must be exactly schema_version={} rule_id={:?}",
                WORK_EQUIVALENCE_SCHEMA_VERSION,
                EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID
            );
        }
        Ok(evidence)
    }

    pub(super) fn is_exact_exhaustive_holds_rule(&self) -> bool {
        self.schema_version == WORK_EQUIVALENCE_SCHEMA_VERSION
            && self.rule_id == EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID
    }

    pub(super) fn qualifies(&self, verdict: WorkEquivalenceVerdict) -> bool {
        self.is_exact_exhaustive_holds_rule() && verdict == WorkEquivalenceVerdict::Holds
    }
}

/// Semantic outcome to which a work-equivalence rule is being applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WorkEquivalenceVerdict {
    Holds,
    ExpectedViolation,
    Deadlock,
    Simulation,
    RandomizedExternalOperator,
    Timeout,
    Other,
}

/// Exact schema-v1 rule definition embedded in the strict-corpus manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExhaustiveWorkEquivalenceContractV1 {
    kind: String,
    required_verdict: String,
    require_complete_exploration: bool,
    distinct_state_parity: String,
    raw_initial_state_generation_parity: String,
    raw_successor_generation_parity: String,
    total_state_generation_parity: String,
    count_arm: String,
}

impl ExhaustiveWorkEquivalenceContractV1 {
    fn is_exact(&self) -> bool {
        self.kind == "exhaustive_state_space"
            && self.required_verdict == REQUIRED_VERDICT_HOLDS
            && self.require_complete_exploration
            && self.distinct_state_parity == "exact"
            && self.raw_initial_state_generation_parity == "exact"
            && self.raw_successor_generation_parity == "exact"
            && self.total_state_generation_parity == "exact"
            && self.count_arm == "bfs_no_reduction_single_worker"
    }
}

/// Exact schema-v1 outcome dispositions embedded in the strict manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkEquivalenceOutcomeDispositionsV1 {
    expected_violation: String,
    deadlock: String,
    simulation: String,
    randomized_external_operator: String,
    external_io: String,
    timeout: String,
}

impl WorkEquivalenceOutcomeDispositionsV1 {
    fn is_exact(&self) -> bool {
        self.expected_violation == "exclude_unless_predeclared_typed_rule"
            && self.deadlock == "exclude_unless_predeclared_typed_rule"
            && self.simulation == "exclude"
            && self.randomized_external_operator == "exclude"
            && self.external_io == "exclude"
            && self.timeout == "missing_or_stale"
    }
}

/// Strict-corpus work-equivalence policy.
///
/// The map shape is retained because rule IDs are JSON object keys. Validation
/// requires that it contain exactly the sole schema-v1 rule.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkEquivalencePolicyV1 {
    schema_version: u64,
    default_eligible_rule_id: String,
    rules: BTreeMap<String, ExhaustiveWorkEquivalenceContractV1>,
    outcome_dispositions: WorkEquivalenceOutcomeDispositionsV1,
}

impl WorkEquivalencePolicyV1 {
    pub(super) fn validate_exact(&self) -> Result<()> {
        if self.schema_version != WORK_EQUIVALENCE_SCHEMA_VERSION {
            bail!(
                "unsupported work-equivalence policy schema {}; expected {}",
                self.schema_version,
                WORK_EQUIVALENCE_SCHEMA_VERSION
            );
        }
        if self.default_eligible_rule_id != EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID {
            bail!(
                "work-equivalence default rule must be {:?}",
                EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID
            );
        }
        if self.rules.len() != 1 {
            bail!(
                "work-equivalence policy schema {} must define exactly one rule",
                WORK_EQUIVALENCE_SCHEMA_VERSION
            );
        }
        let rule = self
            .rules
            .get(EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID)
            .context("work-equivalence policy is missing its default exhaustive rule")?;
        if !rule.is_exact() {
            bail!(
                "work-equivalence rule {:?} does not match the schema-v1 exhaustive contract",
                EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID
            );
        }
        if !self.outcome_dispositions.is_exact() {
            bail!("work-equivalence outcome dispositions do not match the schema-v1 contract");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn exact_policy_value() -> Value {
        json!({
            "schema_version": 1,
            "default_eligible_rule_id": "exhaustive_generated_work_parity_v1",
            "rules": {
                "exhaustive_generated_work_parity_v1": {
                    "kind": "exhaustive_state_space",
                    "required_verdict": "holds",
                    "require_complete_exploration": true,
                    "distinct_state_parity": "exact",
                    "raw_initial_state_generation_parity": "exact",
                    "raw_successor_generation_parity": "exact",
                    "total_state_generation_parity": "exact",
                    "count_arm": "bfs_no_reduction_single_worker"
                }
            },
            "outcome_dispositions": {
                "expected_violation": "exclude_unless_predeclared_typed_rule",
                "deadlock": "exclude_unless_predeclared_typed_rule",
                "simulation": "exclude",
                "randomized_external_operator": "exclude",
                "external_io": "exclude",
                "timeout": "missing_or_stale"
            }
        })
    }

    #[test]
    fn exact_evidence_round_trips_and_only_qualifies_holds() {
        let evidence = WorkEquivalenceEvidence::parse_exact(&json!({
            "schema_version": 1,
            "rule_id": "exhaustive_generated_work_parity_v1"
        }))
        .unwrap();
        assert_eq!(evidence, WorkEquivalenceEvidence::exhaustive_holds());
        assert!(evidence.qualifies(WorkEquivalenceVerdict::Holds));
        for verdict in [
            WorkEquivalenceVerdict::ExpectedViolation,
            WorkEquivalenceVerdict::Deadlock,
            WorkEquivalenceVerdict::Simulation,
            WorkEquivalenceVerdict::RandomizedExternalOperator,
            WorkEquivalenceVerdict::Timeout,
            WorkEquivalenceVerdict::Other,
        ] {
            assert!(!evidence.qualifies(verdict));
        }
        assert_eq!(
            serde_json::to_value(evidence).unwrap(),
            json!({
                "schema_version": 1,
                "rule_id": "exhaustive_generated_work_parity_v1"
            })
        );
    }

    #[test]
    fn evidence_rejects_legacy_prose_wrong_identity_and_extensions() {
        for value in [
            json!("same enough"),
            json!({
                "schema_version": 2,
                "rule_id": "exhaustive_generated_work_parity_v1"
            }),
            json!({
                "schema_version": 1,
                "rule_id": "whatever"
            }),
            json!({
                "schema_version": 1,
                "rule_id": "exhaustive_generated_work_parity_v1",
                "free_form_exception": "close enough"
            }),
        ] {
            assert!(WorkEquivalenceEvidence::parse_exact(&value).is_err());
        }
    }

    #[test]
    fn manifest_policy_requires_the_exact_schema_v1_contract() {
        let policy: WorkEquivalencePolicyV1 = serde_json::from_value(exact_policy_value()).unwrap();
        policy.validate_exact().unwrap();

        let mut wrong_verdict = exact_policy_value();
        wrong_verdict["rules"][EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID]["required_verdict"] =
            json!("holds_or_expected_violation");
        let policy: WorkEquivalencePolicyV1 = serde_json::from_value(wrong_verdict).unwrap();
        assert!(policy.validate_exact().is_err());

        let mut extension = exact_policy_value();
        extension["rules"][EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID]["exception"] =
            json!("free form");
        assert!(serde_json::from_value::<WorkEquivalencePolicyV1>(extension).is_err());

        let mut ambiguous_legacy_accounting = exact_policy_value();
        let rule = ambiguous_legacy_accounting["rules"][EXHAUSTIVE_GENERATED_WORK_PARITY_RULE_ID]
            .as_object_mut()
            .unwrap();
        rule.remove("raw_initial_state_generation_parity");
        rule.remove("raw_successor_generation_parity");
        rule.remove("total_state_generation_parity");
        rule.insert("generated_transition_parity".to_string(), json!("exact"));
        assert!(
            serde_json::from_value::<WorkEquivalencePolicyV1>(ambiguous_legacy_accounting).is_err()
        );
    }
}
