// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pure-data descriptors for the compiled BFS step pipeline.
//!
//! These are backend-agnostic shapes describing actions, invariants, and
//! compiled function bundles. They live in `tla-jit-abi` so `tla-check` and
//! trust-codegen share one call-boundary contract.
//!
//! Part of #4395.

use crate::{ActionDescriptor, InvariantDescriptor, JitInvariantFn, JitNextStateFn, StateLayout};

/// Configuration for compiling a BFS step function for a specific spec.
#[derive(Debug, Clone)]
pub struct BfsStepSpec {
    /// Number of i64 slots per state (flat representation).
    pub state_len: usize,
    /// Layout of the state variables.
    pub state_layout: StateLayout,
    /// One entry per action instance to unroll.
    pub actions: Vec<ActionDescriptor>,
    /// Invariants to check on each new successor.
    pub invariants: Vec<InvariantDescriptor>,
}

/// A pre-compiled action function plus the descriptor metadata it was built for.
#[derive(Debug, Clone)]
pub struct CompiledActionFn {
    /// Metadata for the specialized action instance.
    pub descriptor: ActionDescriptor,
    /// Native function pointer for the compiled next-state action.
    pub func: JitNextStateFn,
}

impl CompiledActionFn {
    /// Create a compiled action wrapper from a descriptor and function pointer.
    #[must_use]
    pub fn new(descriptor: ActionDescriptor, func: JitNextStateFn) -> Self {
        Self { descriptor, func }
    }
}

/// A pre-compiled invariant function plus the descriptor metadata it was built for.
#[derive(Debug, Clone)]
pub struct CompiledInvariantFn {
    /// Metadata for the invariant.
    pub descriptor: InvariantDescriptor,
    /// Native function pointer for the compiled invariant.
    pub func: JitInvariantFn,
}

impl CompiledInvariantFn {
    /// Create a compiled invariant wrapper from a descriptor and function pointer.
    #[must_use]
    pub fn new(descriptor: InvariantDescriptor, func: JitInvariantFn) -> Self {
        Self { descriptor, func }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_descriptor_clone() {
        let d = ActionDescriptor {
            name: "Send".to_string(),
            action_idx: 1,
            binding_values: vec![0, 1],
            formal_values: vec![1],
            read_vars: vec![0, 1],
            write_vars: vec![2],
            compound_read_vars: Vec::new(),
        };
        let d2 = d.clone();
        assert_eq!(d2.name, "Send");
        assert_eq!(d2.action_idx, 1);
        assert_eq!(d2.binding_values, vec![0, 1]);
        assert_eq!(d2.formal_values, vec![1]);
    }

    #[test]
    fn bfs_step_spec_clone() {
        let spec = BfsStepSpec {
            state_len: 3,
            state_layout: StateLayout::new(vec![]),
            actions: vec![],
            invariants: vec![],
        };
        let spec2 = spec.clone();
        assert_eq!(spec2.state_len, 3);
    }
}
