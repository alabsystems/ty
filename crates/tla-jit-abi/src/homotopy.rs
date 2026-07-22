// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Evidence of topological stability for a TLA+ action.
//! Part of the "Algebraic Geometry" program for TY 1.0.

use serde::{Deserialize, Serialize};

/// Evidence of topological stability for a TLA+ action.
/// An action is stable if its memory access pattern is invariant
/// under symmetric permutations of the named groups.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionHomotopy {
    /// The canonical name of the action.
    pub action_name: String,
    /// If true, the action's memory access pattern is proven to be
    /// invariant under symmetric permutations of the named groups.
    pub is_stable: bool,
    /// The symmetry groups this stability claim applies to.
    pub symmetry_groups: Vec<String>,
    /// The indices of state variables that form a symmetric group.
    #[serde(default)]
    pub symmetric_var_groups: Vec<Vec<u16>>,
}

impl ActionHomotopy {
    /// Create a new stability claim.
    pub fn new(
        action_name: impl Into<String>,
        is_stable: bool,
        symmetry_groups: Vec<String>,
        symmetric_var_groups: Vec<Vec<u16>>,
    ) -> Self {
        Self {
            action_name: action_name.into(),
            is_stable,
            symmetry_groups,
            symmetric_var_groups,
        }
    }
}
