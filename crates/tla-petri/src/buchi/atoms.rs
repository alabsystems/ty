// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! LTL atom context: precomputed atom satisfaction for all system states.

use crate::explorer::FullReachabilityGraph;
use crate::model::PropertyAliases;
use crate::petri_net::PetriNet;
#[cfg(test)]
use crate::petri_net::{PlaceIdx, TransitionIdx};
use crate::property_xml::StatePredicate;
use crate::resolved_predicate::{
    eval_predicate, resolve_predicate_with_aliases, ResolvedPredicate,
};

#[cfg(test)]
use std::collections::HashMap;

/// Preserve the old Buchi atom-resolution entry point as a thin compatibility
/// wrapper over the shared predicate resolver.
pub(crate) fn resolve_atom_with_aliases(
    predicate: &StatePredicate,
    aliases: &PropertyAliases,
) -> ResolvedPredicate {
    resolve_predicate_with_aliases(predicate, aliases)
}

#[cfg(test)]
pub(crate) fn resolve_atom(
    predicate: &StatePredicate,
    place_map: &HashMap<&str, PlaceIdx>,
    trans_map: &HashMap<&str, TransitionIdx>,
) -> ResolvedPredicate {
    crate::resolved_predicate::resolve_predicate(predicate, place_map, trans_map)
}

/// Context for LTL model checking: resolved atoms + system graph.
///
/// Legacy path: used by `product.rs` (pre-built product emptiness).
/// New on-the-fly path evaluates atoms lazily and does not use this struct.
pub(crate) struct LtlContext<'a> {
    pub full: &'a FullReachabilityGraph,
    /// Precomputed: atom_sat[atom_id][state_id] = whether atom holds at state.
    atom_sat: Vec<Vec<bool>>,
}

impl<'a> LtlContext<'a> {
    pub fn new(
        atoms: Vec<ResolvedPredicate>,
        full: &'a FullReachabilityGraph,
        net: &'a PetriNet,
    ) -> Self {
        let n = full.graph.num_states as usize;
        let atom_sat: Vec<Vec<bool>> = atoms
            .iter()
            .map(|atom| {
                let mut scratch = Vec::new();
                (0..n)
                    .map(|s| {
                        full.markings.unpack_into(s, &mut scratch);
                        eval_predicate(atom, &scratch, net)
                    })
                    .collect()
            })
            .collect();
        Self { full, atom_sat }
    }

    pub(super) fn atom_holds(&self, atom_id: usize, state_id: u32) -> bool {
        self.atom_sat[atom_id][state_id as usize]
    }

    /// Number of atoms (for the `TY_LTL_DUMP_LASSO` diagnostic).
    pub(super) fn num_atoms(&self) -> usize {
        self.atom_sat.len()
    }
}
