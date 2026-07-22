// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

mod fixpoint;
mod prefire;
mod structural;

#[cfg(test)]
pub(crate) use fixpoint::reduce_iterative;
#[cfg(test)]
pub(crate) use fixpoint::reduce_iterative_structural;
#[cfg(test)]
pub(crate) use fixpoint::reduce_iterative_structural_deadlock_safe_with_protected;
#[cfg(test)]
pub(crate) use fixpoint::reduce_iterative_structural_with_protected;
#[cfg(test)]
pub(crate) use fixpoint::reduce_iterative_temporal_projection_candidate;
pub(crate) use fixpoint::{
    reduce_iterative_structural_one_safe, reduce_iterative_structural_query_with_protected,
    reduce_iterative_structural_with_mode, reduce_query_guarded,
};
#[cfg(test)]
pub(crate) use prefire::apply_query_guarded_prefire;
#[cfg(test)]
pub(crate) use structural::reduce;
#[cfg(test)]
pub(crate) fn reduce_with_mode(
    net: &crate::petri_net::PetriNet,
    protected_places: &[bool],
    mode: super::model::ReductionMode,
) -> super::model::ReducedNet {
    structural::reduce_with_mode(net, protected_places, mode)
}
