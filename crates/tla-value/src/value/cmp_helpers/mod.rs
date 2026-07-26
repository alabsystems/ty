// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Comparison helper functions for `Value` ordering and equality.

mod cross_type;
mod equality;
mod primitives;
mod same_type;
mod set_like;

pub(super) use cross_type::{
    cmp_cross_type, cmp_tuple2_refs_with_value, cmp_tuple_elements_with_value, eq_cross_type,
    eq_tuple2_refs_with_value, eq_tuple_elements_with_value,
};
pub(super) use equality::eq_same_type;
pub(super) use primitives::{cmp_i64_with_value, type_order};
pub(super) use same_type::cmp_same_type;
