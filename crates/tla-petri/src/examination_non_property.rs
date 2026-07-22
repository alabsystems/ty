// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Non-property examination verdict and output helpers.

#[path = "examination_non_property/mod.rs"]
mod split;

#[allow(unused_imports)]
pub(crate) use self::split::{
    deadlock_verdict, liveness_verdict, liveness_verdict_with_groups, one_safe_verdict,
    one_safe_verdict_with_nupn, quasi_liveness_verdict, quasi_liveness_verdict_with_groups,
    stable_marking_verdict, state_space_stats, state_space_stats_with_nupn,
};

#[doc(hidden)]
pub use self::split::tier1_crosscheck_hook;
