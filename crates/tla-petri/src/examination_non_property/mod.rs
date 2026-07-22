// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Non-property examination verdict and output helpers.

mod common;
#[cfg(feature = "dd-backend")]
mod dd_fastpath;
mod deadlock_one_safe;
mod liveness;
mod stable_marking;
mod state_space;

pub(crate) use self::deadlock_one_safe::{
    deadlock_verdict, one_safe_verdict, one_safe_verdict_with_nupn,
};
pub(crate) use self::liveness::{
    liveness_verdict, liveness_verdict_with_groups, quasi_liveness_verdict,
    quasi_liveness_verdict_with_groups,
};
pub(crate) use self::stable_marking::stable_marking_verdict;
pub(crate) use self::state_space::{state_space_stats, state_space_stats_with_nupn};

#[doc(hidden)]
pub use self::state_space::tier1_crosscheck_hook;
