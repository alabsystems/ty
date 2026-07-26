// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![deny(missing_docs)]

//! TY GPU engine: CUDA runtime + device-resident BFS for flat-primary state
//! spaces.
//!
//! This crate provides the *device side* of TY's GPU model-checking tier:
//!
//! - a minimal, `dlopen`-based binding to the CUDA driver API ([`cuda`]) and
//!   NVRTC ([`nvrtc`]) — no build-time CUDA dependency, so `ty` builds and
//!   runs unchanged on machines without a GPU (the engine probes as
//!   unavailable and callers fall back to the CPU path, fail-closed);
//! - a generic level-synchronous BFS driver ([`bfs_driver`]) that keeps the
//!   frontier arenas and the 128-bit fingerprint dedup table resident in GPU
//!   memory and launches one expansion kernel per BFS level;
//! - the CUDA C engine template ([`kernel_template`]) that spec-generated
//!   action/invariant device functions are spliced into before NVRTC
//!   compilation.
//!
//! The caller (the model checker's GPU dispatch) is responsible for producing
//! the per-action CUDA functions — generated from the same trust-ir the CPU
//! JIT compiles — and the enumerated initial states in flat row form.
//!
//! Measured on NVIDIA GB10 (sm_121, 48 SMs): DijkstraMutex N=4 exhaustive
//! search, 33,288,512 distinct states / 146M transitions, state-exact vs the
//! CPU engine, in ~0.6 s — vs 302 s for TLC and >900 s for the CPU
//! interpreter single-threaded (see `docs/perf/gpu-cuda-plan-2026-07-02.md`).

pub mod bfs_driver;
pub mod circuit_exhaust;
pub mod circuit_sim;
pub mod ctl_engine;
mod cuda;
pub mod emit_cuda;
mod kernel_template;
mod nvrtc;

pub use bfs_driver::{
    probe, projected_device_bytes, run_bfs, GpuBfsConfig, GpuBfsOutcome, GpuBfsSpec,
};
pub use circuit_exhaust::{
    exhaustive_sat_cpu, run_circuit_exhaust, CircuitExhaustConfig, CircuitExhaustSpec,
    ExhaustOutcome,
};
pub use circuit_sim::{
    run_circuit_sim, thread_rng_seed, xorshift64, CircuitSimConfig, CircuitSimHit, CircuitSimSpec,
};
pub use ctl_engine::{run_ctl, CtlOp, GpuCtlConfig, GpuCtlOutcome, GpuCtlSpec};
pub use cuda::allocation_headroom_bytes;
pub use emit_cuda::{
    emit_atom_adapters, emit_program, emit_program_with_constraints, GpuProgramSource,
};

/// Why the GPU engine could not run (or run to completion).
#[derive(Debug, Clone, thiserror::Error)]
pub enum GpuError {
    /// CUDA is not usable on this host (missing driver/library/device).
    /// Callers treat this as "engine not offered", never as a check failure.
    #[error("GPU unavailable: {0}")]
    Unavailable(String),
    /// A driver-API call failed after the engine was admitted.
    #[error("CUDA driver error: {0}")]
    Driver(String),
    /// Generated CUDA source failed to compile (NVRTC) or was malformed.
    #[error("GPU codegen error: {0}")]
    Codegen(String),
    /// Host-side byte-layout arithmetic overflowed before a CUDA allocation.
    /// Retrying with a larger engine capacity cannot repair this condition.
    #[error("GPU allocation size overflow: {0}")]
    AllocationOverflow(&'static str),
    /// A fallible host allocation failed while preparing or retrieving GPU
    /// data.  The engine declines instead of letting an infallible `vec!`
    /// allocation abort or invoke the host OOM killer.
    #[error("host allocation failed for {what}: {bytes} bytes")]
    HostAllocationFailed {
        /// Purpose of the requested host buffer.
        what: &'static str,
        /// Requested payload size, saturated to `u64::MAX`.
        bytes: u64,
    },
    /// The process-wide CUDA/unified-memory safety budget was exhausted.
    /// This is intentionally distinct from growable engine capacities so
    /// callers fall back instead of multiplying the requested allocation.
    #[error("GPU memory budget exceeded: needed {needed} bytes, capacity {capacity} bytes")]
    MemoryBudgetExceeded {
        /// Aggregate live bytes the allocation would require.
        needed: u64,
        /// Current safe aggregate budget.
        capacity: u64,
    },
    /// A fail-closed capacity bound was hit; the run must be retried with a
    /// larger allocation or routed back to the CPU engine.
    #[error("GPU capacity exceeded: {what} needed {needed}, capacity {capacity}")]
    CapacityExceeded {
        /// Which allocation overflowed.
        what: &'static str,
        /// Required size (rows / entries).
        needed: u64,
        /// Configured capacity.
        capacity: u64,
    },
}

/// Description of the CUDA device the engine would run on.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Marketing name reported by the driver (e.g. "NVIDIA GB10").
    pub device_name: String,
    /// Compute capability major version.
    pub cc_major: i32,
    /// Compute capability minor version.
    pub cc_minor: i32,
    /// Number of streaming multiprocessors.
    pub multiprocessors: i32,
}
