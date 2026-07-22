// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU bit-parallel EXHAUSTIVE SAT for a combinational AIG over a small free-
//! variable set — a *complete* bounded check, unlike the random-simulation
//! falsification lane ([`crate::circuit_sim`]).
//!
//! Given AND gates (topologically ordered), a set of `V` free input variables,
//! bad-state literals, and constraint literals, this enumerates **all** `2^V`
//! assignments and reports:
//! - [`ExhaustOutcome::Sat`] — some assignment makes `bad ∧ constraints` true
//!   (a witness); or
//! - [`ExhaustOutcome::Unsat`] — no assignment does. Because the enumeration is
//!   provably complete, this is a genuine UNSAT proof (bounded-safety), not the
//!   "no random hit" Unknown the simulation lane returns.
//!
//! The bit-parallel scheme: the low `min(6, V)` free vars are mapped to the 64
//! lanes of a `u64` via the fixed truth-table columns (var `i` = bit `i` of the
//! lane index), so one word covers 64 assignments. The remaining `V-6` vars are
//! enumerated across the launch grid (one thread per outer combination). Thread
//! `n` therefore tests 64 assignments and the whole grid over `n ∈ 0..2^(V-6)`
//! covers all `2^V` — the completeness witness for UNSAT.
//!
//! **Soundness rule:** [`ExhaustOutcome::Unsat`] is returned ONLY after every
//! one of the `2^(V-6)` outer combinations has been launched with no hit. Any
//! cancel/deadline/capacity short-circuit returns [`ExhaustOutcome::Declined`]
//! (Unknown), never Unsat.

use std::ffi::c_void;
use std::time::Instant;

use crate::cuda::{
    check, checked_allocation_bytes, checked_allocation_sum, cuda_api, device_allocation_headroom,
    try_reserve_host, CuDeviceptr, CuFunction, CudaApi, DeviceBuffer,
};
use crate::{probe, GpuError};

/// The largest free-variable count the GPU lane will attempt; above this the
/// `2^V` enumeration is impractical, so callers must fall back to a real SAT
/// solver. `2^30` inner×outer evaluations is ~1e9 — sub-second on GB10-class.
pub const MAX_FREE_VARS: u32 = 30;

/// A combinational AIG plus the free set to enumerate. Latch-free: BMC callers
/// unroll the transition relation into `gates` first.
pub struct CircuitExhaustSpec {
    /// Number of variables (`max_var + 1`). Variable 0 is the constant false.
    pub num_vars: u32,
    /// `(out_var, lit0, lit1)` AND gates in topological order.
    pub gates: Vec<[u32; 3]>,
    /// Free input variables to enumerate, in bit order (low bits first go to
    /// the 64 lanes). `V = free_vars.len()`.
    pub free_vars: Vec<u32>,
    /// Bad-state literals (any true = the property fails).
    pub bad_lits: Vec<u32>,
    /// Constraint literals (all must hold for an assignment to count).
    pub constraint_lits: Vec<u32>,
}

/// Result of an exhaustive check. `Unsat` is a proof; `Declined` is Unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExhaustOutcome {
    /// No assignment satisfies `bad ∧ constraints` — provably safe (complete).
    Unsat,
    /// A satisfying assignment (values for `free_vars`, same order).
    Sat {
        /// The witness assignment (one bool per `free_vars` entry).
        assignment: Vec<bool>,
    },
    /// The check did not complete (too many free vars, cancelled, or driver
    /// error). Never a proof.
    Declined(String),
}

/// Tuning + limits for the GPU lane.
pub struct CircuitExhaustConfig {
    /// Threads per launch (each tests 64 lane assignments).
    pub threads: u32,
    /// Free-variable cap; above it the lane declines. Never above
    /// [`MAX_FREE_VARS`].
    pub max_free_vars: u32,
    /// Optional wall-clock deadline (checked between launches).
    pub deadline: Option<Instant>,
}

impl Default for CircuitExhaustConfig {
    fn default() -> Self {
        CircuitExhaustConfig {
            threads: 65_536,
            max_free_vars: MAX_FREE_VARS,
            deadline: None,
        }
    }
}

/// Evaluate one scalar assignment (`assign[i]` = value of `free_vars[i]`) over
/// the gates and return `bad ∧ constraints`. The CPU reference / oracle used to
/// validate the kernel and to run on non-CUDA hosts.
fn eval_assignment(spec: &CircuitExhaustSpec, assign: &[bool]) -> bool {
    let mut vals = vec![false; spec.num_vars as usize];
    // var 0 = false (constant); free vars from the assignment.
    for (i, &var) in spec.free_vars.iter().enumerate() {
        vals[var as usize] = assign[i];
    }
    let lit = |vals: &[bool], l: u32| -> bool {
        let v = vals[(l >> 1) as usize];
        if l & 1 == 1 {
            !v
        } else {
            v
        }
    };
    for g in &spec.gates {
        vals[g[0] as usize] = lit(&vals, g[1]) & lit(&vals, g[2]);
    }
    let ok = spec.constraint_lits.iter().all(|&l| lit(&vals, l));
    let bad = spec.bad_lits.iter().any(|&l| lit(&vals, l));
    bad && ok
}

/// CPU exhaustive SAT — the oracle and the non-CUDA fallback. Enumerates all
/// `2^V` assignments scalar-ly (intended for small `V`; callers cap it).
///
/// # Errors
///
/// Never errors; returns `Declined` if `V` exceeds `max_free_vars`.
#[must_use]
pub fn exhaustive_sat_cpu(spec: &CircuitExhaustSpec, max_free_vars: u32) -> ExhaustOutcome {
    let Ok(v) = u32::try_from(spec.free_vars.len()) else {
        return ExhaustOutcome::Declined("free-variable count exceeds u32".into());
    };
    if v > max_free_vars || v >= 63 {
        return ExhaustOutcome::Declined(format!("V={v} exceeds cap {max_free_vars}"));
    }
    let total: u64 = 1u64 << v;
    for n in 0..total {
        let assign: Vec<bool> = (0..v).map(|i| (n >> i) & 1 == 1).collect();
        if eval_assignment(spec, &assign) {
            return ExhaustOutcome::Sat { assignment: assign };
        }
    }
    ExhaustOutcome::Unsat
}

const EXHAUST_KERNEL_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;

struct TyExhaustArgs {
  u64* values;        // [num_vars][threads] var-major
  const u32* gates;   // 3*num_gates: out_var, lit0, lit1 (topo order)
  u32 num_gates;
  const u32* free_vars;
  u32 num_free;       // V
  u32 num_inner;      // min(6, V)
  const u32* bad;
  u32 num_bad;
  const u32* constr;
  u32 num_constr;
  u32 threads;
  u64 combo_base;     // global outer-combo index of this launch's first thread
  u64 num_combos;     // total 2^(V-6) (1 if V<=6)
  u64* hit;           // [4]: state(0/1/2), combo, 0, lane_mask
};

static __device__ __forceinline__ u64 ty_elit(const TyExhaustArgs& a, u32 t, u32 lit) {
  u64 v = a.values[(u64)(lit >> 1) * a.threads + t];
  return (lit & 1u) ? ~v : v;
}

// One thread per outer combination; each tests all 64 inner-lane assignments.
extern "C" __global__ void ty_circuit_exhaust(TyExhaustArgs a) {
  u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.threads) return;
  u64 combo = a.combo_base + (u64)t;
  if (combo >= a.num_combos) return;
  if (a.hit[0]) return;
  // Truth-table columns: inner var i takes bit i of the lane index.
  const u64 TT[6] = {
    0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
    0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL
  };
  a.values[t] = 0ULL; // var 0 = constant false
  for (u32 i = 0; i < a.num_free; i++) {
    u64 val;
    if (i < a.num_inner) val = TT[i];
    else val = ((combo >> (i - a.num_inner)) & 1ULL) ? ~0ULL : 0ULL;
    a.values[(u64)a.free_vars[i] * a.threads + t] = val;
  }
  for (u32 g = 0; g < a.num_gates; g++) {
    u32 ov = a.gates[g * 3];
    u64 v = ty_elit(a, t, a.gates[g * 3 + 1]) & ty_elit(a, t, a.gates[g * 3 + 2]);
    a.values[(u64)ov * a.threads + t] = v;
  }
  u64 ok = ~0ULL;
  for (u32 i = 0; i < a.num_constr; i++) ok &= ty_elit(a, t, a.constr[i]);
  u64 bad = 0;
  for (u32 i = 0; i < a.num_bad; i++) bad |= ty_elit(a, t, a.bad[i]);
  bad &= ok;
  if (bad) {
    if (atomicCAS(&a.hit[0], 0ULL, 1ULL) == 0ULL) {
      a.hit[1] = combo;
      a.hit[2] = 0;
      a.hit[3] = bad;
      __threadfence_system();
      atomicExch(&a.hit[0], 2ULL);
    }
  }
}
"#;

#[repr(C)]
struct ExhaustArgs {
    values: CuDeviceptr,
    gates: CuDeviceptr,
    num_gates: u32,
    free_vars: CuDeviceptr,
    num_free: u32,
    num_inner: u32,
    bad: CuDeviceptr,
    num_bad: u32,
    constr: CuDeviceptr,
    num_constr: u32,
    threads: u32,
    combo_base: u64,
    num_combos: u64,
    hit: CuDeviceptr,
}

fn upload<T: Copy>(api: &CudaApi, data: &[T]) -> Result<DeviceBuffer, GpuError> {
    let bytes = std::mem::size_of_val(data).max(1);
    let buf = DeviceBuffer::device(api, bytes)?;
    if !data.is_empty() {
        check(
            api,
            unsafe { (api.cuMemcpyHtoD_v2)(buf.ptr, data.as_ptr().cast::<c_void>(), bytes) },
            "upload exhaust data",
        )?;
    }
    Ok(buf)
}

fn exhaust_static_device_bytes(spec: &CircuitExhaustSpec) -> Result<usize, GpuError> {
    let gates = checked_allocation_bytes(
        "exhaust gate bytes",
        &[spec.gates.len(), std::mem::size_of::<[u32; 3]>()],
    )?
    .max(1);
    let free = checked_allocation_bytes(
        "exhaust free-variable bytes",
        &[spec.free_vars.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    let bad = checked_allocation_bytes(
        "exhaust bad-literal bytes",
        &[spec.bad_lits.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    let constraints = checked_allocation_bytes(
        "exhaust constraint-literal bytes",
        &[spec.constraint_lits.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    checked_allocation_sum(
        "exhaust static device bytes",
        &[gates, free, bad, constraints, 32],
    )
}

/// Right-size the launch scratch to useful work and the process's conservative
/// CUDA budget.  Fewer threads only produce more batches; the outer-combination
/// loop still exhausts the identical assignment space before returning Unsat.
fn plan_exhaust_threads(
    requested: u32,
    num_combos: u64,
    num_vars: u32,
    static_bytes: usize,
    headroom: usize,
) -> Result<u32, GpuError> {
    let per_thread = checked_allocation_bytes(
        "exhaust values bytes per thread",
        &[num_vars as usize, std::mem::size_of::<u64>()],
    )?;
    let scratch_budget = headroom.saturating_sub(static_bytes);
    let by_memory = scratch_budget / per_thread.max(1);
    let useful = num_combos.min(u64::from(u32::MAX)).max(1) as u32;
    let threads = requested
        .max(1)
        .min(useful)
        .min(u32::try_from(by_memory).unwrap_or(u32::MAX));
    if threads == 0 {
        let needed =
            checked_allocation_sum("minimum exhaust device bytes", &[static_bytes, per_thread])?;
        return Err(GpuError::MemoryBudgetExceeded {
            needed: u64::try_from(needed).unwrap_or(u64::MAX),
            capacity: u64::try_from(headroom).unwrap_or(u64::MAX),
        });
    }
    Ok(threads)
}

/// GPU bit-parallel exhaustive SAT. Returns [`ExhaustOutcome::Unsat`] only after
/// the *entire* `2^V` space has been enumerated (the soundness rule).
///
/// # Errors
///
/// [`GpuError`] on driver/nvrtc failure (caller treats as "run on CPU").
pub fn run_circuit_exhaust(
    spec: &CircuitExhaustSpec,
    config: &CircuitExhaustConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<ExhaustOutcome, GpuError> {
    if spec.num_vars == 0 {
        return Err(GpuError::Codegen("empty circuit".into()));
    }
    let v = u32::try_from(spec.free_vars.len())
        .map_err(|_| GpuError::Codegen("free-variable count exceeds u32".into()))?;
    let num_gates = u32::try_from(spec.gates.len())
        .map_err(|_| GpuError::Codegen("gate count exceeds u32".into()))?;
    let num_bad = u32::try_from(spec.bad_lits.len())
        .map_err(|_| GpuError::Codegen("bad-literal count exceeds u32".into()))?;
    let num_constr = u32::try_from(spec.constraint_lits.len())
        .map_err(|_| GpuError::Codegen("constraint count exceeds u32".into()))?;
    if v > config.max_free_vars.min(MAX_FREE_VARS) {
        return Ok(ExhaustOutcome::Declined(format!(
            "V={v} exceeds exhaustive cap {}",
            config.max_free_vars.min(MAX_FREE_VARS)
        )));
    }
    let lit_ok = |lit: u32| (lit >> 1) < spec.num_vars;
    if !spec
        .gates
        .iter()
        .all(|g| g[0] < spec.num_vars && lit_ok(g[1]) && lit_ok(g[2]))
        || !spec.free_vars.iter().all(|&v| v < spec.num_vars)
        || !spec.bad_lits.iter().all(|&l| lit_ok(l))
        || !spec.constraint_lits.iter().all(|&l| lit_ok(l))
    {
        return Err(GpuError::Codegen("literal out of range".into()));
    }

    let api = cuda_api()?;
    let info = probe()?;
    let mut dev = 0;
    check(
        api,
        unsafe { (api.cuDeviceGet)(&mut dev, 0) },
        "cuDeviceGet",
    )?;
    let mut ctx = std::ptr::null_mut();
    check(
        api,
        unsafe { (api.cuDevicePrimaryCtxRetain)(&mut ctx, dev) },
        "cuDevicePrimaryCtxRetain",
    )?;
    check(
        api,
        unsafe { (api.cuCtxSetCurrent)(ctx) },
        "cuCtxSetCurrent",
    )?;

    let ptx = crate::nvrtc::compile_to_ptx(EXHAUST_KERNEL_SRC, info.cc_major, info.cc_minor)?;
    let mut module = std::ptr::null_mut();
    check(
        api,
        unsafe { (api.cuModuleLoadData)(&mut module, ptx.as_ptr().cast::<c_void>()) },
        "cuModuleLoadData",
    )?;
    struct ModuleGuard<'a>(&'a CudaApi, crate::cuda::CuModule);
    impl Drop for ModuleGuard<'_> {
        fn drop(&mut self) {
            unsafe { (self.0.cuModuleUnload)(self.1) };
        }
    }
    let module = ModuleGuard(api, module);
    let name = std::ffi::CString::new("ty_circuit_exhaust").expect("static");
    let mut kern: CuFunction = std::ptr::null_mut();
    if unsafe { (api.cuModuleGetFunction)(&mut kern, module.1, name.as_ptr()) }
        != crate::cuda::CUDA_SUCCESS
    {
        return Err(GpuError::Driver("exhaust kernel symbol missing".into()));
    }

    let num_inner = v.min(6);
    let num_combos: u64 = 1u64 << (v.saturating_sub(6));
    let static_bytes = exhaust_static_device_bytes(spec)?;
    let threads = plan_exhaust_threads(
        config.threads,
        num_combos,
        spec.num_vars,
        static_bytes,
        device_allocation_headroom(api)?,
    )?;
    let values_bytes = checked_allocation_bytes(
        "exhaust values bytes",
        &[
            spec.num_vars as usize,
            threads as usize,
            std::mem::size_of::<u64>(),
        ],
    )?;
    let values = DeviceBuffer::device(api, values_bytes)?;
    let gate_words = checked_allocation_bytes("exhaust gate word count", &[spec.gates.len(), 3])?;
    let mut gates_flat = Vec::new();
    try_reserve_host(&mut gates_flat, gate_words, "exhaust flattened gates")?;
    gates_flat.extend(spec.gates.iter().flatten().copied());
    let gates = upload(api, &gates_flat)?;
    let free_vars = upload(api, &spec.free_vars)?;
    let bad = upload(api, &spec.bad_lits)?;
    let constr = upload(api, &spec.constraint_lits)?;
    let hit = DeviceBuffer::managed(api, 32)?;
    unsafe {
        std::ptr::write_bytes(hit.ptr as *mut u8, 0, 32);
    }

    let block = 128u32;
    let mut combo_base = 0u64;
    while combo_base < num_combos {
        if cancelled() {
            return Ok(ExhaustOutcome::Declined("cancelled".into()));
        }
        if config.deadline.is_some_and(|d| Instant::now() >= d) {
            return Ok(ExhaustOutcome::Declined("deadline".into()));
        }
        let this = (num_combos - combo_base).min(u64::from(threads));
        let grid = u32::try_from((this).div_ceil(u64::from(block)).min(65_535)).expect("bounded");
        let mut args = ExhaustArgs {
            values: values.ptr,
            gates: gates.ptr,
            num_gates,
            free_vars: free_vars.ptr,
            num_free: v,
            num_inner,
            bad: bad.ptr,
            num_bad,
            constr: constr.ptr,
            num_constr,
            threads,
            combo_base,
            num_combos,
            hit: hit.ptr,
        };
        let mut params = [std::ptr::from_mut(&mut args).cast::<c_void>()];
        check(
            api,
            unsafe {
                (api.cuLaunchKernel)(
                    kern,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        check(api, unsafe { (api.cuCtxSynchronize)() }, "cuCtxSynchronize")?;

        let state = unsafe { std::ptr::read_volatile(hit.ptr as *const u64) };
        if state == 2 {
            let words = unsafe { std::slice::from_raw_parts(hit.ptr as *const u64, 4) };
            let combo = words[1];
            let lane = words[3].trailing_zeros() as u64; // any set lane is a witness
                                                         // Reconstruct the assignment: inner var i = bit i of the lane;
                                                         // outer var (i>=num_inner) = bit (i-num_inner) of the combo.
            let assignment: Vec<bool> = (0..v)
                .map(|i| {
                    if i < num_inner {
                        (lane >> i) & 1 == 1
                    } else {
                        (combo >> (i - num_inner)) & 1 == 1
                    }
                })
                .collect();
            debug_assert!(
                eval_assignment(spec, &assignment),
                "GPU witness must replay"
            );
            return Ok(ExhaustOutcome::Sat { assignment });
        }
        combo_base += this;
    }
    // Every outer combination launched with no hit → the full 2^V space is
    // exhausted with no witness → provably UNSAT.
    Ok(ExhaustOutcome::Unsat)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vars: 0=const-false, 1=input a, 2=input b, 3=gate a&b.
    fn and_ab(bad_lit: u32) -> CircuitExhaustSpec {
        CircuitExhaustSpec {
            num_vars: 4,
            gates: vec![[3, 2, 4]], // var3 = lit(a=var1=lit2) & lit(b=var2=lit4)
            free_vars: vec![1, 2],
            bad_lits: vec![bad_lit],
            constraint_lits: vec![],
        }
    }

    #[test]
    fn cpu_sat_when_bad_is_reachable() {
        // bad = (a & b) = var3 = lit 6. SAT at a=b=1.
        let spec = and_ab(6);
        match exhaustive_sat_cpu(&spec, 30) {
            ExhaustOutcome::Sat { assignment } => assert_eq!(assignment, vec![true, true]),
            other => panic!("expected Sat, got {other:?}"),
        }
    }

    #[test]
    fn cpu_unsat_when_bad_is_a_contradiction() {
        // bad = (a & b) AND constraint = !(a & b): no assignment → UNSAT proof.
        let mut spec = and_ab(6);
        spec.constraint_lits = vec![7]; // !var3
        assert_eq!(exhaustive_sat_cpu(&spec, 30), ExhaustOutcome::Unsat);
    }

    #[test]
    fn cpu_unsat_when_bad_never_holds() {
        // bad = var3 & !var3 is not expressible in one lit; instead bad = a&b
        // with constraint !a: forces a=0 so a&b=0 → UNSAT.
        let mut spec = and_ab(6);
        spec.constraint_lits = vec![3]; // !var1 (a = 0)
        assert_eq!(exhaustive_sat_cpu(&spec, 30), ExhaustOutcome::Unsat);
    }

    #[test]
    fn thread_plan_avoids_idle_scratch_for_small_assignment_spaces() {
        assert_eq!(
            plan_exhaust_threads(65_536, 1, 100_000, 32, usize::MAX).unwrap(),
            1
        );
        assert_eq!(
            plan_exhaust_threads(65_536, 8, 100_000, 32, usize::MAX).unwrap(),
            8
        );
    }

    #[test]
    fn thread_plan_scales_to_memory_and_declines_below_one_thread() {
        // 100 vars * 8 bytes = 800 bytes per worker.
        assert_eq!(
            plan_exhaust_threads(65_536, 1_000, 100, 200, 8_200).unwrap(),
            10
        );
        assert!(plan_exhaust_threads(65_536, 1_000, 100, 200, 999).is_err());
    }

    #[test]
    fn static_plan_counts_minimum_upload_buffers() {
        let spec = CircuitExhaustSpec {
            num_vars: 1,
            gates: Vec::new(),
            free_vars: Vec::new(),
            bad_lits: Vec::new(),
            constraint_lits: Vec::new(),
        };
        // Four empty uploads still allocate one byte apiece, plus the 32-byte
        // managed hit record.
        assert_eq!(exhaust_static_device_bytes(&spec).unwrap(), 36);
    }
}
