// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bit-parallel random-simulation engine for AIGER-style AND-inverter
//! circuits (the GPU twin of `tla-aiger`'s SAT-free `random_sim` lane).
//!
//! One device thread drives 64 independent random walks (one per bit lane):
//! every variable's value is a `u64` whose bit L is lane L's Boolean, an AND
//! gate is a single `&`, and negation is `~` — the classic bit-parallel
//! circuit simulation, at `threads × 64` concurrent walks.
//!
//! The circuit is passed as DATA (gate triples in topological order, literal
//! arrays), not codegen: one fixed kernel source compiles once per run and
//! every thread streams the same gate list (broadcast reads, L2-resident).
//!
//! Falsification-only contract: the engine reports the first `(thread,
//! attempt, lane_mask)` where some bad literal held under all constraint
//! literals. It NEVER proves safety — the caller treats "no hit" as Unknown.
//! Walks are deterministic in `(base_seed, thread)`: the host re-derives the
//! per-thread RNG stream (`thread_rng_seed` + `xorshift64`) and replays the
//! hit lane on the CPU to build and verify the standard counterexample
//! trace, so no trace storage exists device-side (mirroring the BFS engine's
//! violation policy).
//!
//! Constraint semantics mirror the CPU walker: an attempt whose constraints
//! fail on a lane does not advance that lane's latches (fresh random inputs
//! are drawn next attempt) — implemented as a per-lane blend mask, while the
//! RNG stream advances every attempt on every lane (which is what makes the
//! scalar replay exact).

use std::ffi::c_void;
use std::time::Instant;

use crate::cuda::{
    check, checked_allocation_bytes, checked_allocation_sum, cuda_api, device_allocation_headroom,
    try_reserve_host, try_zeroed_host_vec, CuDeviceptr, CuFunction, CudaApi, DeviceBuffer,
};
use crate::{bfs_driver::probe, GpuError};

/// AND-inverter circuit in `var*2 + negated` literal encoding. Variable 0 is
/// the constant-false variable (so literal 0 = FALSE, literal 1 = TRUE) —
/// the same encoding as `tla-aiger`'s `Lit`.
pub struct CircuitSimSpec {
    /// Number of variables (`max_var + 1`); the values array size.
    pub num_vars: u32,
    /// `(out_var, lit0, lit1)` AND gates in topological order.
    pub gates: Vec<[u32; 3]>,
    /// Primary-input variables (drawn randomly each attempt, in this order —
    /// the replay contract).
    pub input_vars: Vec<u32>,
    /// `(latch_var, next_state_lit, init_value)` per latch.
    pub latches: Vec<(u32, u32, bool)>,
    /// Bad-state literals (any true = bug candidate).
    pub bad_lits: Vec<u32>,
    /// Invariant-constraint literals (all must hold for a state to count).
    pub constraint_lits: Vec<u32>,
}

/// Engine tuning knobs.
pub struct CircuitSimConfig {
    /// Walker threads (each simulates 64 lanes).
    pub threads: u32,
    /// Attempts (input draws) per kernel launch — the host polls
    /// cancellation/deadline between launches.
    pub attempts_per_launch: u32,
    /// Total attempt budget per thread. Also bounds the replayed trace
    /// length, so keep it within what a witness consumer accepts.
    pub max_attempts: u64,
    /// Base RNG seed (portfolio diversity + replay determinism).
    pub base_seed: u64,
    /// Optional wall-clock deadline checked between launches.
    pub deadline: Option<Instant>,
}

impl Default for CircuitSimConfig {
    fn default() -> Self {
        CircuitSimConfig {
            threads: 4096,
            attempts_per_launch: 512,
            max_attempts: 8192,
            base_seed: 0x517C_C1B7_2722_0A95,
            deadline: None,
        }
    }
}

/// A falsification hit: lane `lane_mask.trailing_zeros()` (any set bit) of
/// `thread` saw `bad & constraints-ok` at `attempt` (0-based input draw).
#[derive(Debug, Clone, Copy)]
pub struct CircuitSimHit {
    /// Walker thread whose RNG stream produced the hit.
    pub thread: u32,
    /// 0-based input-draw index at which `bad & ok` held (the replay bound).
    pub attempt: u64,
    /// Per-lane bad-under-constraints mask; any set bit is a witness lane.
    pub lane_mask: u64,
}

/// The per-thread RNG stream seed for `thread` under `base_seed`. The device
/// kernel and the host replay MUST both use exactly this (then step with
/// [`xorshift64`] once per input draw).
#[must_use]
pub fn thread_rng_seed(base_seed: u64, thread: u32) -> u64 {
    // splitmix64-style scramble so adjacent threads decorrelate.
    let mut z = base_seed
        ^ u64::from(thread)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The RNG step shared by the device kernel and the host replay (identical
/// to `tla-aiger`'s scalar walker).
#[must_use]
pub fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

const SIM_KERNEL_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;

struct TySimArgs {
  u64* values;       // [num_vars][threads] var-major (coalesced per var)
  u64* latch_next;   // [num_latches][threads] two-phase advance scratch
  u64* rng;          // [threads] xorshift states (persist across launches)
  const u32* gates;  // 3*num_gates: out_var, lit0, lit1 (topo order)
  u32 num_gates;
  const u32* inputs; // input vars, draw order = replay order
  u32 num_inputs;
  const u32* latch_vars;
  const u32* next_lits;
  u32 num_latches;
  const u32* bad;
  u32 num_bad;
  const u32* constr;
  u32 num_constr;
  u32 threads;
  u32 attempts;      // attempts this launch
  u64 attempt_base;  // global attempt index of this launch's first attempt
  u64* hit;          // [4]: state (0 none / 1 claiming / 2 published), thread, attempt, lane_mask
};

static __device__ __forceinline__ u64 ty_xs64(u64 s) {
  s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}

static __device__ __forceinline__ u64 ty_lit(const TySimArgs& a, u32 t, u32 lit) {
  u64 v = a.values[(u64)(lit >> 1) * a.threads + t];
  return (lit & 1u) ? ~v : v;
}

extern "C" __global__ void ty_circuit_sim(TySimArgs a) {
  u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.threads) return;
  u64 rng = a.rng[t];
  for (u32 k = 0; k < a.attempts; k++) {
    if (a.hit[0]) break; // someone published a hit; stop burning the budget
    // Random inputs: one fresh u64 per input var = 64 lane bits.
    for (u32 i = 0; i < a.num_inputs; i++) {
      rng = ty_xs64(rng);
      a.values[(u64)a.inputs[i] * a.threads + t] = rng;
    }
    // Evaluate AND gates in topological order.
    for (u32 g = 0; g < a.num_gates; g++) {
      u32 ov = a.gates[g * 3];
      u64 v = ty_lit(a, t, a.gates[g * 3 + 1]) & ty_lit(a, t, a.gates[g * 3 + 2]);
      a.values[(u64)ov * a.threads + t] = v;
    }
    // Constraint mask (lanes whose invariant constraints all hold).
    u64 ok = ~0ULL;
    for (u32 i = 0; i < a.num_constr; i++) ok &= ty_lit(a, t, a.constr[i]);
    // Bad mask under constraints.
    u64 bad = 0;
    for (u32 i = 0; i < a.num_bad; i++) bad |= ty_lit(a, t, a.bad[i]);
    bad &= ok;
    if (bad) {
      if (atomicCAS(&a.hit[0], 0ULL, 1ULL) == 0ULL) {
        a.hit[1] = t;
        a.hit[2] = a.attempt_base + k;
        a.hit[3] = bad;
        __threadfence_system();
        atomicExch(&a.hit[0], 2ULL);
      }
      break;
    }
    // Advance latches on constraint-ok lanes only (two-phase: next-state
    // literals all read the pre-advance values).
    for (u32 i = 0; i < a.num_latches; i++)
      a.latch_next[(u64)i * a.threads + t] = ty_lit(a, t, a.next_lits[i]);
    for (u32 i = 0; i < a.num_latches; i++) {
      u64 cur = a.values[(u64)a.latch_vars[i] * a.threads + t];
      u64 nxt = a.latch_next[(u64)i * a.threads + t];
      a.values[(u64)a.latch_vars[i] * a.threads + t] = (nxt & ok) | (cur & ~ok);
    }
  }
  a.rng[t] = rng;
}

// Broadcast per-var initial values (latch init masks; everything else 0).
extern "C" __global__ void ty_circuit_sim_init(u64* values, const u64* var_init,
                                               u32 num_vars, u32 threads) {
  u64 total = (u64)num_vars * threads;
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * (u64)blockDim.x) {
    values[i] = var_init[i / threads];
  }
}
"#;

#[repr(C)]
struct SimArgs {
    values: CuDeviceptr,
    latch_next: CuDeviceptr,
    rng: CuDeviceptr,
    gates: CuDeviceptr,
    num_gates: u32,
    inputs: CuDeviceptr,
    num_inputs: u32,
    latch_vars: CuDeviceptr,
    next_lits: CuDeviceptr,
    num_latches: u32,
    bad: CuDeviceptr,
    num_bad: u32,
    constr: CuDeviceptr,
    num_constr: u32,
    threads: u32,
    attempts: u32,
    attempt_base: u64,
    hit: CuDeviceptr,
}

fn upload<T: Copy>(api: &CudaApi, data: &[T]) -> Result<DeviceBuffer, GpuError> {
    let bytes = std::mem::size_of_val(data).max(1);
    let buf = DeviceBuffer::device(api, bytes)?;
    if !data.is_empty() {
        check(
            api,
            unsafe { (api.cuMemcpyHtoD_v2)(buf.ptr, data.as_ptr().cast::<c_void>(), bytes) },
            "upload circuit data",
        )?;
    }
    Ok(buf)
}

fn sim_static_device_bytes(spec: &CircuitSimSpec) -> Result<usize, GpuError> {
    let var_init = checked_allocation_bytes(
        "simulation init bytes",
        &[spec.num_vars as usize, std::mem::size_of::<u64>()],
    )?
    .max(1);
    let gates = checked_allocation_bytes(
        "simulation gate bytes",
        &[spec.gates.len(), std::mem::size_of::<[u32; 3]>()],
    )?
    .max(1);
    let inputs = checked_allocation_bytes(
        "simulation input bytes",
        &[spec.input_vars.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    let latch_words = checked_allocation_bytes(
        "simulation latch metadata bytes",
        &[spec.latches.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    let bad = checked_allocation_bytes(
        "simulation bad-literal bytes",
        &[spec.bad_lits.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    let constraints = checked_allocation_bytes(
        "simulation constraint-literal bytes",
        &[spec.constraint_lits.len(), std::mem::size_of::<u32>()],
    )?
    .max(1);
    checked_allocation_sum(
        "simulation static device bytes",
        &[
            var_init,
            gates,
            inputs,
            latch_words,
            latch_words,
            bad,
            constraints,
            32,
        ],
    )
}

fn sim_bytes_per_thread(spec: &CircuitSimSpec) -> Result<usize, GpuError> {
    let words = checked_allocation_sum(
        "simulation words per thread",
        &[
            spec.num_vars as usize,
            spec.latches.len().max(1),
            1, // persistent RNG state
        ],
    )?;
    checked_allocation_bytes(
        "simulation bytes per thread",
        &[words, std::mem::size_of::<u64>()],
    )
}

/// Reduce walker count to fit the fail-closed process budget.  This can only
/// reduce falsification coverage; the random-simulation lane never proves
/// safety, so a smaller plan cannot create an unsound verdict.
fn plan_sim_threads(
    requested: u32,
    static_bytes: usize,
    bytes_per_thread: usize,
    headroom: usize,
) -> Result<u32, GpuError> {
    let by_memory = headroom.saturating_sub(static_bytes) / bytes_per_thread.max(1);
    let threads = requested.min(u32::try_from(by_memory).unwrap_or(u32::MAX));
    if threads == 0 {
        let needed = checked_allocation_sum(
            "minimum simulation device bytes",
            &[static_bytes, bytes_per_thread],
        )?;
        return Err(GpuError::MemoryBudgetExceeded {
            needed: u64::try_from(needed).unwrap_or(u64::MAX),
            capacity: u64::try_from(headroom).unwrap_or(u64::MAX),
        });
    }
    Ok(threads)
}

fn launch_1d(
    api: &CudaApi,
    func: CuFunction,
    grid: u32,
    block: u32,
    params: &mut [*mut c_void],
) -> Result<(), GpuError> {
    check(
        api,
        unsafe {
            (api.cuLaunchKernel)(
                func,
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
    check(api, unsafe { (api.cuCtxSynchronize)() }, "cuCtxSynchronize")
}

/// Run bit-parallel random simulation until a hit, the attempt budget, the
/// deadline, or `cancelled` (polled between launches).
///
/// `Ok(None)` = no bug found within budget (Unknown — NEVER a safety proof).
///
/// # Errors
///
/// [`GpuError`] on driver/nvrtc failure or malformed spec — callers treat any
/// error as "lane unavailable" and fall back to the CPU walker.
pub fn run_circuit_sim(
    spec: &CircuitSimSpec,
    config: &CircuitSimConfig,
    cancelled: &dyn Fn() -> bool,
) -> Result<Option<CircuitSimHit>, GpuError> {
    if spec.num_vars == 0 || config.threads == 0 || config.attempts_per_launch == 0 {
        return Err(GpuError::Codegen(
            "empty circuit, zero threads, or zero attempts per launch".into(),
        ));
    }
    let lit_ok = |lit: u32| (lit >> 1) < spec.num_vars;
    if !spec
        .gates
        .iter()
        .all(|g| g[0] < spec.num_vars && lit_ok(g[1]) && lit_ok(g[2]))
        || !spec.input_vars.iter().all(|&v| v < spec.num_vars)
        || !spec
            .latches
            .iter()
            .all(|&(v, n, _)| v < spec.num_vars && lit_ok(n))
        || !spec.bad_lits.iter().all(|&l| lit_ok(l))
        || !spec.constraint_lits.iter().all(|&l| lit_ok(l))
    {
        return Err(GpuError::Codegen("literal out of range".into()));
    }
    let num_gates = u32::try_from(spec.gates.len())
        .map_err(|_| GpuError::Codegen("gate count exceeds u32".into()))?;
    let num_inputs = u32::try_from(spec.input_vars.len())
        .map_err(|_| GpuError::Codegen("input count exceeds u32".into()))?;
    let num_latches = u32::try_from(spec.latches.len())
        .map_err(|_| GpuError::Codegen("latch count exceeds u32".into()))?;
    let num_bad = u32::try_from(spec.bad_lits.len())
        .map_err(|_| GpuError::Codegen("bad-literal count exceeds u32".into()))?;
    let num_constr = u32::try_from(spec.constraint_lits.len())
        .map_err(|_| GpuError::Codegen("constraint count exceeds u32".into()))?;

    let api = cuda_api()?;
    let info = probe()?;
    // Reuse the BFS driver's context activation path by allocating through
    // the same primary context.
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

    let ptx = crate::nvrtc::compile_to_ptx(SIM_KERNEL_SRC, info.cc_major, info.cc_minor)?;
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
    let sim_name = std::ffi::CString::new("ty_circuit_sim").expect("static");
    let init_name = std::ffi::CString::new("ty_circuit_sim_init").expect("static");
    let mut sim = std::ptr::null_mut();
    let mut init = std::ptr::null_mut();
    let rc1 = unsafe { (api.cuModuleGetFunction)(&mut sim, module.1, sim_name.as_ptr()) };
    let rc2 = unsafe { (api.cuModuleGetFunction)(&mut init, module.1, init_name.as_ptr()) };
    if rc1 != crate::cuda::CUDA_SUCCESS || rc2 != crate::cuda::CUDA_SUCCESS {
        return Err(GpuError::Driver(
            "circuit-sim kernel symbols missing".into(),
        ));
    }

    let static_bytes = sim_static_device_bytes(spec)?;
    let bytes_per_thread = sim_bytes_per_thread(spec)?;
    let threads = plan_sim_threads(
        config.threads,
        static_bytes,
        bytes_per_thread,
        device_allocation_headroom(api)?,
    )?;
    let values_bytes = checked_allocation_bytes(
        "simulation values bytes",
        &[
            spec.num_vars as usize,
            threads as usize,
            std::mem::size_of::<u64>(),
        ],
    )?;
    let latch_next_bytes = checked_allocation_bytes(
        "simulation next-latch bytes",
        &[
            spec.latches.len().max(1),
            threads as usize,
            std::mem::size_of::<u64>(),
        ],
    )?;
    let values = DeviceBuffer::device(api, values_bytes)?;
    let latch_next = DeviceBuffer::device(api, latch_next_bytes)?;

    // Per-var broadcast init: latches get their init mask, everything else 0
    // (var 0 stays 0 = constant false; inputs are freshly drawn per attempt).
    let mut var_init = try_zeroed_host_vec(spec.num_vars as usize, "simulation host init words")?;
    for &(latch_var, _, init_value) in &spec.latches {
        var_init[latch_var as usize] = if init_value { u64::MAX } else { 0 };
    }
    let var_init_buf = upload(api, &var_init)?;

    let mut rng_states = Vec::new();
    try_reserve_host(
        &mut rng_states,
        threads as usize,
        "simulation host RNG words",
    )?;
    rng_states.extend((0..threads).map(|t| thread_rng_seed(config.base_seed, t)));
    let rng = upload(api, &rng_states)?;

    let gate_words =
        checked_allocation_bytes("simulation gate word count", &[spec.gates.len(), 3])?;
    let mut gates_flat = Vec::new();
    try_reserve_host(&mut gates_flat, gate_words, "simulation flattened gates")?;
    gates_flat.extend(spec.gates.iter().flatten().copied());
    let gates = upload(api, &gates_flat)?;
    let inputs = upload(api, &spec.input_vars)?;
    let mut latch_vars = Vec::new();
    let mut next_lits = Vec::new();
    try_reserve_host(
        &mut latch_vars,
        spec.latches.len(),
        "simulation latch variables",
    )?;
    try_reserve_host(
        &mut next_lits,
        spec.latches.len(),
        "simulation next literals",
    )?;
    latch_vars.extend(spec.latches.iter().map(|&(v, _, _)| v));
    next_lits.extend(spec.latches.iter().map(|&(_, n, _)| n));
    let latch_vars_buf = upload(api, &latch_vars)?;
    let next_lits_buf = upload(api, &next_lits)?;
    let bad = upload(api, &spec.bad_lits)?;
    let constr = upload(api, &spec.constraint_lits)?;
    let hit = DeviceBuffer::managed(api, 32)?;
    unsafe {
        std::ptr::write_bytes(hit.ptr as *mut u8, 0, 32);
    }

    // Broadcast initial values.
    {
        let mut p_values = values.ptr;
        let mut p_init = var_init_buf.ptr;
        let mut p_vars = spec.num_vars;
        let mut p_threads = threads;
        let mut params: [*mut c_void; 4] = [
            std::ptr::from_mut(&mut p_values).cast(),
            std::ptr::from_mut(&mut p_init).cast(),
            std::ptr::from_mut(&mut p_vars).cast(),
            std::ptr::from_mut(&mut p_threads).cast(),
        ];
        let total = u64::from(spec.num_vars) * u64::from(threads);
        let grid = u32::try_from(total.div_ceil(256).min(65_535)).expect("bounded");
        launch_1d(api, init, grid, 256, &mut params)?;
    }

    let block = 128u32;
    let grid = threads.div_ceil(block).min(65_535);
    let mut attempt_base = 0u64;
    while attempt_base < config.max_attempts {
        if cancelled() {
            return Ok(None);
        }
        if let Some(deadline) = config.deadline {
            if Instant::now() >= deadline {
                return Ok(None);
            }
        }
        let attempts = u32::try_from(
            (config.max_attempts - attempt_base).min(u64::from(config.attempts_per_launch)),
        )
        .expect("bounded by attempts_per_launch");
        let mut args = SimArgs {
            values: values.ptr,
            latch_next: latch_next.ptr,
            rng: rng.ptr,
            gates: gates.ptr,
            num_gates,
            inputs: inputs.ptr,
            num_inputs,
            latch_vars: latch_vars_buf.ptr,
            next_lits: next_lits_buf.ptr,
            num_latches,
            bad: bad.ptr,
            num_bad,
            constr: constr.ptr,
            num_constr,
            threads,
            attempts,
            attempt_base,
            hit: hit.ptr,
        };
        let mut params = [std::ptr::from_mut(&mut args).cast::<c_void>()];
        launch_1d(api, sim, grid, block, &mut params)?;

        let state = unsafe { std::ptr::read_volatile(hit.ptr as *const u64) };
        if state == 2 {
            let words = unsafe { std::slice::from_raw_parts(hit.ptr as *const u64, 4) };
            return Ok(Some(CircuitSimHit {
                thread: u32::try_from(words[1]).unwrap_or(0),
                attempt: words[2],
                lane_mask: words[3],
            }));
        }
        attempt_base += u64::from(attempts);
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_spec(num_vars: u32, latches: usize) -> CircuitSimSpec {
        CircuitSimSpec {
            num_vars,
            gates: Vec::new(),
            input_vars: Vec::new(),
            latches: (0..latches).map(|_| (0, 0, false)).collect(),
            bad_lits: Vec::new(),
            constraint_lits: Vec::new(),
        }
    }

    #[test]
    fn simulation_thread_plan_scales_to_budget() {
        let spec = empty_spec(100, 10);
        let per_thread = sim_bytes_per_thread(&spec).unwrap();
        assert_eq!(per_thread, 111 * 8);
        assert_eq!(
            plan_sim_threads(4096, 200, per_thread, 200 + per_thread * 7).unwrap(),
            7
        );
    }

    #[test]
    fn simulation_thread_plan_declines_below_one_worker() {
        let spec = empty_spec(100, 10);
        let per_thread = sim_bytes_per_thread(&spec).unwrap();
        assert!(plan_sim_threads(4096, 200, per_thread, 200 + per_thread - 1).is_err());
    }

    #[test]
    fn simulation_static_plan_counts_empty_upload_buffers() {
        let spec = empty_spec(1, 0);
        // var_init=8; six empty metadata uploads=1 byte each; hit=32.
        assert_eq!(sim_static_device_bytes(&spec).unwrap(), 46);
    }
}
