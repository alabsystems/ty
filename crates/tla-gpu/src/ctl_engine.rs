// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Retained-graph CTL fixpoint engine: deep temporal operators on the GPU.
//!
//! The level-sync BFS engine ([`crate::bfs_driver`]) discards each frontier
//! once expanded, which is exactly right for reachability-shaped questions
//! but useless for nested fixpoints. This engine enumerates the same
//! reachable set while RETAINING it:
//!
//! - a monotone **state arena** holds every distinct state row at a
//!   permanent index (arena order = discovery order; the deduped initial
//!   states occupy the front indices `0..deduped_init`, and the per-formula
//!   verdict is the conjunction of the formula's truth over exactly those),
//! - the fingerprint table gains an **index lane** mapping a state's
//!   fingerprint slot to its arena index (published together with the
//!   fingerprint), so successor lookups during fixpoints resolve to indices,
//! - a per-state **deadlock byte** records "no action fired here" (the
//!   maximal-path semantics carrier for `EG`).
//!
//! Formula evaluation is bottom-up over [`CtlOp`] trees with one device
//! byte-mask per operand set (1 byte per retained state):
//!
//! - `Atom(k)` evaluates predicate `k` (an invariant-ABI device function —
//!   the same predicate compiler the reachability lane uses) over the arena;
//! - `Not/And/Or` are elementwise kernels;
//! - `EX Z` is one expansion pass over the arena: `out[s] = ∃ action fired:
//!   Z[succ_idx]`;
//! - least fixpoints (`EF`, `EU`) iterate `Z ∨= step(Z)` to convergence;
//! - the greatest fixpoint (`EG`) iterates `Z ∧= φ ∧ (deadlock ∨ EX Z)` —
//!   a deadlocked φ-state is a maximal path that stays in φ, matching the
//!   finite-maximal-path CTL semantics the CPU checker implements;
//! - universal operators are derived by duality on the host
//!   (`AG = ¬EF¬`, `AF = ¬EG¬`, `A[φ U ψ] = ¬(E[¬ψ U ¬φ∧¬ψ] ∨ EG¬ψ)`,
//!   `AX φ = ¬EX¬φ ∧ (¬deadlock ∨ ???)` — see [`CtlOp`] docs; `EX`/`AX` on
//!   deadlock states follow the standard convention: `EX` is false, `AX` is
//!   true).
//!
//! One retained set serves MANY formulas ([`run_ctl`] takes a batch), which
//! is what the Liveness examination needs (`AG(EF(fireable(t)))` per
//! transition over the same reachable set).
//!
//! Fail-closed like every sibling engine: capacity/driver/nvrtc errors
//! surface as [`GpuError`] and callers fall back to the CPU checker.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use crate::cuda::{
    check, checked_allocation_bytes, checked_allocation_sum, checked_allocation_usize,
    checked_power_of_two_u64, cuda_api, try_reserve_host, try_zeroed_host_vec, CuDeviceptr,
    CuFunction, CudaApi, DeviceBuffer,
};
use crate::kernel_template::assemble_engine_source;
use crate::{bfs_driver::probe, GpuError};

/// A CTL formula over atom indices, in the exact shape the device evaluator
/// executes. Universal operators are pre-lowered by the caller or via
/// [`CtlOp`] constructors here; the engine core evaluates
/// `Atom/Not/And/Or/EX/EF/EU/EG` natively.
#[derive(Debug, Clone)]
pub enum CtlOp {
    /// Truth of predicate `k` (index into the spec's atom list) at a state.
    Atom(usize),
    /// Constant truth.
    True,
    /// Negation.
    Not(Box<CtlOp>),
    /// Conjunction (empty = true).
    And(Vec<CtlOp>),
    /// Disjunction (empty = false).
    Or(Vec<CtlOp>),
    /// Some successor satisfies the operand (false at deadlock states).
    EX(Box<CtlOp>),
    /// Some path reaches the operand.
    EF(Box<CtlOp>),
    /// `E[φ U ψ]`.
    EU(Box<CtlOp>, Box<CtlOp>),
    /// Some MAXIMAL path stays in the operand (a deadlocked operand-state
    /// qualifies).
    EG(Box<CtlOp>),
    /// `E(GF a)` — some path along which the operand holds infinitely often
    /// (the fair-cycle / Büchi-non-emptiness fixpoint, evaluated via the
    /// Emerson–Lei `νZ.μY.(a ∧ EXˢ Z) ∨ EXˢ Y` with a deadlock-stutter EXˢ).
    /// This is the deep-LTL persistence/recurrence carrier: `A(FG p)` is FALSE
    /// iff `EGF(¬p)` holds at some initial state (see [`CtlOp::afg`]).
    EGF(Box<CtlOp>),
}

impl CtlOp {
    /// `AX φ = ¬EX¬φ` (true at deadlock states — no successor violates φ).
    #[must_use]
    pub fn ax(inner: CtlOp) -> CtlOp {
        CtlOp::Not(Box::new(CtlOp::EX(Box::new(CtlOp::Not(Box::new(inner))))))
    }
    /// `AG φ = ¬EF¬φ`.
    #[must_use]
    pub fn ag(inner: CtlOp) -> CtlOp {
        CtlOp::Not(Box::new(CtlOp::EF(Box::new(CtlOp::Not(Box::new(inner))))))
    }
    /// `AF φ = ¬EG¬φ` (maximal-path semantics on both sides).
    #[must_use]
    pub fn af(inner: CtlOp) -> CtlOp {
        CtlOp::Not(Box::new(CtlOp::EG(Box::new(CtlOp::Not(Box::new(inner))))))
    }
    /// `A(FG p) = ¬EGF(¬p)` — the persistence dual (no path visits `¬p`
    /// infinitely often, i.e. every path eventually stays in `p`).
    #[must_use]
    pub fn afg(inner: CtlOp) -> CtlOp {
        CtlOp::Not(Box::new(CtlOp::EGF(Box::new(CtlOp::Not(Box::new(inner))))))
    }
    /// `A[φ U ψ] = ¬( E[¬ψ U (¬φ ∧ ¬ψ)] ∨ EG ¬ψ )`.
    #[must_use]
    pub fn au(phi: CtlOp, psi: CtlOp) -> CtlOp {
        let not_psi = || CtlOp::Not(Box::new(psi.clone()));
        let not_phi = CtlOp::Not(Box::new(phi));
        CtlOp::Not(Box::new(CtlOp::Or(vec![
            CtlOp::EU(
                Box::new(not_psi()),
                Box::new(CtlOp::And(vec![not_phi, not_psi()])),
            ),
            CtlOp::EG(Box::new(not_psi())),
        ])))
    }
}

/// Spec for a retained-graph CTL run: the same action source contract as
/// [`crate::GpuBfsSpec`], plus atom predicates emitted as invariant-ABI
/// functions named `ty_gpu_atom_<k>` (the caller emits them through
/// [`crate::emit_program`]-style plumbing — see `atoms_src`).
pub struct GpuCtlSpec {
    /// i64 slots per state.
    pub slots: usize,
    /// Number of `ty_gpu_action_<k>` device functions in `actions_src`.
    pub action_count: usize,
    /// CUDA C source defining the per-action functions and
    /// `ty_gpu_invariants_ok` (the CTL engine installs no invariants, so the
    /// standard emitter's always-1 combined check is expected).
    pub actions_src: String,
    /// CUDA C source defining `static __device__ int ty_gpu_atom_<k>(const
    /// long long* s)` for `k in 0..atom_count`, returning 0/1 (negative =
    /// runtime fault, fail-closed).
    pub atoms_src: String,
    /// Number of atom predicates.
    pub atom_count: usize,
    /// Row-major initial states (deduped by the engine; they occupy arena
    /// indices `0..distinct_init`).
    pub init_rows: Vec<i64>,
}

/// Tuning knobs (mirrors [`crate::GpuBfsConfig`] where meaningful).
pub struct GpuCtlConfig {
    /// log2 of fingerprint-table slots.
    pub table_bits: u32,
    /// Retained-arena capacity in states (also the frontier bound). The
    /// fail-closed distinct-state cap.
    pub max_states: u64,
    /// CUDA block size.
    pub block: u32,
    /// Abort past this fingerprint-table load factor.
    pub max_load_factor: f64,
    /// Wall-clock deadline checked between kernel launches (fail-closed
    /// decline, mirroring the CPU checker's budget behavior).
    pub deadline: Option<Instant>,
}

impl Default for GpuCtlConfig {
    fn default() -> Self {
        GpuCtlConfig {
            table_bits: 22,
            max_states: 1 << 21,
            block: 256,
            max_load_factor: 0.7,
            deadline: None,
        }
    }
}

/// Result of a retained-graph CTL run.
#[derive(Debug)]
pub struct GpuCtlOutcome {
    /// Distinct reachable states (the retained-arena length).
    pub distinct_states: u64,
    /// Per-formula truth AT EVERY deduped initial state (conjunction over
    /// the initial set — MCC nets have a single initial marking).
    pub verdicts: Vec<bool>,
    /// Wall time of enumeration + all fixpoints (excludes nvrtc).
    pub wall: Duration,
    /// nvrtc + module-load time.
    pub compile_wall: Duration,
}

const CTL_DRIVER: &str = r#"
// ---- retained-graph CTL engine (appended to the standard engine source) ----

struct TyGpuCtlArgs {
  i64* arena;            // retained state rows, row-major
  u64 arena_len;         // states discovered so far
  u64 arena_cap;
  u64* fp_lo;
  u64* fp_hi;
  u64* fp_idx;           // arena index per fingerprint slot (published after hi)
  u64 fp_mask;
  u64* next_count;       // arena append cursor (monotone across levels)
  u64 level_base;        // first arena index of the current frontier level
  u64 level_len;         // frontier length (arena[level_base..level_base+level_len])
  unsigned char* deadlock; // per-state "no action fired" byte (1 = deadlock)
  u64* error_flag;
  // Fixpoint operands (byte masks over arena indices):
  unsigned char* mask_out;
  const unsigned char* mask_a;
  const unsigned char* mask_b;
  int atom_index;
  int op_code;           // elementwise op: 0=copy,1=not,2=and,3=or,4=or_and (out |= a&b)
  u64* changed;          // nonzero when a fixpoint pass changed anything
  // Materialized transition relation (parent arena idx -> successor arena idx),
  // recorded once after enumeration so fixpoint EX passes gather over a flat
  // edge array instead of re-firing every action + re-probing the fp table.
  u64* edge_p;           // parent arena index per edge
  u64* edge_c;           // successor arena index per edge
  u64* edge_count;       // append cursor / total edge count
  u64 edge_cap;          // capacity of edge_p/edge_c (fail-closed if exceeded)
};

// Claim/publish insert that also records the arena index. Returns the
// state's arena index in *out_idx and 1 if newly inserted (this thread owns
// the arena append), 0 for an existing copy, and -1 after a full-table probe.
static __device__ __forceinline__ int ty_gpu_fp_insert_idx(TyGpuCtlArgs& a, u64 hi, u64 lo,
                                                           u64* out_idx) {
  u64 lokey = lo | 1ULL;
  u64 slot = (hi ^ lo) & a.fp_mask;
  for (u64 probe = 0; probe <= a.fp_mask; probe++) {
    u64 prev = atomicCAS(&a.fp_lo[slot], 0ULL, lokey);
    if (prev == 0ULL) {
      u64 idx = atomicAdd(a.next_count, 1ULL);
      if (idx < a.arena_cap) {
        a.fp_idx[slot] = idx;
      }
      __threadfence();
      atomicExch(&a.fp_hi[slot], hi);
      *out_idx = idx;
      return 1;
    }
    if (prev == lokey) {
      u64 h;
      do { h = atomicAdd(&a.fp_hi[slot], 0ULL); } while (h == 0ULL);
      if (h == hi) { *out_idx = a.fp_idx[slot]; return 0; }
    }
    slot = (slot + 1) & a.fp_mask;
  }
  return -1;
}

// Lookup after enumeration has synchronized. Returns 0 after either the first
// empty slot or a complete table probe, so a corrupt/missing successor cannot
// spin a fixpoint kernel forever.
static __device__ __forceinline__ int ty_gpu_fp_lookup_idx(const TyGpuCtlArgs& a,
                                                           u64 hi, u64 lo, u64* out_idx) {
  u64 lokey = lo | 1ULL;
  u64 slot = (hi ^ lo) & a.fp_mask;
  for (u64 probe = 0; probe <= a.fp_mask; probe++) {
    u64 plo = a.fp_lo[slot];
    if (plo == 0ULL) return 0;
    if (plo == lokey && a.fp_hi[slot] == hi) {
      *out_idx = a.fp_idx[slot];
      return 1;
    }
    slot = (slot + 1) & a.fp_mask;
  }
  return 0;
}

// Expand the current frontier level, appending new states to the arena and
// recording per-parent deadlock bytes.
extern "C" __global__ void ty_gpu_ctl_expand(TyGpuCtlArgs a) {
  const u64 pairs = a.level_len * (u64)ACTION_COUNT;
  for (u64 gi = blockIdx.x * (u64)blockDim.x + threadIdx.x; gi < pairs;
       gi += gridDim.x * (u64)blockDim.x) {
    const u64 parent = a.level_base + gi / (u64)ACTION_COUNT;
    const int action = (int)(gi % (u64)ACTION_COUNT);
    const i64* s = a.arena + parent * SLOTS;
    i64 t[SLOTS];
    #pragma unroll
    for (int i = 0; i < SLOTS; i++) t[i] = s[i];
    int fired = ty_gpu_dispatch(action, s, t);
    if (fired < 0) { atomicExch(a.error_flag, (u64)(-fired)); return; }
    if (fired == 0) continue;
    a.deadlock[parent] = 0; // benign race; initialized to 1 per level below
    // Count this transition. Every (parent,action) pair is expanded exactly
    // once across all levels, so after Phase A `edge_count` is the exact total
    // edge count used to size the materialized relation (see record_edges).
    atomicAdd(a.edge_count, 1ULL);
    u64 hi, lo;
    ty_gpu_fp128(t, &hi, &lo);
    u64 idx;
    int inserted = ty_gpu_fp_insert_idx(a, hi, lo, &idx);
    if (inserted < 0) { atomicExch(a.error_flag, 92ULL); return; }
    if (inserted > 0) {
      if (idx < a.arena_cap) {
        i64* dst = a.arena + idx * SLOTS;
        #pragma unroll
        for (int i = 0; i < SLOTS; i++) dst[i] = t[i];
      }
      // idx >= cap: host sees next_count > cap and aborts fail-closed.
    }
  }
}

// Seed: dedup the initial rows into arena[0..], single-threaded (init sets
// are tiny). Frontier for level 0 = arena[0..next_count].
extern "C" __global__ void ty_gpu_ctl_seed(TyGpuCtlArgs a) {
  for (u64 r = 0; r < a.level_len; r++) {
    const i64* row = a.arena + a.arena_cap * SLOTS + r * SLOTS; // staging area
    u64 hi, lo;
    ty_gpu_fp128(row, &hi, &lo);
    u64 idx;
    int inserted = ty_gpu_fp_insert_idx(a, hi, lo, &idx);
    if (inserted < 0) { atomicExch(a.error_flag, 92ULL); return; }
    if (inserted > 0) {
      i64* dst = a.arena + idx * SLOTS;
      for (int i = 0; i < SLOTS; i++) dst[i] = row[i];
    }
  }
}

// Mark a level's deadlock bytes as 1 before expansion clears them.
extern "C" __global__ void ty_gpu_ctl_mark_deadlock(TyGpuCtlArgs a) {
  for (u64 gi = blockIdx.x * (u64)blockDim.x + threadIdx.x; gi < a.level_len;
       gi += gridDim.x * (u64)blockDim.x) {
    a.deadlock[a.level_base + gi] = 1;
  }
}

// Atom evaluation over the whole arena: mask_out[s] = ty_gpu_atoms(atom_index, s).
extern "C" __global__ void ty_gpu_ctl_eval_atom(TyGpuCtlArgs a) {
  for (u64 s = blockIdx.x * (u64)blockDim.x + threadIdx.x; s < a.arena_len;
       s += gridDim.x * (u64)blockDim.x) {
    int v = ty_gpu_atoms(a.atom_index, a.arena + s * SLOTS);
    if (v < 0) { atomicExch(a.error_flag, (u64)(-v)); return; }
    a.mask_out[s] = (unsigned char)(v != 0);
  }
}

// Elementwise mask ops (op_code: 0=copy a,1=not a,2=a&b,3=a|b,4=out|=a&b).
extern "C" __global__ void ty_gpu_ctl_elementwise(TyGpuCtlArgs a) {
  for (u64 s = blockIdx.x * (u64)blockDim.x + threadIdx.x; s < a.arena_len;
       s += gridDim.x * (u64)blockDim.x) {
    unsigned char r;
    switch (a.op_code) {
      case 0: r = a.mask_a[s]; break;
      case 1: r = !a.mask_a[s]; break;
      case 2: r = a.mask_a[s] & a.mask_b[s]; break;
      case 3: r = a.mask_a[s] | a.mask_b[s]; break;
      default: r = a.mask_out[s] | (a.mask_a[s] & a.mask_b[s]); break;
    }
    if (a.mask_out[s] != r) { a.mask_out[s] = r; atomicExch(a.changed, 1ULL); }
  }
}

// EX pass: mask_out[s] = 1 iff some fired action leads to a mask_a-state.
// (mask_out must be zeroed by the host first; deadlock states stay 0.)
extern "C" __global__ void ty_gpu_ctl_ex(TyGpuCtlArgs a) {
  const u64 pairs = a.arena_len * (u64)ACTION_COUNT;
  for (u64 gi = blockIdx.x * (u64)blockDim.x + threadIdx.x; gi < pairs;
       gi += gridDim.x * (u64)blockDim.x) {
    const u64 parent = gi / (u64)ACTION_COUNT;
    if (a.mask_out[parent]) continue; // already known
    const int action = (int)(gi % (u64)ACTION_COUNT);
    const i64* s = a.arena + parent * SLOTS;
    i64 t[SLOTS];
    #pragma unroll
    for (int i = 0; i < SLOTS; i++) t[i] = s[i];
    int fired = ty_gpu_dispatch(action, s, t);
    if (fired < 0) { atomicExch(a.error_flag, (u64)(-fired)); return; }
    if (fired == 0) continue;
    u64 hi, lo;
    ty_gpu_fp128(t, &hi, &lo);
    // Lookup (state must exist: the arena is the full reachable set).
    u64 succ;
    if (!ty_gpu_fp_lookup_idx(a, hi, lo, &succ)) {
      atomicExch(a.error_flag, 90ULL); return; // missing successor
    }
    if (a.mask_a[succ]) {
      if (!a.mask_out[parent]) { a.mask_out[parent] = 1; atomicExch(a.changed, 1ULL); }
    }
  }
}

// Materialize the transition relation ONCE: fire every action from every
// arena state, resolve the successor's arena index (exactly as ty_gpu_ctl_ex
// does), and append (parent, succ) to the edge arrays. edge_count was
// pre-counted during Phase A expansion, so the append cursor never exceeds it;
// edge_cap is a belt-and-braces fail-closed bound.
extern "C" __global__ void ty_gpu_ctl_record_edges(TyGpuCtlArgs a) {
  const u64 pairs = a.arena_len * (u64)ACTION_COUNT;
  for (u64 gi = blockIdx.x * (u64)blockDim.x + threadIdx.x; gi < pairs;
       gi += gridDim.x * (u64)blockDim.x) {
    const u64 parent = gi / (u64)ACTION_COUNT;
    const int action = (int)(gi % (u64)ACTION_COUNT);
    const i64* s = a.arena + parent * SLOTS;
    i64 t[SLOTS];
    #pragma unroll
    for (int i = 0; i < SLOTS; i++) t[i] = s[i];
    int fired = ty_gpu_dispatch(action, s, t);
    if (fired < 0) { atomicExch(a.error_flag, (u64)(-fired)); return; }
    if (fired == 0) continue;
    u64 hi, lo;
    ty_gpu_fp128(t, &hi, &lo);
    u64 succ;
    if (!ty_gpu_fp_lookup_idx(a, hi, lo, &succ)) {
      atomicExch(a.error_flag, 90ULL); return; // missing successor
    }
    u64 pos = atomicAdd(a.edge_count, 1ULL);
    if (pos >= a.edge_cap) { atomicExch(a.error_flag, 91ULL); return; } // over cap
    a.edge_p[pos] = parent;
    a.edge_c[pos] = succ;
  }
}

// EX pass over the materialized relation: mask_out[p] |= 1 iff some edge
// p->c has mask_a[c]. Pure gather — no dispatch, no fp probe — so every
// fixpoint iteration is a flat scan of edge_count edges.
extern "C" __global__ void ty_gpu_ctl_ex_edges(TyGpuCtlArgs a) {
  const u64 n = *a.edge_count;
  for (u64 e = blockIdx.x * (u64)blockDim.x + threadIdx.x; e < n;
       e += gridDim.x * (u64)blockDim.x) {
    const u64 parent = a.edge_p[e];
    if (a.mask_out[parent]) continue; // already known this pass
    if (a.mask_a[a.edge_c[e]]) {
      if (!a.mask_out[parent]) { a.mask_out[parent] = 1; atomicExch(a.changed, 1ULL); }
    }
  }
}
"#;

#[repr(C)]
struct CtlArgs {
    arena: CuDeviceptr,
    arena_len: u64,
    arena_cap: u64,
    fp_lo: CuDeviceptr,
    fp_hi: CuDeviceptr,
    fp_idx: CuDeviceptr,
    fp_mask: u64,
    next_count: CuDeviceptr,
    level_base: u64,
    level_len: u64,
    deadlock: CuDeviceptr,
    error_flag: CuDeviceptr,
    mask_out: CuDeviceptr,
    mask_a: CuDeviceptr,
    mask_b: CuDeviceptr,
    atom_index: i32,
    op_code: i32,
    changed: CuDeviceptr,
    edge_p: CuDeviceptr,
    edge_c: CuDeviceptr,
    edge_count: CuDeviceptr,
    edge_cap: u64,
}

fn launch(
    api: &CudaApi,
    func: CuFunction,
    grid: u32,
    block: u32,
    args: &mut CtlArgs,
) -> Result<(), GpuError> {
    let mut params = [std::ptr::from_mut(args).cast::<c_void>()];
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

unsafe fn read_u64(buf: &DeviceBuffer) -> u64 {
    unsafe { std::ptr::read_volatile(buf.ptr as *const u64) }
}

unsafe fn write_u64(buf: &DeviceBuffer, v: u64) {
    unsafe { std::ptr::write_volatile(buf.ptr as *mut u64, v) }
}

/// Assemble the CTL engine source: the standard engine (for
/// `ty_gpu_dispatch`, `ty_gpu_fp128`, `ty_gpu_fp_insert`) + an atom dispatch
/// table + the CTL driver.
fn assemble_ctl_source(spec: &GpuCtlSpec) -> String {
    let mut src = assemble_engine_source(
        spec.slots,
        spec.action_count,
        &spec.actions_src,
        false,
        false,
    );
    src.push_str(&spec.atoms_src);
    src.push_str(
        "\nstatic __device__ __forceinline__ int ty_gpu_atoms(int k, const i64* s) {\n  switch (k) {\n",
    );
    for k in 0..spec.atom_count {
        src.push_str(&format!("    case {k}: return ty_gpu_atom_{k}(s);\n"));
    }
    src.push_str("    default: return -1;\n  }\n}\n");
    src.push_str(CTL_DRIVER);
    src
}

struct CtlDevice<'a> {
    api: &'a CudaApi,
    module: crate::cuda::CuModule,
    expand: CuFunction,
    seed: CuFunction,
    mark_deadlock: CuFunction,
    eval_atom: CuFunction,
    elementwise: CuFunction,
    ex: CuFunction,
    record_edges: CuFunction,
    ex_edges: CuFunction,
}

impl Drop for CtlDevice<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.cuModuleUnload)(self.module) };
    }
}

/// Run the retained-graph engine: enumerate the reachable set, then evaluate
/// every formula in the batch. See the module docs for semantics.
///
/// # Errors
///
/// [`GpuError`] on any capacity/driver/nvrtc failure (fail-closed; callers
/// fall back to the CPU checker). A deadline expiry surfaces as
/// `GpuError::Driver("deadline...")`.
pub fn run_ctl(
    spec: &GpuCtlSpec,
    config: &GpuCtlConfig,
    formulas: &[CtlOp],
) -> Result<GpuCtlOutcome, GpuError> {
    if spec.slots == 0 || spec.init_rows.is_empty() || spec.init_rows.len() % spec.slots != 0 {
        return Err(GpuError::Codegen("malformed init rows".into()));
    }
    if spec.action_count == 0 {
        return Err(GpuError::Codegen(
            "GPU CTL requires at least one action".into(),
        ));
    }
    if !(1..=1024).contains(&config.block) {
        return Err(GpuError::Codegen(
            "GPU CTL block size must be in 1..=1024".into(),
        ));
    }
    if !config.max_load_factor.is_finite()
        || !(0.0..1.0).contains(&config.max_load_factor)
        || config.max_load_factor == 0.0
    {
        return Err(GpuError::Codegen(
            "GPU CTL max_load_factor must be finite and in (0, 1)".into(),
        ));
    }
    let table_slots = checked_power_of_two_u64("CTL fingerprint table slots", config.table_bits)?;
    let check_formula_atoms = |op: &CtlOp| -> bool {
        fn walk(op: &CtlOp, n: usize) -> bool {
            match op {
                CtlOp::Atom(k) => *k < n,
                CtlOp::True => true,
                CtlOp::Not(a) | CtlOp::EX(a) | CtlOp::EF(a) | CtlOp::EG(a) | CtlOp::EGF(a) => {
                    walk(a, n)
                }
                CtlOp::And(cs) | CtlOp::Or(cs) => cs.iter().all(|c| walk(c, n)),
                CtlOp::EU(a, b) => walk(a, n) && walk(b, n),
            }
        }
        walk(op, spec.atom_count)
    };
    if !formulas.iter().all(check_formula_atoms) {
        return Err(GpuError::Codegen("atom index out of range".into()));
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

    let compile_start = Instant::now();
    let source = assemble_ctl_source(spec);
    if let Ok(path) = std::env::var("TY_GPU_DUMP_CTL") {
        let _ = std::fs::write(path, &source);
    }
    let ptx = crate::nvrtc::compile_to_ptx(&source, info.cc_major, info.cc_minor)?;
    let mut module = std::ptr::null_mut();
    check(
        api,
        unsafe { (api.cuModuleLoadData)(&mut module, ptx.as_ptr().cast::<c_void>()) },
        "cuModuleLoadData",
    )?;
    let get = |name: &str| -> Result<CuFunction, GpuError> {
        let cname = std::ffi::CString::new(name).expect("static");
        let mut f = std::ptr::null_mut();
        let rc = unsafe { (api.cuModuleGetFunction)(&mut f, module, cname.as_ptr()) };
        if rc == crate::cuda::CUDA_SUCCESS {
            Ok(f)
        } else {
            Err(GpuError::Driver(format!("kernel symbol {name} missing")))
        }
    };
    let device = CtlDevice {
        api,
        module,
        expand: get("ty_gpu_ctl_expand")?,
        seed: get("ty_gpu_ctl_seed")?,
        mark_deadlock: get("ty_gpu_ctl_mark_deadlock")?,
        eval_atom: get("ty_gpu_ctl_eval_atom")?,
        elementwise: get("ty_gpu_ctl_elementwise")?,
        ex: get("ty_gpu_ctl_ex")?,
        record_edges: get("ty_gpu_ctl_record_edges")?,
        ex_edges: get("ty_gpu_ctl_ex_edges")?,
    };
    let compile_wall = compile_start.elapsed();

    let table_slots_usize = checked_allocation_usize("CTL fingerprint table slots", table_slots)?;
    let table_bytes =
        checked_allocation_bytes("CTL fingerprint table bytes", &[table_slots_usize, 8])?;
    let row_bytes = checked_allocation_bytes("CTL state row bytes", &[spec.slots, 8])?;
    let row_bytes_u64 = u64::try_from(row_bytes)
        .map_err(|_| GpuError::AllocationOverflow("CTL state row bytes"))?;
    let init_count_usize = spec.init_rows.len() / spec.slots;
    let init_count = u64::try_from(init_count_usize)
        .map_err(|_| GpuError::AllocationOverflow("CTL initial-state count"))?;
    if init_count > config.max_states {
        return Err(GpuError::CapacityExceeded {
            what: "initial states",
            needed: init_count,
            capacity: config.max_states,
        });
    }
    let fingerprint_capacity = (table_slots as f64 * config.max_load_factor) as u64;
    if init_count > fingerprint_capacity {
        return Err(GpuError::CapacityExceeded {
            what: "fingerprint table",
            needed: init_count,
            capacity: fingerprint_capacity,
        });
    }

    let max_states_usize = checked_allocation_usize("CTL maximum state count", config.max_states)?;
    let arena_rows =
        checked_allocation_sum("CTL arena rows", &[max_states_usize, init_count_usize])?;
    let arena_bytes = checked_allocation_bytes("CTL arena bytes", &[arena_rows, row_bytes])?;
    let init_bytes =
        checked_allocation_bytes("CTL initial-state bytes", &[spec.init_rows.len(), 8])?;
    let table_action_count = u64::try_from(spec.action_count)
        .map_err(|_| GpuError::AllocationOverflow("CTL action count"))?;
    let max_transition_pairs = config
        .max_states
        .checked_mul(table_action_count)
        .ok_or(GpuError::AllocationOverflow("CTL maximum transition pairs"))?;
    let staging_offset = config
        .max_states
        .checked_mul(row_bytes_u64)
        .ok_or(GpuError::AllocationOverflow("CTL init staging offset"))?;

    // Arena carries a staging area of init_count rows past the cap (the seed
    // kernel reads the raw init rows from there).
    let arena = DeviceBuffer::device(api, arena_bytes)?;
    let fp_lo = DeviceBuffer::device(api, table_bytes)?;
    let fp_hi = DeviceBuffer::device(api, table_bytes)?;
    let fp_idx = DeviceBuffer::device(api, table_bytes)?;
    let deadlock = DeviceBuffer::device(api, max_states_usize)?;
    let next_count = DeviceBuffer::managed(api, 8)?;
    let error_flag = DeviceBuffer::managed(api, 8)?;
    let changed = DeviceBuffer::managed(api, 8)?;
    // edge_count doubles as (Phase A) the free transition counter and (record
    // pass) the append cursor. edge_p/edge_c are sized + filled only after
    // enumeration, so mk_args reads their pointers/cap from Cells that stay
    // null until then (the expand kernel never touches them).
    let edge_count = DeviceBuffer::managed(api, 8)?;
    let edge_p_cell = std::cell::Cell::new(0u64 as CuDeviceptr);
    let edge_c_cell = std::cell::Cell::new(0u64 as CuDeviceptr);
    let edge_cap_cell = std::cell::Cell::new(0u64);
    check(
        api,
        unsafe { (api.cuMemsetD8_v2)(fp_lo.ptr, 0, fp_lo.bytes) },
        "memset fp_lo",
    )?;
    check(
        api,
        unsafe { (api.cuMemsetD8_v2)(fp_hi.ptr, 0, fp_hi.bytes) },
        "memset fp_hi",
    )?;
    unsafe {
        write_u64(&next_count, 0);
        write_u64(&error_flag, 0);
        write_u64(&changed, 0);
        write_u64(&edge_count, 0);
    }
    // Upload raw init rows to the staging area (arena[max_states..]).
    check(
        api,
        unsafe {
            (api.cuMemcpyHtoD_v2)(
                arena.ptr + staging_offset,
                spec.init_rows.as_ptr().cast::<c_void>(),
                init_bytes,
            )
        },
        "upload init rows",
    )?;

    let search_start = Instant::now();
    let mk_args = |arena_len: u64, level_base: u64, level_len: u64| CtlArgs {
        arena: arena.ptr,
        arena_len,
        arena_cap: config.max_states,
        fp_lo: fp_lo.ptr,
        fp_hi: fp_hi.ptr,
        fp_idx: fp_idx.ptr,
        fp_mask: table_slots - 1,
        next_count: next_count.ptr,
        level_base,
        level_len,
        deadlock: deadlock.ptr,
        error_flag: error_flag.ptr,
        mask_out: 0,
        mask_a: 0,
        mask_b: 0,
        atom_index: 0,
        op_code: 0,
        changed: changed.ptr,
        edge_p: edge_p_cell.get(),
        edge_c: edge_c_cell.get(),
        edge_count: edge_count.ptr,
        edge_cap: edge_cap_cell.get(),
    };
    let check_error = |what: &str| -> Result<(), GpuError> {
        let err = unsafe { read_u64(&error_flag) };
        if err != 0 {
            return Err(GpuError::Driver(format!(
                "GPU CTL kernel signaled fault {err} during {what}"
            )));
        }
        Ok(())
    };
    let check_deadline = || -> Result<(), GpuError> {
        if config.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(GpuError::Driver("deadline expired mid-run".into()));
        }
        Ok(())
    };

    // --- Phase A: enumerate + retain the reachable set ---
    let mut args = mk_args(0, 0, init_count);
    launch(api, device.seed, 1, 1, &mut args)?;
    check_error("seed")?;
    let mut level_base = 0u64;
    let mut arena_len = unsafe { read_u64(&next_count) };
    // The seed deduped the initial rows into arena[0..deduped_init]; every
    // one of those is a distinct initial state, and arena indices at/after it
    // are strictly non-initial states discovered by level-0 expansion. This
    // is the count the initial-state verdict conjunction reads (NOT the raw
    // `init_count`, which would over-include non-initial states when the
    // caller passes duplicate initial rows).
    let deduped_init = arena_len;
    while arena_len > level_base {
        check_deadline()?;
        if arena_len > config.max_states {
            return Err(GpuError::CapacityExceeded {
                what: "retained arena",
                needed: arena_len,
                capacity: config.max_states,
            });
        }
        if arena_len > fingerprint_capacity {
            return Err(GpuError::CapacityExceeded {
                what: "fingerprint table",
                needed: arena_len,
                capacity: fingerprint_capacity,
            });
        }
        let level_len = arena_len - level_base;
        let mut args = mk_args(arena_len, level_base, level_len);
        let grid_l = u32::try_from(level_len.div_ceil(u64::from(config.block)).min(65_535))
            .expect("bounded");
        launch(api, device.mark_deadlock, grid_l, config.block, &mut args)?;
        let pairs = level_len
            .checked_mul(table_action_count)
            .ok_or(GpuError::AllocationOverflow("CTL frontier/action pairs"))?;
        let grid =
            u32::try_from(pairs.div_ceil(u64::from(config.block)).min(65_535)).expect("bounded");
        launch(api, device.expand, grid, config.block, &mut args)?;
        check_error("expand")?;
        level_base = arena_len;
        arena_len = unsafe { read_u64(&next_count) };
    }
    if arena_len > config.max_states {
        return Err(GpuError::CapacityExceeded {
            what: "retained arena",
            needed: arena_len,
            capacity: config.max_states,
        });
    }
    let distinct = arena_len;

    // --- Materialize the transition relation (once) for fixpoint EX passes ---
    // Only worthwhile when the batch has a temporal operator (otherwise EX is
    // never launched). On allocation failure we fall back to the re-firing EX
    // kernel — correct, just slower — rather than declining the whole check.
    fn has_temporal(op: &CtlOp) -> bool {
        match op {
            CtlOp::EX(_) | CtlOp::EF(_) | CtlOp::EG(_) | CtlOp::EU(..) | CtlOp::EGF(_) => true,
            CtlOp::Not(a) => has_temporal(a),
            CtlOp::And(cs) | CtlOp::Or(cs) => cs.iter().any(has_temporal),
            CtlOp::Atom(_) | CtlOp::True => false,
        }
    }
    let needs_edges = formulas.iter().any(has_temporal);
    let mut use_edges = false;
    let mut edge_grid = 0u32;
    let mut _edge_bufs: Option<(DeviceBuffer, DeviceBuffer)> = None;
    if needs_edges {
        let edge_total = unsafe { read_u64(&edge_count) };
        if edge_total > max_transition_pairs {
            return Err(GpuError::Driver(format!(
                "GPU CTL transition counter out of range ({edge_total} > {max_transition_pairs})"
            )));
        }
        let cap = edge_total.max(1);
        let edge_bytes = checked_allocation_usize("CTL edge count", cap)
            .and_then(|count| checked_allocation_bytes("CTL edge bytes", &[count, 8]));
        if let Ok((ep, ec)) = edge_bytes.and_then(|bytes| {
            let ep = DeviceBuffer::device(api, bytes)?;
            let ec = DeviceBuffer::device(api, bytes)?;
            Ok((ep, ec))
        }) {
            edge_p_cell.set(ep.ptr);
            edge_c_cell.set(ec.ptr);
            edge_cap_cell.set(cap);
            // The counter now serves as the record-pass append cursor.
            unsafe { write_u64(&edge_count, 0) };
            let record_pairs = distinct
                .checked_mul(table_action_count)
                .ok_or(GpuError::AllocationOverflow("CTL record-edge pairs"))?;
            let rec_grid = u32::try_from(
                record_pairs
                    .max(1)
                    .div_ceil(u64::from(config.block))
                    .min(65_535),
            )
            .expect("bounded");
            let mut args = mk_args(distinct, 0, 0);
            launch(api, device.record_edges, rec_grid, config.block, &mut args)?;
            check_error("record edges")?;
            edge_grid = u32::try_from(
                edge_total
                    .max(1)
                    .div_ceil(u64::from(config.block))
                    .min(65_535),
            )
            .expect("bounded");
            use_edges = true;
            _edge_bufs = Some((ep, ec));
        }
    }

    // --- Phase B: evaluate the formula batch ---
    let mask_bytes = checked_allocation_usize("CTL formula mask bytes", distinct.max(1))?;
    let grid_s = u32::try_from(
        distinct
            .max(1)
            .div_ceil(u64::from(config.block))
            .min(65_535),
    )
    .expect("bounded");
    let alloc_mask = || DeviceBuffer::device(api, mask_bytes);
    let zero_mask = |m: &DeviceBuffer| -> Result<(), GpuError> {
        check(
            api,
            unsafe { (api.cuMemsetD8_v2)(m.ptr, 0, m.bytes) },
            "memset mask",
        )
    };

    // Recursive evaluator: returns a device mask for the sub-formula.
    fn eval(
        op: &CtlOp,
        api: &CudaApi,
        device: &CtlDevice<'_>,
        mk_args: &dyn Fn(u64, u64, u64) -> CtlArgs,
        distinct: u64,
        grid_s: u32,
        block: u32,
        alloc_mask: &dyn Fn() -> Result<DeviceBuffer, GpuError>,
        zero_mask: &dyn Fn(&DeviceBuffer) -> Result<(), GpuError>,
        deadlock_buf: CuDeviceptr,
        changed: &DeviceBuffer,
        error_check: &dyn Fn(&str) -> Result<(), GpuError>,
        deadline_check: &dyn Fn() -> Result<(), GpuError>,
        action_count: u64,
        use_edges: bool,
        edge_grid: u32,
    ) -> Result<DeviceBuffer, GpuError> {
        let elementwise = |out: &DeviceBuffer,
                           a: CuDeviceptr,
                           b: CuDeviceptr,
                           op_code: i32|
         -> Result<(), GpuError> {
            let mut args = mk_args(distinct, 0, 0);
            args.mask_out = out.ptr;
            args.mask_a = a;
            args.mask_b = b;
            args.op_code = op_code;
            launch(api, device.elementwise, grid_s, block, &mut args)
        };
        let ex_pass = |out: &DeviceBuffer, z: CuDeviceptr| -> Result<(), GpuError> {
            let mut args = mk_args(distinct, 0, 0);
            args.mask_out = out.ptr;
            args.mask_a = z;
            if use_edges {
                // Gather over the materialized relation (edge_grid threads).
                launch(api, device.ex_edges, edge_grid, block, &mut args)?;
            } else {
                // Fall back to re-firing every action (correct, slower).
                let pairs = distinct
                    .checked_mul(action_count)
                    .ok_or(GpuError::AllocationOverflow("CTL EX action pairs"))?;
                let grid = u32::try_from(pairs.max(1).div_ceil(u64::from(block)).min(65_535))
                    .expect("bounded");
                launch(api, device.ex, grid, block, &mut args)?;
            }
            error_check("EX pass")
        };
        let fixpoint = |body: &mut dyn FnMut() -> Result<(), GpuError>| -> Result<(), GpuError> {
            loop {
                deadline_check()?;
                unsafe { write_u64(changed, 0) };
                body()?;
                if unsafe { read_u64(changed) } == 0 {
                    return Ok(());
                }
            }
        };

        match op {
            CtlOp::Atom(k) => {
                let out = alloc_mask()?;
                let mut args = mk_args(distinct, 0, 0);
                args.mask_out = out.ptr;
                args.atom_index = i32::try_from(*k).expect("checked");
                launch(api, device.eval_atom, grid_s, block, &mut args)?;
                error_check("atom eval")?;
                Ok(out)
            }
            CtlOp::True => {
                let out = alloc_mask()?;
                check(
                    api,
                    unsafe { (api.cuMemsetD8_v2)(out.ptr, 1, out.bytes) },
                    "memset true",
                )?;
                Ok(out)
            }
            CtlOp::Not(a) => {
                let ma = eval(
                    a,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                let out = alloc_mask()?;
                zero_mask(&out)?;
                elementwise(&out, ma.ptr, ma.ptr, 1)?;
                Ok(out)
            }
            CtlOp::And(cs) | CtlOp::Or(cs) => {
                let is_and = matches!(op, CtlOp::And(_));
                let out = alloc_mask()?;
                check(
                    api,
                    unsafe { (api.cuMemsetD8_v2)(out.ptr, u8::from(is_and), out.bytes) },
                    "memset unit",
                )?;
                for c in cs {
                    let mc = eval(
                        c,
                        api,
                        device,
                        mk_args,
                        distinct,
                        grid_s,
                        block,
                        alloc_mask,
                        zero_mask,
                        deadlock_buf,
                        changed,
                        error_check,
                        deadline_check,
                        action_count,
                        use_edges,
                        edge_grid,
                    )?;
                    elementwise(&out, out.ptr, mc.ptr, if is_and { 2 } else { 3 })?;
                }
                Ok(out)
            }
            CtlOp::EX(a) => {
                let ma = eval(
                    a,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                let out = alloc_mask()?;
                zero_mask(&out)?;
                ex_pass(&out, ma.ptr)?;
                Ok(out)
            }
            CtlOp::EF(a) => {
                // μZ. a ∨ EX Z — start Z = a, accumulate EX into Z.
                let z = eval(
                    a,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                fixpoint(&mut || {
                    // ex_out |= "some successor in Z"; then Z |= ex_out.
                    ex_pass(&z, z.ptr)
                })?;
                Ok(z)
            }
            CtlOp::EU(a, b) => {
                // μZ. b ∨ (a ∧ EX Z) — start Z = b.
                let ma = eval(
                    a,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                let z = eval(
                    b,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                let ex_out = alloc_mask()?;
                fixpoint(&mut || {
                    zero_mask(&ex_out)?;
                    // ex_out = EX Z (fresh each round; `changed` tracks Z only).
                    unsafe { write_u64(changed, 0) };
                    ex_pass(&ex_out, z.ptr)?;
                    unsafe { write_u64(changed, 0) };
                    // Z |= a ∧ ex_out
                    elementwise(&z, ma.ptr, ex_out.ptr, 4)
                })?;
                Ok(z)
            }
            CtlOp::EG(a) => {
                // νZ. a ∧ (deadlock ∨ EX Z) — start Z = a, shrink.
                let ma = eval(
                    a,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                let z = alloc_mask()?;
                elementwise(&z, ma.ptr, ma.ptr, 0)?; // Z = a
                let ex_out = alloc_mask()?;
                let keep = alloc_mask()?;
                fixpoint(&mut || {
                    zero_mask(&ex_out)?;
                    unsafe { write_u64(changed, 0) };
                    ex_pass(&ex_out, z.ptr)?;
                    // keep = ex_out | deadlock
                    zero_mask(&keep)?;
                    elementwise(&keep, ex_out.ptr, deadlock_buf, 3)?;
                    unsafe { write_u64(changed, 0) };
                    // Z = Z & (a & keep): shrink; changed tracks the Z write.
                    elementwise(&keep, ma.ptr, keep.ptr, 2)?;
                    unsafe { write_u64(changed, 0) };
                    elementwise(&z, z.ptr, keep.ptr, 2)
                })?;
                Ok(z)
            }
            CtlOp::EGF(a) => {
                // E(GF a): a path visiting `a` infinitely often — the
                // Emerson–Lei fair-cycle νZ. μY. (a ∧ EXˢZ) ∨ EXˢY, with the
                // STUTTER-aware successor EXˢ(M) = EX(M) ∨ (deadlock ∧ M) so a
                // deadlocked `a`-state (an infinite `a`-stutter) is a witness —
                // matching the CPU Büchi product's deadlock self-loop. Two
                // manual nested fixpoints (no re-entrant closure borrow).
                let ma = eval(
                    a,
                    api,
                    device,
                    mk_args,
                    distinct,
                    grid_s,
                    block,
                    alloc_mask,
                    zero_mask,
                    deadlock_buf,
                    changed,
                    error_check,
                    deadline_check,
                    action_count,
                    use_edges,
                    edge_grid,
                )?;
                let z = alloc_mask()?;
                check(
                    api,
                    unsafe { (api.cuMemsetD8_v2)(z.ptr, 1, z.bytes) },
                    "EGF z=true",
                )?;
                let y = alloc_mask()?;
                let ex_z = alloc_mask()?;
                let ex_y = alloc_mask()?;
                // Outer greatest fixpoint: Z shrinks to states that start a
                // fair path. Z ⊇ Y ⊇ ... is monotone decreasing → terminates.
                loop {
                    deadline_check()?;
                    zero_mask(&y)?; // Y := ∅
                                    // Inner least fixpoint: Y grows to states that reach an
                                    // `a`-state-with-an-EXˢZ-successor, following EXˢ edges.
                    loop {
                        deadline_check()?;
                        // exZ := EXˢ Z ; exY := EXˢ Y  (compute before the reset;
                        // their `changed` writes are noise for convergence).
                        zero_mask(&ex_z)?;
                        ex_pass(&ex_z, z.ptr)?;
                        elementwise(&ex_z, deadlock_buf, z.ptr, 4)?; // |= deadlock & Z
                        zero_mask(&ex_y)?;
                        ex_pass(&ex_y, y.ptr)?;
                        elementwise(&ex_y, deadlock_buf, y.ptr, 4)?; // |= deadlock & Y
                                                                     // Y |= (a ∧ exZ) | exY ; `changed` now tracks only Y.
                        unsafe { write_u64(changed, 0) };
                        elementwise(&y, ma.ptr, ex_z.ptr, 4)?; // y |= a & exZ
                        elementwise(&y, y.ptr, ex_y.ptr, 3)?; // y |= exY
                        if unsafe { read_u64(changed) } == 0 {
                            break;
                        }
                    }
                    // Z := Y (shrink); `changed` set iff Z actually shrank.
                    unsafe { write_u64(changed, 0) };
                    elementwise(&z, y.ptr, y.ptr, 0)?;
                    if unsafe { read_u64(changed) } == 0 {
                        break;
                    }
                }
                Ok(z)
            }
        }
    }

    let mk_args_ref = |a: u64, b: u64, c: u64| mk_args(a, b, c);
    let mut verdicts = Vec::new();
    try_reserve_host(&mut verdicts, formulas.len(), "CTL formula verdicts")?;
    for formula in formulas {
        check_deadline()?;
        let mask = eval(
            formula,
            api,
            &device,
            &mk_args_ref,
            distinct,
            grid_s,
            config.block,
            &alloc_mask,
            &zero_mask,
            deadlock.ptr,
            &changed,
            &check_error,
            &check_deadline,
            table_action_count,
            use_edges,
            edge_grid,
        )?;
        // Truth at every deduped initial state (arena indices
        // 0..deduped_init — the seed appended the distinct initial states
        // first; every later arena index is a non-initial reachable state, so
        // the conjunction MUST NOT read past `deduped_init`).
        let take =
            checked_allocation_usize("CTL initial-state mask bytes", deduped_init.min(distinct))?
                .max(1);
        let mut init_mask: Vec<u8> =
            try_zeroed_host_vec(take.min(mask_bytes), "CTL initial-state mask")?;
        let take = init_mask.len();
        unsafe {
            check(
                api,
                (api.cuMemcpyDtoH_v2)(
                    init_mask.as_mut_ptr().cast::<c_void>(),
                    mask.ptr,
                    take.max(1),
                ),
                "download init mask",
            )?;
        }
        verdicts.push(init_mask[..take].iter().all(|&b| b != 0));
    }

    Ok(GpuCtlOutcome {
        distinct_states: distinct,
        verdicts,
        wall: search_start.elapsed(),
        compile_wall,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctl_fingerprint_insert_and_lookup_probes_are_bounded() {
        assert_eq!(
            CTL_DRIVER
                .matches("for (u64 probe = 0; probe <= a.fp_mask; probe++)")
                .count(),
            2
        );
        assert_eq!(
            CTL_DRIVER
                .matches("atomicExch(a.error_flag, 92ULL)")
                .count(),
            2
        );
        assert!(!CTL_DRIVER.contains("while (true)"));
    }
}
