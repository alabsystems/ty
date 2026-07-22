//! CUDA C source assembly for the generic device-resident BFS engine.
//!
//! The engine source is a fixed driver (fingerprint table, frontier expansion
//! kernel, seed kernel) parameterized by spec-generated pieces:
//!
//! - `SLOTS` — i64 slots per state (`StateLayout::total_slots`)
//! - `ACTION_COUNT` and `ty_gpu_actions` — one `__device__` function per
//!   compiled action; each takes the parent row and a successor scratch row
//!   (pre-copied from the parent) and returns 1 if the action is enabled
//!   (successor written) or 0 (disabled; scratch content ignored)
//! - `ty_gpu_invariants_ok` — 1 if all invariants hold on a row
//!
//! Dedup is a 128-bit-fingerprint two-lane claim/publish open-addressed table
//! (the device mirror of `AtomicFpSet`'s hi/lo lanes).

/// Assemble the full CUDA C source for a spec.
///
/// `actions_src` must define:
/// `static __device__ int ty_gpu_action_<k>(const long long* s, long long* t)`
/// for k in `0..action_count`, plus
/// `static __device__ int ty_gpu_invariants_ok(const long long* s)`.
pub fn assemble_engine_source(
    slots: usize,
    action_count: usize,
    actions_src: &str,
    track_slot_stats: bool,
    trace_parents: bool,
) -> String {
    let mut src = String::with_capacity(actions_src.len() + ENGINE_DRIVER.len() + 4096);
    src.push_str(&format!(
        "#define SLOTS {slots}\n#define ACTION_COUNT {action_count}\n\
         #define TY_GPU_TRACK_SLOT_STATS {stats}\n\
         #define TY_GPU_TRACE {trace}\n\
         typedef unsigned long long u64;\ntypedef long long i64;\n\n",
        stats = i32::from(track_slot_stats),
        trace = i32::from(trace_parents),
    ));
    src.push_str(actions_src);
    // The expand/seed kernels call ty_gpu_constraint_ok for on-device state-
    // constraint pruning. emit_program[_with_constraints] always defines it
    // (a `return 1` no-op when there are no constraints), but other callers
    // that hand-build actions_src (e.g. the CTL engine, which reuses this
    // template for ty_gpu_dispatch/fp128 but drives its own kernels) do not —
    // provide a permissive default so their unused BFS kernels still compile.
    if !actions_src.contains("ty_gpu_constraint_ok") {
        src.push_str(
            "\nstatic __device__ __forceinline__ int ty_gpu_constraint_ok(const i64* s) { (void)s; return 1; }\n",
        );
    }
    src.push_str(ACTION_TABLE_HEADER);
    for k in 0..action_count {
        src.push_str(&format!("    case {k}: return ty_gpu_action_{k}(s, t);\n"));
    }
    src.push_str(ACTION_TABLE_FOOTER);
    src.push_str(ENGINE_DRIVER);
    src
}

const ACTION_TABLE_HEADER: &str = r#"
static __device__ __forceinline__ int ty_gpu_dispatch(int action, const i64* s, i64* t) {
  switch (action) {
"#;

const ACTION_TABLE_FOOTER: &str = r#"    default: return 0;
  }
}
"#;

const ENGINE_DRIVER: &str = r#"
// ---- generic engine below this line (spec-independent) ----

// 128-bit fingerprint of a row: two-lane multiply/rotate mix over SLOTS words.
static __device__ __forceinline__ void ty_gpu_fp128(const i64* row, u64* hi, u64* lo) {
  u64 h1 = 0x9E3779B185EBCA87ULL, h2 = 0xC2B2AE3D27D4EB4FULL;
  #pragma unroll
  for (int i = 0; i < SLOTS; i++) {
    u64 v = (u64)row[i];
    h1 = (h1 ^ (v * 0x9DDFEA08EB382D69ULL));
    h1 = (h1 << 31) | (h1 >> 33);
    h1 *= 0x87C37B91114253D5ULL;
    h2 = (h2 + v) * 0xFF51AFD7ED558CCDULL;
    h2 ^= h2 >> 29;
  }
  h1 ^= (u64)SLOTS * 8; h1 ^= h1 >> 32; h1 *= 0xC4CEB9FE1A85EC53ULL; h1 ^= h1 >> 29;
  h2 ^= h2 >> 32; h2 *= 0x9E3779B97F4A7C15ULL; h2 ^= h2 >> 27;
  *hi = h1 | 1ULL; // hi is never 0 (0 = unpublished sentinel)
  *lo = h2;
}

// Two-lane claim/publish insert. Returns 1 if newly inserted, 0 if already
// present, and -1 if every table slot was probed (fail-closed saturation).
static __device__ __forceinline__ int ty_gpu_fp_insert(u64* lo_lane, u64* hi_lane,
                                                       u64 mask, u64 hi, u64 lo) {
  u64 lokey = lo | 1ULL;
  u64 slot = (hi ^ lo) & mask;
  for (u64 probe = 0; probe <= mask; probe++) {
    u64 prev = atomicCAS(&lo_lane[slot], 0ULL, lokey);
    if (prev == 0ULL) { atomicExch(&hi_lane[slot], hi); return 1; }
    if (prev == lokey) {
      u64 h;
      do { h = atomicAdd(&hi_lane[slot], 0ULL); } while (h == 0ULL);
      if (h == hi) return 0;
    }
    slot = (slot + 1) & mask;
  }
  return -1;
}

#if TY_GPU_TRACK_SLOT_STATS
// Per-new-state slot statistics (Petri StateSpace / UpperBounds / OneSafe:
// max token in a place, max tokens in a marking, per-place maxima). Slots
// are non-negative in tracked domains. The per-slot atomic is skipped for
// zero values (most slots on most states), so the common cost is the two
// row-aggregate atomics.
static __device__ __forceinline__ void ty_gpu_track_slot_stats(const i64* row, u64* max_slot,
                                                               u64* max_slot_sum,
                                                               u64* slot_maxima,
                                                               u64* slot_minima) {
  u64 mx = 0, sum = 0;
  #pragma unroll
  for (int i = 0; i < SLOTS; i++) {
    u64 v = (u64)row[i];
    sum += v;
    if (v > mx) mx = v;
    if (v) atomicMax(&slot_maxima[i], v);
    atomicMin(&slot_minima[i], v);
  }
  atomicMax(max_slot, mx);
  atomicMax(max_slot_sum, sum);
}
#endif

struct TyGpuLevelArgs {
  const i64* frontier;      // parent rows, row-major SLOTS per row
  u64 frontier_len;         // parent row count
  u64* fp_lo;               // fingerprint table lanes
  u64* fp_hi;
  u64 fp_mask;              // table_slots - 1
  i64* next_rows;           // successor arena (capacity checked on host)
  u64 next_cap;             // successor arena capacity in rows
  u64* next_count;          // atomic append cursor
  u64* transitions;         // total candidate successors (enabled action firings)
  u64* deadlocks;           // parents with zero enabled actions
  u64* error_flag;          // fail-closed: nonzero = a kernel signaled a
                            // runtime error / fallback request; the host
                            // aborts the run and no verdict is reported
  u64* max_slot;            // TY_GPU_TRACK_SLOT_STATS: max slot value over all states
  u64* max_slot_sum;        // TY_GPU_TRACK_SLOT_STATS: max per-state slot sum
  u64* slot_maxima;         // TY_GPU_TRACK_SLOT_STATS: per-slot maxima (SLOTS entries)
  u64* slot_minima;         // TY_GPU_TRACK_SLOT_STATS: per-slot minima (SLOTS entries)
  unsigned char* enabled_flags; // per-parent "some action fired" byte
  int* violation;           // set to 1 + violating-row-index-marker on invariant failure
  i64* violation_row;       // first violating row (best-effort single writer)
  // Counterexample-trace support (TY_GPU_TRACE): the frontier is a WINDOW into
  // a monotone arena, so a child's parent has a stable global index.
  u64 level_base;           // arena index of the first frontier row this level
  u64* parent;              // parent[child_idx] = parent arena index (~0 = init)
  u64* violation_index;     // arena index of the first violating state
};

// One thread per (parent, action) pair, parent-major: the ACTION_COUNT
// threads of one parent are consecutive, so a warp broadcasts the parent row
// from L1/L2 (each row is streamed once per level, not once per action).
// Guards diverge within the warp, but at typical enabledness (a few actions
// per state) only a couple of full bodies serialize.
extern "C" __global__ void ty_gpu_expand_level(TyGpuLevelArgs a) {
  u64 local_trans = 0;
  const u64 pairs = a.frontier_len * (u64)ACTION_COUNT;
  for (u64 gi = blockIdx.x * (u64)blockDim.x + threadIdx.x; gi < pairs;
       gi += gridDim.x * (u64)blockDim.x) {
    const u64 parent = gi / (u64)ACTION_COUNT;
    const int action = (int)(gi % (u64)ACTION_COUNT);
    const i64* s = a.frontier + parent * SLOTS;
    i64 t[SLOTS];
    #pragma unroll
    for (int i = 0; i < SLOTS; i++) t[i] = s[i];
    int fired = ty_gpu_dispatch(action, s, t);
    if (fired < 0) { atomicExch(a.error_flag, (u64)(-fired)); return; }
    if (fired == 0) continue;
    a.enabled_flags[parent] = 1; // benign race: any writer suffices
    local_trans++;
    // State-constraint pruning: raw enabledness (enabled_flags) and the raw
    // transition count (local_trans) are already recorded, so deadlock and
    // transition counts match the CPU reference (which computes them from RAW
    // successors before the constraint filter). A constraint-failing successor
    // is then dropped: never fingerprinted, counted distinct, invariant-
    // checked, or expanded.
    int cok = ty_gpu_constraint_ok(t);
    if (cok < 0) { atomicExch(a.error_flag, (u64)(-cok)); return; }
    if (cok == 0) continue;
    u64 hi, lo;
    ty_gpu_fp128(t, &hi, &lo);
    int inserted = ty_gpu_fp_insert(a.fp_lo, a.fp_hi, a.fp_mask, hi, lo);
    if (inserted < 0) { atomicExch(a.error_flag, 92ULL); return; }
    if (inserted > 0) {
#if TY_GPU_TRACK_SLOT_STATS
      ty_gpu_track_slot_stats(t, a.max_slot, a.max_slot_sum, a.slot_maxima, a.slot_minima);
#endif
      u64 idx = atomicAdd(a.next_count, 1ULL);
#if TY_GPU_TRACE
      // Monotone arena: record this child's parent's global index so the host
      // can walk init->violation. (level_base + parent) is the parent's arena
      // index because the frontier is arena[level_base .. level_base+len].
      if (idx < a.next_cap) a.parent[idx] = a.level_base + parent;
#endif
      int inv = ty_gpu_invariants_ok(t);
      if (inv < 0) { atomicExch(a.error_flag, (u64)(-inv)); return; }
      if (inv == 0) {
        if (atomicCAS(a.violation, 0, 1) == 0) {
          for (int i = 0; i < SLOTS; i++) a.violation_row[i] = t[i];
#if TY_GPU_TRACE
          a.violation_index[0] = idx;
#endif
          __threadfence_system();
          atomicExch(a.violation, 2); // 2 = row published
        }
      }
      if (idx < a.next_cap) {
        i64* dst = a.next_rows + idx * SLOTS;
        #pragma unroll
        for (int i = 0; i < SLOTS; i++) dst[i] = t[i];
      }
      // idx >= cap: host detects overflow via next_count > next_cap and
      // aborts the run (fail-closed) — no partial-level results are used.
    }
  }
  if (local_trans) atomicAdd(a.transitions, local_trans);
}

// Count parents whose enabled flag stayed zero (deadlocked states).
extern "C" __global__ void ty_gpu_count_deadlocks(TyGpuLevelArgs a) {
  u64 local_deadlocks = 0;
  for (u64 gi = blockIdx.x * (u64)blockDim.x + threadIdx.x; gi < a.frontier_len;
       gi += gridDim.x * (u64)blockDim.x) {
    if (!a.enabled_flags[gi]) local_deadlocks++;
  }
  if (local_deadlocks) atomicAdd(a.deadlocks, local_deadlocks);
}

// Seed the fingerprint table with the initial states (single thread; init sets
// are tiny). Initial states are checked against the invariants too.
extern "C" __global__ void ty_gpu_seed(TyGpuLevelArgs a) {
  for (u64 r = 0; r < a.frontier_len; r++) {
    const i64* row = a.frontier + r * SLOTS;
    // Drop constraint-failing initial states before they enter the reachable
    // set (matches the CPU check_init_state returning None). Host init
    // enumeration may already prune these; the gate is idempotent.
    int cok = ty_gpu_constraint_ok(row);
    if (cok < 0) { atomicExch(a.error_flag, (u64)(-cok)); return; }
    if (cok == 0) continue;
    u64 hi, lo;
    ty_gpu_fp128(row, &hi, &lo);
    int inserted = ty_gpu_fp_insert(a.fp_lo, a.fp_hi, a.fp_mask, hi, lo);
    if (inserted < 0) { atomicExch(a.error_flag, 92ULL); return; }
    if (inserted > 0) {
#if TY_GPU_TRACK_SLOT_STATS
      ty_gpu_track_slot_stats(row, a.max_slot, a.max_slot_sum, a.slot_maxima, a.slot_minima);
#endif
      u64 idx = atomicAdd(a.next_count, 1ULL);
#if TY_GPU_TRACE
      if (idx < a.next_cap) a.parent[idx] = ~0ULL; // initial state: no parent
#endif
      int inv = ty_gpu_invariants_ok(row);
      if (inv < 0) { atomicExch(a.error_flag, (u64)(-inv)); return; }
      if (inv == 0) {
        if (atomicCAS(a.violation, 0, 1) == 0) {
          for (int i = 0; i < SLOTS; i++) a.violation_row[i] = row[i];
#if TY_GPU_TRACE
          a.violation_index[0] = idx;
#endif
          __threadfence_system();
          atomicExch(a.violation, 2);
        }
      }
      i64* dst = a.next_rows + idx * SLOTS;
      for (int i = 0; i < SLOTS; i++) dst[i] = row[i];
    }
  }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_probe_is_bounded_and_saturation_is_reported() {
        assert!(ENGINE_DRIVER.contains("for (u64 probe = 0; probe <= mask; probe++)"));
        assert!(ENGINE_DRIVER.contains("return -1;"));
        assert_eq!(
            ENGINE_DRIVER
                .matches("atomicExch(a.error_flag, 92ULL)")
                .count(),
            2
        );
        assert!(!ENGINE_DRIVER.contains("while (true)"));
    }
}
