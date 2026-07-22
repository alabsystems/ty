//! Device-resident, level-synchronous BFS driver.
//!
//! The full frontier, successor arena, and fingerprint table live in GPU
//! memory; the host only launches one expansion kernel per BFS level and reads
//! back a few managed counters. On unified-memory parts (GB10 class) counter
//! readback is a cache-coherent load, not a copy.

use std::ffi::c_void;
use std::time::{Duration, Instant};

use crate::cuda::{
    check, checked_allocation_bytes, checked_allocation_usize, checked_power_of_two_u64, cuda_api,
    try_reserve_host, try_zeroed_host_vec, CuDeviceptr, CuFunction, CuModule, CudaApi,
    DeviceBuffer,
};
use crate::kernel_template::assemble_engine_source;
use crate::{GpuError, GpuInfo};

/// Spec-side inputs to the GPU engine: the generated action/invariant CUDA
/// source and the enumerated initial states, over a fixed flat slot layout.
pub struct GpuBfsSpec {
    /// i64 slots per state (`StateLayout::total_slots`).
    pub slots: usize,
    /// Number of `ty_gpu_action_<k>` device functions in `actions_src`.
    pub action_count: usize,
    /// CUDA C source defining the per-action functions and
    /// `ty_gpu_invariants_ok`. See `kernel_template` for the contract.
    pub actions_src: String,
    /// Row-major initial states (`len % slots == 0`). May contain duplicates;
    /// the engine dedups and invariant-checks them.
    pub init_rows: Vec<i64>,
    /// Track per-state slot statistics (max slot value / max per-state slot
    /// sum) across all distinct states — the Petri StateSpace metrics.
    pub track_slot_stats: bool,
}

/// Engine tuning knobs. `Default` is sized for ~100M-state runs.
pub struct GpuBfsConfig {
    /// log2 of fingerprint-table slots. Two u64 lanes per slot.
    pub table_bits: u32,
    /// Successor arena capacity in rows (also frontier capacity).
    pub frontier_cap_rows: u64,
    /// CUDA block size for the expansion kernel.
    pub block: u32,
    /// Abort when the fingerprint table passes this load factor (fail-closed;
    /// probe chains degrade and CAS storms distort results beyond it).
    pub max_load_factor: f64,
    /// Fail-closed cap on distinct states (`CapacityExceeded` with
    /// `what = "distinct-state cap"` past it). Callers with a configured
    /// exploration bound set it so an unbounded/oversized space declines at
    /// the bound instead of grinding to the table capacity. `u64::MAX` = no
    /// cap beyond the table itself.
    pub max_distinct: u64,
    /// When set, retain the reachable set in a monotone arena with per-state
    /// parent pointers so an invariant violation yields the full init->bad
    /// counterexample path ([`GpuBfsOutcome::violation_trace`]) on-device
    /// instead of only the violating row. Costs one extra u64/state plus the
    /// whole arena resident (no level ping-pong); a violation stops the search
    /// early so the arena stays shallow in the case that produces a trace.
    /// Off by default (pure counting keeps the ping-pong fast path).
    pub trace_on_violation: bool,
}

impl Default for GpuBfsConfig {
    fn default() -> Self {
        GpuBfsConfig {
            table_bits: 27,
            // Measured best on the GB10 for the register-heavy expand kernel
            // (dijkstra-4: ~16% faster than 256); auto-clamped down per kernel
            // to the driver-reported max-threads-per-block in `run_bfs`.
            block: 512,
            frontier_cap_rows: 32 << 20,
            max_load_factor: 0.7,
            max_distinct: u64::MAX,
            trace_on_violation: false,
        }
    }
}

/// Result of an exhaustive (or violation-terminated) GPU BFS.
#[derive(Debug)]
pub struct GpuBfsOutcome {
    /// Distinct states discovered (exact modulo 128-bit fingerprint collisions,
    /// the same collision model the CPU engine reports).
    pub distinct_states: u64,
    /// Enabled action firings observed (candidate successors, pre-dedup).
    pub transitions: u64,
    /// BFS levels executed (diameter + 1).
    pub levels: u64,
    /// States with zero enabled successor actions.
    pub deadlock_states: u64,
    /// First invariant-violating state row, if any (verdict only; the CPU
    /// engine owns counterexample-trace construction).
    pub violation: Option<Vec<i64>>,
    /// Full init->violation counterexample path (state rows, first = an initial
    /// state, last = the violating state), reconstructed on-device by walking
    /// parent pointers. `Some` only when [`GpuBfsConfig::trace_on_violation`]
    /// was set AND a violation was found; `None` otherwise.
    pub violation_trace: Option<Vec<Vec<i64>>>,
    /// Wall time of the device search (excludes nvrtc compile).
    pub wall: Duration,
    /// nvrtc + module-load time.
    pub compile_wall: Duration,
    /// Max slot value over all distinct states (when `track_slot_stats`).
    pub max_slot_value: u64,
    /// Max per-state slot sum over all distinct states (when `track_slot_stats`).
    pub max_slot_sum: u64,
    /// Per-slot maxima over all distinct states (when `track_slot_stats`;
    /// empty otherwise). `slots` entries — the Petri UpperBounds/OneSafe
    /// per-place token maxima.
    pub slot_maxima: Vec<u64>,
    /// Per-slot minima over all distinct states (when `track_slot_stats`;
    /// empty otherwise). `slots` entries — with the maxima, the Petri
    /// StableMarking constancy carrier (`min == max` ⟺ the slot never
    /// changes).
    pub slot_minima: Vec<u64>,
}

struct Engine<'a> {
    api: &'a CudaApi,
    module: CuModule,
    expand: CuFunction,
    seed: CuFunction,
    count_deadlocks: CuFunction,
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.cuModuleUnload)(self.module) };
    }
}

/// Mirror of the device-side `TyGpuLevelArgs` (all fields 8-byte).
#[repr(C)]
struct LevelArgs {
    frontier: CuDeviceptr,
    frontier_len: u64,
    fp_lo: CuDeviceptr,
    fp_hi: CuDeviceptr,
    fp_mask: u64,
    next_rows: CuDeviceptr,
    next_cap: u64,
    next_count: CuDeviceptr,
    transitions: CuDeviceptr,
    deadlocks: CuDeviceptr,
    error_flag: CuDeviceptr,
    max_slot: CuDeviceptr,
    max_slot_sum: CuDeviceptr,
    slot_maxima: CuDeviceptr,
    slot_minima: CuDeviceptr,
    enabled_flags: CuDeviceptr,
    violation: CuDeviceptr,
    violation_row: CuDeviceptr,
    level_base: u64,
    parent: CuDeviceptr,
    violation_index: CuDeviceptr,
}

/// Probe CUDA availability and describe device 0.
pub fn probe() -> Result<GpuInfo, GpuError> {
    let api = cuda_api()?;
    let mut count = 0;
    check(
        api,
        unsafe { (api.cuDeviceGetCount)(&mut count) },
        "cuDeviceGetCount",
    )?;
    if count == 0 {
        return Err(GpuError::Unavailable("no CUDA devices".into()));
    }
    let mut dev = 0;
    check(
        api,
        unsafe { (api.cuDeviceGet)(&mut dev, 0) },
        "cuDeviceGet",
    )?;
    let mut name = [0 as std::ffi::c_char; 128];
    check(
        api,
        unsafe { (api.cuDeviceGetName)(name.as_mut_ptr(), 128, dev) },
        "cuDeviceGetName",
    )?;
    let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let mut cc_major = 0;
    let mut cc_minor = 0;
    let mut sms = 0;
    check(
        api,
        unsafe {
            (api.cuDeviceGetAttribute)(
                &mut cc_major,
                crate::cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            )
        },
        "cc major",
    )?;
    check(
        api,
        unsafe {
            (api.cuDeviceGetAttribute)(
                &mut cc_minor,
                crate::cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            )
        },
        "cc minor",
    )?;
    check(
        api,
        unsafe {
            (api.cuDeviceGetAttribute)(
                &mut sms,
                crate::cuda::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                dev,
            )
        },
        "sm count",
    )?;
    Ok(GpuInfo {
        device_name: name,
        cc_major,
        cc_minor,
        multiprocessors: sms,
    })
}

fn activate_context(api: &CudaApi) -> Result<(), GpuError> {
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
    Ok(())
}

fn load_engine<'a>(
    api: &'a CudaApi,
    spec: &GpuBfsSpec,
    info: &GpuInfo,
    trace: bool,
) -> Result<Engine<'a>, GpuError> {
    let source = assemble_engine_source(
        spec.slots,
        spec.action_count,
        &spec.actions_src,
        spec.track_slot_stats,
        trace,
    );
    if let Ok(path) = std::env::var("TY_GPU_DUMP_ASSEMBLED") {
        let _ = std::fs::write(path, &source);
    }
    let ptx = crate::nvrtc::compile_to_ptx(&source, info.cc_major, info.cc_minor)?;
    let mut module = std::ptr::null_mut();
    check(
        api,
        unsafe { (api.cuModuleLoadData)(&mut module, ptx.as_ptr().cast::<c_void>()) },
        "cuModuleLoadData",
    )?;
    let expand_name = std::ffi::CString::new("ty_gpu_expand_level").expect("static");
    let seed_name = std::ffi::CString::new("ty_gpu_seed").expect("static");
    let count_name = std::ffi::CString::new("ty_gpu_count_deadlocks").expect("static");
    let mut expand = std::ptr::null_mut();
    let mut seed = std::ptr::null_mut();
    let mut count_deadlocks = std::ptr::null_mut();
    let rc1 = unsafe { (api.cuModuleGetFunction)(&mut expand, module, expand_name.as_ptr()) };
    let rc2 = unsafe { (api.cuModuleGetFunction)(&mut seed, module, seed_name.as_ptr()) };
    let rc3 =
        unsafe { (api.cuModuleGetFunction)(&mut count_deadlocks, module, count_name.as_ptr()) };
    if rc1 != crate::cuda::CUDA_SUCCESS
        || rc2 != crate::cuda::CUDA_SUCCESS
        || rc3 != crate::cuda::CUDA_SUCCESS
    {
        unsafe { (api.cuModuleUnload)(module) };
        return Err(GpuError::Driver(
            "engine kernel symbols missing from module".into(),
        ));
    }
    Ok(Engine {
        api,
        module,
        expand,
        seed,
        count_deadlocks,
    })
}

fn launch(
    api: &CudaApi,
    func: CuFunction,
    grid: u32,
    block: u32,
    args: &mut LevelArgs,
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

/// Run an exhaustive device-resident BFS for the spec.
///
/// Fail-closed: any capacity overflow, load-factor breach, or driver error
/// aborts with an error — no partial result is ever reported as a verdict.
pub fn run_bfs(spec: &GpuBfsSpec, config: &GpuBfsConfig) -> Result<GpuBfsOutcome, GpuError> {
    if spec.slots == 0 || spec.init_rows.is_empty() || spec.init_rows.len() % spec.slots != 0 {
        return Err(GpuError::Codegen("malformed init rows".into()));
    }
    if spec.action_count == 0 {
        return Err(GpuError::Codegen(
            "GPU BFS requires at least one action".into(),
        ));
    }
    if config.frontier_cap_rows == 0 {
        return Err(GpuError::Codegen(
            "GPU BFS frontier capacity is zero".into(),
        ));
    }
    if !config.max_load_factor.is_finite()
        || !(0.0..1.0).contains(&config.max_load_factor)
        || config.max_load_factor == 0.0
    {
        return Err(GpuError::Codegen(
            "GPU BFS max_load_factor must be finite and in (0, 1)".into(),
        ));
    }
    let table_slots = checked_power_of_two_u64("BFS fingerprint table slots", config.table_bits)?;
    // Occupancy-tuning override (measurement lever; the default is set from the
    // best measured value). Clamped to a valid CUDA block range; falls back to
    // the configured block otherwise. The final block is additionally clamped
    // to the kernel's driver-reported max-threads-per-block below.
    let desired_block = std::env::var("TY_GPU_BLOCK")
        .ok()
        .and_then(|b| b.parse::<u32>().ok())
        .filter(|&b| (32..=1024).contains(&b) && b % 32 == 0)
        .unwrap_or(config.block);
    let api = cuda_api()?;
    let info = probe()?;
    activate_context(api)?;

    let compile_start = Instant::now();
    let trace = config.trace_on_violation;
    let engine = load_engine(api, spec, &info, trace)?;
    let compile_wall = compile_start.elapsed();

    // Clamp the block to the expand kernel's driver-reported max-threads-per-
    // block: a register-heavy generated kernel may not launch at the desired
    // width, so the clamp keeps a large default fast where it fits and falls
    // back safely (rounded down to a warp multiple) where it does not.
    let block = {
        let mut kmax: std::ffi::c_int = desired_block as std::ffi::c_int;
        let _ = unsafe {
            (api.cuFuncGetAttribute)(
                &mut kmax,
                crate::cuda::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                engine.expand,
            )
        };
        let kmax = if kmax >= 32 {
            (kmax as u32 / 32) * 32
        } else {
            32
        };
        desired_block.min(kmax).max(32)
    };

    let table_slots_usize = checked_allocation_usize("BFS fingerprint table slots", table_slots)?;
    let table_bytes =
        checked_allocation_bytes("BFS fingerprint table bytes", &[table_slots_usize, 8])?;
    let row_bytes = checked_allocation_bytes("BFS state row bytes", &[spec.slots, 8])?;
    let action_count = u64::try_from(spec.action_count)
        .map_err(|_| GpuError::AllocationOverflow("BFS action count"))?;
    // Trace mode retains every state in one monotone arena instead of the
    // two-buffer ping-pong, so give that single arena the combined capacity
    // (2x rows for the same total bytes) and add a per-state parent array.
    let arena_rows = if trace {
        config
            .frontier_cap_rows
            .checked_mul(2)
            .ok_or(GpuError::AllocationOverflow("BFS trace arena rows"))?
    } else {
        config.frontier_cap_rows
    };
    let arena_rows_usize = checked_allocation_usize("BFS arena rows", arena_rows)?;
    let arena_bytes = checked_allocation_bytes("BFS arena bytes", &[arena_rows_usize, row_bytes])?;
    let init_bytes =
        checked_allocation_bytes("BFS initial-state bytes", &[spec.init_rows.len(), 8])?;
    let parent_bytes = if trace {
        checked_allocation_bytes("BFS parent bytes", &[arena_rows_usize, 8])?
    } else {
        8
    };
    let slot_stats_bytes = checked_allocation_bytes("BFS slot-stat bytes", &[spec.slots, 8])?;

    let fp_lo = DeviceBuffer::device(api, table_bytes)?;
    let fp_hi = DeviceBuffer::device(api, table_bytes)?;
    let mut arena_a = DeviceBuffer::device(api, arena_bytes)?;
    // Ping-pong needs a second full arena; trace mode reuses `arena_a`
    // monotonically and uses `arena_b` only as the tiny init-staging buffer
    // the seed reads from (avoids a read-after-write hazard on the arena).
    let mut arena_b = DeviceBuffer::device(
        api,
        if trace {
            init_bytes.max(8)
        } else {
            arena_bytes
        },
    )?;
    let parent = DeviceBuffer::device(api, parent_bytes)?;
    let next_count = DeviceBuffer::managed(api, 8)?;
    let transitions = DeviceBuffer::managed(api, 8)?;
    let deadlocks = DeviceBuffer::managed(api, 8)?;
    let error_flag = DeviceBuffer::managed(api, 8)?;
    let max_slot = DeviceBuffer::managed(api, 8)?;
    let max_slot_sum = DeviceBuffer::managed(api, 8)?;
    let slot_maxima = DeviceBuffer::managed(api, slot_stats_bytes)?;
    let slot_minima = DeviceBuffer::managed(api, slot_stats_bytes)?;
    let enabled_flags = DeviceBuffer::device(api, arena_rows_usize)?;
    let violation = DeviceBuffer::managed(api, 8)?;
    let violation_row = DeviceBuffer::managed(api, row_bytes)?;
    let violation_index = DeviceBuffer::managed(api, 8)?;

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
        write_u64(&transitions, 0);
        write_u64(&deadlocks, 0);
        write_u64(&error_flag, 0);
        write_u64(&max_slot, 0);
        write_u64(&max_slot_sum, 0);
        write_u64(&violation, 0);
        std::ptr::write_bytes(slot_maxima.ptr as *mut u8, 0, slot_stats_bytes);
        std::ptr::write_bytes(slot_minima.ptr as *mut u8, 0xFF, slot_stats_bytes);
    }

    let init_count = (spec.init_rows.len() / spec.slots) as u64;
    if init_count > config.frontier_cap_rows {
        return Err(GpuError::CapacityExceeded {
            what: "initial states",
            needed: init_count,
            capacity: config.frontier_cap_rows,
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
    // Trace mode stages init in arena_b and seeds into the monotone arena_a;
    // ping-pong stages in arena_a and seeds into arena_b (swapped below).
    let (init_target, seed_next) = if trace {
        (arena_b.ptr, arena_a.ptr)
    } else {
        (arena_a.ptr, arena_b.ptr)
    };
    check(
        api,
        unsafe {
            (api.cuMemcpyHtoD_v2)(
                init_target,
                spec.init_rows.as_ptr().cast::<c_void>(),
                init_bytes,
            )
        },
        "upload init rows",
    )?;

    let search_start = Instant::now();
    let mut args = LevelArgs {
        frontier: init_target,
        frontier_len: init_count,
        fp_lo: fp_lo.ptr,
        fp_hi: fp_hi.ptr,
        fp_mask: table_slots - 1,
        next_rows: seed_next,
        next_cap: arena_rows,
        next_count: next_count.ptr,
        transitions: transitions.ptr,
        deadlocks: deadlocks.ptr,
        error_flag: error_flag.ptr,
        max_slot: max_slot.ptr,
        max_slot_sum: max_slot_sum.ptr,
        slot_maxima: slot_maxima.ptr,
        slot_minima: slot_minima.ptr,
        enabled_flags: enabled_flags.ptr,
        violation: violation.ptr,
        violation_row: violation_row.ptr,
        level_base: 0,
        parent: parent.ptr,
        violation_index: violation_index.ptr,
    };
    launch(api, engine.seed, 1, 1, &mut args)?;
    if unsafe { read_u64(&error_flag) } != 0 {
        return Err(GpuError::Driver(format!(
            "GPU kernel signaled runtime error (JitStatus {}) while seeding initial states",
            unsafe { read_u64(&error_flag) }
        )));
    }

    let mut frontier_len = unsafe { read_u64(&next_count) };
    let mut distinct = frontier_len;
    if distinct > config.max_distinct {
        return Err(GpuError::CapacityExceeded {
            what: "distinct-state cap",
            needed: distinct,
            capacity: config.max_distinct,
        });
    }
    let mut levels: u64 = 0;
    let row_bytes_u = u64::try_from(row_bytes)
        .map_err(|_| GpuError::AllocationOverflow("BFS state row bytes"))?;
    // Trace mode retains everything in `arena_a`; `level_base`/`arena_len`
    // window the current frontier and the append point. Ping-pong keeps the
    // original two-buffer swap. Both share the level body via a closure.
    let mut level_base: u64 = 0;
    let mut arena_len: u64 = distinct; // trace: init states occupy arena_a[0..distinct]
    if !trace {
        // Seed wrote deduped init rows into arena_b; expansion starts there.
        std::mem::swap(&mut arena_a, &mut arena_b);
    }

    loop {
        if unsafe { read_u64(&violation) } == 2 {
            break; // verdict decided; stop expanding
        }
        // Set up this level's frontier / append windows for the active mode.
        let (frontier_ptr, next_ptr, next_cap, lvl_base, parent_ptr) = if trace {
            if level_base >= arena_len {
                break;
            }
            frontier_len = arena_len - level_base;
            let child_base = arena_len;
            (
                arena_a.ptr + level_base * row_bytes_u,
                arena_a.ptr + child_base * row_bytes_u,
                arena_rows - child_base,
                level_base,
                parent.ptr + child_base * 8,
            )
        } else {
            if frontier_len == 0 {
                break;
            }
            (
                arena_a.ptr,
                arena_b.ptr,
                config.frontier_cap_rows,
                0,
                parent.ptr,
            )
        };
        unsafe { write_u64(&next_count, 0) };
        check(
            api,
            unsafe {
                (api.cuMemsetD8_v2)(
                    enabled_flags.ptr,
                    0,
                    checked_allocation_usize("BFS frontier length", frontier_len)?,
                )
            },
            "clear enabled flags",
        )?;
        let mut args = LevelArgs {
            frontier: frontier_ptr,
            frontier_len,
            fp_lo: fp_lo.ptr,
            fp_hi: fp_hi.ptr,
            fp_mask: table_slots - 1,
            next_rows: next_ptr,
            next_cap,
            next_count: next_count.ptr,
            transitions: transitions.ptr,
            deadlocks: deadlocks.ptr,
            error_flag: error_flag.ptr,
            max_slot: max_slot.ptr,
            max_slot_sum: max_slot_sum.ptr,
            slot_maxima: slot_maxima.ptr,
            slot_minima: slot_minima.ptr,
            enabled_flags: enabled_flags.ptr,
            violation: violation.ptr,
            violation_row: violation_row.ptr,
            level_base: lvl_base,
            parent: parent_ptr,
            violation_index: violation_index.ptr,
        };
        let pairs = frontier_len
            .checked_mul(action_count)
            .ok_or(GpuError::AllocationOverflow("BFS frontier/action pairs"))?;
        let blocks_wanted = pairs.div_ceil(u64::from(block));
        let grid = u32::try_from(blocks_wanted.min(65_535)).expect("bounded");
        launch(api, engine.expand, grid, block, &mut args)?;
        let dl_blocks =
            u32::try_from(frontier_len.div_ceil(u64::from(block)).min(65_535)).expect("bounded");
        launch(api, engine.count_deadlocks, dl_blocks, block, &mut args)?;

        let err = unsafe { read_u64(&error_flag) };
        if err != 0 {
            return Err(GpuError::Driver(format!(
                "GPU action kernel signaled runtime error / fallback (JitStatus {err}); \
                 no verdict reported — rerun on the CPU engine"
            )));
        }
        let produced = unsafe { read_u64(&next_count) };
        let cap = if trace {
            next_cap
        } else {
            config.frontier_cap_rows
        };
        if produced > cap {
            return Err(GpuError::CapacityExceeded {
                what: "successor arena",
                needed: produced,
                capacity: cap,
            });
        }
        distinct = distinct
            .checked_add(produced)
            .ok_or(GpuError::AllocationOverflow("BFS distinct-state count"))?;
        if distinct > config.max_distinct {
            return Err(GpuError::CapacityExceeded {
                what: "distinct-state cap",
                needed: distinct,
                capacity: config.max_distinct,
            });
        }
        if distinct > fingerprint_capacity {
            return Err(GpuError::CapacityExceeded {
                what: "fingerprint table",
                needed: distinct,
                capacity: fingerprint_capacity,
            });
        }
        if trace {
            level_base = arena_len;
            arena_len = arena_len
                .checked_add(produced)
                .ok_or(GpuError::AllocationOverflow("BFS trace arena length"))?;
        } else {
            std::mem::swap(&mut arena_a, &mut arena_b);
            frontier_len = produced;
        }
        levels += 1;
    }
    let wall = search_start.elapsed();

    let violated = unsafe { read_u64(&violation) } >= 1;
    let violation = if violated {
        let mut row = try_zeroed_host_vec(spec.slots, "BFS violation row")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                violation_row.ptr as *const i64,
                row.as_mut_ptr(),
                spec.slots,
            );
        }
        Some(row)
    } else {
        None
    };

    // Trace reconstruction: walk parent pointers from the violating state back
    // to an initial state (parent == u64::MAX), then read those rows out.
    let violation_trace = if trace && violated {
        // The kernel records the violating child's LEVEL-LOCAL index (next_count
        // resets per level). At break, `level_base` is that level's arena base
        // (the loop advances level_base = child_base after each expand, then the
        // next iteration breaks on violation==2), so the global arena index is
        // level_base + local. For a seed-level violation level_base is 0.
        let vidx = level_base
            .checked_add(unsafe { read_u64(&violation_index) })
            .ok_or(GpuError::AllocationOverflow("BFS violation index"))?;
        if vidx < distinct {
            let n = checked_allocation_usize("BFS trace state count", distinct)?;
            // Read only the parent chain. Copying the entire `distinct`-entry
            // parent arena duplicated up to hundreds of MiB in host memory on
            // an integrated-memory machine precisely when memory was tight.
            let mut path: Vec<u64> = Vec::new();
            let mut cur = vidx;
            loop {
                try_reserve_host(&mut path, 1, "BFS trace indices")?;
                path.push(cur);
                if path.len() > n {
                    break; // cycle guard (BFS-by-level parents cannot cycle)
                }
                let parent_offset = cur
                    .checked_mul(8)
                    .ok_or(GpuError::AllocationOverflow("BFS parent offset"))?;
                let parent_ptr = parent
                    .ptr
                    .checked_add(parent_offset)
                    .ok_or(GpuError::AllocationOverflow("BFS parent pointer"))?;
                let mut p = 0u64;
                check(
                    api,
                    unsafe {
                        (api.cuMemcpyDtoH_v2)(
                            std::ptr::from_mut(&mut p).cast::<c_void>(),
                            parent_ptr,
                            8,
                        )
                    },
                    "copy trace parent",
                )?;
                if p == u64::MAX || p >= distinct {
                    break; // reached an initial state (or a safety bound)
                }
                cur = p;
            }
            path.reverse(); // init -> ... -> violating state
            let mut rows = Vec::new();
            try_reserve_host(&mut rows, path.len(), "BFS trace rows")?;
            for &idx in &path {
                let mut row: Vec<i64> = try_zeroed_host_vec(spec.slots, "BFS trace row")?;
                check(
                    api,
                    unsafe {
                        (api.cuMemcpyDtoH_v2)(
                            row.as_mut_ptr().cast::<c_void>(),
                            arena_a.ptr + idx * row_bytes_u,
                            row_bytes,
                        )
                    },
                    "copy trace row",
                )?;
                rows.push(row);
            }
            Some(rows)
        } else {
            None
        }
    } else {
        None
    };

    let (slot_maxima_out, slot_minima_out) = if spec.track_slot_stats {
        let mut maxima = try_zeroed_host_vec(spec.slots, "BFS slot maxima")?;
        let mut minima = try_zeroed_host_vec(spec.slots, "BFS slot minima")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                slot_maxima.ptr as *const u64,
                maxima.as_mut_ptr(),
                spec.slots,
            );
            std::ptr::copy_nonoverlapping(
                slot_minima.ptr as *const u64,
                minima.as_mut_ptr(),
                spec.slots,
            );
        }
        (maxima, minima)
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(GpuBfsOutcome {
        distinct_states: distinct,
        transitions: unsafe { read_u64(&transitions) },
        levels,
        deadlock_states: unsafe { read_u64(&deadlocks) },
        violation,
        violation_trace,
        wall,
        compile_wall,
        max_slot_value: unsafe { read_u64(&max_slot) },
        max_slot_sum: unsafe { read_u64(&max_slot_sum) },
        slot_maxima: slot_maxima_out,
        slot_minima: slot_minima_out,
    })
}
