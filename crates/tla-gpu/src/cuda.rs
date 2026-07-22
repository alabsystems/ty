//! Minimal CUDA driver-API binding, loaded at runtime via `dlopen`.
//!
//! Only the handful of entry points the BFS driver needs. The `_v2` symbol
//! variants are the current 64-bit ABI (the unsuffixed exports are the legacy
//! 32-bit-`CUdeviceptr` ABI and must not be used).

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
#[cfg(unix)]
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::sync::{Mutex, OnceLock};

use crate::GpuError;

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuDeviceptr = u64;

pub const CUDA_SUCCESS: CuResult = 0;
pub const CU_MEM_ATTACH_GLOBAL: c_uint = 1;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
/// Max threads per block this specific kernel supports given its register /
/// local-memory footprint (`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`).
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;

macro_rules! cuda_api {
    ($( fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) ; )*) => {
        /// Resolved driver-API function pointers.
        ///
        /// Field names mirror the CUDA symbol names exactly (they double as
        /// the `dlsym` lookup keys), hence the non-snake-case allowance.
        #[allow(non_snake_case, dead_code)]
        pub struct CudaApi {
            _lib: libloading::Library,
            $( pub $name: unsafe extern "C" fn($($ty),*) -> CuResult, )*
        }

        impl CudaApi {
            #[allow(non_snake_case)]
            fn load_from(lib: libloading::Library) -> Result<Self, GpuError> {
                unsafe {
                    $(
                        let $name = *lib
                            .get::<unsafe extern "C" fn($($ty),*) -> CuResult>(
                                concat!(stringify!($name), "\0").as_bytes(),
                            )
                            .map_err(|e| GpuError::Unavailable(format!(
                                "libcuda missing symbol {}: {e}", stringify!($name)
                            )))?;
                    )*
                    Ok(CudaApi { $( $name, )* _lib: lib })
                }
            }
        }
    };
}

cuda_api! {
    fn cuInit(flags: c_uint);
    fn cuDeviceGetCount(count: *mut c_int);
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int);
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CuDevice);
    fn cuDeviceGetAttribute(value: *mut c_int, attrib: c_int, dev: CuDevice);
    fn cuDevicePrimaryCtxRetain(ctx: *mut CuContext, dev: CuDevice);
    fn cuCtxSetCurrent(ctx: CuContext);
    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void);
    fn cuModuleUnload(module: CuModule);
    fn cuModuleGetFunction(func: *mut CuFunction, module: CuModule, name: *const c_char);
    fn cuFuncGetAttribute(pi: *mut c_int, attrib: c_int, func: CuFunction);
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize);
    fn cuMemAlloc_v2(dptr: *mut CuDeviceptr, bytes: usize);
    fn cuMemAllocManaged(dptr: *mut CuDeviceptr, bytes: usize, flags: c_uint);
    fn cuMemFree_v2(dptr: CuDeviceptr);
    fn cuMemsetD8_v2(dptr: CuDeviceptr, value: u8, count: usize);
    fn cuMemcpyHtoD_v2(dst: CuDeviceptr, src: *const c_void, bytes: usize);
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CuDeviceptr, bytes: usize);
    fn cuLaunchKernel(
        func: CuFunction,
        grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
        block_x: c_uint, block_y: c_uint, block_z: c_uint,
        shared_bytes: c_uint, stream: CuStream,
        params: *mut *mut c_void, extra: *mut *mut c_void,
    );
    fn cuCtxSynchronize();
    fn cuGetErrorString(error: CuResult, str_out: *mut *const c_char);
}

// DGX Spark exposes GPU allocations from the same physical 128-GiB pool used
// by Linux.  CUDA device allocations are not charged to a process's ordinary
// RSS/cgroup memory counter, so letting one process consume the driver's full
// reported capacity can trigger a global host OOM before the allocator fails.
// Keep every tla-gpu process on an integrated device to a deliberately
// conservative share and retain a separate free-memory reserve for the
// desktop, kernel, and concurrent jobs. 1/8 admits the common ~11-GiB trace
// BFS on a 128-GiB Spark while still limiting one process to 16 GiB. Discrete
// devices use a smaller device-local reserve without the UMA process cap.
const CUDA_PROCESS_BUDGET_DIVISOR: usize = 8;
const CUDA_FREE_RESERVE_DIVISOR: usize = 8;
const CUDA_MIN_FREE_RESERVE_BYTES: usize = 1 << 30;
const DISCRETE_FREE_RESERVE_DIVISOR: usize = 16;
const DISCRETE_MIN_FREE_RESERVE_BYTES: usize = 256 << 20;

/// Device + managed bytes currently owned by this process through
/// [`DeviceBuffer`].  A mutex makes the capacity check and CUDA allocation one
/// transaction across Rust test threads.
static CUDA_LIVE_ALLOCATED_BYTES: Mutex<usize> = Mutex::new(0);

static CUDA: OnceLock<Result<CudaApi, GpuError>> = OnceLock::new();

/// Load (once per process) and return the driver API, or why it is unavailable.
pub fn cuda_api() -> Result<&'static CudaApi, GpuError> {
    CUDA.get_or_init(|| {
        let lib = ["libcuda.so.1", "libcuda.so"]
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name).ok() })
            .ok_or_else(|| GpuError::Unavailable("libcuda not found".into()))?;
        let api = CudaApi::load_from(lib)?;
        let rc = unsafe { (api.cuInit)(0) };
        if rc != CUDA_SUCCESS {
            return Err(GpuError::Unavailable(format!(
                "cuInit failed: {}",
                error_name(&api, rc)
            )));
        }
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

pub fn error_name(api: &CudaApi, rc: CuResult) -> String {
    let mut ptr: *const c_char = std::ptr::null();
    let get = unsafe { (api.cuGetErrorString)(rc, &mut ptr) };
    if get == CUDA_SUCCESS && !ptr.is_null() {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    } else {
        format!("CUDA error {rc}")
    }
}

/// Convert a driver-API return code into a `Result`.
pub fn check(api: &CudaApi, rc: CuResult, what: &str) -> Result<(), GpuError> {
    if rc == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::Driver(format!("{what}: {}", error_name(api, rc))))
    }
}

fn as_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Multiply byte-layout factors without permitting a wrapped, undersized GPU
/// allocation. Callers fail closed through [`GpuError::AllocationOverflow`].
pub(crate) fn checked_allocation_bytes(
    what: &'static str,
    factors: &[usize],
) -> Result<usize, GpuError> {
    factors.iter().copied().try_fold(1usize, |bytes, factor| {
        bytes
            .checked_mul(factor)
            .ok_or(GpuError::AllocationOverflow(what))
    })
}

/// Add independently-computed allocation components without wrapping.
pub(crate) fn checked_allocation_sum(
    what: &'static str,
    parts: &[usize],
) -> Result<usize, GpuError> {
    parts.iter().copied().try_fold(0usize, |bytes, part| {
        bytes
            .checked_add(part)
            .ok_or(GpuError::AllocationOverflow(what))
    })
}

/// Convert a device-side count into a host allocation size without truncation.
pub(crate) fn checked_allocation_usize(what: &'static str, value: u64) -> Result<usize, GpuError> {
    usize::try_from(value).map_err(|_| GpuError::AllocationOverflow(what))
}

/// Compute a power-of-two table size without a wrapped release-mode shift.
pub(crate) fn checked_power_of_two_u64(what: &'static str, bits: u32) -> Result<u64, GpuError> {
    1u64.checked_shl(bits)
        .ok_or(GpuError::AllocationOverflow(what))
}

/// Reserve host vector storage through the fallible API. This avoids an
/// allocator abort for large trace/result buffers after a GPU search.
pub(crate) fn try_reserve_host<T>(
    values: &mut Vec<T>,
    additional: usize,
    what: &'static str,
) -> Result<(), GpuError> {
    let final_len = values
        .len()
        .checked_add(additional)
        .ok_or(GpuError::AllocationOverflow(what))?;
    let bytes = checked_allocation_bytes(what, &[final_len, std::mem::size_of::<T>()])?;
    values
        .try_reserve(additional)
        .map_err(|_| GpuError::HostAllocationFailed {
            what,
            bytes: as_u64_saturating(bytes),
        })
}

/// Build a zero/default-filled host vector without an infallible allocation.
pub(crate) fn try_zeroed_host_vec<T: Default>(
    len: usize,
    what: &'static str,
) -> Result<Vec<T>, GpuError> {
    let mut values = Vec::new();
    try_reserve_host(&mut values, len, what)?;
    values.resize_with(len, T::default);
    Ok(values)
}

/// Pure policy: integrated devices get a 1/8 per-process cap and retain 1/8
/// of shared host/device memory; discrete devices only retain a 1/16 VRAM
/// reserve because their allocations cannot invoke the host OOM killer.
fn allocation_limit_from_info(
    cuda_free: usize,
    total: usize,
    shared_host_available: Option<usize>,
    live: usize,
    integrated: bool,
) -> usize {
    if total == 0 {
        return 0;
    }
    let (process_limit, reserve, effective_free) = if integrated {
        (
            total / CUDA_PROCESS_BUDGET_DIVISOR,
            (total / CUDA_FREE_RESERVE_DIVISOR)
                .max(CUDA_MIN_FREE_RESERVE_BYTES)
                .min(total),
            shared_host_available.map_or(cuda_free, |host| cuda_free.min(host)),
        )
    } else {
        // Device-local VRAM cannot invoke the host OOM killer. Retain a small
        // CUDA reserve for the display/other contexts, but do not impose the
        // UMA-specific 1/8 process cap that would disable ordinary 8/24-GiB
        // discrete GPUs.
        (
            total,
            (total / DISCRETE_FREE_RESERVE_DIVISOR)
                .max(DISCRETE_MIN_FREE_RESERVE_BYTES)
                .min(total),
            cuda_free,
        )
    };
    // Driver/host "free" excludes the process's live allocations. Add `live`
    // back when expressing the result as an aggregate ownership limit; the
    // additional headroom itself remains exactly `free - reserve`.
    process_limit.min(live.saturating_add(effective_free.saturating_sub(reserve)))
}

fn host_memory_available() -> Result<usize, GpuError> {
    // Central workspace probe: MemAvailable capped by cgroup availability and
    // any explicit contest confinement. Do not grow a second /proc parser here.
    tla_resource::platform::effective_available_bytes()
        .ok_or_else(|| GpuError::Driver("effective host memory is unavailable".into()))
}

fn device_shares_host_memory(api: &CudaApi) -> Result<bool, GpuError> {
    let mut dev = 0;
    check(
        api,
        unsafe { (api.cuDeviceGet)(&mut dev, 0) },
        "cuDeviceGet",
    )?;
    let mut integrated = 0;
    check(
        api,
        unsafe { (api.cuDeviceGetAttribute)(&mut integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) },
        "query integrated-memory attribute",
    )?;
    Ok(integrated != 0)
}

fn safe_allocation_limit(api: &CudaApi, live: usize) -> Result<usize, GpuError> {
    let (free, total) = memory_info(api)?;
    let integrated = device_shares_host_memory(api)?;
    let host_available = if integrated {
        Some(host_memory_available()?)
    } else {
        None
    };
    Ok(allocation_limit_from_info(
        free,
        total,
        host_available,
        live,
        integrated,
    ))
}

// Serialize the check+allocate transaction across TY processes.  This cannot
// coordinate unrelated CUDA programs, so the independent free-memory reserve
// remains mandatory; it does close the race between parallel cargo tests.
#[cfg(unix)]
struct InterprocessAllocationLock(File);

#[cfg(unix)]
impl InterprocessAllocationLock {
    fn acquire() -> Result<Self, GpuError> {
        unsafe extern "C" {
            fn flock(fd: c_int, operation: c_int) -> c_int;
            fn getuid() -> u32;
        }
        const LOCK_EX: c_int = 2;
        let path = format!("/tmp/ty-cuda-allocation-{}.lock", unsafe { getuid() });
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| GpuError::Driver(format!("open CUDA allocation lock: {e}")))?;
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            return Err(GpuError::Driver(format!(
                "lock CUDA allocation guard: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self(file))
    }
}

#[cfg(unix)]
impl Drop for InterprocessAllocationLock {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn flock(fd: c_int, operation: c_int) -> c_int;
        }
        const LOCK_UN: c_int = 8;
        let _ = unsafe { flock(self.0.as_raw_fd(), LOCK_UN) };
    }
}

#[cfg(not(unix))]
struct InterprocessAllocationLock;

#[cfg(not(unix))]
impl InterprocessAllocationLock {
    fn acquire() -> Result<Self, GpuError> {
        Ok(Self)
    }
}

fn memory_info(api: &CudaApi) -> Result<(usize, usize), GpuError> {
    let mut free = 0usize;
    let mut total = 0usize;
    check(
        api,
        unsafe { (api.cuMemGetInfo_v2)(&mut free, &mut total) },
        "cuMemGetInfo",
    )?;
    Ok((free, total))
}

/// Remaining bytes this process may reserve under the aggregate CUDA budget.
/// The result is advisory for adaptive planners; [`DeviceBuffer`] repeats the
/// check atomically at allocation time.
pub(crate) fn device_allocation_headroom(api: &CudaApi) -> Result<usize, GpuError> {
    let live = CUDA_LIVE_ALLOCATED_BYTES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Ok(safe_allocation_limit(api, *live)?.saturating_sub(*live))
}

/// Device memory allocation that frees on drop.
pub struct DeviceBuffer {
    pub ptr: CuDeviceptr,
    pub bytes: usize,
}

impl DeviceBuffer {
    pub fn device(api: &CudaApi, bytes: usize) -> Result<Self, GpuError> {
        Self::allocate(api, bytes, false)
    }

    pub fn managed(api: &CudaApi, bytes: usize) -> Result<Self, GpuError> {
        Self::allocate(api, bytes, true)
    }

    fn allocate(api: &CudaApi, bytes: usize, managed: bool) -> Result<Self, GpuError> {
        if bytes == 0 {
            return Err(GpuError::Codegen("zero-byte CUDA allocation".into()));
        }

        // Keep the mutex through the driver allocation.  This prevents two
        // parallel Rust tests from both observing the same headroom and each
        // reserving it in full.
        let mut live = CUDA_LIVE_ALLOCATED_BYTES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _interprocess = InterprocessAllocationLock::acquire()?;
        let capacity = safe_allocation_limit(api, *live)?;
        let needed = live
            .checked_add(bytes)
            .ok_or(GpuError::MemoryBudgetExceeded {
                needed: u64::MAX,
                capacity: as_u64_saturating(capacity),
            })?;
        if needed > capacity {
            return Err(GpuError::MemoryBudgetExceeded {
                needed: as_u64_saturating(needed),
                capacity: as_u64_saturating(capacity),
            });
        }

        let mut ptr = 0;
        if managed {
            check(
                api,
                unsafe { (api.cuMemAllocManaged)(&mut ptr, bytes, CU_MEM_ATTACH_GLOBAL) },
                "cuMemAllocManaged",
            )?;
        } else {
            check(
                api,
                unsafe { (api.cuMemAlloc_v2)(&mut ptr, bytes) },
                "cuMemAlloc",
            )?;
        }
        *live = needed;
        Ok(DeviceBuffer { ptr, bytes })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(api) = cuda_api() {
            let mut live = CUDA_LIVE_ALLOCATED_BYTES
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let freed = unsafe { (api.cuMemFree_v2)(self.ptr) } == CUDA_SUCCESS;
            if freed {
                *live = live.saturating_sub(self.bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    #[test]
    fn unified_memory_policy_keeps_a_large_host_reserve() {
        assert_eq!(
            allocation_limit_from_info(120 * GIB, 128 * GIB, Some(80 * GIB), 0, true),
            16 * GIB
        );
        assert_eq!(
            allocation_limit_from_info(120 * GIB, 128 * GIB, Some(20 * GIB), 0, true),
            4 * GIB
        );
        assert_eq!(
            allocation_limit_from_info(120 * GIB, 128 * GIB, Some(16 * GIB), 0, true),
            0
        );
    }

    #[test]
    fn unified_memory_policy_scales_down_on_small_devices() {
        assert_eq!(
            allocation_limit_from_info(7 * GIB, 8 * GIB, None, 0, true),
            GIB
        );
        assert_eq!(allocation_limit_from_info(GIB, 8 * GIB, None, 0, true), 0);
        assert_eq!(allocation_limit_from_info(0, 0, None, 0, true), 0);
        assert_eq!(
            allocation_limit_from_info(2 * GIB, 128 * GIB, None, 14 * GIB, true),
            14 * GIB
        );
    }

    #[test]
    fn discrete_gpu_policy_does_not_apply_the_uma_process_cap() {
        assert_eq!(
            allocation_limit_from_info(23 * GIB, 24 * GIB, Some(GIB), 0, false),
            43 * GIB / 2
        );
        assert_eq!(
            allocation_limit_from_info(7 * GIB, 8 * GIB, Some(GIB), 0, false),
            13 * GIB / 2
        );
    }

    #[test]
    fn byte_layout_math_is_checked() {
        assert_eq!(checked_allocation_bytes("test", &[2, 3, 8]).unwrap(), 48);
        assert!(checked_allocation_bytes("test", &[usize::MAX, 2]).is_err());
        assert_eq!(checked_allocation_sum("test", &[2, 3, 8]).unwrap(), 13);
        assert!(checked_allocation_sum("test", &[usize::MAX, 1]).is_err());
        assert_eq!(checked_allocation_usize("test", 7).unwrap(), 7);
        assert_eq!(checked_power_of_two_u64("test", 63).unwrap(), 1u64 << 63);
        assert!(checked_power_of_two_u64("test", 64).is_err());
    }
}
