//! Minimal NVRTC (runtime CUDA C compilation) binding, loaded via `dlopen`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::OnceLock;

use crate::GpuError;

type NvrtcResult = c_int;
type NvrtcProgram = *mut c_void;

const NVRTC_SUCCESS: NvrtcResult = 0;

struct NvrtcApi {
    _lib: libloading::Library,
    create: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    compile: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    destroy: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

static NVRTC: OnceLock<Result<NvrtcApi, GpuError>> = OnceLock::new();

fn nvrtc_api() -> Result<&'static NvrtcApi, GpuError> {
    NVRTC
        .get_or_init(|| {
            let lib = [
                "libnvrtc.so.13",
                "libnvrtc.so.12",
                "libnvrtc.so.11",
                "libnvrtc.so",
            ]
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name).ok() })
            .ok_or_else(|| GpuError::Unavailable("libnvrtc not found".into()))?;
            unsafe {
                macro_rules! sym {
                    ($name:literal) => {
                        *lib.get(concat!($name, "\0").as_bytes()).map_err(|e| {
                            GpuError::Unavailable(format!("libnvrtc missing symbol {}: {e}", $name))
                        })?
                    };
                }
                Ok(NvrtcApi {
                    create: sym!("nvrtcCreateProgram"),
                    compile: sym!("nvrtcCompileProgram"),
                    ptx_size: sym!("nvrtcGetPTXSize"),
                    ptx: sym!("nvrtcGetPTX"),
                    log_size: sym!("nvrtcGetProgramLogSize"),
                    log: sym!("nvrtcGetProgramLog"),
                    destroy: sym!("nvrtcDestroyProgram"),
                    error_string: sym!("nvrtcGetErrorString"),
                    _lib: lib,
                })
            }
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn error_string(api: &NvrtcApi, rc: NvrtcResult) -> String {
    let ptr = unsafe { (api.error_string)(rc) };
    if ptr.is_null() {
        format!("nvrtc error {rc}")
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

fn program_log(api: &NvrtcApi, prog: NvrtcProgram) -> String {
    let mut size = 0usize;
    if unsafe { (api.log_size)(prog, &mut size) } != NVRTC_SUCCESS || size <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    if unsafe { (api.log)(prog, buf.as_mut_ptr().cast::<c_char>()) } != NVRTC_SUCCESS {
        return String::new();
    }
    buf.pop(); // trailing NUL
    String::from_utf8_lossy(&buf).into_owned()
}

/// Compile CUDA C source to PTX for the given compute capability.
pub fn compile_to_ptx(source: &str, cc_major: i32, cc_minor: i32) -> Result<Vec<u8>, GpuError> {
    let api = nvrtc_api()?;
    let src = CString::new(source).map_err(|_| GpuError::Codegen("NUL in CUDA source".into()))?;
    let name = CString::new("ty_gpu_bfs.cu").expect("static");

    let mut prog: NvrtcProgram = std::ptr::null_mut();
    let rc = unsafe {
        (api.create)(
            &mut prog,
            src.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if rc != NVRTC_SUCCESS {
        return Err(GpuError::Codegen(format!(
            "nvrtcCreateProgram: {}",
            error_string(api, rc)
        )));
    }

    let arch = CString::new(format!("--gpu-architecture=compute_{cc_major}{cc_minor}"))
        .expect("static shape");
    let std_opt = CString::new("--std=c++17").expect("static");
    let opts = [arch.as_ptr(), std_opt.as_ptr()];

    let rc = unsafe { (api.compile)(prog, opts.len() as c_int, opts.as_ptr()) };
    if rc != NVRTC_SUCCESS {
        let log = program_log(api, prog);
        unsafe { (api.destroy)(&mut prog) };
        return Err(GpuError::Codegen(format!(
            "nvrtc compile failed: {}\n{log}",
            error_string(api, rc)
        )));
    }

    let mut size = 0usize;
    let rc = unsafe { (api.ptx_size)(prog, &mut size) };
    if rc != NVRTC_SUCCESS {
        unsafe { (api.destroy)(&mut prog) };
        return Err(GpuError::Codegen(format!(
            "nvrtcGetPTXSize: {}",
            error_string(api, rc)
        )));
    }
    let mut ptx = vec![0u8; size];
    let rc = unsafe { (api.ptx)(prog, ptx.as_mut_ptr().cast::<c_char>()) };
    unsafe { (api.destroy)(&mut prog) };
    if rc != NVRTC_SUCCESS {
        return Err(GpuError::Codegen(format!(
            "nvrtcGetPTX: {}",
            error_string(api, rc)
        )));
    }
    // Keep the trailing NUL: cuModuleLoadData expects a NUL-terminated image
    // for PTX text.
    Ok(ptx)
}
