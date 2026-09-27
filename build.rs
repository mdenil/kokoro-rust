// Compiles kernels/kokoro.cu to PTX with the owner-accepted configuration (owner #25): strict
// rounding, no FMA contraction (-fmad=false), compute capability 8.9. The flags are fixed; only the
// location of nvcc is configurable: $NVCC, else $CUDA_HOME/bin/nvcc, else $CUDA_PATH/bin/nvcc,
// else `nvcc` on PATH, else /usr/local/cuda/bin/nvcc (the CUDA toolkit's default install prefix).
use std::path::PathBuf;

fn find_nvcc() -> Result<PathBuf, String> {
    let env_path = |v: &str| std::env::var_os(v).filter(|s| !s.is_empty()).map(PathBuf::from);
    if let Some(p) = env_path("NVCC") {
        return if p.is_file() { Ok(p) } else { Err(format!("NVCC={} does not exist", p.display())) };
    }
    let mut tried = vec![];
    let mut candidates: Vec<PathBuf> = ["CUDA_HOME", "CUDA_PATH"].iter().filter_map(|v| env_path(v)).map(|d| d.join("bin/nvcc")).collect();
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join("nvcc")));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
        tried.push(c.display().to_string());
    }
    Err(format!("nvcc not found (tried $NVCC, $CUDA_HOME/bin, $CUDA_PATH/bin, PATH, /usr/local/cuda/bin: {})", tried.join(", ")))
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/kokoro.cu");
    for v in ["NVCC", "CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let nvcc = find_nvcc().unwrap_or_else(|e| panic!("{e}. Install the CUDA 12.9 toolkit or set NVCC=/path/to/nvcc (see README)."));
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kokoro.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(["-ptx", "-arch=compute_89", "-fmad=false", "-O3", "kernels/kokoro.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("running {}: {e}", nvcc.display()));
    assert!(status.success(), "nvcc ({}) failed compiling kernels/kokoro.cu", nvcc.display());
}
