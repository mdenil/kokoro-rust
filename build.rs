// Compiles kernels/kokoro.cu to PTX when the `cuda` feature is enabled (nvcc from $NVCC,
// else /usr/local/cuda/bin/nvcc). -fmad=false keeps elementwise rounding identical to the CPU path.
fn main() {
    println!("cargo:rerun-if-changed=kernels/kokoro.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kokoro.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(["-ptx", "-arch=compute_89", "-fmad=false", "-O3", "kernels/kokoro.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed compiling kernels/kokoro.cu");
}
