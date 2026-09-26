// Compiles kernels/kokoro.cu to PTX when the `cuda` feature is enabled (nvcc from $NVCC,
// else /usr/local/cuda/bin/nvcc). Default: -fmad=false (the owner-accepted strict baseline). Set
// KOKORO_FMA=1 at build time for normal FMA contraction (owner #11 exploration; its audio deltas are
// PROVISIONAL until owner listening acceptance — docs/PERF_LEDGER.md PL-005).
fn main() {
    println!("cargo:rerun-if-changed=kernels/kokoro.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=KOKORO_FMA");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let strict = !std::env::var("KOKORO_FMA").map(|v| v == "1").unwrap_or(false);
    println!("cargo:rustc-env=KOKORO_KERNEL_ROUNDING={}", if strict { "strict(-fmad=false)" } else { "fma" });
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kokoro.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(["-ptx", "-arch=compute_89", if strict { "-fmad=false" } else { "-fmad=true" }, "-O3", "kernels/kokoro.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed compiling kernels/kokoro.cu");
}
