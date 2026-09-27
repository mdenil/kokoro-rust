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
    // conv1d_igemm input-channel chunk (PL-010 tuning knob), shared by the kernel and the Rust launcher
    println!("cargo:rerun-if-env-changed=KOKORO_IG_BK");
    let ig_bk = std::env::var("KOKORO_IG_BK").unwrap_or_else(|_| "4".into());
    assert!(ig_bk.parse::<u32>().map(|v| v > 0 && v <= 16).unwrap_or(false), "KOKORO_IG_BK must be 1..=16");
    println!("cargo:rustc-env=KOKORO_IG_BK={ig_bk}");
    // PHASE 2: fused tensor-core conv time tile (64 or 128)
    println!("cargo:rerun-if-env-changed=KOKORO_WMMA_TN");
    let wmma_tn = std::env::var("KOKORO_WMMA_TN").unwrap_or_else(|_| "128".into());
    assert!(wmma_tn == "64" || wmma_tn == "128", "KOKORO_WMMA_TN must be 64 or 128");
    println!("cargo:rustc-env=KOKORO_WMMA_TN={wmma_tn}");
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kokoro.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(["-ptx", "-arch=compute_89", if strict { "-fmad=false" } else { "-fmad=true" }, "-O3", &format!("-DIG_BK={ig_bk}"), &format!("-DWMMA_TN={wmma_tn}"), "kernels/kokoro.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed compiling kernels/kokoro.cu");
}
