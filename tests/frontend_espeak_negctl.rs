//! Negative control (own process: espeak is a process-wide singleton): a non-pinned libespeak-ng
//! must be REFUSED, never silently used with different pronunciations. Needs a non-1.52
//! libespeak-ng on the host: $KOKORO_ESPEAK_NEGCTL_LIB, else the first one found in the standard
//! system library directories (e.g. the distribution package). Without one the control cannot run,
//! which is a failure, not a pass.
use kokoro::frontend::espeak::Espeak;
use std::path::PathBuf;

#[path = "support/paths.rs"]
mod paths;

fn non_pinned_lib() -> PathBuf {
    if let Some(p) = std::env::var_os("KOKORO_ESPEAK_NEGCTL_LIB").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    let dirs = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"];
    let mut found: Vec<PathBuf> = dirs
        .iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().map(|n| n.to_string_lossy().starts_with("libespeak-ng.so.")).unwrap_or(false) && !p.to_string_lossy().ends_with("1.52.0"))
        .collect();
    found.sort();
    found.into_iter().next().expect("no non-pinned libespeak-ng found (install the distribution package or set KOKORO_ESPEAK_NEGCTL_LIB); negative control cannot run - NOT a pass")
}

#[test]
fn non_pinned_espeak_is_refused() {
    let lib = non_pinned_lib();
    let err = match Espeak::get(&lib, &paths::frontend_dir().join("espeak-ng-1.52.0")) {
        Ok(e) => panic!("non-pinned espeak {} accepted", e.version),
        Err(e) => format!("{e:#}"),
    };
    println!("{} refused as expected: {err}", lib.display());
    assert!(err.contains("pinned") || err.contains("Initialize"), "{err}");
}
