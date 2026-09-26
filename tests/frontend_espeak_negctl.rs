//! Negative control (own process: espeak is a process-wide singleton): a non-pinned libespeak-ng
//! (the host's system 1.51) must be REFUSED, never silently used with different pronunciations.
use kokoro::frontend::espeak::{default_dir, Espeak};
use std::path::Path;

#[test]
fn non_pinned_espeak_is_refused() {
    let sys = Path::new("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1.1.51");
    if !sys.exists() {
        panic!("host system libespeak-ng 1.51 not present; negative control cannot run — NOT a pass");
    }
    let err = match Espeak::get(sys, &default_dir()) {
        Ok(e) => panic!("non-pinned espeak {} accepted", e.version),
        Err(e) => format!("{e:#}"),
    };
    println!("refused as expected: {err}");
    assert!(err.contains("pinned") || err.contains("Initialize"), "{err}");
}
