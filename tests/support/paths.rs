//! Test configuration, read from the environment only (no host-specific defaults; docs/PORTABILITY.md).
//! - `KOKORO_DATA` (required): test data root with `fixtures/`, `frontend/`, the model snapshot under
//!   `hf/`, and (private tests only) `bench/private/` and `evidence/private/`.
//! - `KOKORO_MODEL_DIR` (optional): model snapshot directory; default: the pinned snapshot under
//!   `$KOKORO_DATA/hf`.
//! - `KOKORO_FRONTEND_DIR` (optional): frontend data; default `$KOKORO_DATA/frontend`.
//! - `CUDA_VISIBLE_DEVICES`: passed through unchanged to the binary under test when set; never forced.
//! - `KOKORO_BIN` (optional): binary under test; default this crate's `kokoro`.
//! - `KOKORO_PRIVATE_CHAPTER` (optional, private tests only): path of a private long-form text
//!   (relative to `KOKORO_DATA` or absolute); its sha256 is pinned in the tests. Never in Git.
//! Scratch output goes under cargo's per-project `CARGO_TARGET_TMPDIR` (inside the ignored target dir).
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// Pinned Kokoro-82M snapshot revision (docs/truth-pack).
pub const MODEL_REV: &str = "f3ff3571791e39611d31c381e3a41a3af07b4987";

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from)
}

pub fn data() -> PathBuf {
    env_path("KOKORO_DATA").unwrap_or_else(|| panic!("KOKORO_DATA is not set: point it at the test data root (see docs/PORTABILITY.md); a missing data root is NOT a pass"))
}

pub fn model_dir() -> PathBuf {
    env_path("KOKORO_MODEL_DIR").unwrap_or_else(|| data().join(format!("hf/hub/models--hexgrad--Kokoro-82M/snapshots/{MODEL_REV}")))
}

pub fn frontend_dir() -> PathBuf {
    env_path("KOKORO_FRONTEND_DIR").unwrap_or_else(|| data().join("frontend"))
}

/// The optional private long-form text (see module docs). Unset = the private tests fail.
pub fn private_chapter() -> PathBuf {
    let p = env_path("KOKORO_PRIVATE_CHAPTER").unwrap_or_else(|| panic!("KOKORO_PRIVATE_CHAPTER is not set: the private tests need the private text (never in Git) - NOT a pass"));
    if p.is_absolute() { p } else { data().join(p) }
}

pub fn bin() -> PathBuf {
    env_path("KOKORO_BIN").unwrap_or_else(|| PathBuf::from(env!("CARGO_BIN_EXE_kokoro")))
}

/// Fresh scratch directory `<CARGO_TARGET_TMPDIR>/<parts...>` (removed first).
pub fn scratch(parts: &[&str]) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    for p in parts {
        d.push(p);
    }
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Pass the caller's GPU selection through to a child whose environment was cleared.
pub fn pass_cuda_env(cmd: &mut Command) -> &mut Command {
    if let Some(v) = std::env::var_os("CUDA_VISIBLE_DEVICES") {
        cmd.env("CUDA_VISIBLE_DEVICES", v);
    }
    cmd
}

/// First `name` found on the test process's PATH.
pub fn which(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join(name)).find(|c| is_executable(c)))
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata().map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0).unwrap_or(false)
}
