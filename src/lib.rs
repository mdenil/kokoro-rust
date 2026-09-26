//! Native Rust inference engine for hexgrad/Kokoro-82M.
#![deny(unsafe_code)]

pub mod albert;
pub mod model;
pub mod nn;
#[allow(unsafe_code)]
pub mod ops;
pub mod st;
pub mod vocoder;
pub mod weights;

pub fn cli_main() -> anyhow::Result<()> {
    anyhow::bail!("CLI not implemented yet")
}
