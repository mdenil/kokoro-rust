//! Native Rust inference engine for hexgrad/Kokoro-82M.
#![deny(unsafe_code)]

pub mod albert;
pub mod cli;
pub mod engine;
pub mod frontend;
#[allow(unsafe_code)]
pub mod gpu;
pub mod model;
pub mod nn;
pub mod ordered;
pub mod ops;
pub mod st;
pub mod timeline;
pub mod torchpt;
pub mod vocoder;
pub mod wav;
pub mod weights;

pub use cli::cli_main;
