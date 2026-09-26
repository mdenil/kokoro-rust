//! Dev probe: print native frontend chunks for each stdin line (lengths only unless --show).
use kokoro::frontend::pipeline::{EnglishFrontend, FrontendPaths};
use std::io::BufRead;

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("KOKORO_FRONTEND_DIR")?;
    let fe = EnglishFrontend::load(&FrontendPaths::under(std::path::Path::new(&dir)))?;
    let show = std::env::args().any(|a| a == "--show");
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        match fe.line_chunks(&line) {
            Ok(ch) => println!("{} chunks: {:?}", ch.len(), ch.iter().map(|c| if show { format!("{} | {}", c.graphemes, c.phonemes) } else { c.phonemes.chars().count().to_string() }).collect::<Vec<_>>()),
            Err(e) => println!("error: {e:#}"),
        }
    }
    Ok(())
}
