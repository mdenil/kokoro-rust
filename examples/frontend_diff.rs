//! Dev probe: native frontend vs any frontend_oracle.py JSONL (exploratory, unpinned). Prints a summary
//! and the first mismatching line numbers (use --show for text; never on private corpora).
use kokoro::frontend::pipeline::{EnglishFrontend, FrontendPaths};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let show = args.iter().any(|a| a == "--show");
    let fe = EnglishFrontend::load(&FrontendPaths::under(std::path::Path::new(&std::env::var("KOKORO_FRONTEND_DIR")?)))?;
    let (mut n, mut bad) = (0, vec![]);
    for l in std::fs::read_to_string(&args[1])?.lines().skip(1) {
        let r: serde_json::Value = serde_json::from_str(l)?;
        if r["blank"].as_bool() == Some(true) { continue; }
        n += 1;
        let got = fe.line_chunks(r["text"].as_str().unwrap());
        let ok = match (&got, r.get("error")) {
            (Err(_), Some(_)) => true,
            (Ok(ch), None) => {
                let want: Vec<(&str, &str)> = r["chunks"].as_array().unwrap().iter().map(|c| (c["graphemes"].as_str().unwrap(), c["phonemes"].as_str().unwrap())).collect();
                ch.iter().map(|c| (c.graphemes.as_str(), c.phonemes.as_str())).collect::<Vec<_>>() == want
            }
            _ => false,
        };
        if !ok {
            bad.push(r["line"].as_u64().unwrap());
            if show { println!("line {}: {:?}\n  got {:?}\n  want err={:?} {:?}", r["line"], r["text"], got.map(|c| c.into_iter().map(|x| x.phonemes).collect::<Vec<_>>()).map_err(|e| e.to_string()), r.get("error"), r["chunks"].as_array().map(|a| a.iter().map(|c| c["phonemes"].clone()).collect::<Vec<_>>())); }
        }
    }
    println!("{}: {}/{n} lines exact; mismatching lines: {:?}", args[1], n - bad.len(), &bad[..bad.len().min(30)]);
    Ok(())
}
