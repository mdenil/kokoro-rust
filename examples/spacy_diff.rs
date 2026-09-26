//! Dev probe: native spaCy tokens/tags vs an oracle JSONL's recorded "spacy" tokens, for given line numbers.
use kokoro::frontend::spacy_tag::Tagger;
use kokoro::frontend::spacy_tok::Tokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(std::env::var("KOKORO_FRONTEND_DIR")?).join("spacy-en_core_web_sm-3.8.0");
    let (tk, tg) = (Tokenizer::load(&dir)?, Tagger::load(&dir)?);
    let want: Vec<u64> = args[2..].iter().map(|a| a.parse().unwrap()).collect();
    for l in std::fs::read_to_string(&args[1])?.lines().skip(1) {
        let r: serde_json::Value = serde_json::from_str(l)?;
        if !want.contains(&r["line"].as_u64().unwrap()) { continue; }
        let text = r["preprocess"]["text"].as_str().unwrap_or("");
        let toks = tk.tokenize(text);
        let tags = tg.tag(&toks).tags;
        let got: Vec<(String, String, String)> = toks.iter().zip(tags).map(|(t, g)| (t.text.clone(), if t.space { " ".into() } else { String::new() }, g)).collect();
        let exp: Vec<(String, String, String)> = r["spacy"].as_array().map(|a| a.iter().map(|t| (t["text"].as_str().unwrap().into(), t["ws"].as_str().unwrap().into(), t["tag"].as_str().unwrap().into())).collect()).unwrap_or_default();
        println!("== line {} error={:?} tokens got {} want {}", r["line"], r.get("error"), got.len(), exp.len());
        let n = got.len().max(exp.len());
        for i in 0..n {
            if got.get(i) != exp.get(i) { println!("  [{i}] got {:?} want {:?}", got.get(i), exp.get(i)); }
        }
    }
    Ok(())
}
