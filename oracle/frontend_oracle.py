"""Frontend oracle (F0): run the PINNED production English frontend and record every decision.

For each input line (the exact KPipeline('a') path: misaki 0.9.4 en.G2P(trf=False,
fallback=EspeakFallback(british=False), unk='') + KPipeline.en_tokenize chunking):
  - preprocess: misaki G2P.preprocess output (text, tokens, features)
  - spacy: raw spaCy tokens (text, whitespace, tag_) — the tagger-parity target
  - tokens: final misaki MTokens (text, tag, whitespace, phonemes, rating, flags)
  - fallback: every espeak-ng fallback call (input text -> phonemes), instrumented
  - chunks: KPipeline chunks exactly as synthesized (graphemes, phonemes, token span)
  - phonemes: the joined phoneme string misaki returns
Output: JSONL (one record per input line, line = 1-based), plus a meta record with pins/hashes.
Usage: python oracle/frontend_oracle.py <lines.txt> <out.jsonl> [--british]
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def main():
    common.assert_pins()
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    british = "--british" in sys.argv
    from kokoro.pipeline import KPipeline
    from misaki import en
    pipe = KPipeline(lang_code="b" if british else "a", repo_id="hexgrad/Kokoro-82M", model=False)
    g2p = pipe.g2p
    calls = []
    real_fallback = g2p.fallback

    def fallback(tk):
        ps, rating = real_fallback(tk)
        calls.append({"text": tk.text, "phonemes": ps, "rating": rating})
        return ps, rating
    if real_fallback is not None:
        g2p.fallback = fallback

    raw = src.read_bytes()
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    with open(dst, "w", encoding="utf-8") as out:
        out.write(json.dumps({"meta": {
            "input": str(src), "input_sha256": hashlib.sha256(raw).hexdigest(), "lines": len(lines),
            "lang": "b" if british else "a", "fallback": real_fallback is not None,
            "pins": common.runtime_meta()["pins"],
            "lexicon_sha256": {f: hashlib.sha256((pathlib.Path(en.__file__).parent / "data" / f).read_bytes()).hexdigest()
                               for f in ("us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json")},
        }}, ensure_ascii=False) + "\n")
        for no, line in enumerate(lines, 1):
            calls.clear()
            rec = {"line": no, "text": line}
            if not line.strip():
                rec.update(blank=True, chunks=[])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            ptext, ptokens, features = en.G2P.preprocess(line)
            rec["preprocess"] = {"text": ptext, "tokens": ptokens, "features": {str(k): v for k, v in features.items()}}
            doc = g2p.nlp(ptext)
            rec["spacy"] = [{"text": t.text, "ws": t.whitespace_, "tag": t.tag_} for t in doc]
            ps, tokens = g2p(line)
            rec["phonemes"] = ps
            rec["tokens"] = [{"text": t.text, "tag": t.tag, "ws": t.whitespace, "phonemes": t.phonemes,
                              "rating": t._.rating if t._ else None, "stress": t._.stress if t._ else None,
                              "currency": t._.currency if t._ else None, "num_flags": t._.num_flags if t._ else None,
                              "alias": t._.alias if t._ else None, "is_head": t._.is_head if t._ else None} for t in tokens]
            rec["fallback"] = list(calls)
            # chunking exactly as KPipeline.__call__ (English branch), from the same token list
            chunks = []
            for gs, cps, tks in pipe.en_tokenize(tokens):
                if not cps:
                    continue
                chunks.append({"graphemes": gs, "phonemes": cps, "n_tokens": len(tks), "oversize": len(cps) > 510})
            rec["chunks"] = chunks
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{src}: {len(lines)} lines -> {dst}")


if __name__ == "__main__":
    main()
