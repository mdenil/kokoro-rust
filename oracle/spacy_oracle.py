"""spaCy tokenizer/tagger oracle (F2 seams) for the pinned en_core_web_sm 3.8.0, run exactly as misaki
does (enable=['tok2vec', 'tagger']) on misaki-preprocessed text.

Per line: tokens (text, whitespace, norm, prefix, suffix, shape, is_space, tag) + seams:
tok2vec output (doc.tensor [T, 96]) and tagger scores [T, 50]. Tokens as JSONL, tensors as
safetensors keyed "l<line>.tensor" / "l<line>.scores". Also exports BASE_NORMS once.
Usage: python oracle/spacy_oracle.py <lines.txt> <out_prefix>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def main():
    common.assert_pins()
    import numpy as np
    import spacy
    from misaki import en
    from safetensors.numpy import save_file
    from spacy.lang.norm_exceptions import BASE_NORMS
    src, prefix = pathlib.Path(sys.argv[1]), sys.argv[2]
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    tagger = nlp.get_pipe("tagger")
    lines = src.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    tensors = {}
    with open(prefix + ".tokens.jsonl", "w", encoding="utf-8") as out:
        for no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            text, _, _ = en.G2P.preprocess(line)
            doc = nlp(text)
            scores = tagger.model.predict([doc])[0]
            ids = doc.to_array(["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"]).astype(np.uint64)
            toks = [{"ids": [int(x) for x in ids[i]], "text": t.text, "ws": t.whitespace_, "norm": t.norm_, "prefix": t.prefix_, "suffix": t.suffix_,
                     "shape": t.shape_, "is_space": t.is_space, "tag": t.tag_} for i, t in enumerate(doc)]
            out.write(json.dumps({"line": no, "text": text, "tokens": toks}, ensure_ascii=False) + "\n")
            tensors[f"l{no}.tensor"] = np.ascontiguousarray(doc.tensor, dtype=np.float32)
            tensors[f"l{no}.scores"] = np.ascontiguousarray(scores, dtype=np.float32)
    save_file(tensors, prefix + ".seams.safetensors")
    base = common.DATA / "frontend/spacy-en_core_web_sm-3.8.0/base_norms.json"
    base.write_text(json.dumps(BASE_NORMS, ensure_ascii=False))
    print(f"{src.name}: {len(tensors) // 2} lines")


if __name__ == "__main__":
    main()
