"""Export the pinned en_core_web_sm 3.8.0 tokenizer data, lexeme norms and tok2vec+tagger weights
for the native frontend (F2). Data only (MIT-licensed model); the Rust port implements the
algorithms itself. Output: $KOKORO_DATA/frontend/spacy-en_core_web_sm-3.8.0/
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
    from safetensors.numpy import save_file
    from spacy import symbols
    out = common.DATA / "frontend/spacy-en_core_web_sm-3.8.0"
    out.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    tok = nlp.tokenizer
    data = {
        "prefix": tok.prefix_search.__self__.pattern if tok.prefix_search else None,
        "suffix": tok.suffix_search.__self__.pattern if tok.suffix_search else None,
        "infix": tok.infix_finditer.__self__.pattern if tok.infix_finditer else None,
        "token_match": tok.token_match.__self__.pattern if tok.token_match else None,
        "url_match": tok.url_match.__self__.pattern if tok.url_match else None,
        "faster_heuristics": tok.faster_heuristics,
        "rules": {k: [{"ORTH": s.get(65, s.get("ORTH")), "NORM": s.get(67, s.get("NORM"))} for s in v] for k, v in tok.rules.items()},
    }
    # sanity: rules use int attr ids 65 (ORTH) / 67 (NORM) or strings
    (out / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False))
    tables = {name: dict(nlp.vocab.lookups.get_table(name)) for name in nlp.vocab.lookups.tables}
    (out / "lookups.json").write_text(json.dumps({k: {str(a): b for a, b in v.items()} for k, v in tables.items()}, ensure_ascii=False))
    # StringStore.add returns the symbol id (not the string hash) for strings in SYMBOLS_BY_STR
    (out / "symbols.json").write_text(json.dumps({k: int(v) for k, v in symbols.IDS.items()}, ensure_ascii=False, sort_keys=True))
    # model structure + params
    tagger = nlp.get_pipe("tagger")
    tok2vec = nlp.get_pipe("tok2vec")
    arrays, nodes = {}, []
    for label, model in (("tok2vec", tok2vec.model), ("tagger", tagger.model)):
        for i, node in enumerate(model.walk()):
            info = {"model": label, "i": i, "name": node.name, "id": node.id,
                    "dims": {d: node.maybe_get_dim(d) for d in node.dim_names},
                    "attrs": {k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else repr(v)) for k, v in node.attrs.items()},
                    "params": []}
            for p in node.param_names:
                if node.has_param(p):
                    key = f"{label}.{i}.{node.name}.{p}"
                    arrays[key] = np.ascontiguousarray(node.get_param(p)).astype(np.float32)
                    info["params"].append({"name": p, "key": key, "shape": list(arrays[key].shape)})
            nodes.append(info)
    save_file(arrays, str(out / "tagger_weights.safetensors"))
    (out / "model_structure.json").write_text(json.dumps({"labels": list(tagger.labels), "nodes": nodes}, indent=1, default=str))
    print("rules", len(data["rules"]), "lookup tables", {k: len(v) for k, v in tables.items()}, "params", len(arrays),
          "labels", len(tagger.labels), "faster_heuristics", data["faster_heuristics"], "token_match", data["token_match"] is not None)
    for n in nodes:
        print(n["model"], n["i"], n["name"], n["dims"], [ (p["name"], p["shape"]) for p in n["params"]], {k: v for k, v in n["attrs"].items() if k in ("seed", "column", "window_size", "nP", "normalize")})


if __name__ == "__main__":
    main()
