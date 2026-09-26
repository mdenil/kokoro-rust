"""RETIRED 2026-09-26: the `kokoro` binary no longer has a python-bridge frontend (the native
frontend is exact vs the pinned reference; tests/frontend_pipeline.rs). Kept only as a debugging aid.

DEV-ONLY text frontend bridge: misaki 0.9.4 G2P + KPipeline chunking, as a resident process.

This is NOT part of the native product. The Rust CLI can spawn it (`--frontend python-bridge`)
until a native G2P exists; every sidecar produced this way records the bridge explicitly.

Protocol: one JSON request per stdin line {"text": "..."}; one JSON response per stdout line
{"chunks": [{"graphemes": "...", "phonemes": "..."}, ...]} or {"error": "..."}.
"""
import json
import sys


def main():
    from kokoro.pipeline import KPipeline
    lang = sys.argv[1] if len(sys.argv) > 1 else "a"
    quiet = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=False)
    print(json.dumps({"ready": True, "frontend": f"misaki-0.9.4/KPipeline(lang_code={lang!r})"}), flush=True)
    import re
    for line in sys.stdin:
        try:
            req = json.loads(line)
            # Mirrors KPipeline.__call__ (pipeline.py:365-386) for English, EXCEPT the silent
            # ps[:510] truncation: oversize chunks are returned intact so the caller can flag them.
            chunks = []
            for seg in re.split(r"\n+", req["text"].strip()):
                if not seg.strip():
                    continue
                _, tokens = quiet.g2p(seg)
                for gs, ps, _tks in quiet.en_tokenize(tokens):
                    if ps:
                        chunks.append({"graphemes": gs, "phonemes": ps})
            print(json.dumps({"chunks": chunks}, ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
