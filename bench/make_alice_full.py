"""Regenerate the public-domain Alice line file from the pinned Gutenberg download, with the same
construction as corpus_alice_ch1.txt (bench/CORPUS.md): all 12 chapters (not committed, size), or
the first N chapters with --chapters N (bench/corpus_alice_ch1-3.txt is --chapters 3).

python3 bench/make_alice_full.py [--chapters N] <pg11.txt> <out.txt>
"""
import argparse
import hashlib
import re

PG11_SHA256 = "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", type=int, default=None, help="only the first N chapters")
    ap.add_argument("pg11")
    ap.add_argument("out")
    args = ap.parse_args()
    raw = open(args.pg11, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == PG11_SHA256, "pg11.txt does not match the pinned hash"
    lines = raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    heads = [i for i, l in enumerate(lines) if re.fullmatch(r"CHAPTER [IVXL]+\.", l.strip())]
    end = next(i for i, l in enumerate(lines) if l.startswith("*** END OF THE PROJECT GUTENBERG"))
    bounds = list(zip(heads, heads[1:] + [end]))
    if args.chapters is not None:
        bounds = bounds[: args.chapters]
    utts = []
    per_chapter = []
    for a, b in bounds:
        before = len(utts)
        body = "\n".join(lines[a + 2:b])
        for p in body.split("\n\n"):
            p = re.sub(r"\s+", " ", p).strip()
            if not p or set(p) <= set("* "):
                continue
            utts.extend(x.strip() for x in re.split(r"(?<=[.!?’”])\s+(?=[“‘A-Z])", p) if x.strip())
        per_chapter.append((lines[a].strip(), len(utts) - before))
    out = "\n".join(u.replace("_", "") for u in utts) + "\n"
    open(args.out, "w", encoding="utf-8").write(out)
    print(per_chapter)
    print(len(utts), "utterances; sha256", hashlib.sha256(out.encode()).hexdigest())


if __name__ == "__main__":
    main()
