"""Regenerate the full public-domain Alice line file (all 12 chapters) from the pinned Gutenberg
download, with the same construction as corpus_alice_ch1.txt (bench/CORPUS.md). Not committed
(size); reproducible from the pinned source hash.

python3 bench/make_alice_full.py <pg11.txt> <out.txt>
"""
import hashlib
import re
import sys

PG11_SHA256 = "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"


def main():
    raw = open(sys.argv[1], "rb").read()
    assert hashlib.sha256(raw).hexdigest() == PG11_SHA256, "pg11.txt does not match the pinned hash"
    lines = raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    heads = [i for i, l in enumerate(lines) if re.fullmatch(r"CHAPTER [IVXL]+\.", l.strip())]
    end = next(i for i, l in enumerate(lines) if l.startswith("*** END OF THE PROJECT GUTENBERG"))
    utts = []
    for a, b in zip(heads, heads[1:] + [end]):
        body = "\n".join(lines[a + 2:b])
        for p in body.split("\n\n"):
            p = re.sub(r"\s+", " ", p).strip()
            if not p or set(p) <= set("* "):
                continue
            utts.extend(x.strip() for x in re.split(r"(?<=[.!?’”])\s+(?=[“‘A-Z])", p) if x.strip())
    out = "\n".join(u.replace("_", "") for u in utts) + "\n"
    open(sys.argv[2], "w", encoding="utf-8").write(out)
    print(len(utts), "utterances; sha256", hashlib.sha256(out.encode()).hexdigest())


if __name__ == "__main__":
    main()
