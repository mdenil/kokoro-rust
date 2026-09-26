"""Deterministic character-level "soup" corpus for the frontend differential test (targets the
tokenizer prefix/suffix/infix regexes, Unicode handling and reference crash paths).
Usage: python bench/make_frontend_soup.py <out.txt>   (seed 11, 3000 lines)
"""
import random
import sys


def main():
    r = random.Random(11)
    pool = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") * 3 + list(" " * 40)
            + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") * 2
            + list("’‘“”—–…«»¿¡°€£¥©®™§¶•·×÷±½¼¾²³¹µ")
            + list("éèêëàâäôöûüçñøåæœßÉÀÇÑØÅİıŁłŽžŠšČčĞğŞşΑαΣσςΩωЖжЯя漢字かなカナ한글")
            + [" ", " ", "\t", "​", "﻿", "́", "̈"])
    lines = []
    for _ in range(3000):
        n = r.choice([1, 2, 3, 5, 8, 13, 30, 60, 120, 250])
        s = "".join(r.choice(pool) for _ in range(n)).strip()
        if r.random() < 0.5:
            s = " ".join(w for w in s.split(" ") if w)
        lines.append(s if s.strip() else "x")
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(sys.argv[1], len(lines), "lines")


if __name__ == "__main__":
    main()
