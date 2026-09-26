"""misaki EspeakFallback oracle on a synthetic token list (F3 unit fixture): stresses phonemizer's
punctuation preserve/restore paths, clause splitting, digits, non-ASCII and empty outputs.
Usage: python oracle/espeak_oracle.py <out.jsonl>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

CASES = [
    "Kokoro", "U.S.A.", "U.S.A", "...", "…", "(hello)", "hello)", "(hello", "a,b", "a, b", "!!", "?!",
    "“quoted”", "\"quoted\"", "x—y—z", "—", "it!—That", "can;—but", "e.g.", "i.e.,", "Dr.", "St.John",
    "naïve", "café", "Ångström", "Zoë", "ﬁnance", "Ⅻ", "123", "3.14", "1,000", "10:30", "50%", "$5",
    "#hashtag", "@user", "foo_bar", "foo-bar", "foo/bar", "a&b", "C++", "x²", "½", "WWII", "NASA",
    "Llanfairpwllgwyngyll", "Tchaikovsky", "Nguyen", "Xiaoming", "Dvořák", "Łódź", "Ōsaka", "日本",
    "Привет", "hello world", "  spaced  ", "[bracket]", "{brace}", "«guillemets»", "¡Hola!", "¿Qué?",
    "a;b:c", "end.", ".start", "mid.dle", "tab\there", "x", "zzz", "hmm", "brrr", "shh", "pfft",
    "psst", "tsk", "mmhmm", "uh-huh", "ok", "OK", "ok.", "O'Brien", "rock'n'roll", "’tis", "y’all",
]


def main():
    common.assert_pins()
    from misaki import espeak, en
    fb = espeak.EspeakFallback(british=False)
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        for text in CASES:
            tk = en.MToken(text=text, tag="NN", whitespace="")
            ps, rating = fb(tk)
            raw = fb.backend.phonemize([text])
            out.write(json.dumps({"text": text, "phonemes": ps, "rating": rating, "phonemize": raw}, ensure_ascii=False) + "\n")
    print(len(CASES), "cases")


if __name__ == "__main__":
    main()
