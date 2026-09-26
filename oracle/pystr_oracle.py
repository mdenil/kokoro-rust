"""Exhaustive Python str-semantics table (pinned reference interpreter) for the frontend's pystr helpers:
per code point isalpha / isdigit (+ unicodedata.numeric value) / isspace / isupper, and the
lower() / upper() / title() mappings where they differ from identity. Output JSON (compact ranges).
Usage: python oracle/pystr_oracle.py <out.json>
"""
import json
import pathlib
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402


def ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        ok = not (0xD800 <= cp <= 0xDFFF) and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append([start, cp - 1])
            start = None
    if start is not None:
        out.append([start, 0x10FFFF])
    return out


def mapping(f):
    return {str(cp): f(chr(cp)) for cp in range(0x110000) if not (0xD800 <= cp <= 0xDFFF) and f(chr(cp)) != chr(cp)}


def main():
    common.assert_pins()
    digits = {str(cp): int(unicodedata.numeric(chr(cp))) for cp in range(0x110000)
              if not (0xD800 <= cp <= 0xDFFF) and chr(cp).isdigit()}
    table = {
        "python": sys.version.split()[0], "unicodedata": unicodedata.unidata_version,
        "isalpha": ranges(str.isalpha), "isdigit": digits, "isspace": ranges(str.isspace),
        "isupper": ranges(str.isupper), "lower": mapping(str.lower), "upper": mapping(str.upper),
        "title": mapping(str.title),
        # CPython handle_capital_sigma, probed through str.lower() itself: scanning away from U+03A3,
        # case-ignorable chars are skipped, then a cased char decides. P1 = "AΣx" keeps σ <=> x is a
        # (non-ignorable) cased char; P2 = "AΣxA" keeps σ <=> x ignorable or cased.
        "sigma_cased": ranges(lambda c: ("A\u03a3" + c).lower()[1] == "\u03c3"),
        "sigma_ignorable": ranges(lambda c: ("A\u03a3" + c + "A").lower()[1] == "\u03c3" and ("A\u03a3" + c).lower()[1] != "\u03c3"),
    }
    pathlib.Path(sys.argv[1]).write_text(json.dumps(table, ensure_ascii=False, sort_keys=True))
    print({k: len(v) if isinstance(v, (list, dict)) else v for k, v in table.items()})


if __name__ == "__main__":
    main()
