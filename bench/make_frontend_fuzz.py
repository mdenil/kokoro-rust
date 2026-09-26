"""Deterministic synthetic frontend fuzz corpus (public, generated; no third-party text).

Mixes the constructs that drive misaki/spaCy branches: numbers in every format num2words and
misaki handle (cardinals, decimals, negatives, ordinals, years, times, currency, percentages,
ranges, fractions), abbreviations, acronyms, contractions, possessives, hyphenation, quotes and
dashes of every style, brackets, markdown link features, URLs/emails, accented / non-Latin
words, emoji, odd whitespace, all-caps, unicode digits, and very long lines (multi-chunk).

Usage: python bench/make_frontend_fuzz.py <out.txt> [n_lines=4000] [seed=20260926]
"""
import random
import sys

WORDS = (
    "the a an and or but if then because while after before over under into onto from with without "
    "house river garden window letter morning evening question answer reason story music doctor "
    "teacher city country road bridge table chair paper record present object minute content "
    "read lead live wind tear bow close use wound desert produce refuse project permit conduct "
    "quickly slowly never always perhaps certainly almost nearly really very quite rather "
    "walked talked jumped carried wondered believed remembered opened closed reached "
    "running singing reading writing building thinking looking waiting "
    "beautiful terrible wonderful curious important different strange ordinary"
).split()
NAMES = "Alice Bob Charlotte Dmitri Élodie François Nguyen Siobhan Tchaikovsky Zoë O'Brien McDonald MacLeod Xiaoming Łukasz".split()
ABBR = ["Dr.", "Mr.", "Mrs.", "Ms.", "St.", "Jr.", "Sr.", "Prof.", "e.g.", "i.e.", "etc.", "vs.", "a.m.", "p.m.", "U.S.", "U.K.", "Inc.", "No.", "Mt.", "Ave."]
ACRO = ["NASA", "FBI", "UNESCO", "CEO", "HTML", "TTS", "GPU", "DNA", "NATO", "OK", "USA", "EU", "AI", "PhD", "iPhone", "eBay"]
CONTRACT = ["don't", "won't", "can't", "I'm", "you're", "he'd", "she'll", "we've", "they'd've", "y'all", "'tis", "o'clock", "ma'am", "rock'n'roll", "it’s", "isn’t", "we’re"]
QUOTES = [('"', '"'), ("“", "”"), ("‘", "’"), ("'", "'"), ("«", "»"), ("(", ")"), ("[", "]"), ("{", "}")]
PUNCT_END = [".", "!", "?", "...", "…", "?!", "!!", ";", ":", ",", "", "—", " –"]
SYMBOLS = ["&", "%", "+", "=", "/", "#", "@", "*", "~", "^", "|", "<", ">", "°", "©", "™", "§", "†"]
UNI = ["café", "naïve", "résumé", "Ångström", "façade", "jalapeño", "Straße", "Dvořák", "Ōsaka", "İstanbul", "Привет", "日本", "ﬁnance", "Ⅻ", "½", "x²", "①", "٣", "５"]
EMOJI = ["🙂", "🎉", "❤️", "👍🏽", "🇺🇸"]


def number(r):
    k = r.randrange(22)
    if k == 0:
        return str(r.randrange(10))
    if k == 1:
        return str(r.randrange(10, 1000))
    if k == 2:
        return f"{r.randrange(1000, 10**7):,}"
    if k == 3:
        return str(r.randrange(10**6, 10**13))
    if k == 4:
        return f"{r.uniform(0, 1000):.{r.randrange(1, 4)}f}"
    if k == 5:
        return "-" + str(r.randrange(1, 5000))
    if k == 6:
        n = r.randrange(1, 200)
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    if k == 7:
        return str(r.randrange(1000, 2100))
    if k == 8:
        return f"{r.randrange(1, 13)}:{r.randrange(60):02d}"
    if k == 9:
        return f"${r.randrange(0, 5000)}" + (f".{r.randrange(100):02d}" if r.random() < 0.5 else "")
    if k == 10:
        return f"£{r.randrange(1, 900)}" + (f".{r.randrange(100):02d}" if r.random() < 0.3 else "")
    if k == 11:
        return f"€{r.randrange(1, 9000):,}" + (f".{r.randrange(100):02d}" if r.random() < 0.3 else "")
    if k == 12:
        return f"{r.randrange(0, 101)}%"
    if k == 13:
        a = r.randrange(1900, 2030)
        return f"{a}-{a + r.randrange(1, 20)}"
    if k == 14:
        return f"{r.randrange(1, 10)}/{r.randrange(2, 13)}"
    if k == 15:
        return f"{r.randrange(1, 32)}-{r.randrange(1, 13)}-{r.randrange(1990, 2030)}"
    if k == 16:
        return f"{r.randrange(100, 999)}-{r.randrange(1000, 9999)}"
    if k == 17:
        return f"{r.randrange(1, 100)}s" if r.random() < 0.5 else f"'{r.randrange(10, 99)}s"
    if k == 18:
        return f"{r.uniform(-50, 50):.1f}°"
    if k == 19:
        return f"{r.randrange(1, 60)}x"
    if k == 20:
        return f"#{r.randrange(1, 100)}"
    return f"{r.randrange(1, 10)}.{r.randrange(0, 10)}.{r.randrange(0, 10)}"


def link(r):
    w = r.choice(WORDS + NAMES)
    f = r.choice(["/kˈOkəɹO/", "/ɹˈɛd/", "-1", "+2", "0.5", "-0.5", "#y#", "#n#", "nothing", "", "1"])
    return f"[{w}]({f})"


def token(r):
    k = r.random()
    if k < 0.45:
        w = r.choice(WORDS)
        return w.upper() if r.random() < 0.03 else (w.capitalize() if r.random() < 0.08 else w)
    if k < 0.53:
        return r.choice(NAMES) + ("'s" if r.random() < 0.3 else "")
    if k < 0.65:
        return number(r)
    if k < 0.69:
        return r.choice(ABBR)
    if k < 0.73:
        return r.choice(ACRO)
    if k < 0.78:
        return r.choice(CONTRACT)
    if k < 0.81:
        return "-".join(r.choice(WORDS) for _ in range(r.randrange(2, 4)))
    if k < 0.84:
        return r.choice(UNI)
    if k < 0.86:
        return r.choice(SYMBOLS)
    if k < 0.88:
        return link(r)
    if k < 0.89:
        return r.choice(["https://example.com/a-b?c=1", "www.example.org", "someone@example.com", "C:\\temp\\file.txt", "file_name_v2.txt"])
    if k < 0.90:
        return r.choice(EMOJI)
    return r.choice(WORDS) + r.choice(["—", "–", "-", "/", "…", "."]) + r.choice(WORDS)


def sentence(r, n):
    toks = [token(r) for _ in range(n)]
    if r.random() < 0.25:
        i = r.randrange(len(toks))
        j = min(len(toks), i + r.randrange(1, 5))
        a, b = r.choice(QUOTES)
        toks[i] = a + toks[i]
        toks[j - 1] = toks[j - 1] + b
    s = " ".join(toks)
    if r.random() < 0.05:
        s = s.replace(" ", r.choice(["  ", "\t", " \u00a0", "\u2009"]), 1)
    s = s[:1].upper() + s[1:] if r.random() < 0.8 else s
    return s + r.choice(PUNCT_END)


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    r = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 20260926)
    lines = []
    for i in range(n):
        if i % 97 == 0:
            line = " ".join(sentence(r, r.randrange(12, 30)) for _ in range(r.randrange(8, 16)))  # long, multi-chunk
        else:
            line = " ".join(sentence(r, r.randrange(1, 18)) for _ in range(r.randrange(1, 4)))
        line = line.replace("\n", " ").strip()
        lines.append(line if line else "Empty.")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out, len(lines), "lines")


if __name__ == "__main__":
    main()
