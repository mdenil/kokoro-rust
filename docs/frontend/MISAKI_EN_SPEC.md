# Behavioral spec: production English frontend (misaki 0.9.4 en.G2P + KPipeline chunking)

Source: pinned `misaki/en.py` (712 lines), `misaki/espeak.py`, `kokoro/pipeline.py` in the
reference venv. Cites are `en.py:<line>`. Configuration used by production: `KPipeline('a')` →
`en.G2P(trf=False, british=False, fallback=EspeakFallback(british=False), unk='')`, version None.

## S1 Pipeline (en.py:653-712)
1. `preprocess(text)` (509-539): `text.lstrip()`; markdown-style `[word](feature)` links are
   replaced by `word`; features: integer stress (`-2..`), `±0.5`, `/phonemes/` (explicit), `#flags#`
   (number flags); `tokens` = whitespace split of the result (for alignment only).
2. `tokenize` (541-566): spaCy `en_core_web_sm` 3.8.0 with components `tok2vec, tagger` → MTokens
   (text, tag_, whitespace_). Features are applied via spaCy alignment (only when present).
3. `fold_left` (568-573): merge non-head tokens into the previous (only produced by features).
4. `retokenize` (576-617): per token, `subtokenize` (regex en.py:55) unless alias/phonemes set;
   special handling: `$ £ €` with tag `$` → currency marker (phonemes '', rating 4); tag `:` and text
   `-`/`–` → '—'; tags in PUNCT_TAGS (68) and not all-letters → punctuation phonemes (69, PUNCTS);
   `CD` after a currency marker attaches currency to the last number sub-token; a `2` between letters
   → alias `to`. Sub-tokens without whitespace between them are grouped into lists (words).
5. Right-to-left resolution with `TokenContext(future_vowel, future_to)` (659-705): single tokens →
   `Lexicon(tk, ctx)`, then `fallback` if None; lists → longest-left-prefix merge search with the
   lexicon, fallback on unresolved non-junk sub-token (whole list to espeak), else `resolve_tokens`
   (626-651) for inter-subtoken spacing and stress demotion.
6. Merge lists (706); replace `ɾ`→`T`, `ʔ`→`t` (version != 2.0, 707-710); join phonemes + whitespace.

## S2 Lexicon (123-493)
- Data: `us_gold.json`, `us_silver.json` grown by `grow_dictionary` (125-136: add capitalized /
  lowercased variants for len ≥ 2); entries are strings or POS-keyed dicts with `DEFAULT`.
- Lookup order `get_word` (329-359): special cases (165-201: ADD symbols, SYMBOLS, dotted
  abbreviations → spelled NNP, a/am/an/I/by/to/in/the/vs/used with tag + context rules);
  case folding rule (334-343); known word → `lookup` (228-246: uppercase non-gold → lower + NNP
  handling, gold→silver, POS dict resolution incl. 'None' key when future_vowel is None, `get_NNP`
  letter spelling); possessive/plural `s'`/`'`; morphology `stem_s` (258-270), `stem_ed` (288-298),
  `stem_ing` (315-327) with `_s/_ed/_ing` phonological suffix rules (248-313; US flap `ɾ` rules).
- Stress `apply_stress` (93-118) with cap stresses (0.5 capitalized, 2 all-caps; 140, 480).
- Numbers `get_number` (370-449) using num2words cardinal/ordinal/year/float, digit-by-digit rules,
  `O` for internal zero in 3-digit numbers, currency pairs (dollars/cents, pounds/pence, euros/cents),
  suffixes s/'s/ed/'d/ing; `is_number` (466-474); NFKC + unicode numerics (478-479).
- Non-lexicon letters (not in LEXICON_ORDS 71) → None (fallback).

## S3 Fallback (espeak.py)
espeak-ng `en-us` via phonemizer (with_stress, tie '^', preserve punctuation) + E2M mapping table and
post-rules (US: o^ʊ→O, ɜː(ɹ)→ɜɹ, ɪə→iə, drop ː; 'o'→'ɔ'; ɾ→T, ʔ→t); rating 2. GPL-3 (owner decision).

## S4 Chunking (pipeline.py:174-221, 369-386)
Input split on `\n+`; per segment `g2p` then `en_tokenize`: accumulate tokens while phoneme count
(with spaces) ≤ 510; on overflow choose split via `waterfall_last` (174-189: last `!.?…`, then `:;`,
then `,—`, bumping over `)`/`”`, requiring the remainder ≤ 510), yield (graphemes, phonemes);
skip empty; production truncates ps > 510 with a warning (380-382) — the Rust CLI refuses instead.

## S5 Where POS tags matter (drives the tagger-port requirement)
special cases (a/am/an/I/by/to/in/the/vs/used), POS-keyed lexicon entries, NNP handling in lookup
and get_word, punctuation/currency/number handling in retokenize (tags `$`, `:`, PUNCT_TAGS, `CD`),
`future_to` context (tag TO/IN). The frontend oracle (oracle/frontend_oracle.py) records spaCy tags
so the native tagger can be tested for exact agreement, and how often a tag difference changes
phonemes.
