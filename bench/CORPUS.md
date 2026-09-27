# Benchmark corpus

`corpus_alice_ch1.txt` — Lewis Carroll, *Alice's Adventures in Wonderland*, Chapter I
(public domain), from Project Gutenberg eBook #11 (`https://www.gutenberg.org/cache/epub/11/pg11.txt`,
sha256 `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`, fetched 2026-09-26).

Construction (the only transformations): chapter body between "CHAPTER I." and "CHAPTER II.",
paragraphs whitespace-normalized, split into sentences at terminal punctuation followed by an
uppercase letter/opening quote, one utterance per line, Gutenberg `_emphasis_` underscores
removed. Text is otherwise exact (curly quotes, em-dashes, parentheses kept).

65 utterances, 8–590 characters (median 142). Long lines exceed 510 phonemes and exercise the
reference's waterfall chunking. sha256 of the file after underscore removal is recorded
with each benchmark result.

`corpus_alice_ch1-3.txt` — chapters I–III of the same book in one file, built from the same
pinned source with the same construction (`python3 bench/make_alice_full.py --chapters 3 pg11.txt
corpus_alice_ch1-3.txt`). 251 utterances (chapter I: 65, II: 92, III: 94), 8–590 characters (median
86), sha256 `8108bc77fafd71a7ce0bd66c59c75bd591da205eb6855003971134866f9a8ece`. Its first 65
lines are `corpus_alice_ch1.txt` exactly (`--chapters 1` reproduces that file). Used for the speed
comparison in docs/VALIDATION.md.

## Legal / provenance notes
- Project Gutenberg's eBook #11 page states the work is in the public domain in the USA. Users
  elsewhere should check their local law. The excerpt here contains no Project Gutenberg license
  text or trademark and is distributed as a transformed public-domain excerpt (transformations above).
- `corpus_alice_ch1.chunks.jsonl` holds phoneme chunks derived from that excerpt with the pinned
  misaki 0.9.4 frontend (Apache-2.0).
- Some performance figures (docs/VALIDATION.md) come from a private long-form text that is not in
  this repository.
