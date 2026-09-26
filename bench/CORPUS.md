# Benchmark corpus

`corpus_alice_ch1.txt` — Lewis Carroll, *Alice's Adventures in Wonderland*, Chapter I
(public domain), from Project Gutenberg eBook #11 (`https://www.gutenberg.org/cache/epub/11/pg11.txt`,
sha256 `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`, fetched 2026-09-26).

Construction (the only transformations): chapter body between "CHAPTER I." and "CHAPTER II.",
paragraphs whitespace-normalized, split into sentences at terminal punctuation followed by an
uppercase letter/opening quote, one utterance per line, Gutenberg `_emphasis_` underscores
removed. Text is otherwise exact (curly quotes, em-dashes, parentheses kept).

65 utterances, 8–596 characters (median 142). Long lines exceed 510 phonemes and exercise the
reference's waterfall chunking. sha256 of the file after underscore removal is recorded in
the benchmark receipts.
