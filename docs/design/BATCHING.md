# Batched CUDA forward (design)

Goal: many chunks per forward on the RTX 4090 without changing per-item semantics.
- Correctness bar at design time (the retired f32 engine): each item's batched output matched its
  single-item output within fixed per-case drift bounds, with EXACT ids / durations / sample
  counts.
- In the current BF16x product the predictor runs in BF16, so batch shape can change predicted
  durations on some lines.
- For fixed inputs and batching options, outputs are reproducible byte for byte across runs
  (tests/bf16x_regression.rs), as long as every batch fits in GPU memory.
- When a batch runs out of GPU memory it is split in half, down to single items (a message is
  printed to stderr). Because batch shape affects the BF16 predictor, a split can change the
  audio and durations of the affected lines. Other GPU work competing for memory can therefore
  make otherwise identical runs differ.

## Ragged layout with zero gaps (no padding-to-max)
Items are concatenated along time with zero gaps. Two layouts:
- token layout: item b occupies rows [ot_b, ot_b + T_b) with gap Gt >= 2 (text-encoder k5 halo).
- frame layout: item b starts at N-domain offset A_b with gap Gn >= 2. Every other time domain is a
  fixed multiple of it, so all ops that scale time keep items aligned without remapping:
  2N (F0/N curves, decoder tail): 2·A_b · 20N (generator stage 0): 20·A_b ·
  120N+1 (stage 1 and STFT frames): 120·A_b (segment length 120·N_b + 1) · samples: 600·A_b.
  Gap sizes then exceed every halo: stage 0 max conv reach 25 (k11 d5) <= 20·Gn; stage 1 reach 25+1
  (reflect shift) <= 120·Gn; stride-6 noise conv needs stage-1 offsets ≡ 0 (mod 6) — holds.

## Per-op rules
- Pointwise / per-row ops (linear, LN rows, activations, snake, AdaIN apply): whole buffer; gap
  values may become garbage and are never read unmasked.
- Stride-1 convs (per-tap GEMM): gap columns of the INPUT are re-zeroed before the taps (mask
  kernel with per-domain segment table) → identical to per-item zero padding.
- ConvT (ups, k-2p = s) and nearest×2 / depthwise ConvT: position-equivariant; zero input gaps give
  exactly per-item outputs on the scaled layout.
- Stride-6 noise conv: aligned offsets (above) make each output window per-item.
- Reflection pad (1,0) before stage 1: per-item kernel (shift segment right by one, copy x[1]).
- Instance-norm statistics: per (item, channel) over the item's valid span only.
- AdaIN / AdaLN gamma, beta: per item (ref_s = pack[len(phonemes)-1] differs per item).
- LSTMs: one kernel runs all items; step s processes t = s (fwd) or T_b-1-s (bwd) for items with
  s < T_b; W_hh row held in registers, dotted with every item's h (weight reuse).
- ALBERT attention: per item (no cross-item attention); batched over heads and items.
- Durations/alignment: per item on the host (exact); expand kernels use per-frame source tables.
- SineGen / noise / STFT / iSTFT: per item (own f0 span, own counter-based noise stream keyed by the
  item's seed — no cross-item noise coupling), reflect padding at item boundaries.

## Scheduler
Length-bucketed microbatches: sort pending chunks by phoneme length (proxy for frames), fill a batch
up to a frame budget derived from free VRAM (override flags; OOM → halve and retry, single-item
fallback). Results are restored to (line, chunk) order before joining per line.

## Tests (bounds fixed before judging)
batch vs single (short/long/tail batches, reversed and shuffled orders, batch sizes 1..max);
negative controls: disabled gap masking, a neighbour's noise stream or style swapped in, a loud
neighbour item — each must be DETECTED; full coverage (every item present, sample counts exact).
