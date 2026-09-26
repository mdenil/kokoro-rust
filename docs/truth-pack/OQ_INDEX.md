# OQ_INDEX — kokoro-rust open questions register

**Hard rule: no kernel ships against an unresolved [OPEN].** Answers must quote the pinned
source (installed kokoro-0.9.4 / misaki-0.9.4 packages) with file:line cites.

| OQ | Question | Status | Answer (line-cited) |
|---|---|---|---|
| OQ-1 | Exact KModel module graph + tensor shapes (bert, bert_encoder, predictor, text_encoder, decoder)? | [OPEN] | |
| OQ-2 | Stochastic paths: SineGen phase noise / SourceModuleHnNSF additive noise — RNG-driven in 0.9.4? Which ops, which shapes, where seeded? | [OPEN] | |
| OQ-3 | Voice pack layout + ref_s indexing rule (pack[len(ps)-1]?) + style split (s_ref=[:,:128] decoder vs s=[:,128:] prosody)? | [OPEN] | |
| OQ-4 | Phoneme→id mapping: config vocab only? BOS/EOS 0-padding? max length 510 enforcement + truncation behavior? | [OPEN] | |
| OQ-5 | Duration path semantics: sigmoid→sum→/speed→round→clamp exact ops + dtype; alignment matrix construction? | [OPEN] | |
| OQ-6 | ALBERT specifics: shared-layer count (12 iterations of 1 group?), embedding size vs hidden, position ids, attention mask handling, activation fn? | [OPEN] | |
| OQ-7 | LSTM inventory: which are bidirectional, hidden sizes, batch_first, dropout at eval, packed sequences? | [OPEN] | |
| OQ-8 | Conv inventory: weight_norm where, dilations, paddings, reflection pad, InstanceNorm eps/affine, AdaIN formula? | [OPEN] | |
| OQ-9 | Generator: upsample ConvTranspose params, source module harmonics count, noise convs, resblock structure (kernel/dilation), final conv → n_fft/2+1 spec, iSTFT (n_fft=20 hop=5 window?) exact torch.stft/istft semantics incl. center/padding? | [OPEN] | |
| OQ-10 | F0 frame rate ↔ sample rate math: total upsampling 10*6*5=300? F0 curve upsampling factor and where /speed enters output length? | [OPEN] | |
| OQ-11 | Reference device orientation: KModel device handling; does KPipeline hardcode anything CUDA-specific; fp32 vs fp16 default on CUDA? | [OPEN] | |
| OQ-12 | misaki en G2P: dictionary lookup vs espeak fallback split; spacy role; token-level determinism; stress marks; punctuation mapping? | [OPEN] | |
| OQ-13 | KPipeline text chunking: how long inputs split (per line? token limit?); en_callable; what exactly production feeds per synth call? | [OPEN] | |
| OQ-14 | Name traps: weight file dict nesting (module. prefixes?), weight_norm parametrizations (weight_g/weight_v vs parametrizations.*), key typos to preserve? | [OPEN] | |
| OQ-15 | Nondeterminism floor: CUDA vs CPU run-to-run deltas; thread-count sensitivity; dropout truly off at eval? | [OPEN] | |
| OQ-16 | Output normalization: any final clamp/normalize on waveform; sample dtype (f32) and WAV write path (soundfile) semantics? | [OPEN] | |
| OQ-17 | Which weights does 0.9.4 actually load from kokoro-v1_0.pth (unused keys? strict load?) — tensor census both sides | [OPEN] | |
| OQ-18 | License texts of kokoro + misaki wheels (dist-info) for redistribution notes | [OPEN] | |

Exit gate: zero blocking OQs — record date + commit here when reached.
