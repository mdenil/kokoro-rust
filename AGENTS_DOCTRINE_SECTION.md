# kokoro Engineering Doctrine (paste into AGENTS.md; adapt bracketed parts)

<!-- Source: franken_ocr AGENTS.md doctrine, proven over 591 commits. Keep the numbering:
     code comments will cite "doctrine #N". -->

0. **The anti-ceremony counterweight.** A process artifact may exist only as a hard gate for
   a named capability — at creation it names its consumer, the gate it enforces, the OBSERVED
   defect class justifying it, and its deletion condition (the test: does running code or a
   release gate branch on it?). Process work earns zero capability credit; process share is a
   diagnostic, never a quota. Honest-credit floor: real code + real tests in the same work
   item; no fixture-as-live-proof; no assertion weakening; refusal-only stays open; closes
   come only from the verify orchestrator with cited evidence — a false close is reopened
   with an incident comment.

1. **Correctness outranks speed, always (G1 > G2).** Parity gate FIRST, perf second. A faster
   kernel that drifts the output is reverted — no source landed — and memorialized in
   `docs/NEGATIVE_EVIDENCE.md`. We ship speed *on top of* parity, never instead of it.

2. **The quant recipe is fixed and validated: quantize the decoder FFN/expert GEMMs only.**
   Keep high precision (BF16/F32): the entire vision/encoder tower, projector, embed_tokens,
   any router gates, and ALL norms. int8 on attention q/k/v/o and on lm_head go *beyond* the
   validated set — gate them behind a measured-<END_METRIC> kill-switch, never assume lossless.

3. **NEVER hand-roll wide-SIMD over scalar inner loops.** It measured ~5× SLOWER than LLVM
   autovectorization in sibling repos, and LLVM-autovec later beat even the hand-SDOT dense
   GEMV at m=1. The winning levers are (a) full-core-parallel forward + (b) native int8
   *matmul* intrinsics (SDOT/VNNI; register-blocked SMMLA where full-rate), with LLVM
   autovectorizing the elementwise/norm/softmax/dequant glue. Re-prove hand-SIMD vs autovec
   per toolchain major.

4. **The edge is kernels-at-peak, NOT framework overhead.** A naive "fused tape-free forward"
   of scalar-f32 ops regressed 3–10×; an un-blocked SMMLA was slower than SDOT (load-bound);
   accelerator-f32 did not beat CPU-int8. The win is the combination by construction: a fused,
   tape-free, zero-per-op-allocation single-model forward with EVERY op at peak. Honest bar:
   at-or-near the best <ONNX_CLASS_RUNTIME> (the peak-kernel CPU runtime — NOT the reference
   framework, which you should beat outright), portable where it can't build.

5. **NEVER nest rayon under a held lock; NEVER nest a second async runtime inside a task.**
   Single model behind a cache; sequential outer item loop; each forward fans out internally
   via the kernel's rayon pool (pinned physical cores, one live forward at a time). A
   `many_items_without_deadlock` CI watchdog (items ≫ pool) hangs on regression.

6. **int8 i32-accumulation overflow is a proof obligation, not an assumption.** Worst case here
   is <TENSOR> at K=<K> (U8S8 ≤ <BOUND>); it must be proven by a unit test at worst-case K with
   all-extreme operands on every arch. Do NOT inherit another repo's K bound.

7. **Model semantics you didn't read will burn you.** Decode memory/compute laws (window/ring
   semantics, cross-item dependence, generation-processor flags) come from the pinned source
   via the OQ register, never from assumptions. **Hard rule: no kernel ships against an
   unresolved [OPEN]. A phase exit gate cannot pass while it depends on an unresolved OQ.**

8. **Honest, measured everything.** Every accepted numeric divergence → `docs/DISCREPANCIES.md`
   (reference behavior, our impl, MEASURED impact, kill-switch env var, review date). Every
   rejected optimization → `docs/NEGATIVE_EVIDENCE.md` (the 5-pass loop: claim+baseline → one
   lever + bit-exact proof → rebench + score → keep/revert → next hotspot). Head-to-heads use
   thread/allocator/precision fairness controls — never benchmark the reference oversubscribed.
   No silent numerics changes, ever.

9. **Two binaries from one entrypoint** (short + long name): shared `pub fn cli_main()` in the
   lib; each `[[bin]]` is a thin one-line shim pointing at its OWN shim file.

## Adaptive-Controller Contract
No adaptive/heuristic controller (thread-count tuner, admission control, quality voter) ships
without: explicit state/action/loss definition, a calibration metric, a deterministic fallback
trigger, and an evidence artifact. Deterministic fallback is mandatory.

## Session Close Protocol
1. File tracker issues for remaining work. 2. Run quality gates if code changed. 3. Update
issue statuses. 4. Flush tracker to JSONL + `git add` it. 5. Commit, push, hand off with
gates-run + remaining-risks summary.
