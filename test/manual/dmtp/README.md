# DMTP regression checks

The optional `SGLANG_DMTP_FUSED_T1_STATE=1` path requires
`SGLANG_MTP_PREFIX_TWO_DRAFT_GRAPHS=1`. It fuses request-slot T1 state
transport and validation; the default remains off. Slot reuse invalidates
cached proposals. The shared state module has no QSA attention dependency.
Use `test_turbo_fused_state.py` for byte-exact transport, request reuse,
sampling invalidation, and CPU/CUDA coverage.

Run from the repository root with a matching SGLang runtime:

```bash
python -m unittest discover -s test/manual/dmtp -p 'test_*.py' -v
```

The common dense path supports the configured tensor-parallel size; it is not
restricted to TP4. PP, attention DP/CP, and other unsupported recipes still fail
explicitly rather than silently changing the execution path.

`SGLANG_MTP_PREFIX_TWO_DRAFT_GRAPHS=1` records the MTP sampling/conditioning tail
and the DFlash sampling tail in their parent graphs. Request-indexed prefetched
proposals must be invalidated when slots are recycled or sampling state changes.
Graph weight/layout changes require a restart. The regression tests also cover
embedding/conditioner replacement and sampling-row refresh.

These operator and lifecycle tests do not replace real-model CUDA Graph, KV
parity, or tensor-parallel generation checks. QSA-specific optimizations are
maintained on `main`, alongside the non-QSA runtime.

## QSA GPU-state path

The new optimizations are opt-in; default behavior is unchanged. The validated
recipe uses native ratio-four QSA, seven proposals/eight verification slots,
exact batch-size capture, and at most 32 concurrent requests.

```bash
export SGLANG_DMTP_QSA_MTP_GRAPH=1
export SGLANG_DMTP_QSA_TWO_DRAFT_GRAPHS=1
export SGLANG_DMTP_QSA_BATCHED_STATE=1
export SGLANG_DMTP_QSA_FUSED_PREPARE=1
export SGLANG_DMTP_QSA_DIRECT_PACK=1
export SGLANG_DMTP_QSA_GPU_COMMIT=1
export SGLANG_DMTP_QSA_FUSED_STATE_READ=1
export SGLANG_DMTP_QSA_REUSE_SLOT_IDS=1
```

GPU commit packs accepted tokens, the bonus token, HC features, positions, and
KV addresses using device-resident acceptance metadata. The next T1 read checks
request-slot ownership, sequence length, anchor, and sampling state. Slot-ID
reuse is immutable and keyed by slot order, device, and CUDA stream.

Scheduler overlap additionally requires `SGLANG_DMTP_QSA_OVERLAP=1` and the
complete GPU-state recipe above. Do not pass `--disable-overlap-schedule` when
using it. Cached mutable batches, two-batch overlap, attention DP, prefix reuse,
chunked prefill, and quantized weights are not supported by this recipe.
`SGLANG_DMTP_QSA_CACHED_BATCH` remains a separate serial-only experiment.

Prompt initialization remains eager EXTEND. Committed MTP advancement uses the
native DRAFT_EXTEND_V2 path with private MTP KV tensors. These changes do not
alter the proposal budget, sparse proposal probabilities, or rejection sampler.
Real-model testing of the original worktree does not by itself establish
end-to-end distribution equivalence or GPU parity of a newly ported revision.
