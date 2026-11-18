"""Fused QSA metadata versus the original packed-row construction."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.speculative.dmtp_qsa_pack import (
    pack_committed_inputs,
    prepare_append,
    stage_graph_inputs,
)


class PackTest(unittest.TestCase):
    def exercise(self, device):
        for bs in range(1, 33):
            for private in (False, True):
                counts = [(i % 8) + 1 for i in range(bs)]
                prefixes = [2044 + i * 13 for i in range(bs)]
                slots = torch.arange(bs, 0, -1, dtype=torch.int64, device=device)
                loc = (
                    torch.arange(sum(counts), dtype=torch.int64, device=device) + 40000
                )
                table = None
                # Prefix range includes both sides of the QSA selection boundary.
                if private:
                    table = torch.arange(40 * 8192, device=device).view(40, 8192)[
                        :, ::2
                    ]
                actual = prepare_append(slots, prefixes, counts, loc, table)
                lens = torch.tensor(counts, device=device, dtype=torch.int32)
                prefix = torch.tensor(prefixes, device=device, dtype=torch.int32)
                pos = torch.cat(
                    [
                        torch.arange(p, p + n, device=device)
                        for p, n in zip(prefixes, counts)
                    ]
                )
                expected_loc = (
                    table[torch.repeat_interleave(slots, lens.long()), pos].long()
                    if private
                    else loc
                )
                expected = (
                    lens,
                    prefix,
                    prefix + lens,
                    pos,
                    expected_loc,
                    lens.cumsum(0, dtype=torch.int32) - lens,
                    pos[None].expand(3, -1).contiguous(),
                    lens - 1,
                    lens.long().cumsum(0) - 1,
                )
                for a, r in zip(actual, expected):
                    self.assertEqual(a.dtype, r.dtype)
                    self.assertTrue(torch.equal(a, r))

    def test_cpu_original_contract(self):
        self.exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel requires a GPU")
    def test_cuda_original_contract(self):
        self.exercise("cuda")

    def test_invalid_counts_and_context_rejected(self):
        slots, locations = torch.tensor([1]), torch.tensor([7])
        with self.assertRaises(ValueError):
            prepare_append(slots, [1], [0], locations)
        with self.assertRaises(ValueError):
            prepare_append(
                slots, [8], [1], locations, torch.zeros(3, 8, dtype=torch.int64)
            )

    def staging(self, device):
        capacity, width = 256, 10240
        buffers = SimpleNamespace(
            input_ids=torch.empty(capacity, device=device, dtype=torch.int64),
            positions=torch.empty(capacity, device=device, dtype=torch.int64),
            out_cache_loc=torch.empty(capacity, device=device, dtype=torch.int64),
            mrope_positions=torch.empty(3, capacity, device=device, dtype=torch.int64),
            hidden_states=torch.empty(
                capacity, width, device=device, dtype=torch.bfloat16
            ),
        )
        pointers = {k: v.data_ptr() for k, v in vars(buffers).items()}
        for bs in [32, 1, 29, 8, 2, 32]:
            n, count = bs * 8, bs * 5
            ids = torch.arange(count, device=device, dtype=torch.int64)
            batch = SimpleNamespace(
                input_ids=ids,
                positions=ids + 2044,
                out_cache_loc=ids + 50000,
                mrope_positions=(ids + 2044)[None].expand(3, -1),
                spec_info=SimpleNamespace(
                    hidden_states=torch.randn(
                        count, width, device=device, dtype=torch.bfloat16
                    )
                ),
            )
            stage_graph_inputs(batch, buffers, n)
            for name in ["input_ids", "positions", "out_cache_loc"]:
                self.assertTrue(
                    torch.equal(getattr(buffers, name)[:count], getattr(batch, name))
                )
                self.assertEqual(
                    getattr(buffers, name)[count:n].count_nonzero().item(), 0
                )
            self.assertTrue(
                torch.equal(
                    buffers.hidden_states[:count].view(torch.uint8),
                    batch.spec_info.hidden_states.view(torch.uint8),
                )
            )
            self.assertEqual(buffers.hidden_states[count:n].count_nonzero().item(), 0)
            self.assertTrue(
                torch.equal(buffers.mrope_positions[:, :count], batch.mrope_positions)
            )
            self.assertEqual(
                buffers.mrope_positions[:, count:n].count_nonzero().item(), 0
            )
            self.assertEqual(
                pointers, {k: v.data_ptr() for k, v in vars(buffers).items()}
            )

    def test_cpu_staging_and_stable_addresses(self):
        self.staging("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel requires a GPU")
    def test_cuda_staging_and_stable_addresses(self):
        self.staging("cuda")

    def direct(self, device):
        from sglang.srt.speculative.dmtp_qsa_contract import committed_inputs

        for bs in range(1, 33):
            for private in (False, True):
                capacity, width = 256, 64
                counts = [(i % 8) + 1 for i in range(bs)]
                prefix = [2044 + i * 13 for i in range(bs)]
                lens = torch.tensor(counts, device=device, dtype=torch.int32)
                candidates = torch.arange(bs * 16, device=device).view(bs, 16)[:, ::2]
                bonus = torch.arange(bs * 2, device=device)[::2] + 100000
                hc = torch.randn(bs, 8, width * 2, device=device, dtype=torch.bfloat16)[
                    ..., ::2
                ]
                locations = (
                    torch.arange(bs * 16, device=device).view(bs, 16)[:, ::2] + 50000
                )
                reqs = torch.arange(bs, 0, -1, device=device, dtype=torch.int64)
                table = (
                    torch.arange(40 * 8192, device=device).view(40, 8192)[:, ::2]
                    if private
                    else None
                )
                buffers = SimpleNamespace(
                    input_ids=torch.empty(capacity, device=device, dtype=torch.int64),
                    positions=torch.empty(capacity, device=device, dtype=torch.int64),
                    out_cache_loc=torch.empty(
                        capacity, device=device, dtype=torch.int64
                    ),
                    hidden_states=torch.empty(
                        capacity, width, device=device, dtype=torch.bfloat16
                    ),
                    mrope_positions=torch.empty(
                        3, capacity, device=device, dtype=torch.int64
                    ),
                    totals=torch.empty(32, device=device, dtype=torch.int32),
                    correct=torch.empty(32, device=device, dtype=torch.int32),
                    last=torch.empty(32, device=device, dtype=torch.int64),
                )
                packed = pack_committed_inputs(
                    candidates,
                    bonus,
                    hc,
                    locations,
                    reqs,
                    prefix,
                    counts,
                    table,
                    buffers,
                )
                valid = torch.arange(8, device=device)[None] < lens[:, None]
                expected_tokens = committed_inputs(candidates, lens, bonus)[valid]
                expected = prepare_append(reqs, prefix, counts, locations[valid], table)
                for a, r in [
                    (packed.tokens, expected_tokens),
                    (packed.hc, hc[valid]),
                    (packed.positions, expected[3]),
                    (packed.locations, expected[4]),
                    (packed.totals, expected[2]),
                    (packed.correct, expected[7]),
                    (packed.select, expected[8]),
                    (packed.mrope, expected[6]),
                ]:
                    self.assertTrue(
                        torch.equal(
                            a.contiguous().view(torch.uint8),
                            r.contiguous().view(torch.uint8),
                        )
                    )
                for field in [
                    "input_ids",
                    "positions",
                    "out_cache_loc",
                    "hidden_states",
                ]:
                    self.assertEqual(
                        getattr(buffers, field)[sum(counts) : bs * 8]
                        .count_nonzero()
                        .item(),
                        0,
                    )
                self.assertEqual(
                    buffers.mrope_positions[:, sum(counts) : bs * 8]
                    .count_nonzero()
                    .item(),
                    0,
                )
                if device == "cuda":
                    gpu = pack_committed_inputs(
                        candidates,
                        bonus,
                        hc,
                        locations,
                        reqs,
                        None,
                        None,
                        table,
                        buffers,
                        accept_len=lens - 1,
                        new_lengths=torch.tensor(prefix, device=device) + lens,
                    )
                    self.assertTrue(gpu.gpu_only)
                    self.assertIsNone(gpu.host_lengths)
                    self.assertIsNone(gpu.host_prefixes)
                    self.assertEqual(gpu.tokens.numel(), bs * 8)
                    n = sum(counts)
                    for a, r in [
                        (gpu.tokens[:n], expected_tokens),
                        (gpu.hc[:n], hc[valid]),
                        (gpu.positions[:n], expected[3]),
                        (gpu.locations[:n], expected[4]),
                        (gpu.lens, expected[0]),
                        (gpu.prefix, expected[1]),
                        (gpu.start, expected[5]),
                        (gpu.totals, expected[2]),
                        (gpu.correct, expected[7]),
                        (gpu.select, expected[8]),
                        (gpu.mrope[:, :n], expected[6]),
                    ]:
                        self.assertTrue(
                            torch.equal(
                                a.contiguous().view(torch.uint8),
                                r.contiguous().view(torch.uint8),
                            )
                        )
                    for field in [
                        "input_ids",
                        "positions",
                        "out_cache_loc",
                        "hidden_states",
                    ]:
                        self.assertEqual(
                            getattr(buffers, field)[n : bs * 8].count_nonzero().item(),
                            0,
                        )
                    self.assertEqual(
                        buffers.mrope_positions[:, n : bs * 8].count_nonzero().item(), 0
                    )

    def test_cpu_direct_commit(self):
        self.direct("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel requires a GPU")
    def test_cuda_direct_commit(self):
        self.direct("cuda")
