"""Storage lifetime/reorder contracts; CUDA tests exercise the fused kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.speculative.dmtp_qsa_state import PrefetchedT1State


class StateTest(unittest.TestCase):
    def exercise(self, device):
        for bs in range(1, 33):
            bank = PrefetchedT1State(40)
            slots = torch.arange(1, bs + 1, device=device, dtype=torch.int64)
            values = (
                torch.arange(bs, device=device).reshape(bs, 1),
                torch.arange(bs * 16, device=device).reshape(bs, 1, 16),
                torch.randn(bs, 1, 16, device=device),
                torch.randn(bs, 8, 2560, device=device, dtype=torch.bfloat16),
                torch.rand(bs, device=device),
                torch.arange(bs, device=device) % 2 == 0,
                torch.arange(bs, device=device, dtype=torch.int64) + 2048,
                torch.arange(bs, device=device, dtype=torch.int64) + 30000,
            )
            reference = tuple(value.clone() for value in values)
            bank.save(slots, values)
            for value in values:
                value.zero_()
            order = torch.arange(bs - 1, -1, -1, device=device)
            actual = bank.load(slots[order])
            for a, r in zip(actual, reference):
                self.assertTrue(
                    torch.equal(a.view(torch.uint8), r[order].view(torch.uint8))
                )
            # A recycled slot replaces all fields, without affecting its peers.
            replacement = tuple(value[:1] for value in values)
            bank.save(slots[:1], replacement)
            updated = bank.load(slots)
            for a, r in zip(updated, reference):
                self.assertTrue(torch.equal(a[1:], r[1:]))
                self.assertEqual(a[0].count_nonzero().item(), 0)
            # A previously returned batch owns its memory across bank writes.
            for a, r in zip(actual, reference):
                self.assertTrue(torch.equal(a, r[order]))

    def test_cpu_lifetime_and_reorder(self):
        self.exercise("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel requires a GPU")
    def test_fused_cuda_byte_exact(self):
        self.exercise("cuda")

    def test_layout_change_rejected(self):
        bank = PrefetchedT1State(4)
        slots = torch.tensor([1, 2])
        bank.save(slots, (torch.ones(2, 3),))
        with self.assertRaisesRegex(ValueError, "layout changed"):
            bank.save(slots, (torch.ones(2, 4),))

    def _controller_checks(self, fused, reuse=False):
        from sglang.srt.speculative.dmtp_qsa_worker import PrefixState, QSAController
        from sglang.srt.speculative.dspark_components import dspark_draft

        c = QSAController.__new__(QSAController)
        c.gpu_request_state, c.batched_state, c.two_graphs = True, True, True
        c.fused_state_read = fused
        c.reuse_slot_ids = reuse
        c.state_bank = PrefetchedT1State(5)
        a, b = SimpleNamespace(), SimpleNamespace()
        c.states = {
            1: PrefixState(a, 2048, 70, None, prepared=True),
            2: PrefixState(b, 4096, 90, None, prepared=True),
        }
        c.graph_used = {}
        c.mtp_tail = SimpleNamespace(signature=lambda: (), initial_signature=())
        batch = SimpleNamespace(
            reqs=[b, a],
            req_pool_indices=torch.tensor([2, 1]),
            seq_lens=torch.tensor([4096, 2048]),
            sampling_info=SimpleNamespace(temperatures=torch.ones(2, 1)),
        )
        values = (
            torch.tensor([[11], [12]]),
            torch.arange(32).view(2, 1, 16),
            torch.ones(2, 1, 16) / 16,
            torch.randn(2, 8, 16),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            torch.tensor([2048, 4096]),
            torch.tensor([70, 90]),
        )
        c.state_bank.save(torch.tensor([1, 2]), values)
        block = torch.zeros(2, 8, dtype=torch.int64)
        block[:, 0] = torch.tensor([90, 70])
        with patch.object(
            dspark_draft,
            "resolve_greedy_mask",
            return_value=torch.zeros(2, dtype=torch.bool),
        ):
            c.t1(batch, block)
            self.assertEqual(block[:, 1].tolist(), [12, 11])
            self.assertTrue(torch.equal(c.t1_sparse[0], values[1].flip(0)))
            for field in ("length", "anchor", "slot", "identity"):
                with self.subTest(field=field):
                    if field == "length":
                        batch.seq_lens[0] += 1
                    elif field == "anchor":
                        block[0, 0] += 1
                    elif field == "slot":
                        batch.req_pool_indices[0] = 1
                    else:
                        batch.reqs[0] = SimpleNamespace(req_pool_idx=2)
                    with self.assertRaises(RuntimeError):
                        c.t1(batch, block)
                    batch.seq_lens[0] = 4096
                    block[0, 0] = 90
                    batch.req_pool_indices[0] = 2
                    batch.reqs[0] = b

    def test_gpu_state_controller_reorder_and_invalidations(self):
        self._controller_checks(False)

    def test_fused_read_controller_reorder_and_invalidations(self):
        self._controller_checks(True)

    def test_reused_slot_ids_keep_controller_invalidations(self):
        self._controller_checks(True, reuse=True)

    def _slot_id_cache(self, device):
        from sglang.srt.speculative.dmtp_qsa_worker import QSAController

        c = QSAController.__new__(QSAController)
        c.reuse_slot_ids = True
        slots = [2, 1]
        first = c._expected_slot_tensor(slots, device)
        with patch("torch.tensor", side_effect=AssertionError("Unexpected allocation")):
            self.assertIs(c._expected_slot_tensor([2, 1], device), first)
        slots.reverse()
        second = c._expected_slot_tensor(slots, device)
        self.assertEqual(first.tolist(), [2, 1])
        self.assertEqual(second.tolist(), [1, 2])
        self.assertIsNot(first, second)
        self.assertIsNot(c._expected_slot_tensor([2, 1], device), first)
        c.reuse_slot_ids = False
        self.assertIsNot(c._expected_slot_tensor([2, 1], device), first)

    def test_slot_id_cache_cpu_ownership_and_reorder(self):
        self._slot_id_cache("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA cache test requires a GPU")
    def test_slot_id_cache_cuda_stream_and_device(self):
        from sglang.srt.speculative.dmtp_qsa_worker import QSAController

        self._slot_id_cache(torch.device("cuda", torch.cuda.current_device()))
        c = QSAController.__new__(QSAController)
        c.reuse_slot_ids = True
        device = torch.device("cuda", torch.cuda.current_device())
        first = c._expected_slot_tensor([2, 1], device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            second = c._expected_slot_tensor([2, 1], device)
            self.assertIsNot(first, second)
            self.assertIs(c._expected_slot_tensor([2, 1], device), second)
        stream.synchronize()
        self.assertEqual(first.tolist(), second.tolist())
        self.assertIsNot(c._expected_slot_tensor([2, 1], "cpu"), second)

    def _checked_read(self, device):
        for bs in range(1, 33):
            bank = PrefetchedT1State(40)
            slots = torch.arange(1, bs + 1, device=device, dtype=torch.int64)
            values = (
                slots[:, None] + 100,
                torch.arange(bs * 16, device=device).view(bs, 1, 16),
                torch.randn(bs, 1, 16, device=device),
                torch.randn(bs, 8, 2560, device=device, dtype=torch.bfloat16),
                torch.arange(bs, device=device, dtype=torch.float32) + 0.25,
                slots % 2 == 0,
                slots + 2048,
                slots + 30000,
            )
            # Copies must preserve bit patterns, including NaNs and signed zero.
            values[2].view(torch.int32).flatten()[:4] = torch.tensor(
                [0x7FC01234, -2147483648, 0x7F800000, -8388608],
                device=device,
                dtype=torch.int32,
            )
            values[3].view(torch.int16).flatten()[:4] = torch.tensor(
                [0x7FC1, -32768, 0x7F80, -128], device=device, dtype=torch.int16
            )
            bank.save(slots, values)
            order = torch.arange(bs - 1, -1, -1, device=device)
            selected = slots[order]
            block = torch.full((bs, 16), -7, device=device, dtype=torch.int64)[:, ::2]
            block[:, 0] = values[7][order]
            expected = selected.clone()
            lens, temp, greedy = (values[i][order] for i in (6, 4, 5))
            args = (selected, expected, lens, block, temp, greedy)
            result = bank.load_checked(*args)
            for actual, reference in zip(result, values):
                self.assertTrue(
                    torch.equal(
                        actual.view(torch.uint8), reference[order].view(torch.uint8)
                    )
                )
            self.assertTrue(torch.equal(block[:, 1], values[0][order, 0]))
            self.assertTrue(torch.all(block[:, 2:] == -7))
            for field in (
                "slot",
                "low",
                "high",
                "length",
                "anchor",
                "temp",
                "nan_temp",
                "greedy",
            ):
                changed = [x.clone() for x in args]
                if field == "slot":
                    changed[1][0] += 1
                elif field == "low":
                    changed[0][0] = 0
                elif field == "high":
                    changed[0][0] = 40
                elif field == "length":
                    changed[2][0] += 1
                elif field == "anchor":
                    changed[3][0, 0] += 1
                elif field == "temp":
                    changed[4][0] += 1
                elif field == "nan_temp":
                    changed[4][0] = float("nan")
                else:
                    changed[5][0] = ~changed[5][0]
                changed[3][:, 1] = -99
                # Exercise the same device comparison without poisoning the
                # CUDA context with the public API's intentional assertion.
                _, valid = bank._checked_values(*changed)
                self.assertFalse(valid[0].item(), field)
                self.assertTrue(valid[1:].all().item(), field)
                self.assertEqual(changed[3][0, 1].item(), -99)
                if device == "cpu":
                    with self.assertRaises(RuntimeError):
                        bank.load_checked(*changed)
            bank.save(slots, tuple(torch.zeros_like(v) for v in values))
            for actual, reference in zip(result, values):
                self.assertTrue(
                    torch.equal(
                        actual.view(torch.uint8), reference[order].view(torch.uint8)
                    )
                )

    def test_checked_cpu_read(self):
        self._checked_read("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel requires a GPU")
    def test_checked_cuda_byte_exact(self):
        self._checked_read("cuda")

    def test_pool_initialization_uses_passed_mapping_not_outer_proxy(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
        from sglang.srt.speculative.dmtp_qsa_worker import QSAMTPWorker

        worker = QSAMTPWorker.__new__(QSAMTPWorker)
        worker.req_to_token_pool = None
        worker._dmtp_controller = SimpleNamespace(batched_state=True, native=Mock())
        mapping = SimpleNamespace(req_to_token=torch.zeros(41, 8, dtype=torch.int32))
        with patch.object(DFlashWorkerV2, "alloc_memory_pool") as parent:
            worker.alloc_memory_pool(None, mapping, None)
        parent.assert_called_once_with(None, mapping, None)
        worker._dmtp_controller.native.alloc_memory_pool.assert_called_once_with(
            None, mapping, None
        )
        self.assertEqual(worker._dmtp_controller.state_bank.capacity, 41)
