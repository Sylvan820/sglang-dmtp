"""The real controller/native GPU-metadata branch must not read token lengths back."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from sglang.srt.speculative.dmtp_qsa_native import NativeQSAMTP
from sglang.srt.speculative.dmtp_qsa_state import PrefetchedT1State
from sglang.srt.speculative.dmtp_qsa_worker import PrefixState, QSAController


class NoReadTensor(torch.Tensor):
    def tolist(self):
        raise AssertionError("Unexpected device-to-host list")

    def item(self):
        raise AssertionError("Unexpected device-to-host scalar")


class GPUCommitTest(unittest.TestCase):
    def test_controller_does_not_read_acceptance_back(self):
        c = QSAController.__new__(QSAController)
        c.direct_pack = c.gpu_commit = c.gpu_request_state = c.batched_state = (
            c.two_graphs
        ) = True
        c.graph_used = {"target": True, "draft": True}
        c.checked_target = {2}
        c.current_slots = [2, 1]
        requests = [object(), object()]
        c.batch = NS(reqs=requests, req_pool_indices=torch.tensor([2, 1]))
        c.sampling = object()
        c.worker = NS(model_runner=NS(model=NS(model=NS(hc_count=1, hidden_size=4))))
        c.mtp_tail = NS(
            stage=Mock(),
            temperatures=torch.ones(2),
            greedy_mask=torch.zeros(2, dtype=torch.bool),
        )
        c.state_bank = PrefetchedT1State(4)
        c.states = {
            s: PrefixState(r, 10, 20, None, prepared=True)
            for s, r in zip([2, 1], requests)
        }
        result = (
            torch.zeros(2, 1, dtype=torch.int64),
            torch.zeros(2, 1, 16, dtype=torch.int64),
            torch.ones(2, 1, 16) / 16,
            torch.randn(2, 8, 4),
        )
        marker = object()
        c.native = NS(
            graph=NS(input_staging=marker, tail_result=result),
            forward=Mock(return_value=(torch.zeros(2, 4), None)),
        )
        x = torch.arange(16).view(2, 8)
        accept = torch.tensor([1, 4]).as_subclass(NoReadTensor)
        totals = torch.tensor([20, 40]).as_subclass(NoReadTensor)
        bonus = torch.tensor([101, 202]).as_subclass(NoReadTensor)

        def packed(
            candidates, bonus, hc, loc, reqs, prefix, lengths, mapping, buffers, **kw
        ):
            self.assertIsNone(prefix)
            self.assertIsNone(lengths)
            self.assertIs(buffers, marker)
            self.assertIs(kw["accept_len"], accept)
            return NS(
                tokens=candidates.flatten(),
                hc=hc.flatten(0, 1),
                locations=loc.flatten(),
            )

        with patch(
            "sglang.srt.speculative.dmtp_qsa_pack.pack_committed_inputs",
            side_effect=packed,
        ):
            c.commit(x, accept, bonus, None, totals, x + 100, hc=torch.randn(16, 4))
        for slot, req in zip([2, 1], requests):
            self.assertIs(c.states[slot].request, req)
            self.assertIsNone(c.states[slot].length)
            self.assertIsNone(c.states[slot].anchor)
        values = c.state_bank.load(c.batch.req_pool_indices)
        self.assertTrue(torch.equal(values[-2], totals))
        self.assertTrue(torch.equal(values[-1], bonus))
        c.native.forward.assert_called_once()

    def test_native_retains_device_counts_without_cpu_mirror(self):
        native = NativeQSAMTP.__new__(NativeQSAMTP)
        plan = NS(bind=Mock())
        native.runner = NS(runner_ids=NS(plan=Mock(return_value=plan)))
        native.execution_scope = nullcontext
        native.graph_fields = {}
        native.hidden_only = True
        buffers = object()
        native.graph = NS(
            fused_prepare=True,
            input_staging=buffers,
            checked={2},
            execute=Mock(return_value=(torch.ones(2, 4), None)),
        )
        counts = torch.tensor([2, 5], dtype=torch.int32).as_subclass(NoReadTensor)
        prefix = torch.tensor([2047, 4093], dtype=torch.int32).as_subclass(NoReadTensor)
        prepared = NS(
            gpu_only=True,
            buffers=buffers,
            rows=16,
            host_lengths=None,
            host_prefixes=None,
            lens=counts,
            prefix=prefix,
            totals=prefix + counts,
            positions=torch.arange(16),
            start=torch.tensor([0, 2]),
            mrope=torch.zeros(3, 16, dtype=torch.int64),
            correct=counts - 1,
            select=torch.tensor([1, 6]),
        )
        with patch(
            "sglang.srt.speculative.dmtp_qsa_native.scoped_config",
            side_effect=lambda **kw: nullcontext(),
        ):
            native.forward(
                tokens=torch.arange(16),
                hc=torch.zeros(16, 4),
                req_indices=torch.tensor([2, 1]),
                locations=torch.arange(16),
                prefixes=None,
                lengths=None,
                decode=True,
                prepared=prepared,
            )
        batch = native.graph.execute.call_args.args[0]
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.extend_seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)
        self.assertTrue(torch.equal(batch.spec_info.extend_seq_lens_tensor, counts))
        self.assertTrue(torch.equal(batch.spec_info.select_index, torch.tensor([1, 6])))
        self.assertEqual(batch.extend_num_tokens, 16)


if __name__ == "__main__":
    unittest.main()
