"""Cached objects must be equivalent to fresh per-iteration inputs."""

import unittest

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.dmtp_qsa_batch import QSAAppendBatchCache
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


class BatchCacheTest(unittest.TestCase):
    def test_reuse_resets_every_iteration_field_and_keeps_cpu_storage(self):
        cache = QSAAppendBatchCache(32)
        previous = {}
        for step, bs in enumerate([29, 1, 32, 29, 8, 1, 32]):
            lengths = [(i + step) % 8 + 1 for i in range(bs)]
            rows = sum(lengths)
            totals = [2050 + step * 19 + i for i in range(bs)]
            fields = dict(
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                batch_size=bs,
                input_ids=torch.arange(rows) + step * 1000,
                req_pool_indices=torch.arange(bs, 0, -1),
                seq_lens=torch.tensor(totals, dtype=torch.int32),
                out_cache_loc=torch.arange(rows) + step * 10000,
                seq_lens_sum=sum(totals),
                positions=torch.arange(rows) + step * 100,
                extend_seq_lens_cpu=lengths,
            )
            spec_fields = dict(
                hidden_states=torch.randn(rows, 16),
                num_accept_tokens=torch.tensor(lengths),
                select_index=torch.tensor(lengths).cumsum(0) - 1,
            )
            actual = cache.build(fields, spec_fields, totals)
            reference = ForwardBatch(
                **fields,
                seq_lens_cpu=torch.tensor(totals, dtype=torch.int32),
                spec_info=EagleDraftExtendInput(**spec_fields),
            )
            self.assertEqual(set(vars(actual)), set(vars(reference)))
            for key, value in vars(reference).items():
                if key == "spec_info":
                    self.assertEqual(set(vars(actual.spec_info)), set(vars(value)))
                    for name, expected in vars(value).items():
                        got = getattr(actual.spec_info, name)
                        if isinstance(expected, torch.Tensor):
                            self.assertIs(got, expected)
                        else:
                            self.assertEqual(got, expected)
                elif isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(getattr(actual, key), value))
                else:
                    self.assertEqual(getattr(actual, key), value)
            identity = (
                id(actual),
                id(actual.spec_info),
                actual.seq_lens_cpu.data_ptr(),
            )
            if bs in previous:
                self.assertEqual(identity, previous[bs])
            previous[bs] = identity
            actual.iteration_ids = object()
            actual.iteration_cols = object()
            actual.out_cache_loc_virtual = torch.ones(1)
            actual._dmtp_prepared_staging = object()
            actual.spec_info.kv_indptr = torch.ones(1)
            actual.spec_info.extra_from_backend = object()

    def test_rejects_invalid_capacity_or_mode(self):
        cache = QSAAppendBatchCache(1)
        with self.assertRaises(ValueError):
            cache.build(dict(batch_size=2), {}, [1, 1])
        with self.assertRaises(ValueError):
            cache.build(dict(batch_size=1, forward_mode=ForwardMode.EXTEND), {}, [1])
