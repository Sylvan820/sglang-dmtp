"""Exact-B object reuse for the serial private QSA append path."""

from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput


class QSAAppendBatchCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}

    def build(self, fields, spec_fields, totals_cpu):
        bs = fields["batch_size"]
        if not 1 <= bs <= self.capacity or len(totals_cpu) != bs:
            raise ValueError("Invalid cached QSA append batch")
        if fields["forward_mode"] != ForwardMode.DRAFT_EXTEND_V2:
            raise ValueError("Only serial QSA draft append may reuse batches")
        entry = self.entries.get(bs)
        if entry is None:
            batch = ForwardBatch(
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                batch_size=bs,
                input_ids=None,
                req_pool_indices=None,
                seq_lens=None,
                out_cache_loc=None,
                seq_lens_sum=None,
            )
            spec = EagleDraftExtendInput()
            cpu = torch.empty(bs, dtype=torch.int32, device="cpu")
            entry = SimpleNamespace(
                batch=batch,
                spec=spec,
                batch_defaults=vars(batch).copy(),
                spec_defaults=vars(spec).copy(),
                cpu=cpu,
                cpu_array=cpu.numpy(),
            )
            self.entries[bs] = entry
        # Clear fields installed by address translation/backend execution too.
        # Defaults retain no previous request tensors or mutable input lists.
        entry.batch.__dict__.clear()
        entry.batch.__dict__.update(entry.batch_defaults)
        entry.batch.__dict__.update(fields)
        entry.spec.__dict__.clear()
        entry.spec.__dict__.update(entry.spec_defaults)
        entry.spec.__dict__.update(spec_fields)
        entry.cpu_array[:] = totals_cpu
        entry.batch.seq_lens_cpu = entry.cpu
        entry.batch.spec_info = entry.spec
        return entry.batch
