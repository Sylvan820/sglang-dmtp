"""Native draft-extend graphs and complete QSA cache parity snapshots."""

import os
from functools import wraps
from types import SimpleNamespace

import torch
from sglang.srt.layers.dp_attention import set_dp_buffer_len, set_is_extend_in_batch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _QSAExtendRunner(EAGLEDraftExtendCudaGraphRunner):
    def _replay_graph(self, shape_key, forward_batch):
        output = super()._replay_graph(shape_key, forward_batch)
        self.token_hidden = output.hidden_states
        return output


class QSAMTPGraph:
    @torch.no_grad()
    def __init__(self, native):
        self.native = native
        self.runner = native.runner
        self.capacity = native.owner.server_args.max_running_requests
        self.width = 8
        self.runner.graph_shared_output = GraphSharedOutput.create_for_model_runner(
            self.runner
        )
        if self.runner.graph_shared_output is None:
            raise ValueError(
                "Native QSA MTP requires graph-enabled shared output initialization"
            )
        self.batches, self.checked = {}, set()
        worker = SimpleNamespace(
            draft_runner=self.runner,
            draft_extend_attn_backend=self.runner.attn_backend,
            speculative_algorithm=SpeculativeAlgorithm.EAGLE,
            seed_dsa_topk_from_draft_extend=False,
            megakernel=None,
        )

        original_forward = self.runner.model.forward

        @wraps(original_forward)
        def capture_forward(input_ids, positions, batch, **kwargs):
            if getattr(native, "hidden_only", False):
                batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch.spec_info.positions = positions
            self.prepare_sink()
            output = original_forward(input_ids, positions, batch, **kwargs)
            self.batches[batch.batch_size] = batch
            # The controller needs the H-wide LM-head input, not the model's
            # HC-wide recurrent output. Keep the native operator path intact.
            output.hidden_states = native.head_hidden.clone()
            tail = getattr(native, "tail", None)
            if tail is not None:
                tail.forward(
                    output.hidden_states.index_select(0, batch.spec_info.select_index)
                )
            return output

        # Native graph runners call forward directly, bypassing Module hooks.
        self.runner.model.forward = capture_forward
        try:
            self.graph_runner = _QSAExtendRunner(
                worker,
                draft_extend_attn_backend=self.runner.attn_backend,
                speculative_num_steps=7,
            )
        finally:
            self.runner.model.forward = original_forward
        if self.graph_runner.captured_req_width != self.width:
            raise ValueError("Native QSA MTP graph must capture eight rows per request")
        if getattr(native, "tail", None) is not None:
            native.tail.freeze_outputs()
        self.fused_prepare = os.getenv("SGLANG_DMTP_QSA_FUSED_PREPARE", "0") == "1"
        if self.fused_prepare:
            captured = self.graph_runner.buffers
            self.input_staging = SimpleNamespace(
                **{
                    name: torch.empty_like(getattr(captured, name))
                    for name in (
                        "input_ids",
                        "positions",
                        "out_cache_loc",
                        "mrope_positions",
                        "hidden_states",
                    )
                }
            )
            device = captured.input_ids.device
            self.input_staging.totals = torch.empty(
                self.capacity, dtype=torch.int32, device=device
            )
            self.input_staging.correct = torch.empty_like(self.input_staging.totals)
            self.input_staging.last = torch.empty(
                self.capacity, dtype=torch.int64, device=device
            )

    def prepare_sink(self):
        metadata = self.runner.attn_backend.forward_metadata.indexer_metadata
        metadata.graph_ring_group_locs.masked_fill_(
            metadata.graph_write_locs[:, None] == 0, 0
        )

    def staged_batch(self, batch):
        bs, n = batch.batch_size, batch.batch_size * self.width
        count = batch.input_ids.numel()
        if bs > self.capacity or count > n:
            raise ValueError("QSA MTP extend exceeds its exact-B graph")
        from copy import copy

        staged = copy(batch)
        if getattr(self, "fused_prepare", False):
            from sglang.srt.speculative.dmtp_qsa_pack import stage_graph_inputs

            buffers = self.input_staging
            if getattr(batch, "_dmtp_prepared_staging", None) is not buffers:
                stage_graph_inputs(batch, buffers, n)
            for name in ("input_ids", "positions", "out_cache_loc"):
                setattr(staged, name, getattr(buffers, name)[:n])
            staged.mrope_positions = buffers.mrope_positions[:, :n]
            hc = buffers.hidden_states[:n]
            select = batch.spec_info.select_index
            correct = batch.spec_info.num_correct_drafts
        else:
            for name in ("input_ids", "positions", "out_cache_loc"):
                value = getattr(batch, name)
                padded = value.new_zeros(n)
                padded[:count].copy_(value)
                setattr(staged, name, padded)
            staged.mrope_positions = batch.mrope_positions.new_zeros((3, n))
            staged.mrope_positions[:, :count].copy_(batch.mrope_positions)
            hc = batch.spec_info.hidden_states.new_zeros(
                (n, batch.spec_info.hidden_states.shape[1])
            )
            hc[:count].copy_(batch.spec_info.hidden_states)
            select = batch.extend_seq_lens.long().cumsum(0) - 1
            correct = batch.extend_seq_lens - 1
        staged.spec_info = EagleDraftExtendInput(
            hidden_states=hc,
            num_correct_drafts=correct,
            num_accept_tokens=batch.extend_seq_lens,
            num_tokens_per_req=self.width,
            select_index=select,
        )
        staged.spec_info.extend_seq_lens_cpu = batch.extend_seq_lens_cpu
        staged.spec_info.extend_seq_lens_tensor = batch.extend_seq_lens
        staged.spec_info.positions = staged.positions
        staged.extend_num_tokens = n
        batch.iteration_ids.bind_replay(
            staged, self.runner.runner_ids, slots=staged.out_cache_loc
        )
        return staged

    def execute(self, batch):
        staged = self.staged_batch(batch)
        output = self.graph_runner.execute(staged, staged.spec_info.select_index)
        self.live_batch = staged
        if getattr(self.native, "tail", None) is not None:
            self.tail_result = self.native.tail.read(batch.batch_size)
        head = self.graph_runner.token_hidden.index_select(
            0, staged.spec_info.select_index
        )
        return head.clone(), self._copy_logits(output.next_token_logits)

    def _copy_logits(self, logits):
        if getattr(self.native, "hidden_only", False):
            if logits is not None:
                raise RuntimeError(
                    "Hidden-only private MTP graph unexpectedly computed logits"
                )
            return None
        if logits is None:
            raise RuntimeError("Native MTP graph did not expose logits")
        return logits.clone()

    def eager(self, bs):
        batch = self.batches[bs]
        live = self.live_batch
        batch.seq_lens_cpu = live.seq_lens_cpu
        batch.extend_seq_lens_cpu = live.extend_seq_lens_cpu
        for name in ("seq_lens_sum", "extend_prefix_lens_cpu", "max_seq_len_host"):
            if hasattr(live, name):
                setattr(batch, name, getattr(live, name))
        batch.spec_info.extend_seq_lens_cpu = live.extend_seq_lens_cpu
        batch.spec_info.extend_seq_lens_tensor = batch.extend_seq_lens
        batch.spec_info.positions = batch.positions
        live.iteration_ids.bind_replay(
            batch, self.runner.runner_ids, slots=batch.out_cache_loc
        )
        backend = self.runner.attn_backend
        backend.init_forward_metadata_out_graph(batch)
        # QSA selects its indexer overlap path from capture mode, not from
        # torch.cuda.is_current_stream_capturing(). Match the recorded branch
        # without recording another graph or enabling graph replay here.
        with self.graph_runner.model_context(), model_capture_mode():
            batch.dp_local_start_pos = batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                batch.global_dp_buffer_len,
                batch.input_ids.numel(),
                batch.dp_padding_mode.is_max_len(),
                batch.global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)
            with forward_context(
                ForwardContext(attn_backend=backend), forward_batch=batch
            ):
                backend.init_forward_metadata_in_graph(batch)
                self.prepare_sink()
                output = self.runner.model.forward(
                    batch.input_ids, batch.positions, batch
                )
        head = self.native.head_hidden.index_select(
            0, batch.spec_info.select_index
        ).clone()
        tail = getattr(self.native, "tail", None)
        if tail is not None:
            reference = tail.forward(head)
            tail.check(self.tail_result, reference)
        return head, self._copy_logits(output.next_token_logits)


def snapshot_pool(pool, locations, req_indices=None):
    rows = torch.cat((locations.long(), locations.new_zeros(1))).unique()
    snapshots = []
    for layer in pool.full_attention_layer_id_mapping:
        for buffer in pool.get_kv_buffer(layer):
            snapshots.append((buffer, rows, buffer.index_select(0, rows).clone()))
    if req_indices is not None:
        reqs = torch.cat((req_indices.long(), req_indices.new_zeros(1))).unique()
        reqs = (
            reqs[:, None] * pool.qsa_ring_size
            + torch.arange(pool.qsa_ring_size, device=reqs.device)
        ).flatten()
        for buffer in pool.qsa_key_state_buffer_pool:
            snapshots.append((buffer, reqs, buffer.index_select(0, reqs).clone()))
        buffer = pool.qsa_rope_position_buffer
        snapshots.append((buffer, reqs, buffer.index_select(0, reqs).clone()))
        compressed = (rows // pool.qsa_compress_ratio).unique()
        for buffer in pool.qsa_compressed_k_buffer_pool:
            snapshots.append(
                (buffer, compressed, buffer.index_select(0, compressed).clone())
            )
    return snapshots


def restore_pool(snapshots):
    for buffer, rows, value in snapshots:
        buffer.index_copy_(0, rows, value)


def assert_pool(snapshots):
    for buffer, rows, expected in snapshots:
        actual = buffer.index_select(0, rows)
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
