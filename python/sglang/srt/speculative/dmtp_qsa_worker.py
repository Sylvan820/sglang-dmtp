"""HC/QSA implementation behind the same DMTP entry as the dense FA3 path."""

import logging
import os
from copy import deepcopy
from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.runtime_context import get_context, get_exec, get_parallel, get_schedule
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dmtp_qsa_contract import committed_inputs, shifted_prompt
from sglang.srt.speculative.dmtp_qsa_native import NativeQSAMTP, scoped_config

logger = logging.getLogger(__name__)


def validate_recipe(cfg, profile):
    if profile is None or profile.compress_ratio != 4:
        raise ValueError("DMTP HC requires the native ratio-four QSA profile")
    if cfg("speculative_num_draft_tokens") != 8:
        raise ValueError("DMTP HC currently uses seven proposals/eight Target slots")
    if cfg("page_size") < 64 or cfg("page_size") % profile.compress_ratio:
        raise ValueError(
            "DMTP QSA requires native paged KV, not the dense FA3 page-one pool"
        )
    for key in (
        "enable_lora",
        "enable_two_batch_overlap",
        "enable_unified_memory",
        "enable_dp_attention",
    ):
        if cfg(key):
            raise ValueError(f"DMTP QSA has not validated {key}")
    if not cfg("disable_overlap_schedule") or not cfg("disable_radix_cache"):
        raise ValueError(
            "DMTP QSA currently requires synchronous scheduling without prefix reuse"
        )
    if cfg("chunked_prefill_size") != -1 or cfg("quantization") is not None:
        raise ValueError(
            "DMTP QSA requires full prompt prefill and unquantized weights"
        )
    if cfg("speculative_draft_window_size") is not None:
        raise ValueError("DMTP QSA compact draft indices are not validated")
    capacity = cfg("max_running_requests")
    if capacity is None or not 1 <= capacity <= 32:
        raise ValueError("DMTP QSA requires explicit capacity 1..32")


@dataclass
class PrefixState:
    request: object
    length: int
    anchor: int
    head: torch.Tensor
    prepared: tuple = None
    sampling: tuple = None


class QSAController:
    uses_hc = True
    draft_slots = 8

    def __init__(self, worker):
        self.worker = worker
        self.two_graphs = os.getenv("SGLANG_DMTP_QSA_TWO_DRAFT_GRAPHS", "0") == "1"
        self.head = worker.model_runner.model.lm_head
        self.native = NativeQSAMTP(worker)
        self.states = {}
        self.prepared_embeds = None
        self.graph_used = {}
        self.checked_target = set()

    def init_graphs(self):
        if self.two_graphs:
            if (
                self.native.graph is None
                or self.worker._draft_sampler is not self.df_tail
            ):
                raise RuntimeError("DMTP two-graph mode requires MTP and DF capture")
            return
        from sglang.srt.speculative.dmtp_head_conditioner_graph import (
            SampledHeadConditionerGraphs,
        )
        from sglang.srt.speculative.dmtp_selector_graph import DMTPSelectorGraphs
        from sglang.srt.speculative.mtp_conditioner_graph import ConditionerGraphs

        self.conditioner_graphs = ConditionerGraphs(
            self.worker.draft_model.mtp_prefix_conditioner,
            self.worker.server_args.max_running_requests,
            owner=self,
        )
        self.t1_combined = SampledHeadConditionerGraphs(self, enable_graph=True)
        self.tail_selector = DMTPSelectorGraphs(self, t1=False, enable_graph=True)

    def _hc(self, hc, rows):
        target = self.worker.model_runner.model.model
        if hc is None or hc.shape != (rows, target.hc_count * target.hidden_size):
            raise ValueError("MTP requires HC bundle rows, not DFlash layer means")
        return hc

    def prefill(self, batch, anchors, hc):
        lengths = [int(n) for n in batch.extend_lens]
        if any(batch.prefix_lens):
            raise ValueError("DMTP QSA prefix reuse/chunked prefill is not supported")
        hc = self._hc(hc, sum(lengths))
        tokens = shifted_prompt(batch.input_ids, lengths, anchors)
        head, _ = self.native.forward(
            tokens=tokens,
            hc=hc,
            req_indices=batch.req_pool_indices,
            locations=batch.out_cache_loc,
            prefixes=[0] * len(lengths),
            lengths=lengths,
            decode=False,
        )
        for i, slot in enumerate(batch.req_pool_indices.tolist()):
            self.states[slot] = PrefixState(
                batch.reqs[i], lengths[i], int(anchors[i]), head[i].clone()
            )

    def t1(self, batch, block_ids):
        states = []
        for i, (slot, length, anchor) in enumerate(
            zip(
                batch.req_pool_indices.tolist(),
                batch.seq_lens.tolist(),
                block_ids[:, 0].tolist(),
            )
        ):
            state = self.states.get(slot)
            if (
                state is None
                or state.request is not batch.reqs[i]
                or (state.length, state.anchor) != (length, anchor)
            ):
                raise RuntimeError("Stale QSA MTP request/anchor/position")
            states.append(state)
        self.batch = batch
        self.hidden = torch.stack([s.head for s in states])
        self.sampling = batch.sampling_info
        if self.two_graphs:
            if self.mtp_tail.signature() != self.mtp_tail.initial_signature:
                raise RuntimeError(
                    "DMTP prefetched T1 weights/layout changed; restart required"
                )
            fresh = [i for i, state in enumerate(states) if state.prepared is None]
            if fresh:
                indices = torch.tensor(
                    fresh, dtype=torch.int64, device=block_ids.device
                )
                self.mtp_tail.stage(block_ids[indices, 0], self.sampling, indices)
                # Only the first prompt initialization is eager. Subsequent
                # states receive these outputs from the previous MTP graph.
                result = self.mtp_tail.forward(self.hidden.index_select(0, indices))
                for row, index in enumerate(fresh):
                    states[index].prepared = tuple(
                        value[row].clone() for value in result
                    )
                    states[index].sampling = (
                        self.mtp_tail.temperatures[row].clone(),
                        self.mtp_tail.greedy_mask[row].clone(),
                    )
            from sglang.srt.speculative.dspark_components.dspark_draft import (
                resolve_greedy_mask,
            )

            current_temp = (
                self.sampling.temperatures.reshape(-1).float().clamp_min(1e-5)
            )
            current_greedy = resolve_greedy_mask(
                bs=len(states), sampling_info=self.sampling, device=block_ids.device
            )
            torch._assert_async(
                (torch.stack([s.sampling[0] for s in states]) == current_temp).all()
                & (
                    torch.stack([s.sampling[1] for s in states]) == current_greedy
                ).all(),
                "DMTP prefetched T1 sampling parameters changed",
            )
            tokens, ids, q, self.prepared_embeds = (
                torch.stack([s.prepared[j] for s in states]) for j in range(4)
            )
            block_ids[:, 1].copy_(tokens[:, 0])
            self.t1_sparse = (ids, q)
            self.graph_used.update(t1_combined=not fresh, conditioner=not fresh)
            return
        result = self.t1_combined.run(self.hidden, block_ids, self.sampling)
        if result is None or not self.t1_combined.last_used:
            raise RuntimeError("QSA T1 combined graph unexpectedly unavailable")
        tokens, ids, q, self.prepared_embeds = result
        block_ids[:, 1].copy_(tokens[:, 0])
        self.t1_sparse = (ids, q)
        self.graph_used.update(t1_combined=True, conditioner=True)

    def mask(self, seq_lens):
        # The native DFlash2 backend consumes this checkpoint's SWA contract.
        return None

    def proposal(self, hidden, block_ids, positions):
        # The Next export predicts the successors of T1 from scored rows 1..7.
        # Keep all eight query rows; 718 consumes only the first six predictions.
        if self.two_graphs:
            if not self.graph_used.get("draft"):
                raise RuntimeError(
                    "DMTP two-graph selector requires the DF parent graph"
                )
            rest, ids, q = self.df_tail.read(hidden.shape[0])
            if hidden.shape[0] not in self.df_checked:
                reference = self.df_tail.forward(
                    hidden.flatten(0, 1), block_ids.flatten()
                )
                self.df_tail.check((rest, ids, q), reference)
                self.df_checked.add(hidden.shape[0])
        else:
            rest, ids, q = self.tail_selector.run(
                hidden[:, 1:7], block_ids[:, 1], self.sampling
            )
            if not self.tail_selector.last_used:
                raise RuntimeError("QSA selector graph unexpectedly unavailable")
        if not self.sampling.is_all_greedy:
            t1_ids, t1_q = self.t1_sparse
            self.worker._selector_sample = (
                torch.cat((t1_ids, ids), 1),
                torch.cat((t1_q, q), 1),
            )
        self.t1_sparse = None
        self.graph_used["tail_selector"] = True
        return torch.cat((block_ids[:, 1:2], rest), 1)

    def commit(
        self, candidates, accept_len, bonus, features, new_lengths, locations, *, hc
    ):
        if not self.graph_used.get("target") or not self.graph_used.get("draft"):
            raise RuntimeError("DMTP QSA decode must replay both Target and DF graphs")
        bs, width = candidates.shape
        hc = self._hc(hc, bs * width).view(bs, width, -1)
        counts = (accept_len + 1).tolist()
        lengths = torch.tensor(counts, device=candidates.device, dtype=torch.int32)
        tokens = committed_inputs(candidates, lengths, bonus)
        totals = new_lengths.tolist()
        prefixes = [n - c for n, c in zip(totals, counts)]
        slots = self.batch.req_pool_indices.tolist()
        valid = torch.arange(width, device=candidates.device)[None] < lengths[:, None]
        if self.two_graphs:
            self.mtp_tail.stage(bonus, self.sampling)
        head, _ = self.native.forward(
            tokens=tokens[valid],
            hc=hc[valid],
            req_indices=self.batch.req_pool_indices,
            locations=locations[valid],
            prefixes=prefixes,
            lengths=counts,
            decode=True,
        )
        for i in range(bs):
            self.states[slots[i]] = PrefixState(
                self.batch.reqs[i], totals[i], int(bonus[i]), head[i].clone()
            )
            if self.two_graphs:
                self.states[slots[i]].prepared = tuple(
                    value[i].clone() for value in self.native.graph.tail_result
                )
                self.states[slots[i]].sampling = (
                    self.mtp_tail.temperatures[i].clone(),
                    self.mtp_tail.greedy_mask[i].clone(),
                )
        if bs not in self.checked_target:
            logger.info(
                "DMTP_QSA_EXECUTION B=%d target=True draft=True combined=True selector=True mtp_graph=%s",
                bs,
                self.native.graph is not None,
            )
            self.checked_target.add(bs)


class QSAMTPWorker(DFlashWorkerV2):
    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        profile = parse_qsa_profile(target_worker.model_runner.model_config.hf_config)
        validate_recipe(get_context().config_leaf, profile)
        parallel = get_parallel()
        if (
            parallel.pp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.attn_dp_enabled
        ):
            raise ValueError("DMTP QSA currently supports TP-only execution")
        super().__init__(server_args, gpu_id, nccl_port, target_worker)
        if (
            type(self.draft_model).__name__ != "Qwen4DMTPT1DraftModel"
            or self.selector is None
        ):
            raise ValueError("QSA Target requires its matching trained HC DMTP draft")
        self.draft_model.lm_head = self.model_runner.model.lm_head
        self.model_runner.model.model._dmtp_capture_hc = True
        self._dmtp_controller = QSAController(self)

    def _maybe_build_draft_sampler(self):
        if self._dmtp_controller.two_graphs:
            return self._dmtp_controller.df_tail
        return None

    @property
    def spec_v2_attn_backends(self):
        return (
            *super().spec_v2_attn_backends,
            self._dmtp_controller.native.runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        self._dmtp_controller.native.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )

    def init_attention_backends(self):
        super().init_attention_backends()
        self._dmtp_controller.native.init_attention_backends()

    def init_cuda_graphs(self):
        controller = self._dmtp_controller
        if controller.two_graphs:
            from sglang.srt.speculative.dmtp_graph_tails import DMTPGraphSamplingTail

            controller.mtp_tail = DMTPGraphSamplingTail(controller, t1=True)
            controller.df_tail = DMTPGraphSamplingTail(controller, t1=False)
            controller.df_checked = set()
            controller.native.tail = controller.mtp_tail
        config = deepcopy(get_exec().graph.cuda_graph_config)
        config.decode.bs = list(range(1, get_schedule().max_running_requests + 1))
        with scoped_config(cuda_graph_config=config, disable_cuda_graph_padding=True):
            super().init_cuda_graphs()
        if controller.two_graphs:
            controller.df_tail.freeze_outputs()
        self._dmtp_controller.native.init_cuda_graphs()
        self._dmtp_controller.init_graphs()

    def weight_update_runners(self):
        raise NotImplementedError(
            "QSA DMTP live weight updates require joint state invalidation; restart instead"
        )
