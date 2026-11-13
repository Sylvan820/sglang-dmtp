"""Experimental four-slot HC/QSA DMTP worker; ordinary DFlash2 is unchanged."""

from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2, _selector_lattice
from sglang.srt.speculative.dmtp_qsa_contract import (
    committed_inputs, join_sparse_proposal, sample_sparse_t1, shifted_prompt,
)
from sglang.srt.speculative.dmtp_qsa_native import NativeQSAMTP
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask


def validate_recipe(cfg, profile):
    if profile is None or profile.compress_ratio != 4:
        raise ValueError("DMTP QSA requires a ratio-four Target")
    if cfg("speculative_num_draft_tokens") != 4:
        raise ValueError("DMTP QSA retains the native four-slot ring guard")
    if cfg("page_size") % profile.compress_ratio or cfg("page_size") < 64:
        raise ValueError("DMTP QSA requires native paged KV (page >=64, divisible by ratio)")
    for name in ("enable_lora", "enable_two_batch_overlap", "enable_unified_memory", "enable_dp_attention"):
        if cfg(name):
            raise ValueError(f"DMTP QSA has not validated {name}")
    if not cfg("disable_overlap_schedule") or not cfg("disable_radix_cache"):
        raise ValueError("DMTP QSA currently requires synchronous scheduling and no prefix reuse")
    if cfg("chunked_prefill_size") != -1 or cfg("quantization") is not None:
        raise ValueError("DMTP QSA requires full prompt prefill and unquantized weights")
    if cfg("speculative_draft_window_size") is not None:
        raise ValueError("DMTP QSA requires the shared native slot mapping, not compact draft KV")
    capacity = cfg("max_running_requests")
    if capacity is None or not 1 <= capacity <= 32:
        raise ValueError("DMTP QSA requires explicit capacity 1..32")


@dataclass
class PrefixState:
    request: object
    length: int
    anchor: int
    head: torch.Tensor
    logits: torch.Tensor


class QSAMTPController:
    def __init__(self, owner, native):
        self.owner = owner
        self.native = native
        self.states = {}
        self.current = None

    def _hc(self, hc, rows):
        target = self.owner.model_runner.model.model
        width = target.hc_count * target.hidden_size
        if hc is None or hc.shape != (rows, width):
            raise ValueError("Missing graph-owned HC bundle; DFlash layer means are not MTP inputs")
        return hc

    def prefill(self, batch, anchors, hc):
        lengths = [int(n) for n in batch.extend_lens]
        if any(batch.prefix_lens):
            raise ValueError("QSA DMTP does not support cached/chunked MTP prefill")
        hc = self._hc(hc, sum(lengths))
        tokens = shifted_prompt(batch.input_ids, lengths, anchors)
        head, logits = self.native.forward(
            tokens=tokens, hc=hc, req_indices=batch.req_pool_indices,
            locations=batch.out_cache_loc, prefixes=[0] * len(lengths), lengths=lengths, decode=False,
        )
        for i, (slot, anchor) in enumerate(zip(batch.req_pool_indices.tolist(), anchors.tolist())):
            self.states[slot] = PrefixState(batch.reqs[i], lengths[i], anchor, head[i].clone(), logits[i].clone())

    def _params(self, batch, bs, device):
        info = batch.sampling_info
        if batch.has_grammar or batch.return_logprob:
            raise ValueError("QSA DMTP has not validated grammar/logprobs")
        # T1 and selector must use the same proposal sampling semantics.
        if not self.owner._selector_sampling_enabled and not info.is_all_greedy:
            raise ValueError("Sampled QSA DMTP requires native selector sampling")
        greedy = resolve_greedy_mask(bs=bs, sampling_info=info, device=device)
        return info.temperatures.reshape(bs).float().clamp_min(1e-5), greedy

    def _coins(self, bs, steps, device):
        coins = torch.rand(bs, steps, device=device)
        group = get_attention_tp_group()
        if group.world_size > 1:
            group.broadcast(coins, src=0)
        return coins

    def prepare(self, batch, block_ids):
        states = []
        for i, (slot, length, anchor) in enumerate(zip(
            batch.req_pool_indices.tolist(), batch.seq_lens.tolist(), block_ids[:, 0].tolist()
        )):
            state = self.states.get(slot)
            if state is None or state.request is not batch.reqs[i] or (state.length, state.anchor) != (length, anchor):
                raise RuntimeError("Stale/reused QSA MTP request, anchor, or committed length")
            states.append(state)
        head = torch.stack([s.head for s in states])
        logits = torch.stack([s.logits for s in states])
        temperatures, greedy = self._params(batch, len(states), logits.device)
        coins = self._coins(len(states), 3, logits.device)
        t1, ids, q = sample_sparse_t1(logits, temperatures, greedy, coins[:, 0], self.owner.selector.top_k)
        block_ids[:, 1].copy_(t1)
        self.current = (head, t1, ids, q, temperatures, greedy, coins[:, 1:])

    def condition(self, embeddings):
        if self.current is None:
            raise RuntimeError("T1 preparation must precede embedding fusion")
        embeddings[:, 1].copy_(self.owner.draft_model.mtp_prefix_conditioner(embeddings[:, 1], self.current[0]))

    def propose(self, batch, hidden):
        if self.current is None:
            raise RuntimeError("No native T1 distribution for this DF replay")
        _, t1, ids, q, temperatures, greedy, coins = self.current
        # Internal dual-anchor exports score rows 1..7, NOT the old 27B rows 2..7.
        tail_ids, scores = _selector_lattice(self.owner.draft_model, hidden[:, 1:3], t1)
        tail, tail_q = self.owner.selector.sample_path(
            candidate_ids=tail_ids, scores=scores, uniforms=coins,
            temperatures=temperatures, greedy_mask=greedy,
        )
        self.owner._selector_sample = (
            None if batch.sampling_info.is_all_greedy
            else join_sparse_proposal(ids, q, tail_ids, tail_q)
        )
        self.current = None
        return torch.cat((t1[:, None], tail), dim=1)

    def commit(self, batch, candidates, lengths, bonus, new_lengths, locations, hc):
        bs = candidates.shape[0]
        hc = self._hc(hc, bs * 4).view(bs, 4, -1)
        shifted = committed_inputs(candidates, lengths, bonus)
        counts = lengths.tolist()
        totals = new_lengths.tolist()
        prefixes = [n - c for n, c in zip(totals, counts)]
        slots = batch.req_pool_indices.tolist()
        anchors = bonus.tolist()
        # One write per request/DECODE prevents an unaligned multi-row write from
        # overwriting the four-entry ring before the previous group compresses.
        for step in range(max(counts)):
            active = [i for i, n in enumerate(counts) if step < n]
            index = torch.tensor(active, dtype=torch.long, device=candidates.device)
            head, logits = self.native.forward(
                tokens=shifted[index, step], hc=hc[index, step],
                req_indices=batch.req_pool_indices[index], locations=locations[index, step],
                prefixes=[prefixes[i] + step for i in active], lengths=[1] * len(active), decode=True,
            )
            for row, i in enumerate(active):
                if step + 1 == counts[i]:
                    self.states[slots[i]] = PrefixState(batch.reqs[i], totals[i], anchors[i], head[row].clone(), logits[row].clone())


class QSAMTPWorker(DFlashWorkerV2):
    def __init__(self, server_args, gpu_id, nccl_port, target_worker):
        profile = parse_qsa_profile(target_worker.model_runner.model_config.hf_config)
        validate_recipe(get_context().config_leaf, profile)
        parallel = get_parallel()
        if parallel.pp_size != 1 or parallel.attn_cp_size != 1 or parallel.attn_dp_enabled:
            raise ValueError("QSA DMTP currently supports PP1/CP1 without attention DP")
        super().__init__(server_args, gpu_id, nccl_port, target_worker)
        if type(self.draft_model).__name__ != "Qwen4DMTPT1DraftModel" or self.selector is None:
            raise ValueError("QSA DMTP requires a matching trained dual-anchor residual draft")
        if type(self.model_runner.model).__name__ != "Qwen4ExpForConditionalGeneration":
            raise ValueError("QSA DMTP requires the native HC Target")
        self.model_runner.model.model._dmtp_capture_hc = True
        self.native_mtp = NativeQSAMTP(self)
        self._dmtp_controller = QSAMTPController(self, self.native_mtp)

    def _maybe_build_draft_sampler(self):
        # A folded ordinary DF sampler would resample/overwrite the native T1.
        return None

    def alloc_memory_pool(self, memory_pool_config=None, req_to_token_pool=None, token_to_kv_pool_allocator=None):
        super().alloc_memory_pool(memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator)
        self.native_mtp.alloc_memory_pool(memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator)

    def init_attention_backends(self):
        super().init_attention_backends()
        self.native_mtp.init_attention_backends()

    def init_cuda_graphs(self):
        from sglang.srt.speculative.dmtp_qsa_native import scoped_config

        with scoped_config(**self.native_mtp.fields):
            self.native_mtp.worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        super().init_cuda_graphs()

    @property
    def spec_v2_attn_backends(self):
        return (*super().spec_v2_attn_backends, self.native_mtp.runner.attn_backend)

    def weight_update_runners(self):
        raise NotImplementedError("QSA DMTP live weight updates require joint state invalidation; restart instead")
