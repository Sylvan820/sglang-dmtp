"""Committed-only native QSA MTP. No speculative scratch or partial KV moves."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.arg_utils import namespace_of
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.layers.moe.utils import draft_model_build_scope
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@contextmanager
def scoped_config(**fields):
    groups = {}
    for field, value in fields.items():
        groups.setdefault(namespace_of(ServerArgs)[field], {})[field] = value
    with ExitStack() as stack:
        for path, values in groups.items():
            root, *parts = path.split(".")
            bag = get_context().config_bag(root)
            for part in parts:
                bag = getattr(bag, part)
            stack.enter_context(bag.override(**values))
        yield


class NativeQSAMTP:
    def __init__(self, owner):
        self.owner = owner
        graph = deepcopy(get_exec().graph.cuda_graph_config)
        graph.decode.backend = Backend.DISABLED
        graph.prefill.backend = Backend.DISABLED
        self.fields = dict(
            speculative_algorithm="EAGLE",
            speculative_draft_model_path=owner.server_args.model_path,
            speculative_draft_model_revision=owner.server_args.revision,
            speculative_eagle_topk=1,
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
            speculative_draft_attention_backend=None,
            cuda_graph_config=graph,
            disable_cuda_graph=True,
        )
        args = deepcopy(owner.server_args)
        declare_resolution(args, "dmtp_qsa_private", **self.fields)
        with scoped_config(**self.fields), draft_model_build_scope():
            self.worker = TpModelWorker(
                server_args=args, gpu_id=owner.gpu_id, nccl_port=owner.nccl_port,
                is_draft_worker=True, context_length=owner.model_runner.model_config.context_len,
                random_seed=owner.target_worker.random_seed,
            )
        self.runner = self.worker.model_runner
        model = self.runner.model
        if type(model).__name__ != "Qwen4ExpForCausalLMMTP":
            raise ValueError("QSA DMTP requires the Target's native HC MTP weights")
        self.profile = parse_qsa_profile(self.runner.model_config.hf_config)
        if self.profile is None or self.profile.compress_ratio != 4:
            raise ValueError("QSA MTP requires native ratio-four indexing")
        model.set_embed_and_head(*owner.model_runner.model.get_embed_and_head())
        self.head_hidden = None
        model.logits_processor.register_forward_pre_hook(self._capture_head_input)

    def _capture_head_input(self, module, inputs):
        self.head_hidden = inputs[1]

    def alloc_memory_pool(self, config, mapping, allocator):
        if mapping is None or allocator is None:
            raise ValueError("QSA MTP requires the Target's native physical mapping")
        with scoped_config(**self.fields):
            self.worker.alloc_memory_pool(
                memory_pool_config=config, req_to_token_pool=mapping,
                token_to_kv_pool_allocator=allocator,
            )
        if any(self.runner.token_to_kv_pool is runner.token_to_kv_pool for runner in (
            self.owner.model_runner, self.owner.draft_model_runner
        )):
            raise RuntimeError("QSA MTP must own its full/compressed/indexer/ring tensors")

    def init_attention_backends(self):
        with scoped_config(**self.fields):
            self.worker.init_attention_backends()
            backend = DraftBackendFactory(self.runner, 1, 1, qsa_profile=self.profile).create_draft_extend_backend()
            self.runner.attn_backend = backend

    @torch.no_grad()
    def forward(self, *, tokens, hc, req_indices, locations, prefixes, lengths, decode):
        if (tokens.numel() != sum(lengths) or hc.shape[0] != tokens.numel()
                or locations.numel() != tokens.numel() or len(prefixes) != len(lengths)
                or req_indices.numel() != len(lengths) or any(n < 1 for n in lengths)):
            raise ValueError("QSA MTP token/HC/slot rows are misaligned")
        if decode and any(n != 1 for n in lengths):
            raise ValueError("QSA continuation must advance one token per request")
        if not decode and any(prefixes):
            raise ValueError("Only initial zero-prefix MTP prefill may use EXTEND")
        device = tokens.device
        lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        prefix = torch.tensor(prefixes, dtype=torch.int32, device=device)
        totals = prefix + lens
        positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(prefixes, lengths)])
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            batch_size=len(lengths), input_ids=tokens, req_pool_indices=req_indices,
            seq_lens=totals, seq_lens_cpu=torch.tensor([p + n for p, n in zip(prefixes, lengths)], dtype=torch.int32),
            seq_lens_sum=sum(prefixes) + sum(lengths), positions=positions,
            out_cache_loc=locations,
            extend_num_tokens=sum(lengths) if not decode else None,
            extend_seq_lens=lens if not decode else None,
            extend_prefix_lens=prefix if not decode else None,
            extend_start_loc=lens.cumsum(0, dtype=torch.int32) - lens if not decode else None,
            extend_seq_lens_cpu=lengths if not decode else None,
            extend_prefix_lens_cpu=prefixes if not decode else None,
            extend_logprob_start_lens_cpu=[n - 1 for n in lengths] if not decode else None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=SimpleNamespace(hidden_states=hc, custom_mask=None, kv_indptr=None),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            mrope_positions=positions[None].expand(3, -1).contiguous(),
        )
        translator = self.runner.kv_index_translator
        plan = translator.plan(req_pool_indices=req_indices, seq_lens=totals,
                               seq_lens_cpu=batch.seq_lens_cpu, write_virtual=None, write_slots=locations)
        plan.bind(batch, translator)
        self.head_hidden = None
        with scoped_config(**self.fields):
            result = self.runner.forward(batch)
        if result.can_run_graph:
            raise RuntimeError("Private QSA MTP must remain eager until real graph validation")
        logits = result.logits_output.next_token_logits
        head = self.head_hidden
        self.head_hidden = None
        if head is None or head.shape[0] != tokens.numel() or logits.shape[0] != len(lengths):
            raise RuntimeError("Native MTP did not expose its H-wide head input/logits")
        last = lens.long().cumsum(0) - 1
        head = head.index_select(0, last).clone()
        if head.shape[1] != self.owner.draft_model.config.hidden_size:
            raise ValueError("Native MTP head input width mismatch")
        torch._assert_async(torch.isfinite(head).all() & torch.isfinite(logits).all(), "Nonfinite native MTP output")
        return head, logits.clone()
