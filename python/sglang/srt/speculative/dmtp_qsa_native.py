"""Committed-only native QSA MTP. No speculative scratch or partial KV moves."""

import logging
import os
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

import torch
from sglang.srt.arg_groups.arg_utils import namespace_of
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.configs.draft_parallel import DraftParallelLayout
from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.runtime_context import get_context, get_exec, get_parallel, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context


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
        with (
            scoped_config(**self.fields),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            draft_model_build_scope(),
        ):
            self.worker = TpModelWorker(
                server_args=args,
                gpu_id=owner.gpu_id,
                nccl_port=owner.nccl_port,
                is_draft_worker=True,
                context_length=owner.model_runner.model_config.context_len,
                random_seed=owner.target_worker.random_seed,
            )
        self.runner = self.worker.model_runner
        self.graph = None
        model = self.runner.model
        if type(model).__name__ != "Qwen4ExpForCausalLMMTP":
            raise ValueError("QSA DMTP requires the Target's native HC MTP weights")
        self.profile = parse_qsa_profile(self.runner.model_config.hf_config)
        if self.profile is None or self.profile.compress_ratio != 4:
            raise ValueError("QSA MTP requires native ratio-four indexing")
        self._configure_hc_mix()
        model.set_embed_and_head(*owner.model_runner.model.get_embed_and_head())
        self._configure_hidden_only()
        self.head_hidden = None
        model.logits_processor.register_forward_pre_hook(self._capture_head_input)
        self._configure_forward()

    def _configure_hidden_only(self):
        self.hidden_only = (
            os.getenv("SGLANG_DMTP_QSA_HIDDEN_ONLY", "0") == "1"
            or os.getenv("SGLANG_DMTP_QSA_TWO_DRAFT_GRAPHS", "0") == "1"
        )
        if self.hidden_only:
            from sglang.srt.speculative.mtp_hidden_only import HiddenOnlyLogitsProcessor

            model = self.runner.model
            model.logits_processor = HiddenOnlyLogitsProcessor(
                model.logits_processor, enabled=True
            )
            model._dmtp_qsa_normalized_output = True
            logging.getLogger(__name__).info("DMTP_QSA_MTP_HIDDEN_ONLY enabled=True")

    def _configure_forward(self):
        model = self.runner.model
        original = model.forward

        @wraps(original)
        def deterministic_forward(*args, **kwargs):
            # Both runner.forward and native graph capture call this private
            # model. Do not enable deterministic policies on the Target.
            with (
                self.execution_scope(),
                scoped_config(enable_deterministic_inference=True),
            ):
                return original(*args, **kwargs)

        model.forward = deterministic_forward

    @contextmanager
    def execution_scope(self):
        group = getattr(self.runner, "tp_group", None)
        layout = DraftParallelLayout(
            tp_group=group,
            attn_tp_size=1 if group is None else get_parallel().attn_tp_size,
        )
        with (
            draft_tp_context(layout),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            group = layout.tp_group
            comm = None if group is None else group.get_available_pynccl_comm()
            # Match the communicator enabled by native graph capture also in
            # the same-layout eager reference; restore the Target's state.
            with nullcontext() if comm is None else comm.change_state(enable=True):
                yield

    def _configure_hc_mix(self):
        model = self.runner.model.model
        modules = [model.hyper_connection_mixer]
        for layer in model.layers:
            modules.extend((layer.attn_hyper_connection, layer.mlp_hyper_connection))
        # Device-scope atomic HC accumulation is not byte reproducible across
        # forwards. Keep the native MTP graph/reference on the ordered path;
        # the Target's separate modules retain their default fast path.
        for module in modules:
            module.allow_nondeterministic_mix = False
        logging.getLogger(__name__).info(
            "DMTP_QSA_MTP_HC_MIX atomic_accumulation=False modules=%d", len(modules)
        )

    def _capture_head_input(self, module, inputs):
        self.head_hidden = inputs[1]

    def alloc_memory_pool(self, config, mapping, allocator):
        if mapping is None:
            raise ValueError("QSA MTP requires the Target's request row capacity")
        if os.getenv("SGLANG_DMTP_QSA_PRIVATE_MAPPING", "0") != "1":
            self.private_mapping = None
            with scoped_config(**self.fields):
                self.worker.alloc_memory_pool(
                    memory_pool_config=config,
                    req_to_token_pool=mapping,
                    token_to_kv_pool_allocator=allocator,
                )
            if any(
                self.runner.token_to_kv_pool is runner.token_to_kv_pool
                for runner in (self.owner.model_runner, self.owner.draft_model_runner)
            ):
                raise RuntimeError("QSA MTP must own its cache tensors")
            return
        rows, context = mapping.req_to_token.shape
        page = self.owner.server_args.page_size
        self.private_stride = ((context + page - 1) // page) * page
        capacity = page + (rows - 1) * self.private_stride
        self.private_mapping = ReqToTokenPool(
            size=rows - 1,
            max_context_len=context,
            device=mapping.req_to_token.device,
            enable_memory_saver=False,
        )
        table = self.private_mapping.req_to_token
        starts = (
            page + torch.arange(rows - 1, device=table.device) * self.private_stride
        )
        table[1:].copy_(starts[:, None] + torch.arange(context, device=table.device))
        private_config = replace(
            config,
            max_total_num_tokens=capacity,
            full_max_total_num_tokens=capacity,
            swa_max_total_num_tokens=None,
        )
        with scoped_config(**self.fields):
            self.worker.alloc_memory_pool(
                memory_pool_config=private_config,
                req_to_token_pool=self.private_mapping,
                token_to_kv_pool_allocator=None,
            )
        if any(
            self.runner.token_to_kv_pool is runner.token_to_kv_pool
            for runner in (self.owner.model_runner, self.owner.draft_model_runner)
        ):
            raise RuntimeError(
                "QSA MTP must own its full/compressed/indexer/ring tensors"
            )
        if self.runner.runner_ids.req_to_token is not table:
            raise RuntimeError("QSA MTP runner is not bound to its private mapping")
        logging.getLogger(__name__).info(
            "DMTP_QSA_PRIVATE_MAPPING rows=%d stride=%d token_capacity=%d target_capacity=%d",
            rows - 1,
            self.private_stride,
            capacity,
            config.max_total_num_tokens,
        )

    def _private_locations(self, req_indices, lens, positions):
        # output_size avoids a device-to-host shape read for ragged batches.
        rows = torch.repeat_interleave(
            req_indices, lens.long(), output_size=positions.numel()
        )
        return self.private_mapping.req_to_token[rows, positions].long()

    def _reset_private_ring(self, req_indices):
        pool = self.runner.token_to_kv_pool
        ring_rows = (
            req_indices[:, None] * pool.qsa_ring_size
            + torch.arange(pool.qsa_ring_size, device=req_indices.device)
        ).flatten()
        for buffer in (*pool.qsa_key_state_buffer_pool, pool.qsa_rope_position_buffer):
            buffer.index_fill_(0, ring_rows, 0)

    def init_attention_backends(self):
        with scoped_config(**self.fields):
            aux = self.runner.spec_aux_config
            configure_aux_hidden_state_capture(
                model=self.runner.model,
                eagle_use_aux_hidden_state=aux.eagle_use_aux_hidden_state,
                eagle_aux_hidden_state_layer_ids=aux.eagle_aux_hidden_state_layer_ids,
                dflash_use_aux_hidden_state=aux.dflash_use_aux_hidden_state,
                dflash_target_layer_ids=aux.dflash_target_layer_ids,
                is_dspark=aux.is_dspark,
            )
            backend = DraftBackendFactory(
                self.runner, 1, 1, qsa_profile=self.profile
            ).create_draft_extend_backend()
            self.runner.attn_backend = backend
            self.runner.prefill_attention_backend_str = (
                backend.prefill_attention_backend_str
            )
            self.runner.decode_attention_backend_str = (
                backend.decode_attention_backend_str
            )
            self.runner.runner_ids.bind_and_verify_backends([backend])

    def init_cuda_graphs(self):
        from sglang.srt.speculative.dmtp_qsa_graph import QSAMTPGraph

        with scoped_config(**self.fields):
            self.worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        # Match the reference prefix route: real committed rows use the native
        # eager runner until the QSA graph is explicitly selected for diagnosis.
        if (
            os.getenv("SGLANG_DMTP_QSA_MTP_GRAPH", "0") != "1"
            and os.getenv("SGLANG_DMTP_QSA_TWO_DRAFT_GRAPHS", "0") != "1"
        ):
            logging.getLogger(__name__).info(
                "DMTP_QSA_MTP_NATIVE_PACKED_EXTEND graph=False"
            )
            return
        config = deepcopy(get_exec().graph.cuda_graph_config)
        config.decode.backend = Backend.FULL
        config.decode.bs = list(range(1, get_schedule().max_running_requests + 1))
        self.graph_fields = dict(
            self.fields,
            cuda_graph_config=config,
            speculative_num_draft_tokens=8,
            disable_cuda_graph=False,
            disable_cuda_graph_padding=True,
        )
        with self.execution_scope(), scoped_config(**self.graph_fields):
            self.graph = QSAMTPGraph(self)

    @torch.no_grad()
    def forward(
        self,
        *,
        tokens,
        hc,
        req_indices,
        locations,
        prefixes,
        lengths,
        decode,
        prepared=None,
    ):
        gpu_only = prepared is not None and getattr(prepared, "gpu_only", False)
        bs = req_indices.numel()
        if gpu_only:
            if (
                not decode
                or prefixes is not None
                or lengths is not None
                or tokens.numel() != bs * 8
                or hc.shape[0] != bs * 8
                or locations.numel() != bs * 8
            ):
                raise ValueError("Invalid fixed-capacity GPU QSA commit")
        elif (
            tokens.numel() != sum(lengths)
            or hc.shape[0] != tokens.numel()
            or locations.numel() != tokens.numel()
            or len(prefixes) != len(lengths)
            or req_indices.numel() != len(lengths)
            or any(n < 1 for n in lengths)
        ):
            raise ValueError("QSA MTP token/HC/slot rows are misaligned")
        if decode and not gpu_only and any(n > 8 for n in lengths):
            raise ValueError("QSA continuation exceeds the accepted block width")
        if not decode and any(prefixes):
            raise ValueError("Only initial zero-prefix MTP prefill may use EXTEND")
        device = tokens.device
        fused_prepare = prepared is not None or (
            decode and os.getenv("SGLANG_DMTP_QSA_FUSED_PREPARE", "0") == "1"
        )
        if prepared is not None:
            if (
                not decode
                or self.graph is None
                or not self.graph.fused_prepare
                or prepared.buffers is not self.graph.input_staging
                or prepared.host_lengths != lengths
                or prepared.host_prefixes != prefixes
                or prepared.rows != tokens.numel()
            ):
                raise ValueError("QSA prepared commit does not own this graph input")
            lens, prefix, totals = prepared.lens, prepared.prefix, prepared.totals
            positions, starts, mrope = (
                prepared.positions,
                prepared.start,
                prepared.mrope,
            )
            correct, select = prepared.correct, prepared.select
        elif fused_prepare:
            from sglang.srt.speculative.dmtp_qsa_pack import prepare_append

            mapping = getattr(self, "private_mapping", None)
            (
                lens,
                prefix,
                totals,
                positions,
                locations,
                starts,
                mrope,
                correct,
                select,
            ) = prepare_append(
                req_indices,
                prefixes,
                lengths,
                locations,
                mapping.req_to_token if mapping is not None else None,
            )
        else:
            lens = torch.tensor(lengths, dtype=torch.int32, device=device)
            prefix = torch.tensor(prefixes, dtype=torch.int32, device=device)
            totals = prefix + lens
            positions = torch.cat(
                [
                    torch.arange(p, p + n, device=device)
                    for p, n in zip(prefixes, lengths)
                ]
            )
            starts = lens.cumsum(0, dtype=torch.int32) - lens
            mrope = positions[None].expand(3, -1).contiguous()
        if not fused_prepare and getattr(self, "private_mapping", None) is not None:
            if any(
                p + n > self.private_mapping.max_context_len
                for p, n in zip(prefixes, lengths)
            ):
                raise ValueError("Private QSA MTP append exceeds context capacity")
            locations = self._private_locations(req_indices, lens, positions)
            if not decode:
                self._reset_private_ring(req_indices)
        cache_batch = (
            decode
            and not gpu_only
            and os.getenv("SGLANG_DMTP_QSA_CACHED_BATCH", "0") == "1"
        )
        totals_cpu = None if gpu_only else [p + n for p, n in zip(prefixes, lengths)]
        batch_fields = dict(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2 if decode else ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=tokens,
            req_pool_indices=req_indices,
            seq_lens=totals,
            seq_lens_cpu=None,
            seq_lens_sum=None if gpu_only else sum(totals_cpu),
            positions=positions,
            out_cache_loc=locations,
            extend_num_tokens=tokens.numel(),
            extend_seq_lens=lens,
            extend_prefix_lens=prefix,
            extend_start_loc=starts,
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens_cpu=prefixes,
            extend_logprob_start_lens_cpu=None
            if gpu_only
            else [n - 1 for n in lengths],
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if getattr(self, "hidden_only", False)
                else CaptureHiddenMode.LAST
            ),
            mrope_positions=mrope,
        )
        spec_fields = None
        if decode:
            from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

            spec_fields = dict(
                hidden_states=hc,
                num_correct_drafts=correct if fused_prepare else lens - 1,
                num_accept_tokens=lens,
                num_tokens_per_req=8,
                select_index=select if fused_prepare else lens.long().cumsum(0) - 1,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
        if cache_batch:
            if not getattr(self.owner.server_args, "disable_overlap_schedule", False):
                raise ValueError("Cached private QSA batches require serial scheduling")
            cache = getattr(self, "_append_batch_cache", None)
            if cache is None:
                from sglang.srt.speculative.dmtp_qsa_batch import QSAAppendBatchCache

                cache = self._append_batch_cache = QSAAppendBatchCache(
                    self.owner.server_args.max_running_requests
                )
            batch = cache.build(batch_fields, spec_fields, totals_cpu)
        else:
            batch_fields["seq_lens_cpu"] = (
                None if gpu_only else torch.tensor(totals_cpu, dtype=torch.int32)
            )
            batch_fields["spec_info"] = (
                EagleDraftExtendInput(**spec_fields)
                if decode
                else SimpleNamespace(hidden_states=hc, custom_mask=None, kv_indptr=None)
            )
            batch = ForwardBatch(**batch_fields)
        if prepared is not None:
            batch._dmtp_prepared_staging = prepared.buffers
        translator = self.runner.runner_ids
        plan = translator.plan(
            req_pool_indices=req_indices,
            seq_lens=totals,
            seq_lens_cpu=batch.seq_lens_cpu,
            write_virtual=None,
            write_slots=locations,
        )
        plan.bind(batch, translator)
        if decode:
            batch.spec_info.extend_seq_lens_cpu = None if gpu_only else list(lengths)
            batch.spec_info.extend_seq_lens_tensor = lens
        self.head_hidden = None
        if decode and self.graph is not None:
            from sglang.srt.speculative.dmtp_qsa_graph import (
                assert_pool,
                restore_pool,
                snapshot_pool,
            )

            initial_check = bs not in self.graph.checked
            first = initial_check or os.getenv("SGLANG_DMTP_QSA_CHECK_ALL") == "1"
            before = (
                snapshot_pool(self.runner.token_to_kv_pool, locations, req_indices)
                if first
                else None
            )
            with self.execution_scope(), scoped_config(**self.graph_fields):
                head, logits = self.graph.execute(batch)
            if first:
                after = snapshot_pool(
                    self.runner.token_to_kv_pool, locations, req_indices
                )
                restore_pool(before)
                with self.execution_scope(), scoped_config(**self.graph_fields):
                    reference_head, reference_logits = self.graph.eager(bs)
                parity_passed = True
                try:
                    torch.testing.assert_close(head, reference_head, rtol=0, atol=0)
                    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
                    assert_pool(after)
                except AssertionError:
                    parity_passed = False
                    directory = os.getenv("SGLANG_DMTP_QSA_FAILURE_DIR")
                    if directory:
                        from qsa_parity_debug import save_failure

                        save_failure(
                            directory,
                            batch,
                            before,
                            after,
                            head,
                            reference_head,
                            logits,
                            reference_logits,
                        )
                    if (
                        initial_check
                        or os.getenv("SGLANG_DMTP_QSA_WARN_RECHECK") != "1"
                    ):
                        raise
                    logging.getLogger(__name__).warning(
                        "DMTP_QSA_MTP_GRAPH_RECHECK_DIFFERENCE B=%d capacity_rows=%d continuing_generation=True strict_parity=False",
                        bs,
                        tokens.numel(),
                    )
                restore_pool(after)
                self.graph.checked.add(bs)
                if parity_passed:
                    logging.getLogger(__name__).info(
                        "DMTP_QSA_MTP_GRAPH_STRICT_PASS B=%d mode=DRAFT_EXTEND_V2 rows=%d hidden_logits_exact=True full_compressed_ring_exact=True",
                        bs,
                        int(lens.sum()) if gpu_only else sum(lengths),
                    )
            finite = torch.isfinite(head).all()
            if logits is not None:
                finite &= torch.isfinite(logits).all()
            torch._assert_async(finite, "Nonfinite MTP graph output")
            return head, logits
        with self.execution_scope(), scoped_config(**self.fields):
            result = self.runner.forward(batch)
        if result.can_run_graph:
            raise RuntimeError(
                "Private QSA MTP must remain eager until real graph validation"
            )
        logits = result.logits_output.next_token_logits
        head = self.head_hidden
        self.head_hidden = None
        if getattr(self, "hidden_only", False):
            if logits is not None:
                raise RuntimeError(
                    "Hidden-only private MTP unexpectedly computed logits"
                )
        elif logits is None or logits.shape[0] != len(lengths):
            raise RuntimeError("Native MTP did not expose per-request logits")
        if head is None or head.shape[0] != tokens.numel():
            raise RuntimeError("Native MTP did not expose its H-wide head input/logits")
        last = lens.long().cumsum(0) - 1
        head = head.index_select(0, last).clone()
        if head.shape[1] != self.owner.draft_model.config.hidden_size:
            raise ValueError("Native MTP head input width mismatch")
        finite = torch.isfinite(head).all()
        if logits is not None:
            finite &= torch.isfinite(logits).all()
        torch._assert_async(finite, "Nonfinite native MTP output")
        return head, None if logits is None else logits.clone()
