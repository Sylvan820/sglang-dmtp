"""CPU contracts; real HC/QSA graph and cache parity require the GPU smoke."""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.speculative.dmtp_qsa_contract import (
    T1Conditioner,
    committed_inputs,
)
from sglang.srt.speculative.dmtp_qsa_graph import (
    QSAMTPGraph,
    assert_pool,
    restore_pool,
    snapshot_pool,
)
from sglang.srt.speculative.dmtp_qsa_native import NativeQSAMTP
from sglang.srt.speculative.dmtp_qsa_worker import (
    QSAController,
    QSAMTPWorker,
    validate_recipe,
)
from sglang.srt.speculative.dmtp_worker import DMTPWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class QSATest(unittest.TestCase):
    def test_private_execution_scope_restores_communicator_after_exception(self):
        from contextlib import contextmanager

        native = NativeQSAMTP.__new__(NativeQSAMTP)
        state = dict(enabled=False)

        @contextmanager
        def change_state(*, enable):
            old = state["enabled"]
            state["enabled"] = enable
            try:
                yield
            finally:
                state["enabled"] = old

        group = SimpleNamespace(
            get_available_pynccl_comm=lambda: SimpleNamespace(change_state=change_state)
        )
        native.runner = SimpleNamespace(tp_group=group)
        with (
            patch(
                "sglang.srt.speculative.dmtp_qsa_native.draft_tp_context",
                return_value=nullcontext(),
            ) as tp,
            patch(
                "sglang.srt.speculative.dmtp_qsa_native.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with native.execution_scope():
                    self.assertTrue(state["enabled"])
                    raise RuntimeError("intentional")
        self.assertFalse(state["enabled"])
        self.assertIs(tp.call_args.args[0].tp_group, group)

    def test_private_graph_selects_token_hidden_after_replay(self):
        graph = QSAMTPGraph.__new__(QSAMTPGraph)
        graph.native = SimpleNamespace(hidden_only=False)
        tokens = torch.arange(16 * 4).view(16, 4).float()
        select = torch.tensor([1, 6])
        staged = SimpleNamespace(spec_info=SimpleNamespace(select_index=select))
        graph.staged_batch = lambda batch: staged
        graph.graph_runner = SimpleNamespace(
            token_hidden=tokens,
            execute=lambda batch, index: SimpleNamespace(
                hidden_states=tokens[:2], next_token_logits=torch.zeros(2, 10)
            ),
        )
        head, logits = graph.execute(object())
        torch.testing.assert_close(head, tokens[select], rtol=0, atol=0)
        self.assertEqual(logits.shape, (2, 10))
        tokens.zero_()
        self.assertGreater(head.sum().item(), 0)

    def test_private_staging_binds_dynamic_positions_for_exact_bucket(self):
        graph = QSAMTPGraph.__new__(QSAMTPGraph)
        graph.capacity, graph.width = 32, 8
        graph.runner = SimpleNamespace(runner_ids=object())
        positions = torch.tensor([10, 11, 100, 101, 102, 103, 104])
        batch = SimpleNamespace(
            batch_size=2,
            input_ids=torch.arange(7),
            positions=positions,
            out_cache_loc=torch.arange(7) + 64,
            mrope_positions=positions[None].expand(3, -1),
            extend_seq_lens=torch.tensor([2, 5]),
            extend_seq_lens_cpu=[2, 5],
            spec_info=SimpleNamespace(hidden_states=torch.ones(7, 16)),
            iteration_ids=SimpleNamespace(bind_replay=lambda *args, **kwargs: None),
        )
        staged = graph.staged_batch(batch)
        self.assertIs(staged.spec_info.positions, staged.positions)
        self.assertEqual(staged.spec_info.select_index.tolist(), [1, 6])
        self.assertEqual(staged.spec_info.extend_seq_lens_cpu, [2, 5])
        self.assertEqual(staged.positions[:7].tolist(), positions.tolist())
        self.assertEqual(staged.out_cache_loc[7:].count_nonzero().item(), 0)

    def test_private_backend_uses_qsa_factory_without_building_target_gdn(self):
        from unittest.mock import Mock

        native = NativeQSAMTP.__new__(NativeQSAMTP)
        native.fields = {}
        native.profile = object()
        native.worker = SimpleNamespace(
            init_attention_backends=Mock(side_effect=AssertionError("Unused GDN init"))
        )
        binder = Mock()
        native.runner = SimpleNamespace(
            model=object(),
            draft_attention_backend=None,
            runner_ids=SimpleNamespace(bind_and_verify_backends=binder),
            spec_aux_config=SimpleNamespace(
                eagle_use_aux_hidden_state=False,
                eagle_aux_hidden_state_layer_ids=None,
                dflash_use_aux_hidden_state=False,
                dflash_target_layer_ids=None,
                is_dspark=False,
            ),
        )
        backend = SimpleNamespace()
        with (
            patch(
                "sglang.srt.speculative.dmtp_qsa_native.scoped_config",
                lambda **kwargs: nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.dmtp_qsa_native.configure_aux_hidden_state_capture"
            ) as configure,
            patch(
                "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend",
                return_value=backend,
            ),
        ):
            native.init_attention_backends()
        self.assertIs(native.runner.attn_backend, backend)
        self.assertEqual(native.runner.prefill_attention_backend_str, "qsa")
        self.assertEqual(native.runner.decode_attention_backend_str, "qsa")
        binder.assert_called_once_with([backend])
        configure.assert_called_once()
        native.worker.init_attention_backends.assert_not_called()

    def _private_mapping_fixture(self, enabled=True):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

        target = ReqToTokenPool(3, 129, "cpu", False)
        target.req_to_token.fill_(777)
        native = NativeQSAMTP.__new__(NativeQSAMTP)
        native.fields = {}
        native.runner = SimpleNamespace(token_to_kv_pool=object())
        native.owner = SimpleNamespace(
            server_args=SimpleNamespace(page_size=64),
            model_runner=SimpleNamespace(token_to_kv_pool=object()),
            draft_model_runner=SimpleNamespace(token_to_kv_pool=object()),
        )
        calls = []

        def allocate(**kwargs):
            calls.append(kwargs)
            native.runner.runner_ids = SimpleNamespace(
                req_to_token=kwargs["req_to_token_pool"].req_to_token
            )

        native.worker = SimpleNamespace(alloc_memory_pool=allocate)
        with (
            patch.dict(
                os.environ, SGLANG_DMTP_QSA_PRIVATE_MAPPING="1" if enabled else "0"
            ),
            patch(
                "sglang.srt.speculative.dmtp_qsa_native.scoped_config",
                lambda **kwargs: nullcontext(),
            ),
        ):
            native.alloc_memory_pool(
                MemoryPoolConfig(
                    max_total_num_tokens=128,
                    full_max_total_num_tokens=128,
                    max_running_requests=3,
                ),
                target,
                object(),
            )
        return native, target, calls[0]

    def test_failed_private_mapping_candidate_is_opt_in(self):
        native, target, call = self._private_mapping_fixture(enabled=False)
        self.assertIsNone(native.private_mapping)
        self.assertIs(call["req_to_token_pool"], target)
        self.assertEqual(call["memory_pool_config"].max_total_num_tokens, 128)
        self.assertIsNotNone(call["token_to_kv_pool_allocator"])

    def test_private_mapping_owns_aligned_nonoverlapping_chronological_ranges(self):
        native, target, call = self._private_mapping_fixture()
        table = native.private_mapping.req_to_token
        self.assertIsNot(table, target.req_to_token)
        self.assertTrue(
            torch.equal(target.req_to_token, torch.full_like(target.req_to_token, 777))
        )
        self.assertEqual(table[0].count_nonzero().item(), 0)
        self.assertEqual(native.private_stride, 192)
        self.assertEqual(call["memory_pool_config"].max_total_num_tokens, 640)
        self.assertIsNone(call["token_to_kv_pool_allocator"])
        for req in range(1, 4):
            self.assertEqual(int(table[req, 0]) % 64, 0)
            self.assertTrue(
                torch.equal(
                    table[req, 1:] - table[req, :-1], torch.ones(128, dtype=table.dtype)
                )
            )
        self.assertEqual(table[1:].unique().numel(), 3 * 129)

    def test_private_locations_preserve_groups_across_ragged_and_page_boundaries(self):
        native, _, _ = self._private_mapping_fixture()
        for reqs in ([3, 1, 2], [2, 3, 1]):
            for count in range(1, 9):
                prefixes = [3, 61, 119]
                positions = torch.cat([torch.arange(p, p + count) for p in prefixes])
                slots = native._private_locations(
                    torch.tensor(reqs), torch.full((3,), count), positions
                )
                expected = torch.cat(
                    [
                        native.private_mapping.req_to_token[r, p : p + count]
                        for r, p in zip(reqs, prefixes)
                    ]
                ).long()
                torch.testing.assert_close(slots, expected, rtol=0, atol=0)
                self.assertEqual(slots.unique().numel(), 3 * count)
                self.assertTrue(torch.equal(slots % 4, positions % 4))

    def test_private_prefill_ring_reset_is_request_local(self):
        native, _, _ = self._private_mapping_fixture()
        key = torch.ones(64, 4)
        rope = torch.ones(64, dtype=torch.int64)
        native.runner.token_to_kv_pool = SimpleNamespace(
            qsa_ring_size=16,
            qsa_key_state_buffer_pool=[key],
            qsa_rope_position_buffer=rope,
        )
        native._reset_private_ring(torch.tensor([2]))
        self.assertEqual(key[32:48].count_nonzero().item(), 0)
        self.assertEqual(rope[32:48].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(key[:32], torch.ones_like(key[:32])))
        self.assertTrue(torch.equal(key[48:], torch.ones_like(key[48:])))

    def test_private_forward_determinism_scope_restores_on_return_and_exception(self):
        from sglang.srt.runtime_context import get_context, get_exec

        class PrivateModel(torch.nn.Module):
            def forward(inner, x, *, fail=False):
                self.assertTrue(get_exec().deterministic.enable_deterministic_inference)
                if fail:
                    raise RuntimeError("intentional forward failure")
                return x + 1

        private = PrivateModel()
        private_forward = private.forward
        target = torch.nn.Identity()
        target_forward = target.forward
        native = NativeQSAMTP.__new__(NativeQSAMTP)
        native.runner = SimpleNamespace(model=private)
        native._configure_forward()
        self.assertEqual(private.forward.__wrapped__, private_forward)
        with get_context().override_server_args(enable_deterministic_inference=False):
            for forward in (private, private.forward):
                self.assertEqual(forward(torch.tensor(2)).item(), 3)
                self.assertFalse(
                    get_exec().deterministic.enable_deterministic_inference
                )
                with self.assertRaisesRegex(
                    RuntimeError, "intentional forward failure"
                ):
                    forward(torch.tensor(2), fail=True)
                self.assertFalse(
                    get_exec().deterministic.enable_deterministic_inference
                )
            self.assertEqual(target.forward, target_forward)
            self.assertEqual(target(torch.tensor(2)).item(), 2)
        with get_context().override_server_args(enable_deterministic_inference=True):
            private(torch.tensor(2))
            self.assertTrue(get_exec().deterministic.enable_deterministic_inference)

    def test_native_hc_policy_does_not_modify_target_modules(self):
        def model():
            return SimpleNamespace(
                hyper_connection_mixer=SimpleNamespace(allow_nondeterministic_mix=True),
                layers=[
                    SimpleNamespace(
                        attn_hyper_connection=SimpleNamespace(
                            allow_nondeterministic_mix=True
                        ),
                        mlp_hyper_connection=SimpleNamespace(
                            allow_nondeterministic_mix=True
                        ),
                    )
                ],
            )

        target, private = model(), model()
        native = NativeQSAMTP.__new__(NativeQSAMTP)
        native.runner = SimpleNamespace(model=SimpleNamespace(model=private))
        native._configure_hc_mix()
        for component in (
            private.hyper_connection_mixer,
            private.layers[0].attn_hyper_connection,
            private.layers[0].mlp_hyper_connection,
        ):
            self.assertFalse(component.allow_nondeterministic_mix)
        for component in (
            target.hyper_connection_mixer,
            target.layers[0].attn_hyper_connection,
            target.layers[0].mlp_hyper_connection,
        ):
            self.assertTrue(component.allow_nondeterministic_mix)

    def test_hc_atomic_opt_out_uses_existing_ordered_compute(self):
        from sglang.srt.layers.hyperconnection import (
            GatedResidual,
            HyperConnectionConfig,
        )

        module = GatedResidual(
            HyperConnectionConfig(
                hidden_size=16,
                hc_count=4,
                hc_lowrank=8,
                params_dtype=torch.float32,
                hc_per_branch_norm=True,
            ),
            use_combine=False,
        )
        self.assertTrue(module.allow_nondeterministic_mix)
        x = torch.randn(8, 64)

        def ordered(value, down, up, count, width):
            gates = torch.sigmoid(
                torch.nn.functional.linear(
                    torch.nn.functional.silu(
                        torch.nn.functional.linear(value, down) / count
                    ),
                    up,
                )
            )
            return (
                gates.unflatten(-1, (count, width))
                * value.unflatten(-1, (count, width))
            ).mean(-2)

        module._mix_compute = ordered
        module._flydsl_mix_ok = lambda: False
        expected = ordered(
            x,
            module.input_mix_weight_down.weight,
            module.input_mix_weight_up.weight,
            4,
            16,
        )
        with (
            patch(
                "sglang.srt.layers.hyperconnection.flydsl_hc_mix_supported",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.hyperconnection.fused_hc_mix_supported",
                return_value=True,
            ) as supported,
            patch(
                "sglang.srt.layers.hyperconnection.fused_hc_mix", return_value=expected
            ) as atomic,
        ):
            module.mix(x, normalized_input=x)
            atomic.assert_called_once()
            supported.reset_mock()
            atomic.reset_mock()
            module.allow_nondeterministic_mix = False
            result, _ = module.mix(x, normalized_input=x)
            supported.assert_not_called()
            atomic.assert_not_called()
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_reference_route_does_not_capture_private_mtp_by_default(self):
        calls = []
        native = NativeQSAMTP.__new__(NativeQSAMTP)
        native.fields = {}
        native.graph = None
        native.worker = SimpleNamespace(
            init_cuda_graphs=lambda **kwargs: calls.append(kwargs)
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "sglang.srt.speculative.dmtp_qsa_native.scoped_config",
                lambda **kwargs: nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.dmtp_qsa_graph.QSAMTPGraph",
                side_effect=AssertionError("Unexpected MTP capture"),
            ),
        ):
            native.init_cuda_graphs()
        self.assertEqual(calls, [{"capture_decode_cuda_graph": False}])
        self.assertIsNone(native.graph)

    def test_reference_forward_preserves_ragged_rows_and_target_slot_mapping(self):
        for lengths in ([7, 1], [1, 8], [3, 2, 7], [1] * 32):
            native = NativeQSAMTP.__new__(NativeQSAMTP)
            native.fields = {}
            native.graph = None
            native.owner = SimpleNamespace(
                draft_model=SimpleNamespace(config=SimpleNamespace(hidden_size=4))
            )
            n, bs = sum(lengths), len(lengths)
            tokens = torch.arange(n)
            hc = torch.arange(n * 16).view(n, 16).float()
            slots = torch.arange(n) + 1024
            reqs = torch.arange(bs) + 1
            prefixes = [101 + i for i in range(bs)]
            calls = []
            plans = []

            def plan(**kwargs):
                plans.append(kwargs)
                return SimpleNamespace(bind=lambda batch, reader: None)

            def forward(batch):
                calls.append(batch)
                native.head_hidden = torch.arange(n * 4).view(n, 4).float()
                return SimpleNamespace(
                    can_run_graph=False,
                    logits_output=SimpleNamespace(
                        next_token_logits=torch.zeros(bs, 10)
                    ),
                )

            native.runner = SimpleNamespace(
                runner_ids=SimpleNamespace(plan=plan), forward=forward
            )
            with patch(
                "sglang.srt.speculative.dmtp_qsa_native.scoped_config",
                lambda **kwargs: nullcontext(),
            ):
                head, logits = native.forward(
                    tokens=tokens,
                    hc=hc,
                    req_indices=reqs,
                    locations=slots,
                    prefixes=prefixes,
                    lengths=list(lengths),
                    decode=True,
                )
            self.assertEqual(len(calls), 1)
            batch = calls[0]
            self.assertEqual(batch.input_ids.numel(), n)
            self.assertIs(batch.out_cache_loc, slots)
            self.assertIs(batch.spec_info.hidden_states, hc)
            self.assertIs(plans[0]["write_slots"], slots)
            self.assertTrue(batch.forward_mode.is_draft_extend_v2())
            self.assertEqual(batch.extend_prefix_lens_cpu, prefixes)
            last = torch.tensor(lengths).cumsum(0) - 1
            self.assertEqual(
                head.tolist(), torch.arange(n * 4).view(n, 4)[last].tolist()
            )
            self.assertEqual(logits.shape[0], bs)

    def test_native_graph_reference_owns_forward_context_and_live_plan(self):
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            has_forward_context,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        self.assertFalse(has_forward_context())
        self.assertFalse(get_is_capture_mode())
        graph = QSAMTPGraph.__new__(QSAMTPGraph)
        metadata = SimpleNamespace(
            graph_ring_group_locs=torch.ones(8, 4, dtype=torch.int32),
            graph_write_locs=torch.zeros(8, dtype=torch.int32),
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(indexer_metadata=metadata),
            init_forward_metadata_out_graph=lambda batch: None,
            init_forward_metadata_in_graph=lambda batch: None,
        )
        batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([101]),
            extend_seq_lens_cpu=[5],
            extend_seq_lens=torch.tensor([5]),
            out_cache_loc=torch.arange(8),
            spec_info=SimpleNamespace(select_index=torch.tensor([4])),
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_dp_buffer_len=None,
            dp_padding_mode=SimpleNamespace(is_max_len=lambda: True),
            global_num_tokens_cpu=None,
            input_ids=torch.arange(8),
            positions=torch.arange(8),
            moe_num_token_non_padded=lambda: None,
        )
        bound = []
        graph.live_batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([105]),
            extend_seq_lens_cpu=[5],
            iteration_ids=SimpleNamespace(
                bind_replay=lambda batch, reader, slots: bound.append(slots)
            ),
        )
        graph.native = SimpleNamespace()

        def forward(ids, positions, forward_batch):
            self.assertIs(get_attn_backend(), backend)
            self.assertTrue(get_is_capture_mode())
            graph.native.head_hidden = torch.arange(32).view(8, 4)
            return SimpleNamespace(next_token_logits=torch.ones(1, 10))

        graph.runner = SimpleNamespace(
            attn_backend=backend,
            runner_ids=object(),
            model=SimpleNamespace(forward=forward),
        )
        graph.graph_runner = SimpleNamespace(model_context=nullcontext)
        graph.batches = {1: batch}
        flags = SimpleNamespace(capture=SimpleNamespace(disable_dispose_tensor=False))
        with (
            patch("sglang.srt.speculative.dmtp_qsa_graph.set_dp_buffer_len"),
            patch("sglang.srt.speculative.dmtp_qsa_graph.set_is_extend_in_batch"),
            patch(
                "sglang.srt.model_executor.runner_utils.capture_mode.get_flags",
                return_value=flags,
            ),
        ):
            hidden, logits = graph.eager(1)
        self.assertEqual(hidden.tolist(), [[16, 17, 18, 19]])
        self.assertEqual(len(bound), 1)
        self.assertFalse(has_forward_context())
        self.assertFalse(get_is_capture_mode())
        self.assertFalse(flags.capture.disable_dispose_tensor)

    def test_native_graph_direct_forward_captures_head_and_restores_method(self):
        native = SimpleNamespace(
            owner=SimpleNamespace(server_args=SimpleNamespace(max_running_requests=1))
        )
        batch = SimpleNamespace(
            batch_size=1,
            positions=torch.arange(3),
            spec_info=SimpleNamespace(select_index=torch.tensor([2])),
        )
        metadata = SimpleNamespace(
            graph_ring_group_locs=torch.ones(3, 4, dtype=torch.int32),
            graph_write_locs=torch.tensor([0, 1, 0]),
        )

        class Model(torch.nn.Module):
            def forward(self, input_ids, positions, forward_batch):
                native.head_hidden = torch.arange(12).view(3, 4)
                return SimpleNamespace(
                    hidden_states=torch.zeros(1, 16),
                    next_token_logits=torch.zeros(1, 10),
                )

        model = Model()
        original = model.forward
        native.runner = SimpleNamespace(
            model=model,
            attn_backend=SimpleNamespace(
                forward_metadata=SimpleNamespace(indexer_metadata=metadata)
            ),
        )
        outputs = []

        class Runner:
            captured_req_width = 8

            def __init__(self, worker, **kwargs):
                if worker.draft_runner.graph_shared_output is None:
                    raise AssertionError(
                        "Native runner requires shared output before capture"
                    )
                outputs.append(
                    worker.draft_runner.model.forward(None, batch.positions, batch)
                )

        with (
            patch("sglang.srt.speculative.dmtp_qsa_graph._QSAExtendRunner", Runner),
            patch(
                "sglang.srt.speculative.dmtp_qsa_graph.GraphSharedOutput.create_for_model_runner",
                return_value=object(),
            ),
        ):
            graph = QSAMTPGraph(native)
        self.assertIs(graph.batches[1], batch)
        self.assertEqual(
            outputs[0].hidden_states.tolist(), torch.arange(12).view(3, 4).tolist()
        )
        self.assertIs(batch.spec_info.positions, batch.positions)
        self.assertEqual(model.forward, original)
        self.assertEqual(
            metadata.graph_ring_group_locs.tolist(), [[0] * 4, [1] * 4, [0] * 4]
        )

    def test_commit_uses_one_packed_extend_with_reference_pairing(self):
        for counts in ([1], [8], [1, 8], [7, 2, 4, 8]):
            bs = len(counts)
            c = QSAController.__new__(QSAController)
            c.two_graphs = False
            c.batched_state = False
            candidates = torch.arange(bs * 8).view(bs, 8)
            hc = torch.arange(bs * 8 * 4).view(bs, 8, 4).float()
            slots = torch.arange(bs * 8).view(bs, 8) + 64
            bonus = torch.arange(bs) + 1000
            prefixes = torch.arange(bs) * 13 + 101
            calls = []

            def forward(**kwargs):
                calls.append(kwargs)
                return torch.zeros(bs, 4), torch.zeros(bs, 10)

            c.native = SimpleNamespace(forward=forward, graph=None)
            c.graph_used = dict(target=True, draft=True)
            c._hc = lambda value, rows: value
            c.batch = SimpleNamespace(
                req_pool_indices=torch.arange(bs) + 1, reqs=list(range(bs))
            )
            c.states, c.checked_target = {}, set()
            counts_t = torch.tensor(counts)
            c.commit(
                candidates,
                counts_t - 1,
                bonus,
                None,
                prefixes + counts_t,
                slots,
                hc=hc.flatten(0, 1),
            )
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call["lengths"], list(counts))
            self.assertEqual(call["prefixes"], prefixes.tolist())
            self.assertEqual(
                call["tokens"].tolist(),
                sum(
                    [
                        candidates[i, 1:n].tolist() + [int(bonus[i])]
                        for i, n in enumerate(counts)
                    ],
                    [],
                ),
            )
            torch.testing.assert_close(
                call["hc"], torch.cat([hc[i, :n] for i, n in enumerate(counts)])
            )
            torch.testing.assert_close(
                call["locations"],
                torch.cat([slots[i, :n] for i, n in enumerate(counts)]),
            )
            self.assertEqual(len(c.states), bs)

    def test_joint_residual_uses_the_complete_strict_weight_stream(self):
        from sglang.srt.speculative.dflash2_contract import checked_weights

        model = torch.nn.Module()
        model.mtp_prefix_conditioner = T1Conditioner(16, 1e-6)
        weights = [
            (name, value.detach().clone()) for name, value in model.named_parameters()
        ]
        self.assertEqual(len(list(checked_weights(model, iter(weights)))), 3)
        with self.assertRaises(ValueError):
            list(checked_weights(model, iter(weights[:-1])))
        with self.assertRaises(ValueError):
            list(checked_weights(model, iter(weights + weights[:1])))
        bad = list(weights)
        bad[0] = (bad[0][0], torch.full_like(bad[0][1], float("nan")))
        with self.assertRaises(ValueError):
            list(checked_weights(model, iter(bad)))

    def test_config_selects_worker_without_changing_dense_path(self):
        args = SimpleNamespace(
            model_path="/target", trust_remote_code=False, revision=None
        )
        for qsa, expected in ((False, DMTPWorkerV2), (True, QSAMTPWorker)):
            with (
                patch.dict(os.environ, {"SGLANG_DMTP_T1_RESIDUAL": "1"}),
                patch.object(
                    SpeculativeAlgorithm, "get_platform_worker_cls", return_value=None
                ),
                patch(
                    "sglang.srt.speculative.spec_info.resolving_view",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "sglang.srt.hf_transformers.config.get_config",
                    return_value=object(),
                ),
                patch(
                    "sglang.srt.layers.attention.qsa.config.is_qwen_qsa",
                    return_value=qsa,
                ),
            ):
                self.assertIs(SpeculativeAlgorithm.DFLASH.create_worker(args), expected)

    def test_t1_joint_residual_preserves_anchor_and_masks(self):
        torch.manual_seed(42)
        module = T1Conditioner(16, 1e-6)
        for bs in (1, 2, 8, 31, 32):
            blocks = torch.randn(bs, 8, 16)
            hidden = torch.randn(bs, 16)
            before = blocks.clone()
            expected = blocks[:, 1] + module.linear_fc(
                torch.cat((module.norm1(blocks[:, 1]), module.norm2(hidden)), -1)
            )
            actual = module(blocks, hidden)
            torch.testing.assert_close(actual[:, 1], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                actual[:, [0, 2, 3, 4, 5, 6, 7]],
                before[:, [0, 2, 3, 4, 5, 6, 7]],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(blocks, before, rtol=0, atol=0)

    def test_every_commit_length_has_one_bonus(self):
        candidates = torch.arange(64).view(8, 8)
        counts = torch.arange(1, 9)
        actual = committed_inputs(candidates, counts, torch.arange(100, 108))
        for i in range(8):
            self.assertEqual(
                actual[i, : i + 1].tolist(),
                candidates[i, 1 : i + 1].tolist() + [100 + i],
            )

    def test_snapshot_covers_entire_request_ring_and_compressed_writes(self):
        k, v = torch.arange(256).view(64, 4).float(), torch.zeros(64, 4)
        state, rope, compressed = (
            torch.zeros(64, 4),
            torch.zeros(64, dtype=torch.int32),
            torch.zeros(16, 4),
        )
        pool = SimpleNamespace(
            full_attention_layer_id_mapping={0: 0},
            get_kv_buffer=lambda _: (k, v),
            qsa_key_state_buffer_pool=[state],
            qsa_rope_position_buffer=rope,
            qsa_compressed_k_buffer_pool=[compressed],
            qsa_ring_size=16,
            qsa_compress_ratio=4,
        )
        slots, reqs = torch.tensor([12, 20]), torch.tensor([2, 3])
        before = snapshot_pool(pool, slots, reqs)
        self.assertEqual(before[2][1].tolist(), list(range(16)) + list(range(32, 64)))
        self.assertIn(0, before[0][1].tolist())
        self.assertIn(0, before[-1][1].tolist())
        k[slots] = 5
        state[32:64] = 8
        compressed[slots // 4] = 9
        after = snapshot_pool(pool, slots, reqs)
        restore_pool(before)
        assert_pool(before)
        restore_pool(after)
        assert_pool(after)

    def test_qsa_guards_keep_unsupported_modes_closed(self):
        cfg = dict(
            page_size=64,
            speculative_num_draft_tokens=8,
            enable_lora=False,
            enable_two_batch_overlap=False,
            enable_unified_memory=False,
            enable_dp_attention=False,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            quantization=None,
            speculative_draft_window_size=None,
            max_running_requests=32,
        )
        profile = SimpleNamespace(compress_ratio=4)
        validate_recipe(cfg.__getitem__, profile)
        for key, value in [
            ("page_size", 1),
            ("speculative_num_draft_tokens", 4),
            ("max_running_requests", 33),
            ("enable_lora", True),
            ("disable_overlap_schedule", False),
            ("disable_radix_cache", False),
            ("chunked_prefill_size", 2048),
            ("quantization", "fp8"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(dict(cfg, **{key: value}).__getitem__, profile)

    def test_overlap_requires_owned_gpu_state_and_no_cached_batch(self):
        from unittest.mock import patch

        cfg = dict(
            page_size=64,
            speculative_num_draft_tokens=8,
            enable_lora=False,
            enable_two_batch_overlap=False,
            enable_unified_memory=False,
            enable_dp_attention=False,
            disable_overlap_schedule=False,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            quantization=None,
            speculative_draft_window_size=None,
            max_running_requests=32,
        )
        flags = {
            "SGLANG_DMTP_QSA_" + name: "1"
            for name in (
                "OVERLAP",
                "MTP_GRAPH",
                "TWO_DRAFT_GRAPHS",
                "BATCHED_STATE",
                "FUSED_PREPARE",
                "DIRECT_PACK",
                "GPU_COMMIT",
                "FUSED_STATE_READ",
            )
        }
        profile = SimpleNamespace(compress_ratio=4)
        with patch.dict("os.environ", flags, clear=True):
            validate_recipe(cfg.__getitem__, profile)
            for name in flags:
                with self.subTest(flag=name), patch.dict("os.environ", {name: "0"}):
                    with self.assertRaises(ValueError):
                        validate_recipe(cfg.__getitem__, profile)
            with patch.dict("os.environ", {"SGLANG_DMTP_QSA_CACHED_BATCH": "1"}):
                with self.assertRaises(ValueError):
                    validate_recipe(cfg.__getitem__, profile)
            for key in ("enable_two_batch_overlap", "enable_dp_attention"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_recipe(dict(cfg, **{key: True}).__getitem__, profile)


if __name__ == "__main__":
    unittest.main()
