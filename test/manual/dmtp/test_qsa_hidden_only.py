"""Private QSA MTP hidden-only contracts; other MTP callers retain logits/HC."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def model_output_method():
    path = ROOT / "python/sglang/srt/models/qwen4_exp_mtp.py"
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_set_hc_logits_hidden_states"
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
        scope := dict(torch=torch, Optional=Optional, ForwardBatch=object),
    )
    return scope[node.name]


class HiddenOnlyTest(unittest.TestCase):
    def test_normalized_output_is_private_and_default_keeps_hc(self):
        method = model_output_method()
        normalized = torch.arange(8, dtype=torch.float32).view(4, 2)
        hc = torch.arange(32, dtype=torch.float32).view(4, 8)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: True),
            spec_info=SimpleNamespace(select_index=None),
            capture_hidden_mode=SimpleNamespace(is_full=lambda: True),
        )
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                output = SimpleNamespace(
                    hidden_states=normalized, next_token_logits=None
                )
                owner = SimpleNamespace(
                    hc_count=4, hidden_size=2, _dmtp_qsa_normalized_output=enabled
                )
                method(owner, output, hc, batch)
                self.assertIs(output.hidden_states, normalized if enabled else hc)

    def test_hidden_processor_skips_head_only_for_explicit_full_capture(self):
        from sglang.srt.speculative.mtp_hidden_only import HiddenOnlyLogitsProcessor

        original = Mock(return_value="reference")
        processor = HiddenOnlyLogitsProcessor(original, enabled=True)
        hidden = torch.arange(6, dtype=torch.float32).view(3, 2)
        batch = SimpleNamespace(
            capture_hidden_mode=SimpleNamespace(is_full=lambda: True)
        )
        result = processor(torch.arange(3), hidden, object(), batch)
        self.assertIs(result.hidden_states, hidden)
        self.assertIsNone(result.next_token_logits)
        original.assert_not_called()
        batch.capture_hidden_mode.is_full = lambda: False
        self.assertEqual(
            processor(torch.arange(3), hidden, object(), batch), "reference"
        )
        original.assert_called_once()

    def test_private_configuration_is_opt_in_and_preserves_original_processor(self):
        from sglang.srt.speculative.dmtp_qsa_native import NativeQSAMTP
        from sglang.srt.speculative.mtp_hidden_only import HiddenOnlyLogitsProcessor

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                native = NativeQSAMTP.__new__(NativeQSAMTP)
                original = torch.nn.Identity()
                model = torch.nn.Module()
                model.logits_processor = original
                native.runner = SimpleNamespace(model=model)
                with patch.dict(
                    "os.environ", SGLANG_DMTP_QSA_HIDDEN_ONLY="1" if enabled else "0"
                ):
                    native._configure_hidden_only()
                self.assertEqual(native.hidden_only, enabled)
                if enabled:
                    self.assertIsInstance(
                        model.logits_processor, HiddenOnlyLogitsProcessor
                    )
                    self.assertIs(model.logits_processor.original, original)
                    self.assertTrue(model._dmtp_qsa_normalized_output)
                else:
                    self.assertIs(model.logits_processor, original)
                    self.assertFalse(
                        getattr(model, "_dmtp_qsa_normalized_output", False)
                    )

    def test_hidden_graph_returns_selected_rows_without_cloning_missing_logits(self):
        from sglang.srt.speculative.dmtp_qsa_graph import QSAMTPGraph

        graph = QSAMTPGraph.__new__(QSAMTPGraph)
        staged = SimpleNamespace(
            spec_info=SimpleNamespace(select_index=torch.tensor([0, 3]))
        )
        graph.staged_batch = lambda batch: staged
        all_hidden = torch.arange(8, dtype=torch.float32).view(4, 2)
        graph.native = SimpleNamespace(hidden_only=True)
        graph.graph_runner = SimpleNamespace(
            token_hidden=all_hidden,
            execute=lambda *args: SimpleNamespace(next_token_logits=None),
        )
        hidden, logits = graph.execute(object())
        self.assertTrue(torch.equal(hidden, all_hidden[[0, 3]]))
        self.assertIsNone(logits)
        all_hidden.zero_()
        self.assertGreater(hidden.sum().item(), 0)
