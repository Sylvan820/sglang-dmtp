"""CPU dispatch contracts; graph numerics require the separate GPU regression."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/layers/attention/qsa"


def load_function(filename, name, scope):
    tree = ast.parse((SOURCE / filename).read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope[name]


class DeterministicDispatchTest(unittest.TestCase):
    def test_topk_respects_scoped_determinism_and_preserves_default_and_hip(self):
        for deterministic in (False, True):
            for invariant in (False, True):
                for hip in (False, True):
                    with self.subTest(
                        deterministic=deterministic, invariant=invariant, hip=hip
                    ):
                        calls = []
                        scope = dict(
                            torch=torch,
                            is_hip=lambda: hip,
                            is_batch_invariant_mode_enabled=lambda: invariant,
                            get_exec=lambda: SimpleNamespace(
                                deterministic=SimpleNamespace(
                                    enable_deterministic_inference=deterministic
                                )
                            ),
                        )
                        function = load_function("kernel.py", "qsa_fast_topk", scope)
                        modules = {
                            "flashinfer": SimpleNamespace(
                                TopKTieBreak=SimpleNamespace(SMALL="small"),
                                top_k_ragged_transform=lambda *a, **kw: calls.append(
                                    ("stable", kw)
                                ),
                            ),
                            "sglang.kernels.ops.attention.fast_topk": SimpleNamespace(
                                fast_topk=lambda *a, **kw: calls.append(("atomic", kw))
                            ),
                        }
                        with patch.dict(sys.modules, modules):
                            function(
                                SimpleNamespace(is_cuda=True, device="cpu"),
                                torch.tensor([3]),
                                torch.tensor([544]),
                                512,
                            )
                        expected = (
                            "stable"
                            if (deterministic or invariant) and not hip
                            else "atomic"
                        )
                        self.assertEqual(calls[0][0], expected)
                        if expected == "stable":
                            self.assertTrue(calls[0][1]["deterministic"])
                            self.assertTrue(calls[0][1]["dsa_graph_safe"])
                            self.assertEqual(calls[0][1]["tie_break"], "small")
                            self.assertEqual(calls[0][1]["row_starts"].tolist(), [3])

    def test_stable_decode_scores_short_rows_and_preserves_default_fast_path(self):
        for deterministic in (False, True):
            for invariant in (False, True):
                with self.subTest(deterministic=deterministic, invariant=invariant):
                    calls = []

                    def score(*args, **kwargs):
                        calls.append(("score", kwargs["min_scored_len"]))
                        return SimpleNamespace(is_cuda=True)

                    scope = dict(
                        torch=torch,
                        qsa_mqa_decode=score,
                        is_batch_invariant_mode_enabled=lambda: invariant,
                        get_exec=lambda: SimpleNamespace(
                            deterministic=SimpleNamespace(
                                enable_deterministic_inference=deterministic
                            )
                        ),
                        qsa_fast_topk=lambda *a, **kw: calls.append(("stable", None)),
                    )
                    function = load_function(
                        "qsa_indexer.py", "select_decode_tokens", scope
                    )
                    with patch.dict(
                        sys.modules,
                        {
                            "sglang.kernels.ops.attention.fast_topk": SimpleNamespace(
                                fast_topk=lambda *a, **kw: calls.append(
                                    ("atomic", None)
                                )
                            )
                        },
                    ):
                        function(
                            SimpleNamespace(block_topk=512),
                            SimpleNamespace(is_cuda=True),
                            None,
                            None,
                            torch.tensor([0, 511, 512, 513]),
                            4096,
                            None,
                            None,
                            defer_expansion=True,
                        )
                    stable = deterministic or invariant
                    self.assertEqual(
                        calls,
                        [
                            ("score", 0 if stable else 512),
                            ("stable" if stable else "atomic", None),
                        ],
                    )
