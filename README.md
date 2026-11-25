# DMTP: A Better First Token for Block Drafting via MTP

[![Blog](https://img.shields.io/badge/Blog-EN%20%2F%20ZH-0f766e?style=flat-square&logo=readthedocs&logoColor=white)](https://sylvan820.github.io/D-MTP/)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-ffcc4d?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Piggy12)
[![SGLang PR](https://img.shields.io/badge/SGLang-PR%20%2350270-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/sgl-project/sglang/pull/50270)
[![SpecForge PR](https://img.shields.io/badge/SpecForge-PR%20%231189-7c3aed?style=flat-square&logo=github&logoColor=white)](https://github.com/sgl-project/SpecForge/pull/1189)

DMTP combines pretrained native MTP with parallel DFlash2 drafting. MTP proposes the first token; DFlash2 uses that proposal and its predictive representation to draft the remaining block in parallel. The target verifies the combined proposals.

This is a DMTP-enabled SGLang fork, supporting the dense FA3 path for Qwen3.6-27B and the HC/QSA path for Qwen3.8 Flash Next.

![DMTP architecture](https://huggingface.co/Piggy12/Qwen3.6-27B-DMTP/resolve/main/architecture.svg)

## Method

1. **MTP first:** native MTP predicts the first candidate, T1.
2. **Parallel drafting:** DFlash2 receives `[A, T1, MASK x 6]`, where A is the last emitted token. A learned residual injects the MTP representation into the T1 embedding.
3. **Target verification:** DFlash2 proposes T2 through T7. The target verifies all seven proposals using token-level rejection sampling.

The original target-context path and A/MASK embeddings remain unchanged. The 27B checkpoint uses a low-rank hidden-state residual; the 120B checkpoint fuses separately normalized T1 embeddings and MTP hidden states inside its residual branch. The checkpoint configuration determines the adapter.

See the [blog](https://sylvan820.github.io/D-MTP/) for motivation, ablations, and results. Inference performance is still being optimized.

## Models

| Draft | Target | Runtime |
| :--- | :--- | :--- |
| [Qwen3.6-27B-DMTP](https://huggingface.co/Piggy12/Qwen3.6-27B-DMTP) | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Dense FA3 |
| [Qwen3.8-27B-DMTP](https://huggingface.co/Piggy12/Qwen3.8-27B-DMTP) | Qwen3.8-27B | See model card |
| [Qwen3.8-next-120b-dmtp](https://huggingface.co/Piggy12/Qwen3.8-next-120b-dmtp) | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | HC/QSA |

These repositories contain draft weights only. Matching target weights, native MTP, shared embeddings, and the language-model head are supplied separately.

## Installation

Use a CUDA environment with GPU kernels and dependencies matching this SGLang checkout.

```bash
git clone --branch main https://github.com/Sylvan820/sglang-dmtp.git
cd sglang-dmtp
pip install -e python
pip install -U huggingface_hub
```

## Inference

### Qwen3.6-27B

```bash
bash "$(hf download Piggy12/Qwen3.6-27B-DMTP serve_dmtp.sh)"
```

Defaults: TP8, concurrency up to 32, 3072-token context, decode CUDA Graphs, eager prefill, serial scheduling. T1 sampling and conditioning run inside the MTP update graph; candidate selection runs inside the DFlash graph.

### Qwen3.8 Flash Next 120B

```bash
bash "$(hf download Piggy12/Qwen3.8-next-120b-dmtp serve_dmtp.sh)"
```

Defaults: unquantized target with native MTP, TP4, concurrency up to 32, 64K context, paged KV with page size 64, decode CUDA Graphs, eager full-prompt prefill, and serial scheduling. The launcher enables device-side commit, fused state reads, and packed graph inputs. Do not substitute a dense FA3 page-one configuration or an FP8 target.

Both launchers use eight verification slots for seven proposals and listen on `127.0.0.1:39532`. Model locations and hardware settings can be overridden:

```bash
TARGET_MODEL=/path/to/target \
DRAFT_MODEL=/path/to/draft \
TP_SIZE=4 CONTEXT_LENGTH=8192 PORT=39532 \
bash "$(hf download Piggy12/Qwen3.8-next-120b-dmtp serve_dmtp.sh)"
```

Choose TP, context length, and KV capacity to fit your GPUs. These are reference launch configurations; validate the exact hardware, checkpoint, and serving configuration before deployment.

### Send a Request

```bash
curl http://127.0.0.1:39532/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a Python binary search function.","sampling_params":{"temperature":1,"top_p":1,"top_k":-1,"max_new_tokens":256}}'
```

## Implementation

- [DMTP worker](python/sglang/srt/speculative/dmtp_worker.py): shared entry point and dense drafting.
- [QSA worker](python/sglang/srt/speculative/dmtp_qsa_worker.py): HC features, native MTP state, QSA cache management, and guarded fast paths.
- [Regression tests and optimization flags](test/manual/dmtp/README.md).

The upstream PR contains the common DMTP path; architecture-specific QSA support is provided by this fork. Optional overlap and other fast paths have explicit guards. CPU regression coverage does not replace GPU correctness and performance validation of a deployed configuration.

## License

Runtime code retains SGLang's [Apache-2.0 license](LICENSE). DMTP draft weight licenses are provided in their model repositories. Qwen3.6-27B uses Apache-2.0; the separately supplied Qwen3.8-Flash-Next target and native MTP are governed by the Qwen Community License.

## Citation

```bibtex
@misc{shen2026dmtp,
  author = {Shen, Yuhao and Huang, Jianuo},
  title  = {{DMTP}: A Better First Token for Block Drafting via MTP},
  year   = {2026},
  url    = {https://sylvan820.github.io/D-MTP/},
  note   = {Technical blog}
}
```

Built on [SGLang](https://github.com/sgl-project/sglang), [DFlash2](https://github.com/z-lab/dflash), and native Qwen MTP. Please also acknowledge the upstream projects when using them.
