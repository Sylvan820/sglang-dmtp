"""Tensor contracts for the opt-in, four-slot QSA DMTP adapter."""

import torch
from torch import nn


def validate_draft_contract(config):
    dc = config.dflash_config
    required = {
        "mtp_t1_fusion": True,
        "mtp_t1_residual": True,
        "mtp_t1_input_slot": 1,
        "scored_query_slots": list(range(1, 8)),
        "total_proposal_token_count": 8,
        "mtp_fusion_norm_zero_centered_gamma": False,
    }
    if dc.get("mtp_drop_anchor", False) or any(
        key not in dc or dc[key] != value for key, value in required.items()
    ):
        raise ValueError("QSA DMTP requires the trained dual-anchor T1 FC-residual contract")
    if dc.get("target_layer_ids") != [5, 15, 25, 35, 45]:
        raise ValueError("QSA DMTP requires the five completed HC-mean features")
    if dc.get("target_hidden_source") != "post_layer_hc_mean":
        raise ValueError("QSA DMTP requires explicit target_hidden_source=post_layer_hc_mean")


class T1Conditioner(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.linear_fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, embedding, hidden):
        if embedding.shape != hidden.shape:
            raise ValueError("T1 embedding requires the H-wide MTP head input, not HC")
        return embedding + self.linear_fc(
            torch.cat((self.norm1(embedding), self.norm2(hidden)), dim=-1)
        )


def shifted_prompt(tokens, lengths, anchors):
    if (len(lengths) != anchors.numel() or any(n < 1 for n in lengths)
            or sum(lengths) != tokens.numel()):
        raise ValueError("Unaligned MTP prompt")
    chunks, offset = [], 0
    for i, n in enumerate(lengths):
        chunks.append(torch.cat((tokens[offset + 1:offset + n], anchors[i:i + 1])))
        offset += n
    return torch.cat(chunks)


def committed_inputs(candidates, commit_lens, bonus):
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError("QSA DMTP retains exactly four Target slots")
    bs = candidates.shape[0]
    if commit_lens.numel() != bs or bonus.numel() != bs:
        raise ValueError("Unaligned MTP commit")
    torch._assert_async(((commit_lens >= 1) & (commit_lens <= 4)).all(), "Invalid MTP commit length")
    shifted = torch.zeros_like(candidates)
    shifted[:, :-1] = candidates[:, 1:]
    shifted.scatter_(1, (commit_lens.long() - 1)[:, None], bonus[:, None])
    return shifted


def sample_sparse_t1(logits, temperatures, greedy_mask, uniforms, top_k):
    if logits.ndim != 2 or not 1 <= top_k <= logits.shape[1]:
        raise ValueError("Invalid native T1 proposal shape")
    bs = logits.shape[0]
    temperatures = temperatures.reshape(bs).float()
    greedy = greedy_mask.reshape(bs) | (temperatures <= 0)
    values, ids = logits.topk(top_k, dim=-1, sorted=True)
    safe = torch.where(greedy, torch.ones_like(temperatures), temperatures)
    q = torch.softmax(values.float() / safe[:, None], dim=-1)
    point = torch.zeros_like(q)
    point[:, 0] = 1
    q = torch.where(greedy[:, None], point, q)
    torch._assert_async(torch.isfinite(q).all(), "Nonfinite T1 proposal")
    cdf = q.cumsum(-1)
    index = torch.searchsorted(cdf.contiguous(), (uniforms.reshape(bs, 1) * cdf[:, -1:]).contiguous(), right=True)
    index.clamp_max_(top_k - 1)
    return ids.gather(1, index)[:, 0], ids, q


def join_sparse_proposal(t1_ids, t1_q, tail_ids, tail_q):
    if (t1_ids.shape != t1_q.shape or tail_ids.shape != tail_q.shape
            or tail_ids.shape != (t1_ids.shape[0], 2, t1_ids.shape[1])):
        raise ValueError("Expected one native T1 and two selector proposals with common K")
    return torch.cat((t1_ids[:, None], tail_ids), 1), torch.cat((t1_q[:, None], tail_q), 1)
