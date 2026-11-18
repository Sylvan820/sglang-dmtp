"""Packed committed-row metadata for QSA MTP; logical positions stay unshifted."""

from itertools import accumulate
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B", "ROWS"])
def _prepare(
    META,
    REQS,
    TABLE,
    POS,
    LOC,
    MROPE,
    TOTAL,
    CORRECT,
    LAST,
    B,
    ROWS,
    STRIDE: tl.constexpr,
    COLUMN: tl.constexpr,
    PRIVATE: tl.constexpr,
):
    request = tl.program_id(0)
    prefix = tl.load(META + request)
    count = tl.load(META + B + request)
    start = tl.load(META + 2 * B + request)
    offset = tl.arange(0, 8)
    row = start + offset
    position = prefix + offset
    valid = offset < count
    tl.store(POS + row, position, valid)
    for axis in tl.static_range(3):
        tl.store(MROPE + axis * ROWS + row, position, valid)
    if PRIVATE:
        slot = tl.load(REQS + request)
        location = tl.load(TABLE + slot * STRIDE + position * COLUMN, valid, other=0)
        tl.store(LOC + row, location, valid)
    tl.store(TOTAL + request, prefix + count)
    tl.store(CORRECT + request, count - 1)
    tl.store(LAST + request, start + count - 1)


def prepare_append(req_indices, prefixes, lengths, locations, mapping=None):
    bs, rows = len(lengths), sum(lengths)
    if (
        not bs
        or len(prefixes) != bs
        or req_indices.numel() != bs
        or locations.numel() != rows
        or any(n < 1 or n > 8 for n in lengths)
    ):
        raise ValueError("QSA packed metadata requires aligned 1..8 committed rows")
    if mapping is not None and any(
        p < 0 or p + n > mapping.shape[1] for p, n in zip(prefixes, lengths)
    ):
        raise ValueError("Private QSA MTP append exceeds context capacity")
    starts = [0, *accumulate(lengths[:-1])]
    device = req_indices.device
    metadata = torch.tensor(
        [prefixes, lengths, starts], dtype=torch.int32, device=device
    )
    prefix, lens, start = metadata.unbind(0)
    if device.type == "cpu":
        positions = torch.cat(
            [torch.arange(p, p + n, device=device) for p, n in zip(prefixes, lengths)]
        )
        if mapping is not None:
            locations = mapping[
                torch.repeat_interleave(req_indices, lens.long()), positions
            ].long()
        return (
            lens,
            prefix,
            prefix + lens,
            positions,
            locations,
            start,
            positions[None].expand(3, -1).contiguous(),
            lens - 1,
            lens.long().cumsum(0) - 1,
        )
    positions = torch.empty(rows, dtype=torch.int64, device=device)
    mrope = torch.empty((3, rows), dtype=torch.int64, device=device)
    totals = torch.empty_like(lens)
    correct = torch.empty_like(lens)
    last = torch.empty(bs, dtype=torch.int64, device=device)
    if mapping is not None:
        locations = torch.empty(rows, dtype=torch.int64, device=device)
    _prepare[(bs,)](
        metadata,
        req_indices,
        mapping if mapping is not None else locations,
        positions,
        locations,
        mrope,
        totals,
        correct,
        last,
        bs,
        rows,
        mapping.stride(0) if mapping is not None else 0,
        mapping.stride(1) if mapping is not None else 0,
        mapping is not None,
        num_warps=1,
    )
    return lens, prefix, totals, positions, locations, start, mrope, correct, last


@triton.jit(do_not_specialize=["COUNT", "N", "M0"])
def _pad_small(
    IDS,
    POS,
    LOC,
    MROPE,
    OUT_IDS,
    OUT_POS,
    OUT_LOC,
    OUT_MROPE,
    COUNT,
    N,
    CAP: tl.constexpr,
    M0,
    M1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < COUNT
    tl.store(OUT_IDS + row, tl.load(IDS + row, valid, other=0), row < N)
    tl.store(OUT_POS + row, tl.load(POS + row, valid, other=0), row < N)
    tl.store(OUT_LOC + row, tl.load(LOC + row, valid, other=0), row < N)
    for axis in tl.static_range(3):
        value = tl.load(MROPE + axis * M0 + row * M1, valid, other=0)
        tl.store(OUT_MROPE + axis * CAP + row, value, row < N)


def stage_graph_inputs(batch, buffers, n):
    count = batch.input_ids.numel()
    if count > n or n > buffers.input_ids.numel():
        raise ValueError("QSA graph input staging exceeds capacity")
    fields = ("input_ids", "positions", "out_cache_loc")
    if batch.input_ids.is_cuda:
        _pad_small[(triton.cdiv(n, 128),)](
            batch.input_ids,
            batch.positions,
            batch.out_cache_loc,
            batch.mrope_positions,
            buffers.input_ids,
            buffers.positions,
            buffers.out_cache_loc,
            buffers.mrope_positions,
            count,
            n,
            buffers.input_ids.numel(),
            *batch.mrope_positions.stride(),
            128,
        )
    else:
        for name in fields:
            getattr(buffers, name)[:count].copy_(getattr(batch, name))
            getattr(buffers, name)[count:n].zero_()
        buffers.mrope_positions[:, :count].copy_(batch.mrope_positions)
        buffers.mrope_positions[:, count:n].zero_()
    # Keep the native runner's large contiguous hidden-copy DMA path.
    buffers.hidden_states[:count].copy_(batch.spec_info.hidden_states)
    buffers.hidden_states[count:n].zero_()


@triton.jit(do_not_specialize=["B", "ROWS"])
def _pack_committed(
    CANDIDATES,
    BONUS,
    HC,
    LOC,
    REQS,
    META,
    TABLE,
    IDS,
    HIDDEN,
    OUT_LOC,
    POS,
    MROPE,
    TOTAL,
    CORRECT,
    LAST,
    B,
    ROWS,
    CAP: tl.constexpr,
    H: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    BONUS_STRIDE: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    R0: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    PRIVATE: tl.constexpr,
    BLOCK: tl.constexpr,
    DEVICE_ROWS: tl.constexpr = False,
):
    total_rows = tl.load(ROWS) if DEVICE_ROWS else ROWS
    request, tile = tl.program_id(0), tl.program_id(1)
    prefix = tl.load(META + request)
    count = tl.load(META + B + request)
    start = tl.load(META + 2 * B + request)
    j, d = tl.arange(0, 8), tile * BLOCK + tl.arange(0, BLOCK)
    row = start + j
    padding_row = request * 8 + j
    valid = j < count
    padding = padding_row >= total_rows
    hidden = tl.load(
        HC + request * H0 + j[:, None] * H1 + d[None, :] * H2,
        valid[:, None] & (d[None, :] < H),
        other=0,
    )
    tl.store(
        HIDDEN + row[:, None] * H + d[None, :],
        hidden,
        valid[:, None] & (d[None, :] < H),
    )
    tl.store(
        HIDDEN + padding_row[:, None] * H + d[None, :],
        0,
        padding[:, None] & (d[None, :] < H),
    )
    if tile == 0:
        token = tl.load(
            CANDIDATES + request * C0 + (j + 1) * C1, j < count - 1, other=0
        )
        token = tl.where(j == count - 1, tl.load(BONUS + request * BONUS_STRIDE), token)
        position = prefix + j
        if PRIVATE:
            slot = tl.load(REQS + request * R0)
            location = tl.load(TABLE + slot * T0 + position * T1, valid, other=0)
        else:
            location = tl.load(LOC + request * L0 + j * L1, valid, other=0)
        tl.store(IDS + row, token, valid)
        tl.store(POS + row, position, valid)
        tl.store(OUT_LOC + row, location, valid)
        tl.store(IDS + padding_row, 0, padding)
        tl.store(POS + padding_row, 0, padding)
        tl.store(OUT_LOC + padding_row, 0, padding)
        for axis in tl.static_range(3):
            tl.store(MROPE + axis * CAP + row, position, valid)
            tl.store(MROPE + axis * CAP + padding_row, 0, padding)
        tl.store(TOTAL + request, prefix + count)
        tl.store(CORRECT + request, count - 1)
        tl.store(LAST + request, start + count - 1)


@triton.jit
def _device_commit_metadata(
    ACCEPT,
    TOTALS,
    META,
    VALID,
    B: tl.constexpr,
    A_STRIDE: tl.constexpr,
    T_STRIDE: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    count = tl.load(ACCEPT + i * A_STRIDE, i < B, other=-1).to(tl.int32) + 1
    total = tl.load(TOTALS + i * T_STRIDE, i < B, other=0).to(tl.int32)
    ok = (count >= 1) & (count <= 8) & (total >= count) & (total <= LIMIT)
    count = tl.where((i < B) & ok, count, 0)
    end = tl.cumsum(count)
    tl.store(META + i, tl.where(ok, total - count, 0), i < B)
    tl.store(META + B + i, count, i < B)
    tl.store(META + 2 * B + i, end - count, i < B)
    tl.store(META + 3 * B, tl.sum(count))
    tl.store(VALID, tl.sum(((i < B) & ~ok).to(tl.int32)) == 0)


def pack_committed_inputs(
    candidates,
    bonus,
    hc,
    locations,
    req_indices,
    prefixes,
    lengths,
    mapping,
    buffers,
    *,
    accept_len=None,
    new_lengths=None,
):
    gpu_only = accept_len is not None
    bs = candidates.shape[0] if gpu_only else len(lengths)
    rows = bs * 8 if gpu_only else sum(lengths)
    if (
        not bs
        or candidates.shape != (bs, 8)
        or locations.shape != (bs, 8)
        or hc.shape[:2] != (bs, 8)
        or bonus.numel() != bs
        or req_indices.numel() != bs
        or (
            not gpu_only
            and (len(prefixes) != bs or any(n < 1 or n > 8 for n in lengths))
        )
        or bs * 8 > buffers.input_ids.numel()
    ):
        raise ValueError("Invalid direct QSA commit pack")
    if (
        not gpu_only
        and mapping is not None
        and any(p < 0 or p + n > mapping.shape[1] for p, n in zip(prefixes, lengths))
    ):
        raise ValueError("Private QSA MTP append exceeds context capacity")
    if gpu_only:
        if (
            not hc.is_cuda
            or prefixes is not None
            or lengths is not None
            or new_lengths is None
            or bs > 32
            or any(
                t.shape != (bs,)
                or t.device != hc.device
                or t.dtype not in (torch.int32, torch.int64)
                for t in (accept_len, new_lengths)
            )
        ):
            raise ValueError(
                "GPU QSA commit requires aligned device acceptance metadata"
            )
        metadata = torch.empty(3 * bs + 1, device=hc.device, dtype=torch.int32)
        valid = torch.empty((), device=hc.device, dtype=torch.bool)
        _device_commit_metadata[(1,)](
            accept_len,
            new_lengths,
            metadata,
            valid,
            bs,
            accept_len.stride(0),
            new_lengths.stride(0),
            mapping.shape[1] if mapping is not None else 2147483647,
            triton.next_power_of_2(bs),
            num_warps=1,
        )
        torch._assert_async(valid, "Invalid QSA committed lengths/context capacity")
        prefix, lens, start = metadata[: 3 * bs].view(3, bs).unbind(0)
    else:
        starts = [0, *accumulate(lengths[:-1])]
        metadata = torch.tensor(
            [prefixes, lengths, starts], device=hc.device, dtype=torch.int32
        )
        prefix, lens, start = metadata.unbind(0)
    if hc.is_cuda:
        _pack_committed[(bs, triton.cdiv(hc.shape[-1], 256))](
            candidates,
            bonus,
            hc,
            locations,
            req_indices,
            metadata,
            mapping if mapping is not None else locations,
            buffers.input_ids,
            buffers.hidden_states,
            buffers.out_cache_loc,
            buffers.positions,
            buffers.mrope_positions,
            buffers.totals,
            buffers.correct,
            buffers.last,
            bs,
            metadata[3 * bs :] if gpu_only else rows,
            buffers.input_ids.numel(),
            hc.shape[-1],
            *candidates.stride(),
            bonus.stride(0),
            *hc.stride(),
            *locations.stride(),
            req_indices.stride(0),
            mapping.stride(0) if mapping is not None else 0,
            mapping.stride(1) if mapping is not None else 0,
            mapping is not None,
            256,
            gpu_only,
        )
    else:
        from sglang.srt.speculative.dmtp_qsa_contract import committed_inputs

        valid = torch.arange(8)[None] < lens[:, None]
        tokens = committed_inputs(candidates, lens, bonus)[valid]
        _, _, totals, pos, loc, _, mrope, correct, last = prepare_append(
            req_indices, prefixes, lengths, locations[valid], mapping
        )
        stage_graph_inputs(
            SimpleNamespace(
                input_ids=tokens,
                positions=pos,
                out_cache_loc=loc,
                mrope_positions=mrope,
                spec_info=SimpleNamespace(hidden_states=hc[valid]),
            ),
            buffers,
            bs * 8,
        )
        buffers.totals[:bs].copy_(totals)
        buffers.correct[:bs].copy_(correct)
        buffers.last[:bs].copy_(last)
    return SimpleNamespace(
        buffers=buffers,
        rows=rows,
        gpu_only=gpu_only,
        host_lengths=None if gpu_only else list(lengths),
        host_prefixes=None if gpu_only else list(prefixes),
        lens=lens,
        prefix=prefix,
        start=start,
        totals=buffers.totals[:bs],
        correct=buffers.correct[:bs],
        select=buffers.last[:bs],
        tokens=buffers.input_ids[:rows],
        hc=buffers.hidden_states[:rows],
        locations=buffers.out_cache_loc[:rows],
        positions=buffers.positions[:rows],
        mrope=buffers.mrope_positions[:, :rows],
    )
