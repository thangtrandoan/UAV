from __future__ import annotations

import random
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_for_speed() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def choose_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate_map_cmc(
    query_feats: torch.Tensor,
    query_ids: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_ids: torch.Tensor,
    topk: Tuple[int, int] = (1, 5),
    q_chunk_size: int = 2048,
    use_fp16_sim: bool = True,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    if q_chunk_size < 1:
        raise ValueError("q_chunk_size must be >= 1")

    if query_feats.is_cuda:
        device = query_feats.device
    elif gallery_feats.is_cuda:
        device = gallery_feats.device
    else:
        device = query_feats.device

    q = F.normalize(query_feats.to(device=device, dtype=torch.float32, non_blocking=True), dim=1)
    g = F.normalize(gallery_feats.to(device=device, dtype=torch.float32, non_blocking=True), dim=1)
    query_ids = query_ids.to(device=device, non_blocking=True)
    gallery_ids = gallery_ids.to(device=device, non_blocking=True)

    num_q = q.size(0)
    num_g = g.size(0)
    max_k = max(topk)
    cmc = torch.zeros(max_k, device=device, dtype=torch.float64)
    ap_sum = 0.0
    ap_count = 0

    sim_dtype = None
    if device.type == "cuda" and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    needs_float_sim = sim_dtype is not None
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    positions = torch.arange(1, num_g + 1, device=device, dtype=torch.float32).unsqueeze(0)

    for start in range(0, num_q, q_chunk_size):
        end = min(start + q_chunk_size, num_q)
        sim = q_mm[start:end] @ g_mm.t()
        if needs_float_sim:
            sim = sim.float()

        order = torch.argsort(sim, dim=1, descending=True)
        del sim

        matches = (gallery_ids[order] == query_ids[start:end].unsqueeze(1)).int()
        has_match = matches.any(dim=1)
        if has_match.any():
            first_match = torch.argmax(matches, dim=1)[has_match]
            first_match_topk = first_match[first_match < max_k]
            if first_match_topk.numel() > 0:
                counts = torch.bincount(first_match_topk, minlength=max_k).to(cmc.dtype)
                cmc += torch.cumsum(counts, dim=0)

            valid_matches = matches[has_match].to(torch.float32)
            precision = torch.cumsum(valid_matches, dim=1) / positions
            ap = (precision * valid_matches).sum(dim=1) / valid_matches.sum(dim=1)
            ap_sum += ap.sum().item()
            ap_count += ap.numel()

        if verbose:
            print(f"[eval] processed queries: {end}/{num_q}")

    cmc = cmc / num_q
    mAP = (ap_sum / ap_count) if ap_count > 0 else 0.0
    r1 = float(cmc[topk[0] - 1].item())
    r5 = float(cmc[topk[1] - 1].item()) if topk[1] <= max_k else 0.0
    return mAP, r1, r5


@torch.no_grad()
def self_check_evaluate_map_cmc() -> Tuple[float, float, float]:
    """
    Small synthetic self-check for chunked mAP/CMC implementation.
    Returns absolute differences (mAP, Rank-1, Rank-5) against a reference implementation.
    """

    def _reference_eval(
        qf: torch.Tensor,
        qid: torch.Tensor,
        gf: torch.Tensor,
        gid: torch.Tensor,
        tk: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        q_ref = F.normalize(qf, dim=1)
        g_ref = F.normalize(gf, dim=1)
        sim_ref = q_ref @ g_ref.t()
        num_q_ref = q_ref.size(0)
        max_k_ref = max(tk)
        cmc_ref = torch.zeros(max_k_ref)
        ap_list_ref = []

        for i in range(num_q_ref):
            order_ref = torch.argsort(sim_ref[i], descending=True)
            matches_ref = (gid[order_ref] == qid[i]).int()
            if matches_ref.sum() == 0:
                continue
            first_ref = torch.where(matches_ref == 1)[0][0].item()
            if first_ref < max_k_ref:
                cmc_ref[first_ref:] += 1
            idxs_ref = torch.where(matches_ref == 1)[0]
            prec_ref = torch.cumsum(matches_ref, 0)[idxs_ref] / (idxs_ref + 1)
            ap_list_ref.append(prec_ref.mean().item())

        cmc_ref = cmc_ref / num_q_ref
        m_ap_ref = float(np.mean(ap_list_ref)) if ap_list_ref else 0.0
        r1_ref = float(cmc_ref[tk[0] - 1].item())
        r5_ref = float(cmc_ref[tk[1] - 1].item()) if tk[1] <= max_k_ref else 0.0
        return m_ap_ref, r1_ref, r5_ref

    torch.manual_seed(42)
    q_feat = torch.randn(17, 32)
    g_feat = torch.randn(23, 32)
    q_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 9, 9, 7, 8, 8, 11, 12, 13, 100], dtype=torch.long)
    g_ids = torch.tensor([9, 1, 7, 0, 4, 2, 6, 1, 3, 5, 0, 8, 10, 11, 8, 12, 13, 14, 15, 16, 4, 5, 7], dtype=torch.long)

    baseline = _reference_eval(q_feat, q_ids, g_feat, g_ids, (1, 5))
    chunked = evaluate_map_cmc(
        q_feat,
        q_ids,
        g_feat,
        g_ids,
        topk=(1, 5),
        q_chunk_size=4,
        use_fp16_sim=False,
        verbose=False,
    )

    return tuple(abs(a - b) for a, b in zip(baseline, chunked))
