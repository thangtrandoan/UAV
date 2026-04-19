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
) -> Tuple[float, float, float]:
    q = F.normalize(query_feats, dim=1)
    g = F.normalize(gallery_feats, dim=1)
    sim = q @ g.t()

    num_q = q.size(0)
    max_k = max(topk)
    cmc = torch.zeros(max_k)
    ap_list = []

    for i in range(num_q):
        order = torch.argsort(sim[i], descending=True)
        matches = (gallery_ids[order] == query_ids[i]).int()

        if matches.sum() == 0:
            continue

        first = torch.where(matches == 1)[0][0].item()
        if first < max_k:
            cmc[first:] += 1

        idxs = torch.where(matches == 1)[0]
        prec = torch.cumsum(matches, 0)[idxs] / (idxs + 1)
        ap_list.append(prec.mean().item())

    cmc = cmc / num_q
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    r1 = float(cmc[topk[0] - 1])
    r5 = float(cmc[topk[1] - 1]) if max_k >= 5 else 0.0
    return mAP, r1, r5
