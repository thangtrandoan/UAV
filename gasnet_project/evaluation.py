from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Sample, VRUDataset, build_query_gallery, read_split_file
from utils import evaluate_map_cmc

# VRU benchmark subsets.
SPLIT_FILE_MAP: Dict[str, str] = {
    "Small": "test_list_1200.txt",
    "Medium": "test_list_2400.txt",
    "Big": "test_list_8000.txt",
}


def _make_eval_loader(
    samples: List[Sample],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    ds = VRUDataset(samples=samples, transform=transform, relabel=False, max_retries=3)
    loader_kwargs = dict(
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **loader_kwargs,
    )


@torch.no_grad()
def _query_expansion_rerank_features(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    k1: int,
    alpha: float,
    q_chunk_size: int,
    use_fp16_sim: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k1 < 1:
        return q_feat, g_feat

    device = q_feat.device if q_feat.is_cuda else g_feat.device
    q = F.normalize(q_feat.to(device=device, dtype=torch.float32), dim=1)
    g = F.normalize(g_feat.to(device=device, dtype=torch.float32), dim=1)
    k = min(k1, g.size(0))

    sim_dtype = None
    if device.type == "cuda" and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    q_new_chunks = []
    for start in range(0, q.size(0), q_chunk_size):
        end = min(start + q_chunk_size, q.size(0))
        sim = q_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False).indices
        q_new_chunks.append(F.normalize(q[start:end] + alpha * g[idx].mean(dim=1), dim=1))

    g_new_chunks = []
    for start in range(0, g.size(0), q_chunk_size):
        end = min(start + q_chunk_size, g.size(0))
        sim = g_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False).indices
        g_new_chunks.append(F.normalize(g[start:end] + alpha * g[idx].mean(dim=1), dim=1))

    return torch.cat(q_new_chunks, dim=0), torch.cat(g_new_chunks, dim=0)


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
    output_device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    ids = []
    out_device = torch.device(output_device)
    if out_device.type == "cuda" and not torch.cuda.is_available():
        out_device = torch.device("cpu")

    for imgs, _, vehicle_ids, _ in loader:
        if use_channels_last:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)

        autocast_kwargs = dict(device_type=device.type, enabled=(use_amp and device.type == "cuda"))
        if device.type == "cuda":
            autocast_kwargs["dtype"] = amp_dtype

        with torch.autocast(**autocast_kwargs):
            outputs = model(imgs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            emb = torch.cat(list(outputs), dim=1)

        feats.append(emb.float().to(out_device, non_blocking=(out_device.type == "cuda")))
        ids.append(vehicle_ids.to(out_device, non_blocking=(out_device.type == "cuda")))

    return torch.cat(feats, dim=0), torch.cat(ids, dim=0)


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    data_root: Path,
    test_transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
    q_chunk_size: int = 2048,
    use_fp16_sim: bool = True,
    verbose_eval: bool = False,
    rerank: bool = False,
    rerank_k1: int = 20,
    rerank_alpha: float = 0.3,
    tta_flip: bool = False,
) -> Dict[str, Tuple[float, float, float]]:
    vru_dir = data_root / "VRU"
    pic_dir = vru_dir / "Pic"
    split_dir = vru_dir / "train_test_split"

    metrics_by_split: Dict[str, Tuple[float, float, float]] = {}

    was_training = model.training
    model.eval()

    for split_name, split_file in SPLIT_FILE_MAP.items():
        test_samples = read_split_file(pic_dir, split_dir / split_file)
        query_samples, gallery_samples = build_query_gallery(test_samples)

        q_loader = _make_eval_loader(
            query_samples,
            test_transform,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )
        g_loader = _make_eval_loader(
            gallery_samples,
            test_transform,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            prefetch_factor,
        )

        feat_out_device = device
        q_feat, q_ids = _extract_features(
            model,
            q_loader,
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            output_device=feat_out_device,
        )
        g_feat, g_ids = _extract_features(
            model,
            g_loader,
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            output_device=feat_out_device,
        )
        if tta_flip:
            flip_transform = test_transform
            try:
                from torchvision import transforms

                flip_transform = transforms.Compose([
                    test_transform,
                    transforms.RandomHorizontalFlip(p=1.0),
                ])
            except Exception:
                flip_transform = test_transform
            q_flip_loader = _make_eval_loader(
                query_samples,
                flip_transform,
                batch_size,
                num_workers,
                pin_memory,
                persistent_workers,
                prefetch_factor,
            )
            g_flip_loader = _make_eval_loader(
                gallery_samples,
                flip_transform,
                batch_size,
                num_workers,
                pin_memory,
                persistent_workers,
                prefetch_factor,
            )
            q_flip_feat, _ = _extract_features(
                model,
                q_flip_loader,
                device,
                use_amp,
                amp_dtype,
                use_channels_last,
                output_device=feat_out_device,
            )
            g_flip_feat, _ = _extract_features(
                model,
                g_flip_loader,
                device,
                use_amp,
                amp_dtype,
                use_channels_last,
                output_device=feat_out_device,
            )
            q_feat = F.normalize(F.normalize(q_feat, dim=1) + F.normalize(q_flip_feat, dim=1), dim=1)
            g_feat = F.normalize(F.normalize(g_feat, dim=1) + F.normalize(g_flip_feat, dim=1), dim=1)
        if rerank:
            q_feat, g_feat = _query_expansion_rerank_features(
                q_feat=q_feat,
                g_feat=g_feat,
                k1=rerank_k1,
                alpha=rerank_alpha,
                q_chunk_size=q_chunk_size,
                use_fp16_sim=use_fp16_sim,
            )

        m_ap, rank1, rank5 = evaluate_map_cmc(
            q_feat,
            q_ids,
            g_feat,
            g_ids,
            topk=(1, 5),
            q_chunk_size=q_chunk_size,
            use_fp16_sim=use_fp16_sim,
            verbose=(verbose_eval and split_name == "Big"),
        )
        metrics_by_split[split_name] = (m_ap, rank1, rank5)

    if was_training:
        model.train()

    return metrics_by_split


def print_eval_report(metrics_by_split: Dict[str, Tuple[float, float, float]], title: str = "Evaluation") -> None:
    print(f"\n=== {title} ===")
    print(f"{'Split':<10} {'mAP':>10} {'Rank-1':>10} {'Rank-5':>10}")
    for split_name in ("Small", "Medium", "Big"):
        m_ap, rank1, rank5 = metrics_by_split[split_name]
        print(f"{split_name:<10} {m_ap:>10.4f} {rank1:>10.4f} {rank5:>10.4f}")
