from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
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

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            _, (bn_global, bn_fs) = model(imgs)
            emb = torch.cat([bn_global, bn_fs], dim=1)

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
