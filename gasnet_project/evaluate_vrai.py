from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
from train import GASNet, Logger
from utils import choose_amp_dtype, configure_cuda_for_speed, evaluate_map_cmc


COLOR_NAMES = {
    0: "White",
    1: "Black",
    2: "Gray",
    3: "Red",
    4: "Green",
    5: "Blue",
    6: "Yellow",
    7: "Brown",
    8: "Others",
}
TYPE_NAMES = {
    0: "Sedan",
    1: "Hatchback",
    2: "SUV",
    3: "Bus",
    4: "Lorry",
    5: "Truck",
    6: "Others",
}
ATTRIBUTE_KEYS = ("color_label", "type_label", "bumper_label", "wheel_label", "sky_label", "luggage_label")


class VRAIDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform=None, max_retries: int = 3, use_cv2: bool = False):
        self.image_paths = image_paths
        self.transform = transform
        self.max_retries = max_retries
        self.use_cv2 = use_cv2

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: Path) -> Image.Image:
        last_err = None
        for _ in range(self.max_retries):
            try:
                if self.use_cv2:
                    import cv2
                    img = cv2.imread(str(path))
                    if img is None:
                        raise OSError(f"Cannot read image with cv2: {path}")
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(img_rgb)
                else:
                    with Image.open(path) as img:
                        return img.convert("RGB")
            except OSError as exc:
                last_err = exc
        raise last_err

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = self._load_rgb(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


def _normalize_state_dict(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    out = state_dict
    if all(k.startswith("module.") for k in out.keys()):
        out = {k[len("module.") :]: v for k, v in out.items()}
    if all(k.startswith("_orig_mod.") for k in out.keys()):
        out = {k[len("_orig_mod.") :]: v for k, v in out.items()}
    return out


def _infer_num_classes(state_dict: dict) -> int:
    for key in ("classifier_global.weight", "classifier_fs.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError("Could not infer num_classes from checkpoint (missing classifier weights)")


def _load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def _load_annotation(annotation_path: Path) -> dict:
    with annotation_path.open("rb") as f:
        return pickle.load(f)


def _load_id_map(path: Path) -> Dict[str, int] | List[int]:
    if path.suffix.lower() in {".json", ".js"}:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_train_id(image_name: str) -> str:
    stem = Path(image_name).stem
    if "_" not in stem:
        raise ValueError(f"Unexpected train image name format: {image_name}")
    return stem.split("_", 1)[0]


def _parse_train_camera(image_name: str) -> str:
    parts = Path(image_name).stem.split("_")
    if len(parts) < 2:
        return ""
    return parts[1]


def _parse_camera(image_name: str, split: str) -> str:
    if split == "train":
        return _parse_train_camera(image_name)
    stem = Path(image_name).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].upper().startswith("C"):
        return parts[1]
    return ""


def _parse_train_frame(image_name: str) -> str:
    parts = Path(image_name).stem.split("_")
    if len(parts) < 3:
        return ""
    return parts[2]


def _identity_key(image_name: str, split: str) -> int | str:
    if split == "train":
        return int(_parse_train_id(image_name))
    return image_name


def _label_name(key: str, value) -> str | int | None:
    if value is None:
        return None
    value = int(value)
    if key == "color_label":
        return COLOR_NAMES.get(value, value)
    if key == "type_label":
        return TYPE_NAMES.get(value, value)
    return value


def _get_attrs(annotation: dict, image_name: str, split: str) -> dict:
    identity_key = _identity_key(image_name, split)
    attrs = {}
    for key in ATTRIBUTE_KEYS:
        labels = annotation.get(key, {})
        value = None
        if isinstance(labels, dict):
            if identity_key in labels:
                value = labels[identity_key]
            elif str(identity_key) in labels:
                value = labels[str(identity_key)]
            elif image_name in labels:
                value = labels[image_name]
        attrs[key] = _label_name(key, value)
    return attrs


def _attr_mismatches(a: dict, b: dict) -> List[str]:
    mismatches = []
    for key in ATTRIBUTE_KEYS:
        if a.get(key) is not None and b.get(key) is not None and a[key] != b[key]:
            mismatches.append(key)
    return mismatches


@torch.no_grad()
def _query_expansion_rerank_features(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    k1: int,
    alpha: float,
    q_chunk_size: int,
    use_fp16_sim: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lightweight re-ranking by query expansion over gallery neighbors.

    This avoids materializing the full query+gallery distance matrix, which is
    too large for VRAI train evaluation, while still using local neighborhood
    context before mAP/CMC and top-k diagnostics are computed.
    """
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
        neighbor = g[idx].mean(dim=1)
        q_new_chunks.append(F.normalize(q[start:end] + alpha * neighbor, dim=1))
    q_new = torch.cat(q_new_chunks, dim=0)

    g_new_chunks = []
    for start in range(0, g.size(0), q_chunk_size):
        end = min(start + q_chunk_size, g.size(0))
        sim = g_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False).indices
        neighbor = g[idx].mean(dim=1)
        g_new_chunks.append(F.normalize(g[start:end] + alpha * neighbor, dim=1))
    g_new = torch.cat(g_new_chunks, dim=0)
    return q_new, g_new


@torch.no_grad()
def _collect_retrieval_diagnostics(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    query_ids: List[int],
    gallery_ids: List[int],
    query_idx: List[int],
    gallery_idx: List[int],
    image_names: List[str],
    annotation: dict,
    split: str,
    topk: int,
    q_chunk_size: int,
    use_fp16_sim: bool,
) -> dict:
    q = F.normalize(q_feat.float(), dim=1)
    g = F.normalize(g_feat.float(), dim=1)
    q_ids = torch.tensor(query_ids, dtype=torch.long, device=q.device)
    g_ids = torch.tensor(gallery_ids, dtype=torch.long, device=q.device)
    k = min(max(10, topk), g.size(0))

    sim_dtype = None
    if q.is_cuda and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    summary = {
        "num_queries": len(query_ids),
        "num_rank1_failures": 0,
        "no_match_in_topk": 0,
        "gt_rank_2_10": 0,
        "gt10": 0,
        "type_mismatch": 0,
        "color_mismatch": 0,
        "cross_camera_failures": 0,
        "diagnostic_topk": k,
    }

    for start in range(0, q_mm.size(0), q_chunk_size):
        end = min(start + q_chunk_size, q_mm.size(0))
        sim = q_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        order = torch.topk(sim, k=k, dim=1, largest=True, sorted=True).indices
        matches = g_ids[order] == q_ids[start:end].unsqueeze(1)

        for local_row in range(order.size(0)):
            q_local = start + local_row
            match_positions = torch.where(matches[local_row])[0]
            first_correct_rank = int(match_positions[0].item() + 1) if match_positions.numel() else None
            if first_correct_rank is None:
                summary["no_match_in_topk"] += 1
            elif 2 <= first_correct_rank <= 10:
                summary["gt_rank_2_10"] += 1
            elif first_correct_rank > 10:
                summary["gt10"] += 1

            pred_gallery_local = int(order[local_row, 0].item())
            q_id = int(query_ids[q_local])
            pred_id = int(gallery_ids[pred_gallery_local])
            if pred_id == q_id:
                continue

            summary["num_rank1_failures"] += 1
            q_name = image_names[query_idx[q_local]]
            pred_name = image_names[gallery_idx[pred_gallery_local]]
            q_attrs = _get_attrs(annotation, q_name, split)
            pred_attrs = _get_attrs(annotation, pred_name, split)
            mismatches = set(_attr_mismatches(q_attrs, pred_attrs))
            if "type_label" in mismatches:
                summary["type_mismatch"] += 1
            if "color_label" in mismatches:
                summary["color_mismatch"] += 1
            if _parse_camera(q_name, split) != _parse_camera(pred_name, split):
                summary["cross_camera_failures"] += 1

    return summary


def _print_diagnostic_report(summary: dict) -> None:
    total = max(1, int(summary["num_queries"]))
    failures = max(1, int(summary["num_rank1_failures"]))
    topk = int(summary["diagnostic_topk"])
    print("\n=== VRAI Retrieval Diagnostics ===")
    print(f"Rank-1 failures:          {summary['num_rank1_failures']}/{summary['num_queries']}")
    print(f"no_match_in_top{topk}:       {summary['no_match_in_topk'] / total:.4f}")
    print(f"gt rank 2-10 rate:        {summary['gt_rank_2_10'] / total:.4f}")
    print(f"type mismatch rate:       {summary['type_mismatch'] / failures:.4f}")
    print(f"color mismatch rate:      {summary['color_mismatch'] / failures:.4f}")
    print(f"cross-camera failure rate:{summary['cross_camera_failures'] / failures:.4f}")


def _human_obvious_hint(mismatches: List[str], same_camera: bool, q_parts: int, pred_parts: int) -> str:
    semantic = {"color_label", "type_label"}
    if semantic.intersection(mismatches):
        return "likely_obvious_color_or_type_mismatch"
    if len(mismatches) >= 3:
        return "likely_obvious_many_attribute_mismatches"
    if not mismatches and same_camera:
        return "likely_hard_same_camera_similar_attributes"
    if q_parts > 0 and pred_parts > 0:
        return "needs_visual_check_discriminative_parts_available"
    return "ambiguous_needs_visual_check"


def _build_case_from_order(
    q_local: int,
    order_row: torch.Tensor,
    score_row: torch.Tensor,
    query_ids: List[int],
    gallery_ids: List[int],
    query_idx: List[int],
    gallery_idx: List[int],
    image_names: List[str],
    annotation: dict,
    split: str,
) -> dict:
    q_name = image_names[query_idx[q_local]]
    q_id = int(query_ids[q_local])
    top_matches = []
    first_correct_rank = None

    for rank, (g_local_t, score_t) in enumerate(zip(order_row.tolist(), score_row.tolist()), 1):
        g_local = int(g_local_t)
        g_name = image_names[gallery_idx[g_local]]
        g_id = int(gallery_ids[g_local])
        is_correct = g_id == q_id
        if is_correct and first_correct_rank is None:
            first_correct_rank = rank
        top_matches.append(
            {
                "rank": rank,
                "gallery_index": int(gallery_idx[g_local]),
                "name": g_name,
                "id": g_id,
                "camera": _parse_camera(g_name, split),
                "score": float(score_t),
                "is_correct": is_correct,
            }
        )

    pred_gallery_local = int(order_row[0].item())
    pred_name = image_names[gallery_idx[pred_gallery_local]]
    pred_id = int(gallery_ids[pred_gallery_local])
    q_attrs = _get_attrs(annotation, q_name, split)
    pred_attrs = _get_attrs(annotation, pred_name, split)
    mismatches = _attr_mismatches(q_attrs, pred_attrs) if pred_id != q_id else []
    same_camera = _parse_camera(q_name, split) == _parse_camera(pred_name, split)
    q_parts = len(annotation.get("d_part_label", {}).get(q_name, []))
    pred_parts = len(annotation.get("d_part_label", {}).get(pred_name, []))
    human_hint = "rank1_correct" if pred_id == q_id else _human_obvious_hint(mismatches, same_camera, q_parts, pred_parts)

    return {
        "query_local_index": q_local,
        "query_index": int(query_idx[q_local]),
        "query_name": q_name,
        "query_id": q_id,
        "query_camera": _parse_camera(q_name, split),
        "query_frame": _parse_train_frame(q_name) if split == "train" else "",
        "query_attrs": q_attrs,
        "query_discriminative_parts": q_parts,
        "rank1_gallery_index": int(gallery_idx[pred_gallery_local]),
        "rank1_name": pred_name,
        "rank1_id": pred_id,
        "rank1_camera": _parse_camera(pred_name, split),
        "rank1_frame": _parse_train_frame(pred_name) if split == "train" else "",
        "rank1_attrs": pred_attrs,
        "rank1_discriminative_parts": pred_parts,
        "rank1_is_correct": pred_id == q_id,
        "first_correct_rank": first_correct_rank,
        "attribute_mismatches": mismatches,
        "same_camera_as_rank1": same_camera,
        "human_obvious_hint": human_hint,
        "top_matches": top_matches,
    }


@torch.no_grad()
def _collect_selected_query_cases(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    query_indices: List[int],
    query_ids: List[int],
    gallery_ids: List[int],
    query_idx: List[int],
    gallery_idx: List[int],
    image_names: List[str],
    annotation: dict,
    split: str,
    topk: int,
    use_fp16_sim: bool,
) -> List[dict]:
    q = F.normalize(q_feat.float(), dim=1)
    g = F.normalize(g_feat.float(), dim=1)
    index_to_local = {int(global_idx): local for local, global_idx in enumerate(query_idx)}
    selected_local = []
    for query_index in query_indices:
        if int(query_index) not in index_to_local:
            print(f"Warning: query_index {query_index} is not in the current query split; skipping")
            continue
        selected_local.append(index_to_local[int(query_index)])
    if not selected_local:
        return []

    k = min(topk, g.size(0))
    sim_dtype = None
    if q.is_cuda and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    cases = []
    for q_local in selected_local:
        q_row = q[q_local : q_local + 1]
        q_mm = q_row.to(sim_dtype) if sim_dtype is not None else q_row
        sim = q_mm @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        scores, order = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
        cases.append(
            _build_case_from_order(
                q_local=q_local,
                order_row=order[0],
                score_row=scores[0],
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_idx=query_idx,
                gallery_idx=gallery_idx,
                image_names=image_names,
                annotation=annotation,
                split=split,
            )
        )
    return cases


def _build_train_query_gallery(
    image_names: List[str],
    random_seed: int = 42,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    by_id: Dict[str, List[int]] = {}
    for idx, name in enumerate(image_names):
        id_str = _parse_train_id(name)
        by_id.setdefault(id_str, []).append(idx)

    query_idx: List[int] = []
    gallery_idx: List[int] = []
    id_map: Dict[str, int] = {}
    next_id = 0

    for id_str, idxs in by_id.items():
        if len(idxs) < 2:
            continue
        id_map[id_str] = next_id
        next_id += 1
        idxs_sorted = sorted(idxs, key=lambda i: image_names[i])
        rng = random.Random(f"{random_seed}:{id_str}")
        query_choice = rng.choice(idxs_sorted)
        query_idx.append(query_choice)
        gallery_idx.extend(i for i in idxs_sorted if i != query_choice)

    if not query_idx or not gallery_idx:
        raise ValueError("Not enough images per identity to build query/gallery for train split")

    query_ids = [id_map[_parse_train_id(image_names[i])] for i in query_idx]
    gallery_ids = [id_map[_parse_train_id(image_names[i])] for i in gallery_idx]
    return query_idx, gallery_idx, query_ids, gallery_ids


def _build_train_query_gallery_centroid(
    image_names: List[str],
    features: torch.Tensor,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    by_id: Dict[str, List[int]] = {}
    for idx, name in enumerate(image_names):
        id_str = _parse_train_id(name)
        by_id.setdefault(id_str, []).append(idx)

    query_idx: List[int] = []
    gallery_idx: List[int] = []
    id_map: Dict[str, int] = {}
    next_id = 0

    feat_norm = F.normalize(features.float(), dim=1)

    for id_str, idxs in by_id.items():
        if len(idxs) < 2:
            continue
        id_map[id_str] = next_id
        next_id += 1

        idxs_tensor = torch.tensor(idxs, dtype=torch.long, device=feat_norm.device)
        id_feats = feat_norm[idxs_tensor]  # [N, D]
        centroid = id_feats.mean(dim=0, keepdim=True)  # [1, D]
        centroid = F.normalize(centroid, dim=1)

        sims = (id_feats @ centroid.t()).squeeze(1)  # [N]
        max_idx = torch.argmax(sims).item()

        query_choice = idxs[max_idx]
        query_idx.append(query_choice)
        gallery_idx.extend(i for i in idxs if i != query_choice)

    if not query_idx or not gallery_idx:
        raise ValueError('Not enough images per identity to build query/gallery for train split')

    query_ids = [id_map[_parse_train_id(image_names[i])] for i in query_idx]
    gallery_ids = [id_map[_parse_train_id(image_names[i])] for i in gallery_idx]
    return query_idx, gallery_idx, query_ids, gallery_ids



def _make_loader(
    image_paths: List[Path],
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    use_cv2: bool = False,
) -> DataLoader:
    ds = VRAIDataset(image_paths=image_paths, transform=transform, max_retries=3, use_cv2=use_cv2)
    loader_kwargs = dict(pin_memory=pin_memory, drop_last=False)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, **loader_kwargs)


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
    log_every: int,
) -> torch.Tensor:
    feats = []
    total = len(loader)
    for step, imgs in enumerate(loader, 1):
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

        feats.append(emb.float())
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == total):
            print(f"  [feat] batch {step}/{total}")

    return torch.cat(feats, dim=0)


def _infer_eval_lists(
    split: str,
    annotation: dict,
    id_map_path: Path | None,
    train_random_seed: int = 42,
) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    if split == "train":
        image_names = annotation.get("train_im_names")
        if image_names is None:
            raise KeyError("train_im_names not found in train_annotation.pkl")
        query_idx, gallery_idx, query_ids, gallery_ids = _build_train_query_gallery(
            image_names,
            random_seed=train_random_seed,
        )
        return image_names, query_idx, gallery_idx, query_ids, gallery_ids

    image_names = annotation.get("test_im_names") or annotation.get("dev_im_names")
    if image_names is None:
        raise KeyError("Annotation file missing test_im_names or dev_im_names")
    query_idx = annotation["query_order"]
    gallery_idx = annotation["gallery_order"]
    if query_idx and isinstance(query_idx[0], str):
        index_map = {name: i for i, name in enumerate(image_names)}
        query_idx = [index_map[name] for name in query_idx]
        gallery_idx = [index_map[name] for name in gallery_idx]

    if id_map_path is None:
        raise ValueError(
            "No identity labels available for test/test-dev. Provide --id-map or use --split train."
        )
    id_map = _load_id_map(id_map_path)
    if isinstance(id_map, list):
        if len(id_map) != len(image_names):
            raise ValueError("id-map list length does not match number of images")
        query_ids = [int(id_map[i]) for i in query_idx]
        gallery_ids = [int(id_map[i]) for i in gallery_idx]
    else:
        query_ids = [int(id_map[image_names[i]]) for i in query_idx]
        gallery_ids = [int(id_map[image_names[i]]) for i in gallery_idx]

    return image_names, query_idx, gallery_idx, query_ids, gallery_ids


@torch.no_grad()
def _build_reid_result(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    topk: int,
    q_chunk_size: int,
    use_fp16_sim: bool,
) -> List[dict]:
    q = F.normalize(q_feat, dim=1)
    g = F.normalize(g_feat, dim=1)
    num_g = g.size(0)
    k = min(topk, num_g)

    sim_dtype = None
    if q.is_cuda and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    results: List[dict] = []
    for start in range(0, q_mm.size(0), q_chunk_size):
        end = min(start + q_chunk_size, q_mm.size(0))
        sim = q_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        topk_idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True).indices
        for row in topk_idx.tolist():
            results.append({"query_id": len(results), "ans_ids": row})

    return results


@torch.no_grad()
def _collect_failure_cases(
    q_feat: torch.Tensor,
    g_feat: torch.Tensor,
    query_ids: List[int],
    gallery_ids: List[int],
    query_idx: List[int],
    gallery_idx: List[int],
    image_names: List[str],
    annotation: dict,
    split: str,
    topk: int,
    q_chunk_size: int,
    use_fp16_sim: bool,
) -> Tuple[List[dict], dict]:
    q = F.normalize(q_feat.float(), dim=1)
    g = F.normalize(g_feat.float(), dim=1)
    q_ids = torch.tensor(query_ids, dtype=torch.long, device=q.device)
    g_ids = torch.tensor(gallery_ids, dtype=torch.long, device=q.device)
    k = min(topk, g.size(0))

    sim_dtype = None
    if q.is_cuda and use_fp16_sim:
        sim_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q_mm = q.to(sim_dtype) if sim_dtype is not None else q
    g_mm = g.to(sim_dtype) if sim_dtype is not None else g

    failures: List[dict] = []
    summary = {
        "num_queries": len(query_ids),
        "num_gallery": len(gallery_ids),
        "num_rank1_failures": 0,
        "first_correct_rank_histogram": Counter(),
        "predicted_wrong_identity_histogram": Counter(),
        "attribute_mismatch_histogram": Counter(),
        "human_obvious_hint_histogram": Counter(),
        "same_camera_rank1_failures": 0,
    }

    for start in range(0, q_mm.size(0), q_chunk_size):
        end = min(start + q_chunk_size, q_mm.size(0))
        sim = q_mm[start:end] @ g_mm.t()
        if sim_dtype is not None:
            sim = sim.float()
        scores, order = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
        matches = g_ids[order] == q_ids[start:end].unsqueeze(1)

        for local_row in range(order.size(0)):
            q_local = start + local_row
            q_name = image_names[query_idx[q_local]]
            q_id = int(query_ids[q_local])
            match_positions = torch.where(matches[local_row])[0]
            first_correct_rank = int(match_positions[0].item() + 1) if match_positions.numel() else None
            if first_correct_rank is None:
                summary["first_correct_rank_histogram"]["no_match_in_topk"] += 1
            else:
                bucket = str(first_correct_rank if first_correct_rank <= 10 else "gt10")
                summary["first_correct_rank_histogram"][bucket] += 1

            pred_gallery_local = int(order[local_row, 0].item())
            pred_name = image_names[gallery_idx[pred_gallery_local]]
            pred_id = int(gallery_ids[pred_gallery_local])
            if pred_id == q_id:
                continue

            q_attrs = _get_attrs(annotation, q_name, split)
            pred_attrs = _get_attrs(annotation, pred_name, split)
            mismatches = _attr_mismatches(q_attrs, pred_attrs)
            same_camera = _parse_train_camera(q_name) == _parse_train_camera(pred_name) if split == "train" else False
            q_parts = len(annotation.get("d_part_label", {}).get(q_name, []))
            pred_parts = len(annotation.get("d_part_label", {}).get(pred_name, []))
            human_hint = _human_obvious_hint(mismatches, same_camera, q_parts, pred_parts)

            top_matches = []
            for rank, (g_local_t, score_t) in enumerate(zip(order[local_row].tolist(), scores[local_row].tolist()), 1):
                g_name = image_names[gallery_idx[int(g_local_t)]]
                g_id = int(gallery_ids[int(g_local_t)])
                top_matches.append(
                    {
                        "rank": rank,
                        "gallery_index": int(gallery_idx[int(g_local_t)]),
                        "name": g_name,
                        "id": g_id,
                        "camera": _parse_train_camera(g_name) if split == "train" else "",
                        "score": float(score_t),
                        "is_correct": g_id == q_id,
                    }
                )

            case = {
                "query_local_index": q_local,
                "query_index": int(query_idx[q_local]),
                "query_name": q_name,
                "query_id": q_id,
                "query_camera": _parse_train_camera(q_name) if split == "train" else "",
                "query_frame": _parse_train_frame(q_name) if split == "train" else "",
                "query_attrs": q_attrs,
                "query_discriminative_parts": q_parts,
                "rank1_gallery_index": int(gallery_idx[pred_gallery_local]),
                "rank1_name": pred_name,
                "rank1_id": pred_id,
                "rank1_camera": _parse_train_camera(pred_name) if split == "train" else "",
                "rank1_frame": _parse_train_frame(pred_name) if split == "train" else "",
                "rank1_attrs": pred_attrs,
                "rank1_discriminative_parts": pred_parts,
                "first_correct_rank": first_correct_rank,
                "attribute_mismatches": mismatches,
                "same_camera_as_rank1": same_camera,
                "human_obvious_hint": human_hint,
                "top_matches": top_matches,
            }
            failures.append(case)
            summary["num_rank1_failures"] += 1
            summary["predicted_wrong_identity_histogram"][str(pred_id)] += 1
            summary["human_obvious_hint_histogram"][human_hint] += 1
            if same_camera:
                summary["same_camera_rank1_failures"] += 1
            for key in mismatches:
                summary["attribute_mismatch_histogram"][key] += 1

    summary = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in summary.items()}
    failures.sort(key=lambda c: (c["first_correct_rank"] is not None, c["first_correct_rank"] or math.inf))
    return failures, summary


def _prepare_display_image(path: Path, size: int = 224, d_parts: list = None) -> Image.Image:
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        orig_w, orig_h = img_rgb.size
        img_resized = img_rgb.resize((256, 256), Image.BILINEAR)
        
        if d_parts:
            draw = ImageDraw.Draw(img_resized)
            for part in d_parts:
                if len(part) == 2 and len(part[0]) == 2 and len(part[1]) == 2:
                    (x1, y1), (x2, y2) = part
                    x1_s = int(x1 * 256 / orig_w)
                    y1_s = int(y1 * 256 / orig_h)
                    x2_s = int(x2 * 256 / orig_w)
                    y2_s = int(y2 * 256 / orig_h)
                    draw.rectangle([x1_s, y1_s, x2_s, y2_s], outline=(255, 255, 0), width=2)
                    
    left = (img_resized.width - size) // 2
    top = (img_resized.height - size) // 2
    return img_resized.crop((left, top, left + size, top + size)).convert("RGB")


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text(xy, text, fill=fill, font=font)


def _save_contact_sheet(case: dict, images_dir: Path, out_path: Path, topk: int, annotation: dict = None) -> None:
    tiles = []
    
    q_name = case["query_name"]
    q_parts = annotation.get("d_part_label", {}).get(q_name, []) if annotation else []
    q_img = _prepare_display_image(images_dir / q_name, d_parts=q_parts)
    q_draw = ImageDraw.Draw(q_img)
    _draw_label(q_draw, (6, 6), f"Q id={case['query_id']} cam={case['query_camera']}", (255, 255, 255))
    tiles.append(q_img)

    for match in case["top_matches"][:topk]:
        m_name = match["name"]
        m_parts = annotation.get("d_part_label", {}).get(m_name, []) if annotation else []
        img = _prepare_display_image(images_dir / m_name, d_parts=m_parts)
        draw = ImageDraw.Draw(img)
        fill = (80, 220, 120) if match["is_correct"] else (255, 90, 90)
        _draw_label(draw, (6, 6), f"R{match['rank']} id={match['id']} {match['score']:.3f}", fill)
        _draw_label(draw, (6, 24), f"cam={match['camera']}", fill)
        tiles.append(img)

    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _make_heat_color(attn: np.ndarray) -> np.ndarray:
    attn = np.asarray(attn)
    attn = np.squeeze(attn)
    if attn.ndim != 2:
        raise ValueError(f"Expected a 2D attention map, got shape {attn.shape}")
    attn = np.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
    attn = np.clip(attn, 0.0, 1.0)
    red = (255 * attn).astype(np.uint8)
    green = (255 * (1.0 - np.abs(attn - 0.5) * 2.0)).astype(np.uint8)
    blue = (255 * (1.0 - attn)).astype(np.uint8)
    return np.ascontiguousarray(np.stack([red, green, blue], axis=-1))


@torch.no_grad()
def _save_attention_heatmaps(
    model: torch.nn.Module,
    image_path: Path,
    out_dir: Path,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
    layers: List[str],
) -> List[dict]:
    base = _prepare_display_image(image_path)
    tensor = transforms.ToTensor()(base)
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor).unsqueeze(0)
    if use_channels_last:
        tensor = tensor.to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        tensor = tensor.to(device, non_blocking=True)

    captured: Dict[str, torch.Tensor] = {}
    handles = []
    def _capture_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().float().cpu()
            return None
        return hook

    for layer in layers:
        module = getattr(model, layer).rga_s.sigmoid
        handles.append(module.register_forward_hook(_capture_hook(layer)))

    autocast_kwargs = dict(device_type=device.type, enabled=(use_amp and device.type == "cuda"))
    if device.type == "cuda":
        autocast_kwargs["dtype"] = amp_dtype
    with torch.autocast(**autocast_kwargs):
        model(tensor)
    for handle in handles:
        handle.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for layer, attn_tensor in captured.items():
        spatial_size = getattr(getattr(model, layer).rga_s, "spatial_size", None)
        attn = np.squeeze(attn_tensor[0].numpy())
        if attn.ndim == 1 and spatial_size is not None:
            expected_nodes = int(spatial_size[0] * spatial_size[1])
            if attn.size == expected_nodes:
                attn = attn.reshape(spatial_size)
        elif attn.ndim == 2 and 1 in attn.shape and spatial_size is not None:
            flat = attn.reshape(-1)
            expected_nodes = int(spatial_size[0] * spatial_size[1])
            if flat.size == expected_nodes:
                attn = flat.reshape(spatial_size)
        elif attn.ndim == 3:
            attn = attn[0]
        if attn.ndim != 2:
            raise ValueError(f"Unexpected attention shape for {layer}: {tuple(attn_tensor.shape)}")
        attn = np.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
        low, high = np.percentile(attn, [1, 99])
        attn_norm = (attn - low) / max(high - low, 1e-6)
        base_rgb = base.convert("RGB")
        heat_array = _make_heat_color(attn_norm)
        heat = Image.fromarray(heat_array)
        heat = heat.resize(base_rgb.size, Image.BILINEAR).convert("RGB")
        overlay = Image.blend(base_rgb, heat, alpha=0.45)

        y_idx, x_idx = np.indices(attn.shape)
        weight = np.maximum(attn - attn.min(), 0.0) + 1e-8
        cx = float((x_idx * weight).sum() / weight.sum() / max(attn.shape[1] - 1, 1))
        cy = float((y_idx * weight).sum() / weight.sum() / max(attn.shape[0] - 1, 1))
        stats.append(
            {
                "layer": layer,
                "mean": float(attn.mean()),
                "max": float(attn.max()),
                "min": float(attn.min()),
                "center_of_mass_x": cx,
                "center_of_mass_y": cy,
            }
        )
        overlay.save(out_dir / f"{image_path.stem}_{layer}_spatial_attention.jpg")

    return stats


def _write_failure_analysis(
    failures: List[dict],
    summary: dict,
    image_names: List[str],
    images_dir: Path,
    annotation: dict,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
) -> None:
    out_dir = args.analysis_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = failures[: args.analysis_max_cases]
    with (out_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for case in selected:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary["written_failure_cases"] = len(selected)
    summary["heatmap_cases"] = min(args.heatmap_cases, len(selected))
    with (out_dir / "failure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # === NEW BLOCK: Lưu contact sheet top 5 của TOÀN BỘ các ca lỗi ===
    all_top5_dir = out_dir / "all_top5_failures"
    all_top5_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Đang lưu {len(failures)} ảnh top 5 của tất cả ca lỗi vào thư mục: {all_top5_dir} ...")
    for idx, case in enumerate(failures):
        # Đặt tên file có chứa ID và gợi ý lỗi để dễ dàng xem qua
        filename = f"case_{idx:04d}_q{case['query_id']}_hint_{case.get('human_obvious_hint', 'none')}.jpg"
        _save_contact_sheet(case, images_dir, all_top5_dir / filename, args.contact_sheet_topk, annotation=annotation)
    print("[INFO] Đã lưu xong tất cả ảnh top 5.")
    # =================================================================

    for idx, case in enumerate(selected[: args.heatmap_cases]):
        case_dir = out_dir / f"case_{idx:04d}_q{case['query_index']}_pred{case['rank1_gallery_index']}"
        _save_contact_sheet(case, images_dir, case_dir / "top_matches.jpg", args.contact_sheet_topk, annotation=annotation)
        heatmap_stats = {}
        q_stats = _save_attention_heatmaps(
            model,
            images_dir / case["query_name"],
            case_dir / "query_heatmaps",
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            args.heatmap_layers,
        )
        heatmap_stats["query"] = q_stats
        rank1_stats = _save_attention_heatmaps(
            model,
            images_dir / case["rank1_name"],
            case_dir / "rank1_heatmaps",
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            args.heatmap_layers,
        )
        heatmap_stats["rank1_wrong_match"] = rank1_stats

        if case["first_correct_rank"] is not None:
            correct = next((m for m in case["top_matches"] if m["is_correct"]), None)
            if correct is not None:
                heatmap_stats["first_correct_match"] = _save_attention_heatmaps(
                    model,
                    images_dir / correct["name"],
                    case_dir / "first_correct_heatmaps",
                    device,
                    use_amp,
                    amp_dtype,
                    use_channels_last,
                    args.heatmap_layers,
                )

        with (case_dir / "case.json").open("w", encoding="utf-8") as f:
            json.dump({"case": case, "heatmap_stats": heatmap_stats, "d_part_label": annotation.get("d_part_label", {}).get(case["query_name"], [])}, f, indent=2, ensure_ascii=False)

    print(f"Wrote failure analysis to {out_dir}")


def _write_selected_query_analysis(
    cases: List[dict],
    images_dir: Path,
    annotation: dict,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
) -> None:
    out_dir = args.selected_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "selected_cases.jsonl").open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    for idx, case in enumerate(cases):
        status = "correct" if case.get("rank1_is_correct") else "wrong"
        case_dir = out_dir / f"selected_{idx:04d}_q{case['query_index']}_{status}_pred{case['rank1_gallery_index']}"
        _save_contact_sheet(case, images_dir, case_dir / "top_matches.jpg", args.contact_sheet_topk, annotation=annotation)

        heatmap_stats = {}
        heatmap_stats["query"] = _save_attention_heatmaps(
            model,
            images_dir / case["query_name"],
            case_dir / "query_heatmaps",
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            args.heatmap_layers,
        )
        heatmap_stats["rank1_match"] = _save_attention_heatmaps(
            model,
            images_dir / case["rank1_name"],
            case_dir / "rank1_heatmaps",
            device,
            use_amp,
            amp_dtype,
            use_channels_last,
            args.heatmap_layers,
        )

        correct = next((m for m in case["top_matches"] if m["is_correct"]), None)
        if correct is not None and correct["name"] != case["rank1_name"]:
            heatmap_stats["first_correct_match"] = _save_attention_heatmaps(
                model,
                images_dir / correct["name"],
                case_dir / "first_correct_heatmaps",
                device,
                use_amp,
                amp_dtype,
                use_channels_last,
                args.heatmap_layers,
            )

        with (case_dir / "case.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "case": case,
                    "heatmap_stats": heatmap_stats,
                    "d_part_label": annotation.get("d_part_label", {}).get(case["query_name"], []),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Wrote selected query analysis to {out_dir}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    vrai_dir = repo_root / "VRAI"
    parser = argparse.ArgumentParser(description="Evaluate GASNet on VRAI and report mAP/Rank-1/Rank-5")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["eval", "submit"], default="eval")
    parser.add_argument("--split", choices=["train", "test", "test-dev"], default="train")
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--id-map", type=Path, default=None)
    parser.add_argument(
        "--train-random-seed",
        type=int,
        default=42,
        help="Seed used to choose one random query image per ID for VRAI train eval",
    )
    parser.add_argument(
        "--train-query-selection",
        choices=["random", "centroid"],
        default="random",
        help="Query selection strategy for train split. 'random' picks one image randomly per ID. 'centroid' picks the image closest to the mean feature per ID.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fp16-sim", action="store_true")
    parser.add_argument("--q-chunk-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("submission_vrai.json"))
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--rerank", action="store_true", help="Apply lightweight gallery-neighbor query expansion re-ranking")
    parser.add_argument("--rerank-k1", type=int, default=20, help="Number of gallery neighbors used for re-ranking")
    parser.add_argument("--rerank-alpha", type=float, default=0.3, help="Neighbor feature weight used for re-ranking")
    parser.add_argument("--diagnostic-topk", type=int, default=20, help="Top-k used for no_match_in_topk diagnostics")
    parser.add_argument("--use-part-branch", action="store_true", help="Load/evaluate a checkpoint trained with the part branch")
    parser.add_argument("--num-parts", type=int, default=4, help="Number of horizontal stripes used by the part branch")
    parser.add_argument("--use-attention-local", action="store_true", help="Load/evaluate a checkpoint trained with the attention local branch")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention maps used by the local branch")
    parser.add_argument("--use-gem", action="store_true", help="Load/evaluate a checkpoint trained with GeM pooling")
    parser.add_argument("--gem-p", type=float, default=3.0, help="GeM pooling exponent")
    parser.add_argument("--backbone", choices=[
        "resnet50", "resnet50_ibn", "swin_t", "dinov3_convnext"
    ], default="resnet50")
    parser.add_argument("--tta-flip", action="store_true", help="Average original and horizontal-flip features")
    parser.add_argument("--analyze-failures", action="store_true", help="Write detailed Rank-1 failure analysis for eval mode")
    parser.add_argument("--analysis-output-dir", type=Path, default=Path("output/vrai_failure_analysis"))
    parser.add_argument("--analysis-topk", type=int, default=20, help="Top-k retrievals kept per failure case")
    parser.add_argument("--analysis-max-cases", type=int, default=200, help="Max failure cases written to JSONL")
    parser.add_argument("--heatmap-cases", type=int, default=12, help="Number of failure cases with saved attention heatmaps")
    parser.add_argument("--heatmap-layers", nargs="+", default=["ga1", "ga2", "ga3", "ga4"], choices=["ga1", "ga2", "ga3", "ga4"])
    parser.add_argument("--contact-sheet-topk", type=int, default=5, help="Top-k gallery images shown next to query")
    parser.add_argument("--selected-query-indices", type=int, nargs="*", default=[], help="Global query indices to export, including corrected cases")
    parser.add_argument("--selected-query-file", type=Path, default=None, help="Text file with one global query index per line")
    parser.add_argument("--selected-output-dir", type=Path, default=Path("output/vrai_selected_query_analysis"))
    parser.add_argument("--log-path", type=Path, default=None, help="Path to save log text file")
    parser.add_argument("--gpu-jetson", action="store_true", help="Optimize dataloader and config for Jetson AGX Orin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.gpu_jetson:
        print("\n--- Jetson AGX Orin Optimization Enabled ---")
        if args.batch_size == 256:
            args.batch_size = 64
        if args.num_workers == 4:
            args.num_workers = 8
        if args.prefetch_factor == 2:
            args.prefetch_factor = 4
            
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = Logger(args.log_path, sys.stdout)
        sys.stderr = Logger(args.log_path, sys.stderr)

    configure_cuda_for_speed()

    if args.annotation is None:
        if args.split == "train":
            annotation_path = vrai_dir / "train_annotation.pkl"
        elif args.split == "test":
            annotation_path = vrai_dir / "test_annotation.pkl"
        else:
            annotation_path = vrai_dir / "test_dev_annotation.pkl"
    else:
        annotation_path = args.annotation

    if args.images_dir is None:
        if args.split == "train":
            images_dir = vrai_dir / "images_train"
        else:
            images_dir = vrai_dir / "images_dev"
    else:
        images_dir = args.images_dir

    annotation = _load_annotation(annotation_path)
    is_centroid = args.mode == "eval" and args.split == "train" and args.train_query_selection == "centroid"

    if is_centroid:
        image_names = annotation.get("train_im_names")
        if image_names is None:
            raise KeyError("train_im_names not found in train_annotation.pkl")
        image_paths = [images_dir / name for name in image_names]
        query_paths = []
        gallery_paths = []
        query_idx, gallery_idx, query_ids, gallery_ids = [], [], [], []
    else:
        if args.mode == "eval":
            image_names, query_idx, gallery_idx, query_ids, gallery_ids = _infer_eval_lists(
                args.split,
                annotation,
                args.id_map,
                train_random_seed=args.train_random_seed,
            )
        else:
            image_names = annotation.get("test_im_names") or annotation.get("dev_im_names")
            if image_names is None:
                raise KeyError("Annotation file missing test_im_names or dev_im_names")
            query_idx = annotation["query_order"]
            gallery_idx = annotation["gallery_order"]
            if query_idx and isinstance(query_idx[0], str):
                index_map = {name: i for i, name in enumerate(image_names)}
                query_idx = [index_map[name] for name in query_idx]
                gallery_idx = [index_map[name] for name in gallery_idx]
        image_paths = [images_dir / name for name in image_names]
        if not image_paths:
            raise ValueError("No images found in annotation list")
        query_paths = [image_paths[idx] for idx in query_idx]
        gallery_paths = [image_paths[idx] for idx in gallery_idx]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_channels_last = torch.cuda.is_available() and (not args.no_channels_last)
    use_amp = torch.cuda.is_available() and (not args.no_amp)
    if args.amp_dtype == "auto":
        amp_dtype = choose_amp_dtype()
    elif args.amp_dtype == "bf16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            print("bf16 not supported on this GPU; falling back to fp16")
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.float16

    test_tfms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pin_memory = torch.cuda.is_available() and (not args.no_pin_memory)
    persistent_workers = (not args.no_persistent_workers) and args.num_workers > 0

    state_dict = _normalize_state_dict(_load_checkpoint(args.model_path))
    try:
        num_classes = _infer_num_classes(state_dict)
    except KeyError:
        if args.num_classes is None:
            raise
        num_classes = args.num_classes
    model = GASNet(
        num_classes=num_classes,
        use_pretrained=False,
        use_part_branch=args.use_part_branch,
        num_parts=args.num_parts,
        use_gem=args.use_gem,
        gem_p=args.gem_p,
        backbone=args.backbone,
        use_attention_local=args.use_attention_local,
        num_attention_heads=args.num_attention_heads,
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    if is_centroid:
        print("Extracting features for all training images to select centroid queries...")
        all_loader = _make_loader(image_paths, test_tfms, args.batch_size, args.num_workers, pin_memory, persistent_workers, args.prefetch_factor, args.gpu_jetson)
        all_feat = _extract_features(model, all_loader, device, use_amp, amp_dtype, use_channels_last, args.log_every).to(device)
        
        if args.tta_flip:
            flip_tfms = transforms.Compose([test_tfms, transforms.RandomHorizontalFlip(p=1.0)])
            all_flip_loader = _make_loader(image_paths, flip_tfms, args.batch_size, args.num_workers, pin_memory, persistent_workers, args.prefetch_factor, args.gpu_jetson)
            all_flip_feat = _extract_features(model, all_flip_loader, device, use_amp, amp_dtype, use_channels_last, args.log_every).to(device)
            all_feat = F.normalize(F.normalize(all_feat, dim=1) + F.normalize(all_flip_feat, dim=1), dim=1)
            
        print("Building query/gallery splits based on feature centroids...")
        query_idx, gallery_idx, query_ids, gallery_ids = _build_train_query_gallery_centroid(image_names, all_feat)
        
        q_feat = all_feat[query_idx]
        g_feat = all_feat[gallery_idx]
        
    else:
        q_loader = _make_loader(query_paths, test_tfms, args.batch_size, args.num_workers, pin_memory, persistent_workers, args.prefetch_factor, args.gpu_jetson)
        g_loader = _make_loader(gallery_paths, test_tfms, args.batch_size, args.num_workers, pin_memory, persistent_workers, args.prefetch_factor, args.gpu_jetson)

        q_feat = _extract_features(model, q_loader, device, use_amp, amp_dtype, use_channels_last, args.log_every).to(device)
        g_feat = _extract_features(model, g_loader, device, use_amp, amp_dtype, use_channels_last, args.log_every).to(device)
        
        if args.tta_flip:
            flip_tfms = transforms.Compose([test_tfms, transforms.RandomHorizontalFlip(p=1.0)])
            q_flip_loader = _make_loader(query_paths, flip_tfms, args.batch_size, args.num_workers, pin_memory, persistent_workers, args.prefetch_factor, args.gpu_jetson)
            g_flip_loader = _make_loader(gallery_paths, flip_tfms, args.batch_size, args.num_workers, pin_memory, persistent_workers, args.prefetch_factor, args.gpu_jetson)
            
            q_flip_feat = _extract_features(model, q_flip_loader, device, use_amp, amp_dtype, use_channels_last, args.log_every).to(device)
            g_flip_feat = _extract_features(model, g_flip_loader, device, use_amp, amp_dtype, use_channels_last, args.log_every).to(device)
            
            q_feat = F.normalize(F.normalize(q_feat, dim=1) + F.normalize(q_flip_feat, dim=1), dim=1)
            g_feat = F.normalize(F.normalize(g_feat, dim=1) + F.normalize(g_flip_feat, dim=1), dim=1)

    if args.mode == "eval":
        q_ids = torch.tensor(query_ids, dtype=torch.long, device=device)
        g_ids = torch.tensor(gallery_ids, dtype=torch.long, device=device)
        m_ap, rank1, rank5 = evaluate_map_cmc(
            q_feat,
            q_ids,
            g_feat,
            g_ids,
            topk=(1, 5),
            q_chunk_size=args.q_chunk_size,
            use_fp16_sim=(not args.no_fp16_sim),
            verbose=True,
        )

        print("\n=== VRAI Evaluation ===")
        print(f"Split: {args.split}")
        print(f"mAP:   {m_ap:.4f}")
        print(f"Rank-1:{rank1:.4f}")
        print(f"Rank-5:{rank5:.4f}")
        diagnostics = _collect_retrieval_diagnostics(
            q_feat=q_feat,
            g_feat=g_feat,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            query_idx=query_idx,
            gallery_idx=gallery_idx,
            image_names=image_names,
            annotation=annotation,
            split=args.split,
            topk=args.diagnostic_topk,
            q_chunk_size=args.q_chunk_size,
            use_fp16_sim=(not args.no_fp16_sim),
        )
        _print_diagnostic_report(diagnostics)
        analysis_q_feat = q_feat
        analysis_g_feat = g_feat

        if args.rerank:
            print(
                f"\nApplying re-ranking: query expansion over gallery neighbors "
                f"(k1={args.rerank_k1}, alpha={args.rerank_alpha:.3f})"
            )
            rerank_q_feat, rerank_g_feat = _query_expansion_rerank_features(
                q_feat=q_feat,
                g_feat=g_feat,
                k1=args.rerank_k1,
                alpha=args.rerank_alpha,
                q_chunk_size=args.q_chunk_size,
                use_fp16_sim=(not args.no_fp16_sim),
            )
            m_ap, rank1, rank5 = evaluate_map_cmc(
                rerank_q_feat,
                q_ids,
                rerank_g_feat,
                g_ids,
                topk=(1, 5),
                q_chunk_size=args.q_chunk_size,
                use_fp16_sim=(not args.no_fp16_sim),
                verbose=True,
            )

            print("\n=== VRAI Evaluation (Re-ranked) ===")
            print(f"Split: {args.split}")
            print(f"mAP:   {m_ap:.4f}")
            print(f"Rank-1:{rank1:.4f}")
            print(f"Rank-5:{rank5:.4f}")
            diagnostics = _collect_retrieval_diagnostics(
                q_feat=rerank_q_feat,
                g_feat=rerank_g_feat,
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_idx=query_idx,
                gallery_idx=gallery_idx,
                image_names=image_names,
                annotation=annotation,
                split=args.split,
                topk=args.diagnostic_topk,
                q_chunk_size=args.q_chunk_size,
                use_fp16_sim=(not args.no_fp16_sim),
            )
            _print_diagnostic_report(diagnostics)
            analysis_q_feat = rerank_q_feat
            analysis_g_feat = rerank_g_feat

        if args.analyze_failures:
            print("\n=== VRAI Failure Analysis ===")
            failure_topk = max(args.analysis_topk, args.contact_sheet_topk, 5)
            failures, summary = _collect_failure_cases(
                q_feat=analysis_q_feat,
                g_feat=analysis_g_feat,
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_idx=query_idx,
                gallery_idx=gallery_idx,
                image_names=image_names,
                annotation=annotation,
                split=args.split,
                topk=failure_topk,
                q_chunk_size=args.q_chunk_size,
                use_fp16_sim=(not args.no_fp16_sim),
            )
            print(f"Rank-1 failures: {summary['num_rank1_failures']}/{summary['num_queries']}")
            _write_failure_analysis(
                failures=failures,
                summary=summary,
                image_names=image_names,
                images_dir=images_dir,
                annotation=annotation,
                model=model,
                args=args,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
            )

        selected_query_indices = list(args.selected_query_indices)
        if args.selected_query_file is not None:
            with args.selected_query_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    selected_query_indices.append(int(line.split()[0]))
        if selected_query_indices:
            print("\n=== VRAI Selected Query Analysis ===")
            selected_topk = max(args.analysis_topk, args.contact_sheet_topk, 20)
            selected_cases = _collect_selected_query_cases(
                q_feat=analysis_q_feat,
                g_feat=analysis_g_feat,
                query_indices=selected_query_indices,
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_idx=query_idx,
                gallery_idx=gallery_idx,
                image_names=image_names,
                annotation=annotation,
                split=args.split,
                topk=selected_topk,
                use_fp16_sim=(not args.no_fp16_sim),
            )
            print(f"Selected query cases: {len(selected_cases)}/{len(selected_query_indices)}")
            _write_selected_query_analysis(
                cases=selected_cases,
                images_dir=images_dir,
                annotation=annotation,
                model=model,
                args=args,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
            )
        return

    if args.split == "train":
        raise ValueError("Submission mode is intended for test/test-dev splits only")

    if args.rerank:
        print(
            f"\nApplying re-ranking: query expansion over gallery neighbors "
            f"(k1={args.rerank_k1}, alpha={args.rerank_alpha:.3f})"
        )
        q_feat, g_feat = _query_expansion_rerank_features(
            q_feat=q_feat,
            g_feat=g_feat,
            k1=args.rerank_k1,
            alpha=args.rerank_alpha,
            q_chunk_size=args.q_chunk_size,
            use_fp16_sim=(not args.no_fp16_sim),
        )

    reid_result = _build_reid_result(
        q_feat=q_feat,
        g_feat=g_feat,
        topk=args.topk,
        q_chunk_size=args.q_chunk_size,
        use_fp16_sim=(not args.no_fp16_sim),
    )

    output = {"reid_result": reid_result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f)

    print(f"Wrote submission to {args.output}")


if __name__ == "__main__":
    main()
