from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train import GASNet
from utils import choose_amp_dtype, configure_cuda_for_speed, evaluate_map_cmc


class VRAIDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform=None, max_retries: int = 3):
        self.image_paths = image_paths
        self.transform = transform
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: Path) -> Image.Image:
        last_err = None
        for _ in range(self.max_retries):
            try:
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
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


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


def _build_train_query_gallery(image_names: List[str]) -> Tuple[List[int], List[int], List[int], List[int]]:
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
        gallery_idx.append(idxs_sorted[0])
        query_idx.extend(idxs_sorted[1:])

    if not query_idx or not gallery_idx:
        raise ValueError("Not enough images per identity to build query/gallery for train split")

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
) -> DataLoader:
    ds = VRAIDataset(image_paths=image_paths, transform=transform, max_retries=3)
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
) -> torch.Tensor:
    feats = []
    for imgs in loader:
        if use_channels_last:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)

        autocast_kwargs = dict(device_type=device.type, enabled=(use_amp and device.type == "cuda"))
        if device.type == "cuda":
            autocast_kwargs["dtype"] = amp_dtype

        with torch.autocast(**autocast_kwargs):
            bn_global, bn_fs = model(imgs)
            emb = torch.cat([bn_global, bn_fs], dim=1)

        feats.append(emb.float())

    return torch.cat(feats, dim=0)


def _infer_eval_lists(
    split: str,
    annotation: dict,
    id_map_path: Path | None,
) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    if split == "train":
        image_names = annotation.get("train_im_names")
        if image_names is None:
            raise KeyError("train_im_names not found in train_annotation.pkl")
        query_idx, gallery_idx, query_ids, gallery_ids = _build_train_query_gallery(image_names)
        return image_names, query_idx, gallery_idx, query_ids, gallery_ids

    image_names = annotation.get("test_im_names") or annotation.get("dev_im_names")
    if image_names is None:
        raise KeyError("Annotation file missing test_im_names or dev_im_names")
    query_idx = annotation["query_order"]
    gallery_idx = annotation["gallery_order"]

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
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if args.mode == "eval":
        image_names, query_idx, gallery_idx, query_ids, gallery_ids = _infer_eval_lists(
            args.split, annotation, args.id_map
        )
    else:
        image_names = annotation.get("test_im_names") or annotation.get("dev_im_names")
        if image_names is None:
            raise KeyError("Annotation file missing test_im_names or dev_im_names")
        query_idx = annotation["query_order"]
        gallery_idx = annotation["gallery_order"]
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

    q_loader = _make_loader(
        query_paths,
        test_tfms,
        args.batch_size,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )
    g_loader = _make_loader(
        gallery_paths,
        test_tfms,
        args.batch_size,
        args.num_workers,
        pin_memory,
        persistent_workers,
        args.prefetch_factor,
    )

    state_dict = _normalize_state_dict(_load_checkpoint(args.model_path))
    num_classes = _infer_num_classes(state_dict)
    model = GASNet(num_classes=num_classes, use_pretrained=False).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    q_feat = _extract_features(model, q_loader, device, use_amp, amp_dtype, use_channels_last).to(device)
    g_feat = _extract_features(model, g_loader, device, use_amp, amp_dtype, use_channels_last).to(device)

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
            verbose=False,
        )

        print("\n=== VRAI Evaluation ===")
        print(f"Split: {args.split}")
        print(f"mAP:   {m_ap:.4f}")
        print(f"Rank-1:{rank1:.4f}")
        print(f"Rank-5:{rank5:.4f}")
        return

    if args.split == "train":
        raise ValueError("Submission mode is intended for test/test-dev splits only")

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
