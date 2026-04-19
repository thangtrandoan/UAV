from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from utils import xyxy_to_yolo


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

UAVDT_NAMES = ["car", "bus", "truck"]
UAVDT_NAME_TO_ID = {n: i for i, n in enumerate(UAVDT_NAMES)}


@dataclass
class Paths:
    root: Path
    visdrone_train: Path
    visdrone_val: Path
    visdrone_test: Path
    uavdt_train: Path
    uavdt_test: Path
    workdir: Path


def _looks_like_dataset_root(root: Path) -> bool:
    return (root / "VisDrone").exists() and (root / "UAVDT").exists()


def resolve_dataset_root(explicit_root: Optional[Path] = None) -> Path:
    if explicit_root is not None and _looks_like_dataset_root(explicit_root):
        return explicit_root

    candidates = [
        Path("/workspace/data"),
        Path("/content/drive/MyDrive/UAV"),
        Path("/content/UAV"),
        Path("d:/UAV"),
    ]

    for c in candidates:
        if _looks_like_dataset_root(c):
            return c

    for c in candidates:
        if not c.exists():
            continue
        for child in c.iterdir():
            if child.is_dir() and _looks_like_dataset_root(child):
                return child

    raise FileNotFoundError(
        "Cannot resolve dataset root. Expected root containing both VisDrone and UAVDT. "
        "Provide --data-root explicitly."
    )


def build_paths(data_root: Optional[Path] = None, workdir: Optional[Path] = None) -> Paths:
    root = resolve_dataset_root(data_root)
    wd = workdir if workdir is not None else (root / "processed")
    return Paths(
        root=root,
        visdrone_train=root / "VisDrone" / "VisDrone2019-DET-train" / "VisDrone2019-DET-train",
        visdrone_val=root / "VisDrone" / "VisDrone2019-DET-val" / "VisDrone2019-DET-val",
        visdrone_test=root / "VisDrone" / "VisDrone2019-DET-test-dev" / "VisDrone2019-DET-test-dev",
        uavdt_train=root / "UAVDT" / "train",
        uavdt_test=root / "UAVDT" / "test",
        workdir=wd,
    )


def convert_visdrone_split(src: Path, dst: Path) -> Tuple[int, int]:
    img_src = src / "images"
    ann_src = src / "annotations"
    lbl_dst = dst / "labels"
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img, n_box = 0, 0
    for ann in ann_src.glob("*.txt"):
        img = img_src / ann.with_suffix(".jpg").name
        if not img.exists():
            continue

        rows = [r.strip().split(",") for r in ann.read_text(encoding="utf-8").splitlines() if r.strip()]
        with Image.open(img) as im:
            w, h = im.size

        lines = []
        for r in rows:
            x, y, bw, bh, _, cls, _, _ = map(int, r[:8])
            if cls == 0:
                continue
            cls_id = cls - 1
            x1, y1 = x, y
            x2, y2 = x + bw, y + bh
            cx, cy, nw, nh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1

        out_lbl = lbl_dst / ann.name
        out_lbl.write_text("\n".join(lines), encoding="utf-8")
        n_img += 1

    return n_img, n_box


def convert_uavdt_split(src: Path, dst: Path) -> Tuple[int, int]:
    ann_src = src / "ann"
    lbl_dst = dst / "labels"
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img, n_box = 0, 0
    for ann in ann_src.glob("*.json"):
        data = json.loads(ann.read_text(encoding="utf-8"))
        w = int(data["size"]["width"])
        h = int(data["size"]["height"])

        lines = []
        for obj in data.get("objects", []):
            cls_name = obj.get("classTitle", "").lower()
            if cls_name not in UAVDT_NAME_TO_ID:
                continue
            p = obj["points"]["exterior"]
            (x1, y1), (x2, y2) = p[0], p[1]
            cx, cy, nw, nh = xyxy_to_yolo(float(x1), float(y1), float(x2), float(y2), w, h)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{UAVDT_NAME_TO_ID[cls_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1

        out_lbl = lbl_dst / ann.with_suffix(".txt").name
        out_lbl.write_text("\n".join(lines), encoding="utf-8")
        n_img += 1

    return n_img, n_box


def build_yolo_lists(img_dir: Path, out_txt: Path) -> int:
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    out_txt.write_text("\n".join(str(p.as_posix()) for p in imgs), encoding="utf-8")
    return len(imgs)


def prepare_data(paths: Paths) -> Dict[str, int]:
    paths.workdir.mkdir(parents=True, exist_ok=True)
    vis = paths.workdir / "visdrone"
    uav = paths.workdir / "uavdt"

    vt, vb = convert_visdrone_split(paths.visdrone_train, vis / "train")
    vv, vb2 = convert_visdrone_split(paths.visdrone_val, vis / "val")
    vte, vb3 = convert_visdrone_split(paths.visdrone_test, vis / "test")

    ut, ub = convert_uavdt_split(paths.uavdt_train, uav / "train")
    uv, ub2 = convert_uavdt_split(paths.uavdt_test, uav / "val")

    build_yolo_lists(paths.visdrone_train / "images", vis / "train_images.txt")
    build_yolo_lists(paths.visdrone_val / "images", vis / "val_images.txt")
    build_yolo_lists(paths.visdrone_test / "images", vis / "test_images.txt")
    build_yolo_lists(paths.uavdt_train / "img", uav / "train_images.txt")
    build_yolo_lists(paths.uavdt_test / "img", uav / "val_images.txt")

    (vis / "visdrone.yaml").write_text(
        "path: " + str(vis.as_posix()) + "\n"
        "train: train_images.txt\n"
        "val: val_images.txt\n"
        "test: test_images.txt\n"
        "names:\n" + "\n".join([f"  {i}: {n}" for i, n in enumerate(VISDRONE_NAMES)]) + "\n",
        encoding="utf-8",
    )

    (uav / "uavdt.yaml").write_text(
        "path: " + str(uav.as_posix()) + "\n"
        "train: train_images.txt\n"
        "val: val_images.txt\n"
        "names:\n" + "\n".join([f"  {i}: {n}" for i, n in enumerate(UAVDT_NAMES)]) + "\n",
        encoding="utf-8",
    )

    return {
        "vis_train_img": vt,
        "vis_val_img": vv,
        "vis_test_img": vte,
        "vis_boxes": vb + vb2 + vb3,
        "uav_train_img": ut,
        "uav_val_img": uv,
        "uav_boxes": ub + ub2,
    }


def letterbox(img: Image.Image, size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[Image.Image, float, Tuple[int, int]]:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), color)
    pad_w, pad_h = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_w, pad_h))
    return canvas, scale, (pad_w, pad_h)


def _yolo_to_xyxy_abs(labels: np.ndarray, w: int, h: int) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    out = labels.copy().astype(np.float32)
    cx = out[:, 1] * w
    cy = out[:, 2] * h
    bw = out[:, 3] * w
    bh = out[:, 4] * h
    out[:, 1] = cx - bw / 2.0
    out[:, 2] = cy - bh / 2.0
    out[:, 3] = cx + bw / 2.0
    out[:, 4] = cy + bh / 2.0
    return out


def _xyxy_abs_to_yolo(labels_xyxy: np.ndarray, size: int) -> np.ndarray:
    if labels_xyxy.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    out = labels_xyxy.copy().astype(np.float32)
    x1, y1, x2, y2 = out[:, 1], out[:, 2], out[:, 3], out[:, 4]
    bw = (x2 - x1).clip(min=0)
    bh = (y2 - y1).clip(min=0)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    out[:, 1] = (cx / size).clip(0.0, 1.0)
    out[:, 2] = (cy / size).clip(0.0, 1.0)
    out[:, 3] = (bw / size).clip(0.0, 1.0)
    out[:, 4] = (bh / size).clip(0.0, 1.0)
    return out


class YoloTxtDataset(Dataset):
    def __init__(
        self,
        image_list_file: Path,
        labels_dir: Path,
        img_size: int = 640,
        train: bool = True,
        mosaic_prob: float = 0.8,
    ):
        raw_paths = [Path(x.strip()) for x in image_list_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.train = train
        self.mosaic_prob = mosaic_prob

        self.img_paths = raw_paths

    def __len__(self) -> int:
        return len(self.img_paths)

    def _read_image_and_labels(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")

        label_path = self.labels_dir / (p.stem + ".txt")
        labels = []
        if label_path.exists():
            for ln in label_path.read_text(encoding="utf-8").splitlines():
                c, cx, cy, w, h = ln.split()
                labels.append([int(c), float(cx), float(cy), float(w), float(h)])
        arr = np.asarray(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        return img, arr

    def _letterbox_with_labels(self, img: Image.Image, labels: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        w0, h0 = img.size
        lb, scale, (pad_w, pad_h) = letterbox(img, self.img_size)

        if labels.size == 0:
            return lb, labels

        lxyxy = _yolo_to_xyxy_abs(labels, w0, h0)
        lxyxy[:, 1] = lxyxy[:, 1] * scale + pad_w
        lxyxy[:, 2] = lxyxy[:, 2] * scale + pad_h
        lxyxy[:, 3] = lxyxy[:, 3] * scale + pad_w
        lxyxy[:, 4] = lxyxy[:, 4] * scale + pad_h

        lxyxy[:, 1:] = np.clip(lxyxy[:, 1:], 0.0, float(self.img_size))
        w_box = lxyxy[:, 3] - lxyxy[:, 1]
        h_box = lxyxy[:, 4] - lxyxy[:, 2]
        keep = (w_box > 1.0) & (h_box > 1.0)
        lxyxy = lxyxy[keep]

        return lb, _xyxy_abs_to_yolo(lxyxy, self.img_size)

    def _load_mosaic(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        s = self.img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        yc = random.randint(s // 2, int(1.5 * s))
        xc = random.randint(s // 2, int(1.5 * s))

        indices = [idx] + random.choices(range(len(self.img_paths)), k=3)
        labels_all = []

        for i, index in enumerate(indices):
            img, labels = self._read_image_and_labels(index)
            w, h = img.size
            img_np = np.asarray(img, dtype=np.uint8)
            labels_xyxy = _yolo_to_xyxy_abs(labels, w, h)

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img_np[y1b:y2b, x1b:x2b]

            if labels_xyxy.size > 0:
                pad_x = x1a - x1b
                pad_y = y1a - y1b
                labels_xyxy[:, 1] += pad_x
                labels_xyxy[:, 2] += pad_y
                labels_xyxy[:, 3] += pad_x
                labels_xyxy[:, 4] += pad_y
                labels_all.append(labels_xyxy)

        if labels_all:
            labels_all = np.concatenate(labels_all, axis=0)
            labels_all[:, 1:] = np.clip(labels_all[:, 1:], 0.0, float(2 * s))
            wh = labels_all[:, 3:5] - labels_all[:, 1:3]
            keep = (wh[:, 0] > 2.0) & (wh[:, 1] > 2.0)
            labels_all = labels_all[keep]
        else:
            labels_all = np.zeros((0, 5), dtype=np.float32)

        out_img = Image.fromarray(mosaic_img).resize((s, s), Image.BILINEAR)
        if labels_all.size > 0:
            labels_all[:, 1:] *= 0.5
            labels_all = _xyxy_abs_to_yolo(labels_all, s)

        return out_img, labels_all

    def __getitem__(self, idx: int):
        p = self.img_paths[idx]
        if self.train and random.random() < self.mosaic_prob and len(self.img_paths) >= 4:
            img, labels = self._load_mosaic(idx)
        else:
            raw_img, raw_labels = self._read_image_and_labels(idx)
            img, labels = self._letterbox_with_labels(raw_img, raw_labels)

        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        t = torch.tensor(labels, dtype=torch.float32) if labels.size > 0 else torch.zeros((0, 5), dtype=torch.float32)
        return x, t, str(p)


def collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets), list(paths)


def build_visdrone_train_loader(paths: Paths, batch_size: int = 16, img_size: int = 640, num_workers: int = 4, mosaic_prob: float = 0.8) -> DataLoader:
    ds = YoloTxtDataset(
        image_list_file=paths.workdir / "visdrone" / "train_images.txt",
        labels_dir=paths.workdir / "visdrone" / "train" / "labels",
        img_size=img_size,
        train=True,
        mosaic_prob=mosaic_prob,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
