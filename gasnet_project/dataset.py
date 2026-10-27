from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    img_path: Path
    vehicle_id: int
    image_id: int


def read_split_file(pic_dir: Path, file_path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id_str, veh_id_str = line.split()
            image_id = int(img_id_str)
            vehicle_id = int(veh_id_str)
            img_path = pic_dir / f"{image_id}.jpg"
            samples.append(Sample(img_path=img_path, vehicle_id=vehicle_id, image_id=image_id))
    return samples


def build_query_gallery(
    samples: List[Sample],
    seed: int | None = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Split into query/gallery with one *random* gallery image per identity.

    Previously the gallery image was always the lowest ``image_id`` of each
    identity (the first frame), which biases the protocol. Now the gallery is
    drawn uniformly at random per identity; the remaining images become
    queries. ``seed`` keeps the split reproducible across runs (set it to
    ``None`` for a fresh split every call).
    """
    rng = random.Random(seed) if seed is not None else random

    by_vehicle: Dict[int, List[Sample]] = {}
    for s in samples:
        by_vehicle.setdefault(s.vehicle_id, []).append(s)

    query: List[Sample] = []
    gallery: List[Sample] = []

    for _, group in by_vehicle.items():
        group = list(group)
        gallery_pos = rng.randrange(len(group))
        gallery.append(group[gallery_pos])
        query.extend(group[:gallery_pos])
        query.extend(group[gallery_pos + 1:])

    return query, gallery


class VRUDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        transform=None,
        relabel: bool = False,
        label_map=None,
        max_retries: int = 3,
    ):
        self.samples = samples
        self.transform = transform
        self.relabel = relabel
        self.label_map = label_map or {}
        self.max_retries = max_retries

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path: Path):
        last_err = None
        for _ in range(self.max_retries):
            try:
                with Image.open(path) as img:
                    return img.convert("RGB")
            except OSError as e:
                last_err = e
        raise last_err

    def __getitem__(self, idx):
        tried = 0
        cur_idx = idx
        while tried < min(10, len(self.samples)):
            s = self.samples[cur_idx]
            try:
                img = self._load_rgb(s.img_path)
                if self.transform is not None:
                    img = self.transform(img)
                label = self.label_map[s.vehicle_id] if self.relabel else s.vehicle_id
                return img, label, s.vehicle_id, s.image_id
            except OSError:
                tried += 1
                cur_idx = (cur_idx + 1) % len(self.samples)

        raise RuntimeError(
            f"Cannot read image after retries, start_idx={idx}, path={self.samples[idx].img_path}"
        )
