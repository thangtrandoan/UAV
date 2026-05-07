from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataset import VISDRONE_NAMES, YoloTxtDataset, build_paths, build_visdrone_train_loader, collate_fn, prepare_data
from evaluate import compute_basic_detection_metrics, evaluate_coco_metrics
from utils import choose_amp_dtype, choose_device, configure_cuda_for_speed, ensure_dir, set_seed


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 16
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01
    mosaic_prob: float = 0.8
    num_workers: int = 4
    img_size: int = 640


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SIEM(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()
        c_half = c_in // 2
        self.branch1 = nn.Sequential(
            ConvBNAct(c_in, c_half, 3, 1),
            ConvBNAct(c_half, c_half, 3, 1),
            ConvBNAct(c_half, c_half, 3, 1),
        )
        self.branch2 = nn.Sequential(
            ConvBNAct(c_in, c_half, 3, 1),
            ConvBNAct(c_half, c_half, 3, 1),
        )

        self.cbs_f1 = ConvBNAct(c_in, c_in, 3, 1)
        self.cbs_attn1 = ConvBNAct(c_in, c_in, 3, 1)
        self.cbs_attn2 = ConvBNAct(c_in, c_in, 3, 1)
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)

        self.conv1x1_1 = nn.Conv2d(c_in, c_in, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_in)
        self.conv1x1_2 = nn.Conv2d(c_in, c_in, 1, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        f1 = torch.cat([b1, b2], dim=1)
        f1_cbs = self.cbs_f1(f1)
        attn_map = F.softmax(self.cbs_attn2(self.cbs_attn1(f1)), dim=1)
        f2 = f1_cbs * attn_map
        f_out_proj = self.conv1x1_2(self.bn(self.conv1x1_1(f2)))
        return f_out_proj + f1_cbs + self.pool(x) + x


class CSPBlock(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c2, 1, 1)
        self.cv2 = ConvBNAct(c1, c2, 1, 1)
        self.m = nn.Sequential(*[ConvBNAct(c2, c2, 3, 1) for _ in range(n)])
        self.cv3 = ConvBNAct(2 * c2, c2, 1, 1)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class MFEFBackbone(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage1 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), CSPBlock(c2, c2, n=2))
        self.siem = SIEM(c2)
        self.stage2 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), CSPBlock(c3, c3, n=3))
        self.stage3 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), CSPBlock(c4, c4, n=3))
        self.stage4 = nn.Sequential(ConvBNAct(c4, c4 * 2, 3, 2), CSPBlock(c4 * 2, c4 * 2, n=2))

    def forward(self, x):
        x = self.stem(x)
        p2 = self.siem(self.stage1(x))
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p2, p3, p4, p5


class AFFM(nn.Module):
    def __init__(self, ch_in: List[int], c_out: int):
        super().__init__()
        self.num_inputs = len(ch_in)
        self.convs = nn.ModuleList([ConvBNAct(c, c_out, 1, 1) for c in ch_in])
        self.weight_conv = nn.Conv2d(c_out * self.num_inputs, self.num_inputs, 1, 1)
        self.out_conv = ConvBNAct(c_out, c_out, 3, 1)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        ys = [conv(x) for conv, x in zip(self.convs, xs)]
        f_cat = torch.cat(ys, dim=1)
        w = F.softmax(self.weight_conv(f_cat), dim=1)
        out = 0.0
        for i, y in enumerate(ys):
            out = out + y * w[:, i : i + 1]
        return self.out_conv(out)


class GAFN(nn.Module):
    def __init__(self, c2, c3, c4, c5, c=128):
        super().__init__()
        self.fuse_p4 = AFFM([c4, c5], c)
        self.fuse_p3 = AFFM([c3, c], c)
        self.fuse_p2 = AFFM([c2, c], c)
        self.fuse_n3 = AFFM([c, c], c)
        self.fuse_n4 = AFFM([c, c], c)
        self.fuse_n5 = AFFM([c, c], c)
        self.p5_lat = ConvBNAct(c5, c, 1, 1)
        self.ds = nn.AvgPool2d(2, 2)

    def forward(self, p2, p3, p4, p5):
        p5u = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        m4 = self.fuse_p4([p4, p5u])
        m4u = F.interpolate(m4, size=p3.shape[-2:], mode="nearest")
        m3 = self.fuse_p3([p3, m4u])
        m3u = F.interpolate(m3, size=p2.shape[-2:], mode="nearest")
        m2 = self.fuse_p2([p2, m3u])

        n3 = self.fuse_n3([m3, self.ds(m2)])
        n4 = self.fuse_n4([m4, self.ds(n3)])
        p5r = self.p5_lat(p5)
        n5 = self.fuse_n5([self.ds(n4), F.interpolate(p5r, size=self.ds(n4).shape[-2:], mode="nearest")])
        return m2, n3, n4, n5


class IDetectLikeHead(nn.Module):
    def __init__(self, c=128, num_classes=10, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.num_anchors = num_anchors
        self.pred2 = nn.Conv2d(c, num_anchors * self.num_outputs, 1)
        self.pred3 = nn.Conv2d(c, num_anchors * self.num_outputs, 1)
        self.pred4 = nn.Conv2d(c, num_anchors * self.num_outputs, 1)
        self.pred5 = nn.Conv2d(c, num_anchors * self.num_outputs, 1)

    def _reshape(self, x):
        b, _, h, w = x.shape
        x = x.view(b, self.num_anchors, self.num_outputs, h, w)
        return x.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, feats):
        m2, n3, n4, n5 = feats
        return [self._reshape(self.pred2(m2)), self._reshape(self.pred3(n3)), self._reshape(self.pred4(n4)), self._reshape(self.pred5(n5))]


class MFEFNet(nn.Module):
    def __init__(self, num_classes=10, base=32, neck_c=128):
        super().__init__()
        self.backbone = MFEFBackbone(base=base)
        c2, c3, c4, c5 = base * 2, base * 4, base * 8, base * 16
        self.neck = GAFN(c2, c3, c4, c5, c=neck_c)
        self.head = IDetectLikeHead(c=neck_c, num_classes=num_classes)

    def forward(self, x):
        return self.head(self.neck(*self.backbone(x)))


DEFAULT_ANCHORS_PX = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)],
    [(200, 280), (300, 380), (420, 520)],
]


def mpdiou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = tgt_xyxy.unbind(-1)

    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    p_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    t_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = p_area + t_area - inter + eps
    iou = inter / union

    ex1 = torch.minimum(px1, tx1)
    ey1 = torch.minimum(py1, ty1)
    ex2 = torch.maximum(px2, tx2)
    ey2 = torch.maximum(py2, ty2)
    diag2 = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    d_tl = (px1 - tx1).pow(2) + (py1 - ty1).pow(2)
    d_br = (px2 - tx2).pow(2) + (py2 - ty2).pow(2)

    mpdiou = iou - (d_tl + d_br) / diag2
    return 1.0 - mpdiou.clamp(min=-1.0, max=1.0)


def decode_boxes_from_logits(pred: torch.Tensor, anchors_wh: Optional[torch.Tensor] = None) -> torch.Tensor:
    _, a, h, w, _ = pred.shape
    device = pred.device
    dtype = pred.dtype

    gy, gx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij")
    gx = gx.view(1, 1, h, w)
    gy = gy.view(1, 1, h, w)

    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]

    cx = (torch.sigmoid(tx) + gx) / max(1.0, float(w))
    cy = (torch.sigmoid(ty) + gy) / max(1.0, float(h))

    if anchors_wh is not None:
        anc = anchors_wh.to(device=device, dtype=dtype)
        bw = torch.sigmoid(tw) * anc[:, 0].view(1, a, 1, 1)
        bh = torch.sigmoid(th) * anc[:, 1].view(1, a, 1, 1)
    else:
        bw = torch.sigmoid(tw)
        bh = torch.sigmoid(th)

    x1 = (cx - bw / 2.0).clamp(0.0, 1.0)
    y1 = (cy - bh / 2.0).clamp(0.0, 1.0)
    x2 = (cx + bw / 2.0).clamp(0.0, 1.0)
    y2 = (cy + bh / 2.0).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _pick_level_by_area(area: float, n_levels: int) -> int:
    bins = [0.02, 0.08, 0.20]
    if n_levels <= 1:
        return 0
    if n_levels == 2:
        return 0 if area < bins[1] else 1
    if n_levels == 3:
        return 0 if area < bins[0] else (1 if area < bins[2] else 2)
    if area < bins[0]:
        return 0
    if area < bins[1]:
        return 1
    if area < bins[2]:
        return 2
    return 3


def _best_anchor_index(gt_w: float, gt_h: float, anchors_wh: torch.Tensor) -> int:
    gt = torch.tensor([gt_w, gt_h], dtype=anchors_wh.dtype, device=anchors_wh.device).clamp(min=1e-6)
    anc = anchors_wh.clamp(min=1e-6)
    ratio = torch.maximum(gt[None, :] / anc, anc / gt[None, :])
    score = ratio.max(dim=1).values
    return int(torch.argmin(score).item())


class MFEFNetLoss(nn.Module):
    def __init__(self, num_classes: int = 10, img_size: int = 640, anchors_px: Optional[List[List[Tuple[int, int]]]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        anchors_px = anchors_px if anchors_px is not None else DEFAULT_ANCHORS_PX
        self.anchors_wh = [torch.tensor(a, dtype=torch.float32) / float(img_size) for a in anchors_px]

    def _allocate_targets(self, preds: List[torch.Tensor]):
        obj_t, cls_t, box_t, pos_mask = [], [], [], []
        for p in preds:
            b, a, h, w, _ = p.shape
            device = p.device
            obj_t.append(torch.zeros((b, a, h, w), device=device, dtype=p.dtype))
            cls_t.append(torch.zeros((b, a, h, w, self.num_classes), device=device, dtype=p.dtype))
            box_t.append(torch.zeros((b, a, h, w, 4), device=device, dtype=p.dtype))
            pos_mask.append(torch.zeros((b, a, h, w), device=device, dtype=torch.bool))
        return obj_t, cls_t, box_t, pos_mask

    def _build_dense_targets(self, preds: List[torch.Tensor], targets: List[torch.Tensor]):
        obj_t, cls_t, box_t, pos_mask = self._allocate_targets(preds)
        n_levels = len(preds)

        for b_ix, t in enumerate(targets):
            if t.numel() == 0:
                continue
            for row in t:
                cls_id = int(row[0].item())
                cx = float(row[1].item())
                cy = float(row[2].item())
                bw = float(row[3].item())
                bh = float(row[4].item())
                if cls_id < 0 or cls_id >= self.num_classes or bw <= 0.0 or bh <= 0.0:
                    continue

                level = min(_pick_level_by_area(bw * bh, n_levels), n_levels - 1)
                _, a, h, w, _ = preds[level].shape
                anchors_l = self.anchors_wh[min(level, len(self.anchors_wh) - 1)].to(preds[level].device)
                anchor_ix = _best_anchor_index(bw, bh, anchors_l) if anchors_l.shape[0] == a else 0

                gx = int(max(0, min(w - 1, math.floor(cx * w))))
                gy = int(max(0, min(h - 1, math.floor(cy * h))))

                x1 = max(0.0, cx - bw / 2.0)
                y1 = max(0.0, cy - bh / 2.0)
                x2 = min(1.0, cx + bw / 2.0)
                y2 = min(1.0, cy + bh / 2.0)

                obj_t[level][b_ix, anchor_ix, gy, gx] = 1.0
                cls_t[level][b_ix, anchor_ix, gy, gx, cls_id] = 1.0
                box_t[level][b_ix, anchor_ix, gy, gx] = torch.tensor([x1, y1, x2, y2], device=preds[level].device, dtype=preds[level].dtype)
                pos_mask[level][b_ix, anchor_ix, gy, gx] = True

        return obj_t, cls_t, box_t, pos_mask

    def forward(self, preds: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        obj_t, cls_t, box_t, pos_mask = self._build_dense_targets(preds, targets)

        l_box = torch.zeros((), device=preds[0].device)
        l_cls = torch.zeros((), device=preds[0].device)
        l_obj = torch.zeros((), device=preds[0].device)

        for s, p in enumerate(preds):
            obj_logits = p[..., 4]
            cls_logits = p[..., 5 : 5 + self.num_classes]
            anchors_l = self.anchors_wh[min(s, len(self.anchors_wh) - 1)]
            pred_boxes = decode_boxes_from_logits(p, anchors_wh=anchors_l)

            l_obj = l_obj + self.bce(obj_logits, obj_t[s])
            pm = pos_mask[s]
            if pm.any():
                l_box = l_box + mpdiou_loss(pred_boxes[pm], box_t[s][pm]).mean()
                l_cls = l_cls + self.bce(cls_logits[pm], cls_t[s][pm])

        num_scales = max(1, len(preds))
        l_box = l_box / num_scales
        l_cls = l_cls / num_scales
        l_obj = l_obj / num_scales
        total = 0.3 * l_box + 0.05 * l_cls + 0.7 * l_obj
        return {"loss": total, "l_box": l_box, "l_cls": l_cls, "l_obj": l_obj}


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=cfg.lr0, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    steps_per_epoch = max(1, int(steps_per_epoch))
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _xywhn_to_xyxy_abs(cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return x1, y1, x2, y2


def build_image_id_map(image_list_file: Path) -> Dict[str, int]:
    img_paths = [Path(x.strip()) for x in image_list_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {str(p.resolve()): i + 1 for i, p in enumerate(img_paths)}


def build_coco_gt_from_yolo(image_list_file: Path, labels_dir: Path, class_names: List[str], output_json: Path) -> Path:
    img_paths = [Path(x.strip()) for x in image_list_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    images = []
    annotations = []
    categories = [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(class_names)]

    ann_id = 1
    for img_id, p in enumerate(img_paths, start=1):
        with Image.open(p) as im:
            w, h = im.size

        images.append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        label_path = labels_dir / f"{p.stem}.txt"
        if not label_path.exists():
            continue

        for ln in label_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            cls_s, cx_s, cy_s, bw_s, bh_s = ln.split()
            cls_id = int(cls_s)
            cx, cy, bw, bh = float(cx_s), float(cy_s), float(bw_s), float(bh_s)

            x1, y1, x2, y2 = _xywhn_to_xyxy_abs(cx, cy, bw, bh, w, h)
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            bw_abs = max(0.0, x2 - x1)
            bh_abs = max(0.0, y2 - y1)
            if bw_abs <= 0.0 or bh_abs <= 0.0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id + 1,
                    "bbox": [x1, y1, bw_abs, bh_abs],
                    "area": bw_abs * bh_abs,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, ensure_ascii=False), encoding="utf-8")
    return output_json


def _box_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    ix1 = torch.maximum(box[0], boxes[:, 0])
    iy1 = torch.maximum(box[1], boxes[:, 1])
    ix2 = torch.minimum(box[2], boxes[:, 2])
    iy2 = torch.minimum(box[3], boxes[:, 3])

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    a1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    a2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (a1 + a2 - inter + eps)


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float = 0.65) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _box_iou_xyxy(boxes[i], boxes[rest])
        order = rest[ious <= iou_thres]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _letterbox_to_original_xyxy(box_xyxy_norm: torch.Tensor, orig_w: int, orig_h: int, input_size: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (box_xyxy_norm * float(input_size)).tolist()
    scale = min(float(input_size) / float(orig_w), float(input_size) / float(orig_h))
    nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
    pad_w = (input_size - nw) / 2.0
    pad_h = (input_size - nh) / 2.0

    x1 = (x1 - pad_w) / max(scale, 1e-7)
    y1 = (y1 - pad_h) / max(scale, 1e-7)
    x2 = (x2 - pad_w) / max(scale, 1e-7)
    y2 = (y2 - pad_h) / max(scale, 1e-7)

    x1 = max(0.0, min(float(orig_w), x1))
    y1 = max(0.0, min(float(orig_h), y1))
    x2 = max(0.0, min(float(orig_w), x2))
    y2 = max(0.0, min(float(orig_h), y2))
    return x1, y1, x2, y2


def _collect_batch_dets(preds: List[torch.Tensor], conf_thres: float, iou_thres: float, max_det: int, num_classes: int) -> List[torch.Tensor]:
    batch_size = preds[0].shape[0]
    dets_per_im = []

    for b in range(batch_size):
        boxes_all = []
        scores_all = []
        cls_all = []

        for p in preds:
            p_b = p[b : b + 1]
            boxes = decode_boxes_from_logits(p_b)[0].reshape(-1, 4)
            obj = torch.sigmoid(p_b[0, ..., 4]).reshape(-1)
            cls_prob = torch.sigmoid(p_b[0, ..., 5 : 5 + num_classes]).reshape(-1, num_classes)
            cls_score, cls_idx = torch.max(cls_prob, dim=1)
            score = obj * cls_score

            m = score >= conf_thres
            if m.any():
                boxes_all.append(boxes[m])
                scores_all.append(score[m])
                cls_all.append(cls_idx[m].float())

        if not boxes_all:
            dets_per_im.append(torch.zeros((0, 6), device=preds[0].device, dtype=preds[0].dtype))
            continue

        boxes = torch.cat(boxes_all, dim=0)
        scores = torch.cat(scores_all, dim=0)
        clses = torch.cat(cls_all, dim=0)

        keep_all = []
        for c in clses.unique():
            cm = clses == c
            keep_c = _nms_xyxy(boxes[cm], scores[cm], iou_thres=iou_thres)
            orig_idx = torch.where(cm)[0][keep_c]
            keep_all.append(orig_idx)

        keep = torch.cat(keep_all) if keep_all else torch.zeros((0,), dtype=torch.long, device=boxes.device)
        if keep.numel() > 0:
            keep = keep[torch.argsort(scores[keep], descending=True)]
            keep = keep[:max_det]

        if keep.numel() > 0:
            det = torch.cat([boxes[keep], scores[keep, None], clses[keep, None]], dim=1)
        else:
            det = torch.zeros((0, 6), device=boxes.device)
        dets_per_im.append(det)

    return dets_per_im


def export_predictions_to_coco_json(
    model: nn.Module,
    image_list_file: Path,
    labels_dir: Path,
    output_json: Path,
    device: torch.device,
    img_size: int = 640,
    batch_size: int = 8,
    num_workers: int = 0,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    max_det: int = 300,
    num_classes: int = 10,
    amp_dtype: Optional[torch.dtype] = None,
) -> Path:
    ds = YoloTxtDataset(image_list_file=image_list_file, labels_dir=labels_dir, img_size=img_size, train=False, mosaic_prob=0.0)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    image_id_map = build_image_id_map(image_list_file)
    records = []

    model.eval()
    use_amp = device.type == "cuda" and amp_dtype is not None
    with torch.no_grad():
        for imgs, _, paths in dl:
            imgs = imgs.to(device)
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp, dtype=amp_dtype):
                preds = model(imgs)
            dets_batch = _collect_batch_dets(preds, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, num_classes=num_classes)

            for dets, p_str in zip(dets_batch, paths):
                p = Path(p_str)
                img_id = image_id_map.get(str(p.resolve()))
                if img_id is None:
                    continue

                with Image.open(p) as im:
                    ow, oh = im.size

                for d in dets:
                    x1, y1, x2, y2 = _letterbox_to_original_xyxy(d[:4], ow, oh, img_size)
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw <= 0.0 or bh <= 0.0:
                        continue

                    records.append(
                        {
                            "image_id": int(img_id),
                            "category_id": int(d[5].item()) + 1,
                            "bbox": [float(x1), float(y1), float(bw), float(bh)],
                            "score": float(d[4].item()),
                        }
                    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return output_json


def _list_has_windows_style_paths(image_list_file: Path) -> bool:
    if not image_list_file.exists():
        return True
    for line in image_list_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        return len(s) > 1 and s[1] == ":"
    return False


def _resolve_eval_splits(raw: str) -> List[str]:
    splits = [s.strip().lower() for s in raw.split(",") if s.strip()]
    return [s for s in splits if s in {"val", "test"}]


def _get_split_paths(paths, split: str) -> Tuple[Path, Path]:
    if split == "test":
        return paths.workdir / "visdrone" / "test_images.txt", paths.workdir / "visdrone" / "test" / "labels"
    return paths.workdir / "visdrone" / "val_images.txt", paths.workdir / "visdrone" / "val" / "labels"


def evaluate_after_training(
    model: nn.Module,
    paths,
    output_dir: Path,
    device: torch.device,
    img_size: int,
    batch_size: int,
    num_classes: int,
    splits: List[str],
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Dict[str, float]]:
    eval_dir = ensure_dir(output_dir / "eval")
    results: Dict[str, Dict[str, float]] = {}

    for split in splits:
        img_list, lbl_dir = _get_split_paths(paths, split)
        if _list_has_windows_style_paths(img_list):
            print("Detected stale/non-container image paths in processed lists. Rebuilding data lists...")
            prepare_data(paths)

        gt_json = eval_dir / f"visdrone_{split}_gt_coco.json"
        pred_json = eval_dir / f"visdrone_{split}_pred_coco.json"
        metrics_json = eval_dir / f"metrics_{split}.json"

        build_coco_gt_from_yolo(img_list, lbl_dir, VISDRONE_NAMES, gt_json)
        export_predictions_to_coco_json(
            model=model,
            image_list_file=img_list,
            labels_dir=lbl_dir,
            output_json=pred_json,
            device=device,
            img_size=img_size,
            batch_size=max(1, min(8, batch_size)),
            num_classes=num_classes,
            num_workers=min(4, max(0, batch_size // 2)),
            amp_dtype=amp_dtype,
        )

        gt_obj = json.loads(gt_json.read_text(encoding="utf-8"))
        pred_obj = json.loads(pred_json.read_text(encoding="utf-8"))
        basic_metrics = compute_basic_detection_metrics(gt_records=gt_obj.get("annotations", []), pred_records=pred_obj, iou_thres=0.5)

        result: Dict[str, Dict[str, float]] = {"basic": basic_metrics}
        try:
            result["coco"] = evaluate_coco_metrics(gt_json, pred_json)
        except Exception as exc:
            print(f"COCO evaluation skipped due to error: {exc}")

        metrics_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Evaluation results saved to:", metrics_json)
        results[split] = result

    return results


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
    max_steps: Optional[int] = None,
    amp_dtype: Optional[torch.dtype] = None,
    channels_last: bool = False,
) -> Dict[str, float]:
    model.train()
    use_amp = device.type == "cuda" and amp_dtype is not None
    autocast_device = "cuda" if use_amp else "cpu"

    sum_loss, sum_box, sum_cls, sum_obj, steps = 0.0, 0.0, 0.0, 0.0, 0
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        targets = [t.to(device) for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp, dtype=amp_dtype):
            preds = model(imgs)
            loss_dict = loss_fn(preds, targets)

        scaler.scale(loss_dict["loss"]).backward()
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and scaler.get_scale() >= prev_scale:
            scheduler.step()

        sum_loss += float(loss_dict["loss"].detach().cpu())
        sum_box += float(loss_dict["l_box"].detach().cpu())
        sum_cls += float(loss_dict["l_cls"].detach().cpu())
        sum_obj += float(loss_dict["l_obj"].detach().cpu())
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    den = max(1, steps)
    return {
        "loss": sum_loss / den,
        "l_box": sum_box / den,
        "l_cls": sum_cls / den,
        "l_obj": sum_obj / den,
        "steps": float(steps),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MFEFNet (VisDrone + UAVDT)")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--workdir", type=Path, default=Path("/workspace/output/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--h100-preset", action="store_true", help="Use recommended settings for H100 training")
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16"], help="AMP dtype for CUDA")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format for CNN speed")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training/inference")
    parser.add_argument("--compile-mode", type=str, default="max-autotune", help="torch.compile mode (e.g., max-autotune, reduce-overhead)")
    parser.add_argument("--prepare-data", action="store_true", help="Convert raw annotations to YOLO labels before training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip automatic evaluation after training")
    parser.add_argument("--eval-splits", type=str, default="val", help="Comma-separated splits to evaluate: val,test")
    parser.add_argument("--eval-best", action=argparse.BooleanOptionalAction, default=True, help="Reload best checkpoint before evaluation")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--neck-c", type=int, default=128)
    parser.add_argument("--classes", type=int, default=len(VISDRONE_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_cuda_for_speed()

    device = choose_device(args.device)
    amp_dtype = choose_amp_dtype()
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    print(f"device={device}, amp_dtype={amp_dtype}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr0=args.lr,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    if args.h100_preset:
        cfg.epochs = 100
        cfg.batch_size = 64
        cfg.lr0 = 0.02
        cfg.num_workers = 8
        cfg.mosaic_prob = 0.8
        print("Applied H100 preset: epochs=100, batch_size=64, lr=0.02, num_workers=8")

    paths = build_paths(data_root=args.data_root, workdir=args.workdir)
    ensure_dir(paths.workdir)
    output_dir = ensure_dir(args.output_dir)

    if args.prepare_data:
        print("[DATA-PREP-START] Converting raw annotations and building YOLO lists...")
        stats = prepare_data(paths)
        print("prepare_data stats:", stats)

    loader = build_visdrone_train_loader(
        paths=paths,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
        num_workers=cfg.num_workers,
        mosaic_prob=cfg.mosaic_prob,
    )

    model = MFEFNet(num_classes=args.classes, base=args.base, neck_c=args.neck_c).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)
    optimizer = build_optimizer(model, cfg)
    loss_fn = MFEFNetLoss(num_classes=args.classes, img_size=cfg.img_size)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(loader)))

    best_loss = float("inf")
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    ckpt_path = ckpt_dir / "best_model.pt"

    print(f"Start training for {cfg.epochs} epochs")
    t0 = time.time()

    max_steps = args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else None
    print(f"[TRAIN-START] epochs={cfg.epochs}, batch_size={cfg.batch_size}, img_size={cfg.img_size}, device={device}")
    for epoch in range(cfg.epochs):
        log = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            max_steps=max_steps,
            amp_dtype=amp_dtype,
            channels_last=args.channels_last,
        )

        if log["loss"] < best_loss:
            best_loss = log["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == cfg.epochs:
            print(
                f"[Epoch {epoch + 1}/{cfg.epochs}] lr={log['lr']:.6f} "
                f"loss={log['loss']:.4f} box={log['l_box']:.4f} cls={log['l_cls']:.4f} obj={log['l_obj']:.4f} "
                f"best={best_loss:.4f}"
            )

    dt_min = (time.time() - t0) / 60.0
    print(f"Training done in {dt_min:.1f} min. Best checkpoint: {ckpt_path}")

    if not args.skip_eval:
        if args.eval_best and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"Loaded best checkpoint for evaluation: {ckpt_path}")

        splits = _resolve_eval_splits(args.eval_splits)
        if not splits:
            splits = ["val"]
        metrics = evaluate_after_training(
            model=model,
            paths=paths,
            output_dir=output_dir,
            device=device,
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_classes=args.classes,
            splits=splits,
            amp_dtype=amp_dtype,
        )
        for split, result in metrics.items():
            print(f"Basic eval ({split}):", result.get("basic", {}))
            if "coco" in result:
                print(f"COCO eval ({split}):", result["coco"])


if __name__ == "__main__":
    main()
