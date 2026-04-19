from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_basic_detection_metrics(
    gt_records: List[Dict],
    pred_records: List[Dict],
    iou_thres: float = 0.5,
) -> Dict[str, float]:
    """Compute Precision, Recall, AP, mAP at IoU=0.5 from COCO-style records.

    Inputs expect:
    - gt_records: [{'image_id', 'category_id', 'bbox': [x,y,w,h]}, ...]
    - pred_records: [{'image_id', 'category_id', 'bbox': [x,y,w,h], 'score'}, ...]
    """
    class_ids = sorted({int(r["category_id"]) for r in gt_records})
    if not class_ids:
        return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "map": 0.0}

    aps: List[float] = []
    tp_total = 0
    fp_total = 0
    gt_total = len(gt_records)

    for cls_id in class_ids:
        gt_cls = [r for r in gt_records if int(r["category_id"]) == cls_id]
        pred_cls = [r for r in pred_records if int(r["category_id"]) == cls_id]
        pred_cls.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        gt_by_img: Dict[int, List[Dict]] = {}
        for g in gt_cls:
            gt_by_img.setdefault(int(g["image_id"]), []).append({"bbox": g["bbox"], "matched": False})

        tp = np.zeros(len(pred_cls), dtype=np.float32)
        fp = np.zeros(len(pred_cls), dtype=np.float32)

        for i, p in enumerate(pred_cls):
            img_id = int(p["image_id"])
            pb = p["bbox"]
            pxyxy = np.array([pb[0], pb[1], pb[0] + pb[2], pb[1] + pb[3]], dtype=np.float32)

            candidates = gt_by_img.get(img_id, [])
            best_iou = 0.0
            best_j = -1

            for j, g in enumerate(candidates):
                gb = g["bbox"]
                gxyxy = np.array([gb[0], gb[1], gb[0] + gb[2], gb[1] + gb[3]], dtype=np.float32)
                iou = _iou_xyxy(pxyxy, gxyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou >= iou_thres and (not candidates[best_j]["matched"]):
                tp[i] = 1.0
                candidates[best_j]["matched"] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        denom = max(1, len(gt_cls))
        recalls = tp_cum / denom
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = _voc_ap(recalls, precisions) if len(pred_cls) > 0 else 0.0
        aps.append(ap)

        tp_total += int(tp.sum())
        fp_total += int(fp.sum())

    precision = float(tp_total / max(1, tp_total + fp_total))
    recall = float(tp_total / max(1, gt_total))
    m_ap = float(np.mean(aps)) if aps else 0.0
    ap = m_ap
    return {"precision": precision, "recall": recall, "ap": ap, "map": m_ap}


def evaluate_coco_metrics(coco_gt_json: Path, coco_pred_json: Path) -> Dict[str, float]:
    """Compute standard COCO metrics: AP, AP50, AP75, APS, APM, APL."""
    coco_mod = importlib.import_module("pycocotools.coco")
    cocoeval_mod = importlib.import_module("pycocotools.cocoeval")
    COCO = coco_mod.COCO
    COCOeval = cocoeval_mod.COCOeval

    coco_gt = COCO(str(coco_gt_json))
    coco_dt = coco_gt.loadRes(str(coco_pred_json))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "AP50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "APS": float(evaluator.stats[3]),
        "APM": float(evaluator.stats[4]),
        "APL": float(evaluator.stats[5]),
    }


def _load_coco_as_records(gt_json: Path, pred_json: Path) -> Tuple[List[Dict], List[Dict]]:
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    pred = json.loads(pred_json.read_text(encoding="utf-8"))
    return gt.get("annotations", []), pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection metrics for MFEFNet")
    parser.add_argument("--gt-json", type=Path, required=True, help="COCO GT json with images/annotations/categories")
    parser.add_argument("--pred-json", type=Path, required=True, help="COCO prediction json list")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for basic Precision/Recall/AP/mAP")
    parser.add_argument("--skip-coco", action="store_true", help="Skip COCO AP evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_records, pred_records = _load_coco_as_records(args.gt_json, args.pred_json)

    basic = compute_basic_detection_metrics(gt_records=gt_records, pred_records=pred_records, iou_thres=args.iou)
    print("Basic metrics")
    print(f"Precision: {basic['precision']:.4f}")
    print(f"Recall:    {basic['recall']:.4f}")
    print(f"AP:        {basic['ap']:.4f}")
    print(f"mAP:       {basic['map']:.4f}")

    if not args.skip_coco:
        coco = evaluate_coco_metrics(args.gt_json, args.pred_json)
        print("\nCOCO metrics")
        print(f"AP50:95: {coco['AP50_95']:.4f}")
        print(f"AP50:    {coco['AP50']:.4f}")
        print(f"AP75:    {coco['AP75']:.4f}")
        print(f"AP_S:    {coco['APS']:.4f}")
        print(f"AP_M:    {coco['APM']:.4f}")
        print(f"AP_L:    {coco['APL']:.4f}")


if __name__ == "__main__":
    main()
