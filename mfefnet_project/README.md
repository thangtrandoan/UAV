# MFEFNet Docker (H100 Ready)

This folder is prepared so others can train and evaluate with Docker on NVIDIA H100.

## 1) Build image

Run from this folder:

```bash
docker build -t mfefnet:h100 .
```

## 2) Train on H100

Linux:

```bash
docker run --gpus all --rm -it \
  -v /path/to/UAV:/workspace/data \
  -v /path/to/output:/workspace/output \
  mfefnet:h100 \
  python train.py \
    --data-root /workspace/data \
    --output-dir /workspace/output \
    --prepare-data \
    --h100-preset
```

Windows PowerShell:

```powershell
docker run --gpus all --rm -it `
  -v d:/UAV:/workspace/data `
  -v d:/UAV/mfefnet_project/output:/workspace/output `
  mfefnet:h100 `
  python train.py --data-root /workspace/data --output-dir /workspace/output --prepare-data --h100-preset
```

## 3) H100 preset

`--h100-preset` applies:

- epochs = 100
- batch_size = 64
- lr = 0.02
- num_workers = 8

## 4) Evaluate metrics

`evaluate.py` reports:

- Basic metrics: Precision, Recall, AP, mAP
- COCO metrics: AP50:95, AP50, AP75, AP_S, AP_M, AP_L

`train.py` now runs evaluation automatically after training and saves:

- `/workspace/output/eval/visdrone_val_gt_coco.json`
- `/workspace/output/eval/visdrone_val_pred_coco.json`
- `/workspace/output/eval/metrics.json`

If you only want training without evaluation, add `--skip-eval`.

Run evaluation with COCO-format json files:

```bash
docker run --gpus all --rm -it \
  -v /path/to/output:/workspace/output \
  mfefnet:h100 \
  python evaluate.py \
    --gt-json /workspace/output/visdrone_val_gt_coco.json \
    --pred-json /workspace/output/visdrone_val_pred_coco.json
```
