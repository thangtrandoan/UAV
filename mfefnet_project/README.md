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
  --data-root /workspace/data \
  --output-dir /workspace/output
```

Windows PowerShell:

```powershell
docker run --gpus all --rm -it `
  -v d:/UAV:/workspace/data `
  -v d:/UAV/mfefnet_project/output:/workspace/output `
  mfefnet:h100 `
  --data-root /workspace/data `
  --output-dir /workspace/output
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

`train.py` runs evaluation automatically after training and saves per split:

- `/workspace/output/eval/visdrone_val_gt_coco.json`
- `/workspace/output/eval/visdrone_val_pred_coco.json`
- `/workspace/output/eval/metrics_val.json`
- `/workspace/output/eval/visdrone_test_gt_coco.json`
- `/workspace/output/eval/visdrone_test_pred_coco.json`
- `/workspace/output/eval/metrics_test.json`

If you only want training without evaluation, add `--skip-eval`.

By default, evaluation reloads the best checkpoint. Disable with `--no-eval-best`.

You can choose evaluation splits with:

```bash
--eval-splits val,test
```

## 5) H100 speed knobs

Optional flags for more speed on H100:

- `--amp-dtype bf16` (default auto uses bf16 if supported)
- `--channels-last`
- `--compile --compile-mode max-autotune`

## 6) One-command entrypoint

The image uses an entrypoint that wraps `train.py` with sensible defaults:

- defaults: `--prepare-data --h100-preset --eval-splits val,test`
- override any setting by passing flags after the image name

Example:

```bash
docker run --gpus all --rm -it \
  -v /path/to/UAV:/workspace/data \
  -v /path/to/output:/workspace/output \
  mfefnet:h100 \
  --data-root /workspace/data \
  --output-dir /workspace/output \
  --eval-splits val,test \
  --amp-dtype bf16 --channels-last --compile
```

Disable defaults if needed:

```bash
--no-prepare-data --no-h100-preset --skip-eval --no-eval-best
```

## 7) Host driver/CUDA checklist

This image is based on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`.

1) Ensure NVIDIA driver is recent enough for CUDA 12.1.
   - Recommended: 535+ (H100 systems typically use 550+)
2) Install NVIDIA Container Toolkit on the host.
3) Validate GPU visibility before running training.

Quick checks:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the host driver is older than required, either upgrade the driver or rebuild the image on a lower CUDA base.

Run evaluation with COCO-format json files:

```bash
docker run --gpus all --rm -it \
  -v /path/to/output:/workspace/output \
  mfefnet:h100 \
  python evaluate.py \
    --gt-json /workspace/output/visdrone_val_gt_coco.json \
    --pred-json /workspace/output/visdrone_val_pred_coco.json
```
