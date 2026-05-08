# GASNet Docker Handoff Guide

This document is for teammates who need to run training and evaluation with Docker.

## 1) Requirements

- NVIDIA GPU with Docker GPU runtime enabled.
- Docker Desktop (or Docker Engine) that supports `--gpus all`.
- VRU dataset available on host machine.

Base image used by this project:

- `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`

## 2) Expected Data Layout

The mounted data root must contain `VRU`:

```text
<DATA_ROOT>/VRU/
  Pic/
  train_test_split/
    train_list.txt
    test_list_1200.txt
    test_list_2400.txt
    test_list_8000.txt
```

## 3) Pull Image (Recommended)

Pull from Docker Hub:

```bash
docker pull trandoanthang/gasnet-h100:v2
```

Alternative tag:

```bash
docker pull trandoanthang/gasnet-h100:latest
```

## 4) Build Image (Optional)

If you want to build locally, run in this folder (`gasnet_project`):

```bash
docker build -t gasnet-h100:v2 .
```

## 5) Run Training + Evaluation

The container runs data checks, prepares environment info, trains, and evaluates on Small/Medium/Big splits at startup.
Pass the VRU root via `--data-root` (or mount it at `/data` and omit the flag).
Use `--output-dir` to control where the checkpoint is saved (default `/workspace/output`).
If the target machine is offline, keep the pretrained ResNet-50 weights cached in the image (default). You can also disable pretrained weights with `--no-pretrained`.

Linux/macOS:

```bash
docker run --gpus all --rm -it \
  -v /path/to/data_root:/data \
  -v /path/to/output:/workspace/output \
  trandoanthang/gasnet-h100:v2 \
  --data-root /data --epochs 60 --batch-size 512 --eval-every 10
```

Windows PowerShell:

```powershell
docker run --gpus all --rm -it `
  -v D:\path\to\data_root:/data `
  -v D:\path\to\output:/workspace/output `
  trandoanthang/gasnet-h100:v2 `
  --data-root /data --epochs 60 --batch-size 512 --eval-every 10
```

## 5.1) H100-Optimized Settings (Recommended)

Linux/macOS:

```bash
docker run --gpus all --rm -it \
  -v /path/to/data_root:/data \
  -v /path/to/output:/workspace/output \
  trandoanthang/gasnet-h100:v2 \
  --data-root /data --epochs 60 --batch-size 512 \
    --amp-dtype bf16 --grad-accum 1 --num-workers 8 --prefetch-factor 4 \
    --eval-every 10
```

Windows PowerShell:

```powershell
docker run --gpus all --rm -it `
  -v D:\path\to\data_root:/data `
  -v D:\path\to\output:/workspace/output `
  trandoanthang/gasnet-h100:v2 `
  --data-root /data --epochs 60 --batch-size 512 `
    --amp-dtype bf16 --grad-accum 1 --num-workers 8 --prefetch-factor 4 `
    --eval-every 10
```

## 6) What Evaluation Prints

With `--run-eval`, the script evaluates all 3 VRU subsets:

- Small (`test_list_1200.txt`)
- Medium (`test_list_2400.txt`)
- Big (`test_list_8000.txt`)

For each subset, it prints:

- `mAP`
- `Rank-1`
- `Rank-5`

For the `Big` split, evaluation now runs in query chunks instead of materializing a full query×gallery matrix. This reduces peak memory and avoids long "no output" periods. Enable `--eval-verbose` to print chunk progress.

## 7) Important Arguments

- `--run-eval`: enable evaluation.
- `--eval-every N`: evaluate every N epochs. Use `0` to skip periodic eval and only run final eval.
- `--batch-size`: training batch size (default `512`).
- `--num-workers`: DataLoader workers (default `8`).
- `--prefetch-factor`: DataLoader prefetch factor per worker (default `4`).
- `--grad-accum`: accumulate gradients over N steps for large effective batches.
- `--amp-dtype`: `auto` (default), `bf16`, or `fp16`.
- `--no-amp`: disable AMP.
- `--no-channels-last`: disable channels-last layout.
- `--no-pin-memory`: disable pinned memory for loaders.
- `--no-persistent-workers`: disable persistent workers for loaders.
- `--no-compile`: disable `torch.compile` if needed for compatibility.
- `--eval-q-chunk-size`: query block size used by chunked retrieval evaluation (default `2048`).
- `--no-fp16-sim`: disable bf16/fp16 similarity matmul during evaluation.
- `--eval-verbose`: print progress while evaluating `Big` split.

## 8) Outputs

Default checkpoint path inside container:

- `/workspace/output/gasnet_vru.pth`

Because `/workspace/output` is mounted to host, checkpoint and logs remain on host after container exits.

## 9) Quick Handoff Checklist

- Image pulled successfully: `trandoanthang/gasnet-h100:v2` (or local build `gasnet-h100:v2`)
- Dataset mount path contains `VRU/Pic` and `VRU/train_test_split`
- Run command includes `--run-eval`
- Terminal shows evaluation table for Small, Medium, Big with `mAP`, `Rank-1`, `Rank-5`
