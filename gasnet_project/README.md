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

Linux/macOS:

```bash
docker run --gpus all --rm -it \
  -v /path/to/data_root:/data \
  -v /path/to/output:/workspace/output \
  trandoanthang/gasnet-h100:v2 \
  python train.py --data-root /data --epochs 60 --batch-size 512 --run-eval --eval-every 10
```

Windows PowerShell:

```powershell
docker run --gpus all --rm -it `
  -v D:\path\to\data_root:/data `
  -v D:\path\to\output:/workspace/output `
  trandoanthang/gasnet-h100:v2 `
  python train.py --data-root /data --epochs 60 --batch-size 512 --run-eval --eval-every 10
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

## 7) Important Arguments

- `--run-eval`: enable evaluation.
- `--eval-every N`: evaluate every N epochs. Use `0` to skip periodic eval and only run final eval.
- `--batch-size`: training batch size (default `512`).
- `--num-workers`: DataLoader workers (default `4`).
- `--no-compile`: disable `torch.compile` if needed for compatibility.

## 8) Outputs

Default checkpoint path inside container:

- `/workspace/output/gasnet_vru.pth`

Because `/workspace/output` is mounted to host, checkpoint and logs remain on host after container exits.

## 9) Quick Handoff Checklist

- Image pulled successfully: `trandoanthang/gasnet-h100:v2` (or local build `gasnet-h100:v2`)
- Dataset mount path contains `VRU/Pic` and `VRU/train_test_split`
- Run command includes `--run-eval`
- Terminal shows evaluation table for Small, Medium, Big with `mAP`, `Rank-1`, `Rank-5`
