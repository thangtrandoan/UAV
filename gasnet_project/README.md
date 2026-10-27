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
- `--save-best` / `--no-save-best`: enable/disable saving `*.best.pth` when eval runs (enabled by default with `--run-eval`).
- `--best-metric`: metric for best checkpoint selection (`big_map` default, `medium_map`, `small_map`).
- `--gate {bcgg,rga}`: stage gate block. Default `bcgg` (new); use `rga` for the legacy block/checkpoints.
- `--gate-hidden` (default `16`), `--gate-groups` (default `4`), `--gate-kernel` (default `7`), `--gate-channel-kernel` (default `5`): BCGG shape controls.
- `--gate-inject-value`: BCGG ablation that adds the `P_value(u)` additive refinement branch.

## 8) Stage gate: BCGG vs RGA

`bcgg.py` implements **Bottleneck Context Group Gate** (a research proposal, not a
published module). It replaces the `RGABlock` interleaved after every backbone stage:

```text
x ──► spatial: 1x1 Conv-BN-ReLU (C→d) → RepDW 7x7 (d) → 1x1 Conv (d→G) → [B,G,1,H,W]
  └─► channel: GAP → Conv1d over channel axis (k=5) → [B,C] → [B,G,C/G,1,1]
gate = 2 * sigmoid(channel + spatial);  y = x * gate
```

Key properties:

- **Identity at init**: both logit heads are zero-initialised, so `gate = 2*sigmoid(0) = 1`
  and the block changes nothing until it learns. Safe to interleave with pretrained stages.
- **Cheap**: the gate params are ~66k for ResNet-50 vs ~14.6M for the 4 RGA blocks,
  and it never builds the `N x N` relation matrices (`N = H*W`) that dominate RGA.
- **Deployable**: `GASNet.to_deploy()` (also used by `export_onnx.py`) folds Conv+BN and
  the three RepDW branches into one depthwise conv; the exported gate sub-graph keeps a
  single `Sigmoid` + two `Mul` and no gate `BatchNormalization`.
- **No fixed spatial size**: unlike RGA, BCGG works at any resolution, so the
  hard-coded `(56,56)..(7,7)` assumptions are gone (RGA still uses them).
- **Not checkpoint-compatible with RGA**: the gate parameters differ, so switching a
  trained model requires retraining (RGA stays available for reproducing old runs).

`evaluate_vrai.py`, `infer.py` and `export_onnx.py` auto-detect the gate type (and the
BCGG hyper-parameters) from the checkpoint, so old RGA checkpoints still load unchanged.
Attention heatmaps work for both blocks.

## 9) Outputs

Default checkpoint path inside container:

- `/workspace/output/gasnet_vru.pth`
- `/workspace/output/gasnet_vru.best.pth` (best checkpoint, when `--run-eval` and best saving is enabled)

Evaluation metrics are printed to stdout/stderr (redirect logs if you want to persist them to a file).

Because `/workspace/output` is mounted to host, checkpoints and redirected logs remain on host after container exits.

## 10) Quick Handoff Checklist

- Image pulled successfully: `trandoanthang/gasnet-h100:v2` (or local build `gasnet-h100:v2`)
- Dataset mount path contains `VRU/Pic` and `VRU/train_test_split`
- Run command includes `--run-eval`
- Terminal shows evaluation table for Small, Medium, Big with `mAP`, `Rank-1`, `Rank-5`
