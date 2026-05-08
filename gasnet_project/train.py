from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import VRUDataset, build_query_gallery, read_split_file
from evaluation import print_eval_report, run_eval
from utils import choose_amp_dtype, configure_cuda_for_speed, set_seed


BEST_METRIC_SPLIT_PRIORITY = {
    "big_map": ("Big", "Medium", "Small"),
    "medium_map": ("Medium", "Big", "Small"),
    "small_map": ("Small", "Medium", "Big"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GASNet on VRU")
    parser.add_argument("--data-root", type=Path, required=True, help="Path containing VRU folder")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-batch-size", type=int, default=128)
    parser.add_argument("--base-lr", type=float, default=3.5e-4)
    parser.add_argument("--grad-accum", type=int, default=1, help="Accumulate gradients over N steps")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--save-path", type=Path, default=Path("/workspace/output/gasnet_vru.pth"))
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-eval", action="store_true", help="Run retrieval evaluation on Small/Medium/Big test splits")
    parser.add_argument("--eval-every", type=int, default=0, help="Run evaluation every N epochs (0 = only final eval if --run-eval)")
    parser.add_argument("--save-best", dest="save_best", action="store_true", help="Save best checkpoint when evaluation runs")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false", help="Disable best-checkpoint saving")
    parser.set_defaults(save_best=None)
    parser.add_argument(
        "--best-metric",
        choices=("big_map", "medium_map", "small_map"),
        default="big_map",
        help="Metric used to select the best checkpoint (fallback order depends on selected split)",
    )
    parser.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained ResNet-50 weights")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--eval-q-chunk-size", type=int, default=2048, help="Query chunk size for retrieval evaluation")
    parser.add_argument("--no-fp16-sim", action="store_true", help="Disable fp16/bf16 similarity matmul during evaluation")
    parser.add_argument("--eval-verbose", action="store_true", help="Print progress while evaluating the Big split")
    return parser.parse_args()


class RGASpatial(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_nodes = spatial_size[0] * spatial_size[1]
        self.spatial_reduction = spatial_reduction
        inter_channels = max(1, channels // channel_reduction)
        relation_features = max(1, self.num_nodes // spatial_reduction)
        self.theta = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.g = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.relation = nn.Sequential(
            nn.Conv1d(self.num_nodes * 2, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(relation_features + 1, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(relation_features, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h * w != self.num_nodes:
            raise ValueError(f"RGASpatial expects spatial size {self.spatial_size}, got {(h, w)}")
        theta = self.theta(x).view(b, -1, self.num_nodes).transpose(1, 2)
        phi = self.phi(x).view(b, -1, self.num_nodes)
        rs = torch.bmm(theta, phi)
        relation = torch.cat([rs, rs.transpose(1, 2)], dim=2).transpose(1, 2)
        relation = self.relation(relation)
        spatial_context = self.g(x).view(b, -1, self.num_nodes).mean(dim=1, keepdim=True)
        attn = self.attn(torch.cat([relation, spatial_context], dim=1))
        attn = self.sigmoid(attn).view(b, 1, h, w)
        return x * attn


class RGAChannel(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_nodes = spatial_size[0] * spatial_size[1]
        inter_spatial = max(1, self.num_nodes // spatial_reduction)
        relation_features = max(1, channels // channel_reduction)
        self.theta = nn.Sequential(
            nn.Conv1d(self.num_nodes, inter_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_spatial),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Conv1d(self.num_nodes, inter_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_spatial),
            nn.ReLU(inplace=True),
        )
        self.relation = nn.Sequential(
            nn.Conv1d(channels * 2, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
        )
        self.channel_embed = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(relation_features + 1, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(relation_features, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h * w != self.num_nodes:
            raise ValueError(f"RGAChannel expects spatial size {self.spatial_size}, got {(h, w)}")
        x_flat = x.view(b, c, self.num_nodes)
        x_perm = x_flat.permute(0, 2, 1)
        theta = self.theta(x_perm).permute(0, 2, 1)
        phi = self.phi(x_perm)
        rc = torch.bmm(theta, phi)
        relation = torch.cat([rc, rc.transpose(1, 2)], dim=2).transpose(1, 2)
        relation = self.relation(relation)
        channel_context = self.channel_embed(x).mean(dim=(2, 3)).unsqueeze(1)
        attn = self.attn(torch.cat([relation, channel_context], dim=1))
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return x * attn


class RGABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.rga_s = RGASpatial(
            channels=channels,
            spatial_size=spatial_size,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction,
        )
        self.rga_c = RGAChannel(
            channels=channels,
            spatial_size=spatial_size,
            spatial_reduction=spatial_reduction,
            channel_reduction=channel_reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rga_s(x)
        x = self.rga_c(x)
        return x


class AggregationGate(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, 4, 1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        stacked = torch.cat(feats, dim=1)
        weights = self.gate(stacked)
        out = sum(feats[i] * weights[:, i : i + 1] for i in range(4))
        return out


class GASNet(nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        base = models.resnet50(weights=weights)
        
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        self.ga1 = RGABlock(256, spatial_size=(56, 56))
        self.ga2 = RGABlock(512, spatial_size=(28, 28))
        self.ga3 = RGABlock(1024, spatial_size=(14, 14))
        self.ga4 = RGABlock(2048, spatial_size=(7, 7))
        
        self.fs1 = AggregationGate(1024, 512)
        self.fs2 = AggregationGate(512, 512)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck_global = nn.BatchNorm1d(2048)
        self.bnneck_fs = nn.BatchNorm1d(512)
        self.classifier_global = nn.Linear(2048, num_classes, bias=False)
        self.classifier_fs = nn.Linear(512, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.ga1(x)

        x = self.layer2(x)
        x = self.ga2(x)

        x = self.layer3(x)
        fs = self.fs1(x)
        fs = self.fs2(fs)
        x = self.ga3(x)

        x = self.layer4(x)
        x = self.ga4(x)

        global_feat = self.gap(x).flatten(1)
        fs_feat = self.gap(fs).flatten(1)
        bn_global = self.bnneck_global(global_feat)
        bn_fs = self.bnneck_fs(fs_feat)

        if self.training:
            return (global_feat, fs_feat), (self.classifier_global(bn_global), self.classifier_fs(bn_fs))
        return (global_feat, fs_feat), (bn_global, bn_fs)


def batch_hard_triplet_loss(feat: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    dist = torch.cdist(feat, feat, p=2)
    labels = labels.view(-1, 1)
    mask_pos = labels.eq(labels.t())
    mask_neg = ~mask_pos

    dist_pos = dist.clone()
    dist_pos[~mask_pos] = -1.0
    dist_pos.fill_diagonal_(-1.0)
    hardest_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist.clone()
    dist_neg[~mask_neg] = 1e9
    hardest_neg, _ = dist_neg.min(dim=1)

    valid = hardest_pos > -0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=feat.device)
    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


def make_loader(
    samples: List,
    transform,
    batch_size: int,
    is_train: bool,
    relabel=False,
    label_map=None,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    ds = VRUDataset(samples=samples, transform=transform, relabel=relabel, label_map=label_map, max_retries=3)
    loader_kwargs = dict(
        pin_memory=pin_memory,
        drop_last=is_train,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        **loader_kwargs,
    )


def _best_checkpoint_path(save_path: Path) -> Path:
    return save_path.with_name(f"{save_path.stem}.best{save_path.suffix}")


def _select_metric(
    metrics_by_split: dict,
    best_metric: str,
) -> tuple[float, str] | tuple[None, None]:
    for split_name in BEST_METRIC_SPLIT_PRIORITY[best_metric]:
        split_metrics = metrics_by_split.get(split_name)
        if split_metrics is not None:
            return float(split_metrics[0]), split_name
    return None, None


def _maybe_update_best_checkpoint(
    model: nn.Module,
    metrics_by_split: dict,
    best_metric: str,
    epoch: int,
    best_score: float | None,
    best_path: Path,
) -> tuple[float | None, int | None, str | None]:
    metric_value, metric_split = _select_metric(metrics_by_split, best_metric)
    if metric_value is not None and (best_score is None or metric_value > best_score):
        torch.save(model.state_dict(), best_path)
        print(f"Updated best checkpoint at epoch {epoch}: {metric_split} mAP={metric_value:.4f} -> {best_path}")
        return metric_value, epoch, metric_split
    return best_score, None, None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_cuda_for_speed()

    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")

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
    use_scaler = use_amp and amp_dtype == torch.float16
    pin_memory = torch.cuda.is_available() and (not args.no_pin_memory)
    persistent_workers = (not args.no_persistent_workers) and args.num_workers > 0
    prefetch_factor = args.prefetch_factor
    if args.save_best is None:
        save_best = args.run_eval
    else:
        save_best = args.run_eval and args.save_best

    vru_dir = args.data_root / "VRU"
    pic_dir = vru_dir / "Pic"
    split_dir = vru_dir / "train_test_split"

    train_samples = read_split_file(pic_dir, split_dir / "train_list.txt")
    train_ids = sorted({s.vehicle_id for s in train_samples})
    train_label_map = {vid: i for i, vid in enumerate(train_ids)}
    num_classes = len(train_label_map)

    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = args.batch_size
    learning_rate = args.base_lr * (batch_size / args.base_batch_size)

    train_loader = make_loader(
        train_samples,
        train_tfms,
        batch_size,
        True,
        relabel=True,
        label_map=train_label_map,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = GASNet(num_classes=num_classes, use_pretrained=not args.no_pretrained).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if torch.cuda.is_available() and hasattr(torch, "compile") and (not args.no_compile):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # New API first, old API fallback for compatibility.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f"device={device}, amp={use_amp}, amp_dtype={amp_dtype}, channels_last={use_channels_last}, "
        f"batch={batch_size}, lr={learning_rate:.6f}, grad_accum={args.grad_accum}, "
        f"workers={args.num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}"
    )

    model.train()
    ce_loss = nn.CrossEntropyLoss()
    best_score = None
    best_epoch = None
    best_split = None
    best_path = None
    if save_best:
        best_path = _best_checkpoint_path(args.save_path)
        best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, labels, _, _) in enumerate(train_loader, 1):
            if use_channels_last:
                imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                (g_feat, f_feat), (logits_g, logits_f) = model(imgs)
                loss_id = ce_loss(logits_g, labels) + ce_loss(logits_f, labels)
                loss_tri = batch_hard_triplet_loss(g_feat, labels) + batch_hard_triplet_loss(f_feat, labels)
                loss = loss_id + loss_tri

            loss_to_backprop = loss / args.grad_accum
            if use_amp and use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if step % args.grad_accum == 0:
                if use_amp and use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * imgs.size(0)

            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        if step % args.grad_accum != 0:
            if use_amp and use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = running / len(train_loader.dataset)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, time={(time.time() - t0)/60:.1f} min")

        should_eval_this_epoch = args.run_eval and args.eval_every > 0 and (epoch % args.eval_every == 0)
        if should_eval_this_epoch:
            metrics = run_eval(
                model=model,
                data_root=args.data_root,
                test_transform=test_tfms,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
                q_chunk_size=args.eval_q_chunk_size,
                use_fp16_sim=(not args.no_fp16_sim),
                verbose_eval=args.eval_verbose,
            )
            print_eval_report(metrics, title=f"Evaluation @ Epoch {epoch}")
            if save_best:
                best_score, maybe_best_epoch, maybe_best_split = _maybe_update_best_checkpoint(
                    model=model,
                    metrics_by_split=metrics,
                    best_metric=args.best_metric,
                    epoch=epoch,
                    best_score=best_score,
                    best_path=best_path,
                )
                if maybe_best_epoch is not None:
                    best_epoch = maybe_best_epoch
                    best_split = maybe_best_split

    if args.run_eval:
        metrics = run_eval(
            model=model,
            data_root=args.data_root,
            test_transform=test_tfms,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
            q_chunk_size=args.eval_q_chunk_size,
            use_fp16_sim=(not args.no_fp16_sim),
            verbose_eval=args.eval_verbose,
        )
        print_eval_report(metrics, title="Final Evaluation")
        if save_best:
            best_score, maybe_best_epoch, maybe_best_split = _maybe_update_best_checkpoint(
                model=model,
                metrics_by_split=metrics,
                best_metric=args.best_metric,
                epoch=args.epochs,
                best_score=best_score,
                best_path=best_path,
            )
            if maybe_best_epoch is not None:
                best_epoch = maybe_best_epoch
                best_split = maybe_best_split

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")
    if save_best:
        if best_score is not None:
            print(f"Best checkpoint summary: epoch={best_epoch}, split={best_split}, mAP={best_score:.4f}, path={best_path}")
        else:
            print("Best checkpoint summary: no evaluation metrics available, best checkpoint not saved")


if __name__ == "__main__":
    main()
