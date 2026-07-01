from __future__ import annotations

import argparse
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from dataset import VRUDataset, build_query_gallery, read_split_file
from evaluation import print_eval_report, run_eval
from utils import choose_amp_dtype, configure_cuda_for_speed, set_seed


BEST_METRIC_SPLIT_PRIORITY = {
    "big_map": ("Big", "Medium", "Small"),
    "medium_map": ("Medium", "Big", "Small"),
    "small_map": ("Small", "Medium", "Big"),
}


@dataclass
class VRAITrainSample:
    img_path: Path
    image_name: str
    vehicle_id: int
    camera: str
    color: int
    vehicle_type: int


def _parse_vrai_train_id(image_name: str) -> int:
    return int(Path(image_name).stem.split("_", 1)[0])


def _parse_vrai_train_camera(image_name: str) -> str:
    parts = Path(image_name).stem.split("_")
    return parts[1] if len(parts) > 1 else ""


def _lookup_attr(labels: dict, vehicle_id: int, image_name: str, default: int = -1) -> int:
    if vehicle_id in labels:
        return int(labels[vehicle_id])
    if str(vehicle_id) in labels:
        return int(labels[str(vehicle_id)])
    if image_name in labels:
        return int(labels[image_name])
    return default


def read_vrai_train_samples(vrai_dir: Path) -> List[VRAITrainSample]:
    annotation_path = vrai_dir / "train_annotation.pkl"
    images_dir = vrai_dir / "images_train"
    with annotation_path.open("rb") as f:
        annotation = pickle.load(f)
    image_names = annotation["train_im_names"]
    samples: List[VRAITrainSample] = []
    for name in image_names:
        vehicle_id = _parse_vrai_train_id(name)
        samples.append(
            VRAITrainSample(
                img_path=images_dir / name,
                image_name=name,
                vehicle_id=vehicle_id,
                camera=_parse_vrai_train_camera(name),
                color=_lookup_attr(annotation.get("color_label", {}), vehicle_id, name),
                vehicle_type=_lookup_attr(annotation.get("type_label", {}), vehicle_id, name),
            )
        )
    return samples


class VRAITrainDataset(Dataset):
    def __init__(self, samples: List[VRAITrainSample], transform=None, label_map=None, max_retries: int = 3):
        self.samples = samples
        self.transform = transform
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
            except OSError as exc:
                last_err = exc
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
                label = self.label_map[s.vehicle_id]
                return img, label, s.vehicle_id, s.image_name, s.camera, s.color, s.vehicle_type
            except OSError:
                tried += 1
                cur_idx = (cur_idx + 1) % len(self.samples)
        raise RuntimeError(f"Cannot read image after retries, start_idx={idx}, path={self.samples[idx].img_path}")


def _concat_relation(relation: torch.Tensor) -> torch.Tensor:
    return torch.cat([relation, relation.transpose(1, 2)], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GASNet on VRU")
    parser.add_argument("--data-root", type=Path, required=True, help="Path containing VRU folder")
    parser.add_argument("--dataset", choices=["vru", "vrai"], default="vru", help="Training dataset layout")
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
    parser.add_argument("--eval-rerank", action="store_true", help="Apply lightweight re-ranking during VRU evaluation")
    parser.add_argument("--eval-rerank-k1", type=int, default=20)
    parser.add_argument("--eval-rerank-alpha", type=float, default=0.3)
    parser.add_argument("--eval-tta-flip", action="store_true", help="Average original and horizontal-flip features during VRU evaluation")
    parser.add_argument("--strong-aug", action="store_true", help="Use stronger domain-randomization augmentation for training")
    parser.add_argument("--pk-k", type=int, default=4, help="Images per identity in PK sampler")
    parser.add_argument("--use-gem", action="store_true", help="Use GeM pooling instead of average pooling")
    parser.add_argument("--gem-p", type=float, default=3.0, help="GeM pooling exponent")
    parser.add_argument("--backbone", choices=["resnet50", "resnet50_ibn", "swin_t"], default="resnet50")
    parser.add_argument("--disable-camera-balanced-sampler", action="store_true")
    parser.add_argument("--disable-attribute-hard-negative-sampler", action="store_true")
    parser.add_argument("--use-part-branch", action="store_true", help="Enable unsupervised horizontal part feature branch")
    parser.add_argument("--num-parts", type=int, default=4, help="Number of horizontal stripes for the part branch")
    parser.add_argument("--attention-reg-weight", type=float, default=0.0, help="Weight for spatial attention regularization")
    parser.add_argument("--attention-target-mean", type=float, default=0.55, help="Target mean activation for spatial attention maps")
    parser.add_argument("--attention-std-margin", type=float, default=0.08, help="Minimum per-map std encouraged for attention maps")
    parser.add_argument("--use-attention-local", action="store_true", help="Enable attention-guided local feature branch (replaces part branch)")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention maps for local branch (1 or 4 for ablation)")
    parser.add_argument("--attn-diversity-weight", type=float, default=0.0, help="Weight for diversity loss between attention heads (0=off)")
    parser.add_argument("--attn-vis-dir", type=Path, default=None, help="Directory to save attention map visualizations")
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
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h * w != self.num_nodes:
            raise ValueError(
                f"Expected spatial dimensions {self.spatial_size} (total={self.num_nodes} nodes), "
                f"but got {(h, w)} (total={h * w} nodes)"
            )
        theta = self.theta(x).view(b, -1, self.num_nodes).transpose(1, 2)
        phi = self.phi(x).view(b, -1, self.num_nodes)
        spatial_relation = torch.bmm(theta, phi)
        relation = _concat_relation(spatial_relation)
        relation = self.relation(relation)
        channel_pooled = self.g(x).flatten(2).mean(dim=1, keepdim=True)
        attn = self.attn(torch.cat([relation, channel_pooled], dim=1))
        attn = self.sigmoid(attn).view(b, 1, h, w)
        self.last_attention = attn
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
            raise ValueError(
                f"Expected spatial dimensions {self.spatial_size} (total={self.num_nodes} nodes), "
                f"but got {(h, w)} (total={h * w} nodes)"
            )
        x_flat = x.view(b, c, self.num_nodes)
        x_perm = x_flat.permute(0, 2, 1)
        theta = self.theta(x_perm).permute(0, 2, 1)
        phi = self.phi(x_perm)
        channel_relation = torch.bmm(theta, phi)
        relation = _concat_relation(channel_relation)
        relation = self.relation(relation)
        spatial_pooled = self.channel_embed(x).mean(dim=(2, 3)).unsqueeze(1)
        attn = self.attn(torch.cat([relation, spatial_pooled], dim=1))
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


class Lite3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gate(x)


class OSBlockFS(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16):
        super().__init__()
        mid_ch = out_ch

        self.stream1 = nn.Sequential(
            Lite3x3(in_ch, mid_ch)
        )
        self.stream2 = nn.Sequential(
            Lite3x3(in_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch)
        )
        self.stream3 = nn.Sequential(
            Lite3x3(in_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch)
        )
        self.stream4 = nn.Sequential(
            Lite3x3(in_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch),
            Lite3x3(mid_ch, mid_ch)
        )

        # shared AG
        self.gate = ChannelGate(mid_ch, reduction)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)

        out = (
            self.gate(s1) * s1 +
            self.gate(s2) * s2 +
            self.gate(s3) * s3 +
            self.gate(s4) * s4
        )

        out = out + self.shortcut(x)
        return self.relu(out)


class BNNeck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), kernel_size=x.shape[-2:]).pow(1.0 / self.p)


class AttentionLocalBranch(nn.Module):
    """Attention-guided local feature learning for UAV Vehicle Re-ID.

    Learns `num_heads` spatial attention maps that weight the backbone
    feature map with residual attention ``x * (1 + att)``, producing
    local features focusing on discriminative vehicle regions without
    fixed horizontal partitioning.
    """

    def __init__(self, in_channels: int, num_heads: int = 4, reduction: int = 16):
        super().__init__()
        self.num_heads = num_heads
        mid = max(1, in_channels // reduction)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_heads, kernel_size=1),
            nn.Sigmoid(),
        )
        # Keep gradient flow - no detach
        self.last_attention: torch.Tensor | None = None

    def forward(
        self, feature_map: torch.Tensor, pool_fn: nn.Module
    ) -> list[torch.Tensor]:
        attention = self.attention(feature_map)      # [B, num_heads, H, W]
        self.last_attention = attention               # gradient flows through

        local_features = []
        for i in range(self.num_heads):
            att = attention[:, i : i + 1, :, :]      # [B, 1, H, W]
            weighted = feature_map * (1 + att)        # residual attention
            pooled = pool_fn(weighted).flatten(1)     # [B, 2048]
            local_features.append(pooled)
        return local_features

    def diversity_loss(self) -> torch.Tensor:
        """Orthogonality loss to encourage diverse attention maps."""
        if self.last_attention is None:
            return torch.tensor(0.0)
        att = self.last_attention                     # [B, K, H, W]
        B, K, H, W = att.shape
        flat = att.view(B, K, -1)                     # [B, K, H*W]
        flat = F.normalize(flat, dim=2)
        gram = torch.bmm(flat, flat.transpose(1, 2))  # [B, K, K]
        eye = torch.eye(K, device=gram.device).unsqueeze(0)
        return ((gram - eye) ** 2).mean()


class IBN(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.5):
        super().__init__()
        split = int(channels * ratio)
        self.split = split
        self.in_norm = nn.InstanceNorm2d(split, affine=True)
        self.bn_norm = nn.BatchNorm2d(channels - split)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.split, x.size(1) - self.split], dim=1)
        return torch.cat([self.in_norm(x1), self.bn_norm(x2)], dim=1)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_ibn: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if use_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNetIBN(nn.Module):
    def __init__(self, layers: tuple[int, int, int, int] = (3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], use_ibn=True)
        self.layer2 = self._make_layer(128, layers[1], stride=2, use_ibn=True)
        self.layer3 = self._make_layer(256, layers[2], stride=2, use_ibn=True)
        self.layer4 = self._make_layer(512, layers[3], stride=2, use_ibn=False)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, use_ibn: bool = True) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckIBN.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BottleneckIBN.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BottleneckIBN.expansion),
            )

        layers = [BottleneckIBN(self.inplanes, planes, stride=stride, downsample=downsample, use_ibn=use_ibn)]
        self.inplanes = planes * BottleneckIBN.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckIBN(self.inplanes, planes, use_ibn=use_ibn))
        return nn.Sequential(*layers)


def _load_matching_resnet50_weights(model: nn.Module) -> None:
    weights = models.ResNet50_Weights.DEFAULT
    state = models.resnet50(weights=weights).state_dict()
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    missing = len(model_state) - len(filtered)
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} matching ResNet50 weights into ResNet50-IBN; skipped {missing} tensors")


class SwinBackbone(nn.Module):
    """Swin-T backbone with 1x1 Conv channel adapters for GASNet compatibility.

    Extracts hierarchical features from 4 Swin stages and projects channels
    to match ResNet50 dimensions (256, 512, 1024, 2048) so that the
    downstream FS branch, Attention Local Branch, and classifiers
    work without modification.
    """

    # Swin-T channel sizes per stage
    SWIN_CHANNELS = (96, 192, 384, 768)
    # Target ResNet-equivalent channel sizes
    TARGET_CHANNELS = (256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        base = swin_t(weights=weights)
        if pretrained:
            print("Loaded Swin-T pretrained weights")

        # Split features into 4 stages.
        # features layout: [patch_embed, stage1_blocks, patch_merge1,
        #   stage2_blocks, patch_merge2, stage3_blocks, patch_merge3,
        #   stage4_blocks]
        self.stage1 = base.features[0:2]   # 96ch,  56x56
        self.stage2 = base.features[2:4]   # 192ch, 28x28
        self.stage3 = base.features[4:6]   # 384ch, 14x14
        self.stage4 = base.features[6:8]   # 768ch, 7x7
        self.norm = base.norm              # LayerNorm(768)

        # 1x1 Conv adapters: Swin channels -> ResNet channels
        adapters = []
        for src_ch, tgt_ch in zip(self.SWIN_CHANNELS, self.TARGET_CHANNELS):
            adapters.append(
                nn.Sequential(
                    nn.Conv2d(src_ch, tgt_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(tgt_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.adapt1, self.adapt2, self.adapt3, self.adapt4 = adapters

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Swin stages output [B, H, W, C]; permute to [B, C, H, W] for Conv2d.
        x = self.stage1(x)
        feat1 = self.adapt1(x.permute(0, 3, 1, 2).contiguous())

        x = self.stage2(x)
        feat2 = self.adapt2(x.permute(0, 3, 1, 2).contiguous())

        x = self.stage3(x)
        feat3 = self.adapt3(x.permute(0, 3, 1, 2).contiguous())

        x = self.stage4(x)
        x = self.norm(x)  # final LayerNorm before last adapter
        feat4 = self.adapt4(x.permute(0, 3, 1, 2).contiguous())

        return feat1, feat2, feat3, feat4


class GASNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool = True,
        use_part_branch: bool = False,
        num_parts: int = 4,
        use_gem: bool = False,
        gem_p: float = 3.0,
        backbone: str = "resnet50",
        use_attention_local: bool = False,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        if num_parts < 1:
            raise ValueError("num_parts must be >= 1")
        self.use_part_branch = use_part_branch
        self.use_attention_local = use_attention_local
        self.num_parts = num_parts
        self.backbone_type = backbone
        if backbone == "swin_t":
            self.swin_backbone = SwinBackbone(pretrained=use_pretrained)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
            base = models.resnet50(weights=weights)
        elif backbone == "resnet50_ibn":
            base = ResNetIBN()
            if use_pretrained:
                _load_matching_resnet50_weights(base)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if backbone != "swin_t":
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.ga1 = RGABlock(256, spatial_size=(56, 56))
            self.ga2 = RGABlock(512, spatial_size=(28, 28))
            self.ga3 = RGABlock(1024, spatial_size=(14, 14))
            self.ga4 = RGABlock(2048, spatial_size=(7, 7))
        
        self.fs1 = OSBlockFS(1024, 512)
        self.fs2 = OSBlockFS(512, 512)
        
        self.gap = GeMPool(p=gem_p) if use_gem else nn.AdaptiveAvgPool2d(1)
        self.bnneck_global = BNNeck(2048)
        self.bnneck_fs = BNNeck(512)
        if self.use_part_branch:
            self.part_reduce = nn.Sequential(
                nn.Linear(2048 * num_parts, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
            )
            self.bnneck_part = BNNeck(512)
            self.classifier_part = nn.Linear(512, num_classes, bias=False)
        if self.use_attention_local:
            self.attn_local = AttentionLocalBranch(
                in_channels=2048,
                num_heads=num_attention_heads,
            )
            # Per-head reduction: 2048 -> 256
            self.attn_head_reduce = nn.ModuleList([
                nn.Linear(2048, 256, bias=False)
                for _ in range(num_attention_heads)
            ])
            # Concat reduction: 256*num_heads -> 512
            self.attn_local_reduce = nn.Sequential(
                nn.BatchNorm1d(256 * num_attention_heads),
                nn.ReLU(inplace=True),
                nn.Linear(256 * num_attention_heads, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
            )
            self.bnneck_attn_local = BNNeck(512)
            self.classifier_attn_local = nn.Linear(512, num_classes, bias=False)
        
        self.classifier_global = nn.Linear(2048, num_classes, bias=False)
        self.classifier_fs = nn.Linear(512, num_classes, bias=False)

    def _part_pool(self, x: torch.Tensor) -> torch.Tensor:
        stripes = torch.chunk(x, self.num_parts, dim=2)
        pooled = [self.gap(stripe).flatten(1) for stripe in stripes]
        return self.part_reduce(torch.cat(pooled, dim=1))

    def attention_regularization_loss(
        self,
        target_mean: float = 0.55,
        std_margin: float = 0.08,
    ) -> torch.Tensor:
        # Swin backbone does not use RGA blocks (has built-in self-attention)
        if self.backbone_type == "swin_t":
            return next(self.parameters()).sum() * 0.0
        losses = []
        for block in (self.ga1, self.ga2, self.ga3, self.ga4):
            attn = block.rga_s.last_attention
            if attn is None:
                continue
            flat = attn.flatten(1)
            mean_loss = (flat.mean(dim=1) - target_mean).pow(2)
            std_loss = F.relu(std_margin - flat.std(dim=1, unbiased=False)).pow(2)
            losses.append((mean_loss + std_loss).mean())
        if not losses:
            return next(self.parameters()).sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, x: torch.Tensor):
        if self.backbone_type == "swin_t":
            feat1, feat2, feat3, feat4 = self.swin_backbone(x)
            # Swin already has self-attention; skip RGA blocks
            fs = self.fs1(feat3)
            fs = self.fs2(fs)
            x = feat4
        else:
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

        logits_global = self.classifier_global(bn_global)
        logits_fs = self.classifier_fs(bn_fs)

        if self.use_attention_local:
            local_feats = self.attn_local(x, self.gap)  # list of [B, 2048]
            reduced = [
                self.attn_head_reduce[i](feat)
                for i, feat in enumerate(local_feats)
            ]  # list of [B, 256]
            attn_local_feat = self.attn_local_reduce(
                torch.cat(reduced, dim=1)
            )  # [B, 512]
            bn_attn_local = self.bnneck_attn_local(attn_local_feat)
            logits_attn_local = self.classifier_attn_local(bn_attn_local)

        if self.use_part_branch and not self.use_attention_local:
            part_feat = self._part_pool(x)
            bn_part = self.bnneck_part(part_feat)
            logits_part = self.classifier_part(bn_part)

        if self.training:
            if self.use_attention_local:
                return (global_feat, fs_feat, attn_local_feat), (logits_global, logits_fs, logits_attn_local)
            if self.use_part_branch:
                return (global_feat, fs_feat, part_feat), (logits_global, logits_fs, logits_part)
            return (global_feat, fs_feat), (logits_global, logits_fs)
        
        if self.use_attention_local:
            return bn_global, bn_fs, bn_attn_local
        if self.use_part_branch:
            return bn_global, bn_fs, bn_part
        return bn_global, bn_fs


def batch_hard_triplet_loss(
    feat: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    cameras=None,
    colors: torch.Tensor | None = None,
    vehicle_types: torch.Tensor | None = None,
) -> torch.Tensor:
    dist = torch.cdist(feat, feat, p=2)
    labels = labels.view(-1, 1)
    mask_pos = labels.eq(labels.t())
    mask_neg = ~mask_pos
    if cameras is not None:
        cam_list = list(cameras)
        cam_equal = torch.tensor(
            [[a == b for b in cam_list] for a in cam_list],
            dtype=torch.bool,
            device=feat.device,
        )
        cross_camera_pos = mask_pos & (~cam_equal)
        has_cross_camera_pos = cross_camera_pos.sum(dim=1) > 0
        mask_pos = torch.where(has_cross_camera_pos.view(-1, 1), cross_camera_pos, mask_pos)

    if colors is not None and vehicle_types is not None:
        colors = colors.to(feat.device).view(-1, 1)
        vehicle_types = vehicle_types.to(feat.device).view(-1, 1)
        same_attr_neg = mask_neg & colors.eq(colors.t()) & vehicle_types.eq(vehicle_types.t())
        has_same_attr_neg = same_attr_neg.sum(dim=1) > 0
        mask_neg = torch.where(has_same_attr_neg.view(-1, 1), same_attr_neg, mask_neg)

    dist_pos = dist.clone()
    dist_pos[~mask_pos] = -1.0
    dist_pos.fill_diagonal_(-1.0)
    hardest_pos, _ = dist_pos.max(dim=1)

    dist_neg = dist.clone()
    dist_neg[~mask_neg] = 1e9
    hardest_neg, _ = dist_neg.min(dim=1)

    valid = hardest_pos > -0.5
    if valid.sum() == 0:
        return feat.sum() * 0.0
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


class PKBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(
        self,
        labels: List[int],
        p: int,
        k: int,
        cameras: List[str] | None = None,
        colors: List[int] | None = None,
        vehicle_types: List[int] | None = None,
        camera_balanced: bool = False,
        attribute_hard_negative: bool = False,
    ):
        self.p = p
        self.k = k
        self.cameras = cameras
        self.camera_balanced = camera_balanced and cameras is not None
        self.attribute_hard_negative = attribute_hard_negative and colors is not None and vehicle_types is not None
        label_to_indices: dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(idx)

        self.label_to_indices = label_to_indices
        self.labels = [label for label, idxs in label_to_indices.items() if len(idxs) >= k]
        if len(self.labels) < p:
            raise ValueError(f"Need at least {p} identities with >= {k} samples, got {len(self.labels)}")
        self.attr_to_labels: dict[tuple[int, int], List[int]] = {}
        if self.attribute_hard_negative:
            label_to_attr = {}
            for idx, label in enumerate(labels):
                if label in self.labels and label not in label_to_attr:
                    label_to_attr[label] = (int(colors[idx]), int(vehicle_types[idx]))
            for label, attr in label_to_attr.items():
                self.attr_to_labels.setdefault(attr, []).append(label)
            self.attr_buckets = [bucket for bucket in self.attr_to_labels.values() if len(bucket) >= 2]
        else:
            self.attr_buckets = []

    def __len__(self) -> int:
        return len(self.labels) // self.p

    def _sample_indices_for_label(self, label: int) -> List[int]:
        indices = self.label_to_indices[label]
        if not self.camera_balanced:
            return random.sample(indices, self.k)

        by_camera: dict[str, List[int]] = {}
        for idx in indices:
            by_camera.setdefault(self.cameras[idx], []).append(idx)
        for camera_indices in by_camera.values():
            random.shuffle(camera_indices)
        camera_keys = list(by_camera.keys())
        random.shuffle(camera_keys)
        selected: List[int] = []
        while len(selected) < self.k and any(by_camera[camera] for camera in camera_keys):
            for camera in camera_keys:
                if len(selected) >= self.k:
                    break
                if by_camera[camera]:
                    selected.append(by_camera[camera].pop())
        if len(selected) < self.k:
            remaining = [idx for idx in indices if idx not in set(selected)]
            selected.extend(random.sample(remaining, min(len(remaining), self.k - len(selected))))
        return selected[: self.k]

    def _make_batch_labels(self, remaining: set[int]) -> List[int]:
        if self.attribute_hard_negative and self.attr_buckets and random.random() < 0.8:
            bucket = [label for label in random.choice(self.attr_buckets) if label in remaining]
            random.shuffle(bucket)
            batch_labels = bucket[: self.p]
            if len(batch_labels) < self.p:
                pool = list(remaining.difference(batch_labels))
                random.shuffle(pool)
                batch_labels.extend(pool[: self.p - len(batch_labels)])
            return batch_labels
        pool = list(remaining)
        random.shuffle(pool)
        return pool[: self.p]

    def __iter__(self):
        remaining = set(self.labels)
        while len(remaining) >= self.p:
            batch_labels = self._make_batch_labels(remaining)
            batch: List[int] = []
            for label in batch_labels:
                batch.extend(self._sample_indices_for_label(label))
                remaining.discard(label)
            yield batch


def make_pk_loader(
    samples: List,
    transform,
    label_map: dict,
    p: int,
    k: int,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    cameras: List[str] | None = None,
    colors: List[int] | None = None,
    vehicle_types: List[int] | None = None,
    camera_balanced: bool = False,
    attribute_hard_negative: bool = False,
    dataset_cls=VRUDataset,
):
    labels = [label_map[s.vehicle_id] for s in samples]
    batch_sampler = PKBatchSampler(
        labels,
        p=p,
        k=k,
        cameras=cameras,
        colors=colors,
        vehicle_types=vehicle_types,
        camera_balanced=camera_balanced,
        attribute_hard_negative=attribute_hard_negative,
    )
    if dataset_cls is VRAITrainDataset:
        ds = dataset_cls(samples=samples, transform=transform, label_map=label_map, max_retries=3)
    else:
        ds = dataset_cls(samples=samples, transform=transform, relabel=True, label_map=label_map, max_retries=3)
    loader_kwargs = dict(pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        ds,
        batch_sampler=batch_sampler,
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


def _pick_vis_images(train_samples: List, max_images: int = 4) -> List[Path]:
    """Select a few training images for attention visualization."""
    ids_seen: set[int] = set()
    selected: List[Path] = []
    for s in train_samples:
        if s.vehicle_id not in ids_seen:
            ids_seen.add(s.vehicle_id)
            selected.append(s.img_path)
            if len(selected) >= max_images:
                break
    return selected


@torch.no_grad()
def _save_attention_local_visualization(
    model: nn.Module,
    sample_images: List[Path],
    output_dir: Path,
    epoch: int,
    device: torch.device,
    test_transform,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_channels_last: bool,
) -> None:
    """Save attention map overlays for a few sample images at given epoch."""
    import numpy as np

    base_model = getattr(model, "_orig_mod", model)
    was_training = base_model.training
    base_model.eval()

    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample_images:
        if not img_path.exists():
            continue
        with Image.open(img_path) as pil_img:
            orig = pil_img.convert("RGB").resize((224, 224))

        tensor = test_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        if use_channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)

        autocast_kwargs = dict(device_type=device.type, enabled=(use_amp and device.type == "cuda"))
        if device.type == "cuda":
            autocast_kwargs["dtype"] = amp_dtype
        with torch.autocast(**autocast_kwargs):
            base_model(tensor)

        att = base_model.attn_local.last_attention  # [1, K, H, W]
        if att is None:
            continue
        att_np = att[0].float().cpu().numpy()  # [K, H, W]

        orig_np = np.array(orig)  # [224, 224, 3]

        for head_idx in range(att_np.shape[0]):
            head_map = att_np[head_idx]  # [H, W]
            low, high = head_map.min(), head_map.max()
            if high - low > 1e-6:
                head_map = (head_map - low) / (high - low)
            else:
                head_map = np.zeros_like(head_map)

            # Resize to image size — simple nearest-neighbor via PIL
            heat_gray = Image.fromarray((head_map * 255).astype(np.uint8))
            heat_gray = heat_gray.resize((224, 224), Image.BILINEAR)
            heat_arr = np.array(heat_gray).astype(np.float32) / 255.0  # [224, 224]

            # Jet-like colormap: blue(0) -> green(0.5) -> red(1)
            red = (255 * heat_arr).astype(np.uint8)
            green = (255 * (1.0 - np.abs(heat_arr - 0.5) * 2.0)).astype(np.uint8)
            blue = (255 * (1.0 - heat_arr)).astype(np.uint8)
            heat_rgb = np.stack([red, green, blue], axis=-1)  # [224, 224, 3]

            alpha = 0.4
            overlay = (orig_np.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
            Image.fromarray(overlay).save(epoch_dir / f"{img_path.stem}_head{head_idx}.jpg")

    print(f"Saved attention visualization to {epoch_dir}")

    if was_training:
        base_model.train()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_cuda_for_speed()

    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.use_attention_local and args.use_part_branch:
        raise ValueError("Cannot use both --use-attention-local and --use-part-branch; they are mutually exclusive")
    if args.eval_every > 0 and not args.run_eval:
        print("--eval-every was set, enabling --run-eval for VRU evaluation")
        args.run_eval = True
    if args.dataset == "vrai" and args.run_eval:
        raise ValueError("--run-eval currently evaluates VRU splits only; use evaluate_vrai.py for VRAI evaluation")

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

    if args.dataset == "vrai":
        train_samples = read_vrai_train_samples(args.data_root / "VRAI")
        dataset_cls = VRAITrainDataset
        train_cameras = [s.camera for s in train_samples]
        train_colors = [s.color for s in train_samples]
        train_types = [s.vehicle_type for s in train_samples]
    else:
        vru_dir = args.data_root / "VRU"
        pic_dir = vru_dir / "Pic"
        split_dir = vru_dir / "train_test_split"
        train_samples = read_split_file(pic_dir, split_dir / "train_list.txt")
        dataset_cls = VRUDataset
        train_cameras = None
        train_colors = None
        train_types = None
    train_ids = sorted({s.vehicle_id for s in train_samples})
    train_label_map = {vid: i for i, vid in enumerate(train_ids)}
    num_classes = len(train_label_map)

    if args.strong_aug:
        train_tfms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.35),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.18, p=1.0)], p=0.25),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04)],
                p=0.75,
            ),
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
        ])
    else:
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

    pk_k = args.pk_k
    if pk_k < 2:
        raise ValueError("--pk-k must be >= 2")
    batch_size = args.batch_size
    if batch_size % pk_k != 0:
        raise ValueError(f"--batch-size must be divisible by k={pk_k}, got {batch_size}")
    pk_p = batch_size // pk_k
    learning_rate = args.base_lr * (batch_size / args.base_batch_size)

    train_loader = make_pk_loader(
        train_samples,
        train_tfms,
        label_map=train_label_map,
        p=pk_p,
        k=pk_k,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        cameras=train_cameras,
        colors=train_colors,
        vehicle_types=train_types,
        camera_balanced=(args.dataset == "vrai" and not args.disable_camera_balanced_sampler),
        attribute_hard_negative=(args.dataset == "vrai" and not args.disable_attribute_hard_negative_sampler),
        dataset_cls=dataset_cls,
    )

    model = GASNet(
        num_classes=num_classes,
        use_pretrained=not args.no_pretrained,
        use_part_branch=args.use_part_branch,
        num_parts=args.num_parts,
        use_gem=args.use_gem,
        gem_p=args.gem_p,
        backbone=args.backbone,
        use_attention_local=args.use_attention_local,
        num_attention_heads=args.num_attention_heads,
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    if torch.cuda.is_available() and hasattr(torch, "compile") and (not args.no_compile):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    warmup_epochs = 5
    milestones = [40, 50]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        drops = sum(epoch + 1 >= m for m in milestones)
        return 0.1 ** drops

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # New API first, old API fallback for compatibility.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f"device={device}, amp={use_amp}, amp_dtype={amp_dtype}, channels_last={use_channels_last}, "
        f"batch={batch_size}, lr={learning_rate:.6f}, grad_accum={args.grad_accum}, "
        f"workers={args.num_workers}, pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
        f"dataset={args.dataset}, camera_balanced={args.dataset == 'vrai' and not args.disable_camera_balanced_sampler}, "
        f"attr_hard_negative={args.dataset == 'vrai' and not args.disable_attribute_hard_negative_sampler}, "
        f"part_branch={args.use_part_branch}, attention_reg_weight={args.attention_reg_weight}, "
        f"attn_local={args.use_attention_local}, attn_heads={args.num_attention_heads}, "
        f"attn_diversity_weight={args.attn_diversity_weight}, "
        f"strong_aug={args.strong_aug}, pk_k={args.pk_k}, gem={args.use_gem}, backbone={args.backbone}"
    )

    model.train()
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
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
        for step, batch in enumerate(train_loader, 1):
            if args.dataset == "vrai":
                imgs, labels, _, _, cameras, colors, vehicle_types = batch
                colors = colors.to(device, non_blocking=True)
                vehicle_types = vehicle_types.to(device, non_blocking=True)
            else:
                imgs, labels, _, _ = batch
                cameras = None
                colors = None
                vehicle_types = None
            if use_channels_last:
                imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                feats, logits = model(imgs)
                loss_id = sum(ce_loss(logit, labels) for logit in logits)
                loss_tri = sum(
                    batch_hard_triplet_loss(
                        feat,
                        labels,
                        cameras=cameras,
                        colors=colors,
                        vehicle_types=vehicle_types,
                    )
                    for feat in feats
                )
                loss = loss_id + 0.5 * loss_tri
                if args.attention_reg_weight > 0:
                    base_model = getattr(model, "_orig_mod", model)
                    loss_attn = base_model.attention_regularization_loss(
                        target_mean=args.attention_target_mean,
                        std_margin=args.attention_std_margin,
                    )
                    loss = loss + args.attention_reg_weight * loss_attn
                if args.use_attention_local and args.attn_diversity_weight > 0:
                    base_model = getattr(model, "_orig_mod", model)
                    div_loss = base_model.attn_local.diversity_loss()
                    loss = loss + args.attn_diversity_weight * div_loss

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
                if args.use_attention_local and step == 1:
                    base_model = getattr(model, "_orig_mod", model)
                    att = base_model.attn_local.last_attention
                    if att is not None:
                        print(
                            f"    attn_local: shape={tuple(att.shape)} "
                            f"mean={att.mean().item():.4f} "
                            f"min={att.min().item():.4f} "
                            f"max={att.max().item():.4f}"
                        )

        if step % args.grad_accum != 0:
            if use_amp and use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = running / len(train_loader.dataset)
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, time={(time.time() - t0)/60:.1f} min")
        scheduler.step()

        if (
            args.use_attention_local
            and args.attn_vis_dir is not None
            and epoch in (1, max(1, args.epochs // 2), args.epochs)
        ):
            _save_attention_local_visualization(
                model=model,
                sample_images=_pick_vis_images(train_samples, max_images=4),
                output_dir=args.attn_vis_dir,
                epoch=epoch,
                device=device,
                test_transform=test_tfms,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
            )

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
                rerank=args.eval_rerank,
                rerank_k1=args.eval_rerank_k1,
                rerank_alpha=args.eval_rerank_alpha,
                tta_flip=args.eval_tta_flip,
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
            rerank=args.eval_rerank,
            rerank_k1=args.eval_rerank_k1,
            rerank_alpha=args.eval_rerank_alpha,
            tta_flip=args.eval_tta_flip,
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
