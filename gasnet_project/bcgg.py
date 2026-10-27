"""Bottleneck Context Group Gate (BCGG).

Research proposal -- this is **not** an established paper/module name.

Design (as used by GASNet now):

    x [B,C,H,W]
      ├─ spatial path : 1x1 Conv-BN-ReLU  C -> d
      │                 RepDepthwiseContext(d, k)   (k x k depthwise, train-time
      │                 branches 1xk / kx1 fold into it for deployment)
      │                 1x1 Conv             d -> G        -> [B,G,1,H,W]
      └─ channel path : GAP -> 1D conv over the channel axis (kernel kc)
                                                           -> [B,C] -> [B,G,C/G,1,1]
      gate = 2 * sigmoid(channel + spatial)                -> [B,G,C/G,H,W]
      y    = x * gate                       (+ value(u) if inject_value=True)

The gate is per-channel: the channel logits are per channel while the spatial
logits are per group and are broadcast over the C/G channels of that group.

With ``identity_init=True`` every logit head is zero-initialised, so the gate is
exactly ``2 * sigmoid(0) = 1`` and the block is an exact identity at step 0.
This matters when the gate is interleaved between pretrained backbone stages.

Cost note: the eager PyTorch graph still materialises a full-size gate
[B,C,H,W] (sigmoid + product). TensorRT can fuse these elementwise ops, but
avoiding the materialisation on the training side would require a custom
kernel. Even so, BCGG is far cheaper than the original RGA block because it
never builds the N x N relation matrices (N = H*W).

``to_deploy()`` folds Conv+BN and the RepDepthwiseContext branches, which is
exact and reduces the deployed graph to three plain convolutions.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "BottleneckContextGroupGate",
    "ChannelLogits",
    "RepDepthwiseContext",
    "infer_gate_config",
]


# --------------------------------------------------------------------------- #
# Conv/BN folding helpers
# --------------------------------------------------------------------------- #
def _fuse_conv_bn_params(
    conv: nn.Conv2d,
    bn: nn.modules.batchnorm._BatchNorm,
) -> tuple[Tensor, Tensor]:
    """Return ``(weight, bias)`` equivalent to ``bn(conv(x))`` in eval mode."""
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError(
            "Cannot fold a BatchNorm without running statistics "
            "(call eval() after training first)"
        )
    if bn.running_mean.numel() != conv.out_channels:
        raise ValueError("BatchNorm/Conv channel mismatch while folding")

    # Fold in fp32 to avoid precision loss when the model runs in fp16/bf16.
    weight = conv.weight.detach().to(torch.float32)
    bias = (
        conv.bias.detach().to(torch.float32)
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=torch.float32, device=weight.device)
    )
    scale = bn.weight.detach().to(torch.float32) / torch.sqrt(
        bn.running_var.detach().to(torch.float32) + float(bn.eps)
    )
    fused_weight = (weight * scale.reshape(-1, *([1] * (weight.dim() - 1)))).to(conv.weight.dtype)
    fused_bias = (
        (bias - bn.running_mean.detach().to(torch.float32)) * scale
        + bn.bias.detach().to(torch.float32)
    ).to(conv.weight.dtype)
    return fused_weight, fused_bias


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.modules.batchnorm._BatchNorm) -> nn.Conv2d:
    """Build a single Conv2d with bias that reproduces ``bn(conv(x))``."""
    fused_weight, fused_bias = _fuse_conv_bn_params(conv, bn)
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=fused_weight.device, dtype=fused_weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(fused_weight)
        fused.bias.copy_(fused_bias)
    return fused


# --------------------------------------------------------------------------- #
# Channel logits: GAP -> 1D conv across the channel axis (ECA style)
# --------------------------------------------------------------------------- #
class ChannelLogits(nn.Module):
    """Per-channel logits from a global descriptor.

    ``GAP`` gives ``[B, 1, C]``; a ``Conv1d(1, 1, k)`` over the channel axis
    mixes neighbouring channels (same idea as ECA-Net). This needs only
    ``k`` parameters and no knowledge of ``C``, which is why the constructor
    only takes the kernel size.
    """

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("channel_kernel must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, self.kernel_size, padding=self.kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[0], x.shape[1]
        y = self.pool(x).reshape(b, 1, c)
        return self.conv(y).reshape(b, c)


# --------------------------------------------------------------------------- #
# Re-parameterizable depthwise context
# --------------------------------------------------------------------------- #
class _DepthwiseBranch(nn.Module):
    """One depthwise conv + BN branch of :class:`RepDepthwiseContext`."""

    def __init__(self, channels: int, kernel_size: int, placement: str) -> None:
        super().__init__()
        if placement == "full":
            kernel, padding = kernel_size, (kernel_size // 2, kernel_size // 2)
        elif placement == "row":
            kernel, padding = (1, kernel_size), (0, kernel_size // 2)
        elif placement == "col":
            kernel, padding = (kernel_size, 1), (kernel_size // 2, 0)
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"Unknown placement: {placement}")
        self.placement = placement
        self.conv = nn.Conv2d(
            channels, channels, kernel, padding=padding, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

    def fused_params(self) -> tuple[Tensor, Tensor]:
        return _fuse_conv_bn_params(self.conv, self.bn)


class RepDepthwiseContext(nn.Module):
    """Depthwise ``k x k`` context block with RepVGG-style training branches.

    Trains three parallel depthwise branches (``k x k``, ``1 x k``, ``k x 1``,
    each followed by BN) and folds them into a single depthwise ``k x k``
    convolution for deployment. The fused result is exact in eval mode.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.branches = nn.ModuleList(
            [
                _DepthwiseBranch(self.channels, self.kernel_size, "full"),
                _DepthwiseBranch(self.channels, self.kernel_size, "row"),
                _DepthwiseBranch(self.channels, self.kernel_size, "col"),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        return out

    @torch.no_grad()
    def fused_conv(self) -> nn.Conv2d:
        """Equivalent single depthwise ``k x k`` conv (eval mode only)."""
        if self.training:
            raise ValueError("Call eval() before fused_conv()")
        k = self.kernel_size
        ref_weight, _ = self.branches[0].fused_params()
        fused_weight = torch.zeros(
            self.channels, 1, k, k, dtype=ref_weight.dtype, device=ref_weight.device
        )
        fused_bias = torch.zeros(
            self.channels, dtype=ref_weight.dtype, device=ref_weight.device
        )
        for branch in self.branches:
            weight, bias = branch.fused_params()  # [C,1,kh,kw], [C]
            if branch.placement == "full":
                fused_weight += weight
            elif branch.placement == "row":
                fused_weight[:, :, k // 2, :] += weight[:, :, 0, :]
            else:  # col
                fused_weight[:, :, :, k // 2] += weight[:, :, :, 0]
            fused_bias += bias

        conv = nn.Conv2d(
            self.channels,
            self.channels,
            k,
            padding=k // 2,
            groups=self.channels,
            bias=True,
        ).to(device=fused_weight.device, dtype=fused_weight.dtype)
        conv.weight.copy_(fused_weight)
        conv.bias.copy_(fused_bias)
        return conv


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #
class BottleneckContextGroupGate(nn.Module):
    """Bottleneck Context Group Gate (BCGG) for a single backbone stage.

    Args:
        channels: stage width (must be divisible by ``groups``).
        hidden: bottleneck width of the spatial context path.
        groups: number of contiguous channel groups sharing one spatial logit.
        kernel_size: depthwise kernel of the context path (odd).
        channel_kernel: 1D kernel of the channel-logit path (odd).
        inject_value: add ``P_value(u)`` as an additive refinement branch
            (separate ablation, off by default).
        identity_init: zero-init both logit heads so the block starts as an
            exact identity.

    Attributes:
        capture_attention: when True, forward stores a per-group spatial
            attention summary in ``last_attention`` and emits it through the
            ``attn_probe`` identity module (used for heatmaps / regularisation).
        last_attention: ``[B,1,H,W]`` sigmoid gate averaged over channels.
    """

    def __init__(
        self,
        channels: int = 384,
        hidden: int = 16,
        groups: int = 4,
        kernel_size: int = 7,
        channel_kernel: int = 5,
        inject_value: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        if min(channels, hidden, groups) <= 0 or channels % groups:
            raise ValueError("Positive sizes required and channels must be divisible by groups")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if channel_kernel <= 0 or channel_kernel % 2 == 0:
            raise ValueError("channel_kernel must be a positive odd integer")

        self.channels = int(channels)
        self.hidden = int(hidden)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.channel_kernel = int(channel_kernel)
        self.inject_value = bool(inject_value)

        self.reduce = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden, 1, bias=False),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=False),
        )
        self.context = RepDepthwiseContext(self.hidden, self.kernel_size)
        self.spatial = nn.Conv2d(self.hidden, self.groups, 1, bias=False)
        self.channel = ChannelLogits(self.channel_kernel)
        self.value = (
            nn.Conv2d(self.hidden, self.channels, 1, bias=False) if self.inject_value else None
        )

        # Attention capture for heatmaps / regularisation. Never part of the
        # numerical path, so it is safe to leave disabled for export.
        self.attn_probe = nn.Identity()
        self.capture_attention = False
        self.last_attention: Tensor | None = None
        self.deployed = False

        if identity_init:
            nn.init.zeros_(self.spatial.weight)
            nn.init.zeros_(self.channel.conv.weight)
            if self.value is not None:
                nn.init.zeros_(self.value.weight)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        # Skipped while tracing/exporting: the check only guards against wiring
        # the block into a stage whose width differs from `channels`.
        if not torch.jit.is_tracing() and c != self.channels:
            raise ValueError(
                f"BottleneckContextGroupGate expects {self.channels} channels, got {c}"
            )

        u = self.context(self.reduce(x))                       # [B,d,H,W]
        spatial = self.spatial(u).unsqueeze(2)                 # [B,G,1,H,W]
        channel = self.channel(x).reshape(b, self.groups, c // self.groups, 1, 1)
        gate = 2.0 * torch.sigmoid(channel + spatial)          # [B,G,C/G,H,W]

        if self.capture_attention:
            # sigmoid gate (without the x2 factor) averaged over all channels.
            attn = (0.5 * gate).mean(dim=(1, 2), keepdim=True).reshape(b, 1, h, w)
            self.last_attention = attn
            self.attn_probe(attn)

        y = x * gate.reshape(b, c, h, w)
        if self.value is not None:
            y = y + self.value(u)
        return y

    @torch.no_grad()
    def to_deploy(self) -> "BottleneckContextGroupGate":
        """Return a copy with Conv/BN folded and the context branches fused."""
        if self.training:
            raise ValueError("Call eval() before to_deploy(); keep a training checkpoint")
        # Drop the inference cache: it may hold a non-leaf tensor from the last
        # forward pass, which the deepcopy protocol rejects.
        self.capture_attention = False
        self.last_attention = None
        module = copy.deepcopy(self)
        module.eval()
        if isinstance(module.context, RepDepthwiseContext):
            module.context = module.context.fused_conv()
        if len(module.reduce) == 3 and isinstance(
            module.reduce[1], (nn.BatchNorm2d, nn.SyncBatchNorm)
        ):
            fused = _fold_conv_bn(module.reduce[0], module.reduce[1])
            module.reduce = nn.Sequential(fused, module.reduce[2])
        module.deployed = True
        module.capture_attention = False
        module.last_attention = None
        return module.eval()

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, hidden={self.hidden}, groups={self.groups}, "
            f"kernel_size={self.kernel_size}, channel_kernel={self.channel_kernel}, "
            f"inject_value={self.inject_value}, deployed={self.deployed}"
        )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def infer_gate_config(
    state_dict: dict,
    default_kind: str = "bcgg",
) -> dict:
    """Infer GASNet gate kwargs from a checkpoint state dict.

    Lets ``evaluate_vrai.py`` / ``infer.py`` / ``export_onnx.py`` load both
    legacy RGA checkpoints and new BCGG checkpoints without extra flags.
    """
    prefix = "ga1."
    keys = [k for k in state_dict if k.startswith(prefix)]
    is_rga = any(".rga_s." in k for k in keys) or any(".rga_c." in k for k in keys)
    is_bcgg = any(k.startswith(prefix + "reduce.") for k in keys) or any(
        k.startswith(prefix + "spatial.") for k in keys
    )

    if is_rga:
        kind = "rga"
    elif is_bcgg:
        kind = "bcgg"
    else:
        kind = default_kind

    config = {
        "gate": kind,
        "gate_hidden": 16,
        "gate_groups": 4,
        "gate_kernel": 7,
        "gate_channel_kernel": 5,
        "gate_inject_value": False,
    }
    if kind != "bcgg":
        return config

    spatial = state_dict.get(prefix + "spatial.weight")
    if spatial is not None:
        config["gate_groups"] = int(spatial.shape[0])
        config["gate_hidden"] = int(spatial.shape[1])

    reduce_weight = state_dict.get(prefix + "reduce.0.weight")
    if reduce_weight is not None:
        config["gate_hidden"] = int(reduce_weight.shape[0])

    channel_weight = state_dict.get(prefix + "channel.conv.weight")
    if channel_weight is not None:
        config["gate_channel_kernel"] = int(channel_weight.shape[-1])

    # The first RepDepthwiseContext branch is the k x k one.
    context_weight = state_dict.get(prefix + "context.branches.0.conv.weight")
    if context_weight is not None:
        config["gate_kernel"] = int(context_weight.shape[-1])

    config["gate_inject_value"] = any(k.startswith(prefix + "value.") for k in keys)
    return config
