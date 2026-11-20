"""ResNet-1D modulation classifier.

The architecture is frozen once trained: the inference engine only has to
support the operators used here.

    input  [B, 2, 1024]   I/Q, unit RMS per window
    stem   conv k7 s2  2->32   + BN + ReLU          L=512
    stage1 BasicBlock  32->32                       L=512
    stage2 BasicBlock  32->64  s2, BasicBlock 64    L=256
    stage3 BasicBlock  64->128 s2                   L=128
    head   global average pool, linear 128->classes

Operators: Conv1d (k=1/3/7, stride 1/2), BatchNorm1d (folded at export),
ReLU, residual Add, global mean, Linear.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + identity)


class AmcResNet1D(nn.Module):
    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(2, w, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(w),
                                  nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(BasicBlock(w, w))
        self.stage2 = nn.Sequential(BasicBlock(w, 2 * w, 2), BasicBlock(2 * w, 2 * w))
        self.stage3 = nn.Sequential(BasicBlock(2 * w, 4 * w, 2))
        self.head = nn.Linear(4 * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(x.mean(dim=-1))


def count_flops(model: nn.Module, length: int = 1024) -> int:
    """Multiply-accumulate FLOPs (2 per MAC) of conv/linear layers for one window."""
    flops = 0

    def conv_hook(m: nn.Conv1d, inp, out):
        nonlocal flops
        flops += 2 * out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal flops
        flops += 2 * m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv1d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p = next(model.parameters())
        model(torch.zeros(1, 2, length, device=p.device, dtype=p.dtype))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return flops
