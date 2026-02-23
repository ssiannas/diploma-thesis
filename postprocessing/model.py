"""Sparse U-Net for point cloud displacement prediction.

3-level encoder-decoder with skip connections. Input: 4-ch features per point
(normalized xyz + curvature). Output: 3-ch displacement vector per point,
bounded by tanh * max_displacement.

GatedSparseUNet adds a learned move/no-move gate head (1-ch sigmoid) that
modulates the displacement at inference. The gate is supervised with BCE
against indicator(||d_gt|| > 0), explicitly modeling the bimodal GT distribution.
"""

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Two sparse conv3x3 + InstanceNorm + ReLU with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, bias=False, dimension=3
        )
        self.bn1 = ME.MinkowskiInstanceNorm(channels)
        self.conv2 = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, bias=False, dimension=3
        )
        self.bn2 = ME.MinkowskiInstanceNorm(channels)

    def forward(self, x):
        residual = x
        out = MF.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = MF.relu(out + residual)
        return out


class SparseUNet(nn.Module):
    """3-level sparse U-Net for displacement prediction.

    Architecture:
        Encoder: conv_in -> ResBlock -> down1 -> ResBlock -> down2 -> ResBlock
        Bottleneck: ResBlock
        Decoder: up2 -> cat(skip) -> conv -> ResBlock ->
                 up1 -> cat(skip) -> conv -> ResBlock
        Head: conv 1x1 -> tanh * max_displacement
    """

    def __init__(self, in_channels: int = 4, max_displacement: float = 5.0):
        super().__init__()
        self.max_displacement = max_displacement

        # Encoder
        self.conv_in = ME.MinkowskiConvolution(
            in_channels, 32, kernel_size=3, stride=1, bias=False, dimension=3
        )
        self.bn_in = ME.MinkowskiInstanceNorm(32)
        self.enc_block1 = ResBlock(32)

        self.down1 = ME.MinkowskiConvolution(
            32, 64, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_down1 = ME.MinkowskiInstanceNorm(64)
        self.enc_block2 = ResBlock(64)

        self.down2 = ME.MinkowskiConvolution(
            64, 128, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_down2 = ME.MinkowskiInstanceNorm(128)
        self.enc_block3 = ResBlock(128)

        # Bottleneck
        self.bottleneck = ResBlock(128)

        # Decoder
        self.up2 = ME.MinkowskiConvolutionTranspose(
            128, 64, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_up2 = ME.MinkowskiInstanceNorm(64)
        self.dec_merge2 = ME.MinkowskiConvolution(
            128, 64, kernel_size=1, stride=1, bias=False, dimension=3
        )
        self.bn_merge2 = ME.MinkowskiInstanceNorm(64)
        self.dec_block2 = ResBlock(64)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            64, 32, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_up1 = ME.MinkowskiInstanceNorm(32)
        self.dec_merge1 = ME.MinkowskiConvolution(
            64, 32, kernel_size=1, stride=1, bias=False, dimension=3
        )
        self.bn_merge1 = ME.MinkowskiInstanceNorm(32)
        self.dec_block1 = ResBlock(32)

        # Head
        self.conv_out = ME.MinkowskiConvolution(
            32, 3, kernel_size=1, stride=1, bias=True, dimension=3
        )
        self.tanh = ME.MinkowskiTanh()

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # Encoder
        e1 = MF.relu(self.bn_in(self.conv_in(x)))
        e1 = self.enc_block1(e1)

        e2 = MF.relu(self.bn_down1(self.down1(e1)))
        e2 = self.enc_block2(e2)

        e3 = MF.relu(self.bn_down2(self.down2(e2)))
        e3 = self.enc_block3(e3)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder level 2
        d2 = MF.relu(self.bn_up2(self.up2(b)))
        d2 = ME.cat(d2, e2)
        d2 = MF.relu(self.bn_merge2(self.dec_merge2(d2)))
        d2 = self.dec_block2(d2)

        # Decoder level 1
        d1 = MF.relu(self.bn_up1(self.up1(d2)))
        d1 = ME.cat(d1, e1)
        d1 = MF.relu(self.bn_merge1(self.dec_merge1(d1)))
        d1 = self.dec_block1(d1)

        # Head
        out = self.tanh(self.conv_out(d1))
        # Scale to max displacement
        out = ME.SparseTensor(
            features=out.F * self.max_displacement,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )

        return out


class GatedSparseUNet(SparseUNet):
    """Sparse U-Net with learned move/no-move gate.

    Inherits encoder-decoder from SparseUNet. Replaces the single head with:
    - Displacement head: 3-ch, tanh-bounded (same as SparseUNet)
    - Gate head: 1-ch, sigmoid (move probability per point)

    At inference: d_final = gate * d_pred.
    During training: displacement loss is NOT gated (all points learn),
    gate is supervised with BCE against indicator(||d_gt|| > 0).
    """

    def __init__(self, in_channels: int = 4, max_displacement: float = 5.0):
        super().__init__(in_channels=in_channels, max_displacement=max_displacement)
        # Replace parent's single head with two heads
        # Displacement head (reuse parent's conv_out and tanh)
        # Gate head
        self.conv_gate = ME.MinkowskiConvolution(
            32, 1, kernel_size=1, stride=1, bias=True, dimension=3
        )
        self.sigmoid = ME.MinkowskiSigmoid()

    def forward(self, x: ME.SparseTensor) -> tuple:
        # Encoder (inherited layers)
        e1 = MF.relu(self.bn_in(self.conv_in(x)))
        e1 = self.enc_block1(e1)

        e2 = MF.relu(self.bn_down1(self.down1(e1)))
        e2 = self.enc_block2(e2)

        e3 = MF.relu(self.bn_down2(self.down2(e2)))
        e3 = self.enc_block3(e3)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder level 2
        d2 = MF.relu(self.bn_up2(self.up2(b)))
        d2 = ME.cat(d2, e2)
        d2 = MF.relu(self.bn_merge2(self.dec_merge2(d2)))
        d2 = self.dec_block2(d2)

        # Decoder level 1
        d1 = MF.relu(self.bn_up1(self.up1(d2)))
        d1 = ME.cat(d1, e1)
        d1 = MF.relu(self.bn_merge1(self.dec_merge1(d1)))
        d1 = self.dec_block1(d1)

        # Displacement head
        disp = self.tanh(self.conv_out(d1))
        disp = ME.SparseTensor(
            features=disp.F * self.max_displacement,
            coordinate_map_key=disp.coordinate_map_key,
            coordinate_manager=disp.coordinate_manager,
        )

        # Gate head
        gate = self.sigmoid(self.conv_gate(d1))

        return disp, gate
