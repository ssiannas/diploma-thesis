"""Sparse U-Net for point cloud displacement prediction.

3-level encoder-decoder with skip connections. Input: 4-ch features per point
(normalized xyz + curvature). Output: 3-ch displacement vector per point,
bounded by tanh * max_displacement.
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


class FiLMGenerator(nn.Module):
    """Maps rate representation to embedding vector for FiLM conditioning."""

    def __init__(self, in_dim: int = 1, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rate: torch.Tensor) -> torch.Tensor:
        # rate: (B, in_dim) -> (B, embed_dim) or (in_dim,) -> (embed_dim,)
        return self.mlp(rate)


class FiLMResBlock(nn.Module):
    """ResBlock with FiLM modulation after second norm, before residual add."""

    def __init__(self, channels: int, embed_dim: int = 64):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, bias=False, dimension=3
        )
        self.bn1 = ME.MinkowskiInstanceNorm(channels)
        self.conv2 = ME.MinkowskiConvolution(
            channels, channels, kernel_size=3, stride=1, bias=False, dimension=3
        )
        self.bn2 = ME.MinkowskiInstanceNorm(channels)
        # FiLM projection: embed -> (gamma, beta) per channel
        self.film_proj = nn.Linear(embed_dim, channels * 2)
        # Init near identity: small weights, bias = [0,...,0,...] + we add 1 to gamma
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x, rate_embeds):
        residual = x
        out = MF.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # FiLM: per-point gamma * features + beta
        gb = self.film_proj(rate_embeds)  # (B, channels*2)
        C = out.F.shape[1]
        batch_indices = out.C[:, 0].long()  # sample index per point
        gb_per_point = gb[batch_indices]  # (N, channels*2)
        gamma = gb_per_point[:, :C] + 1.0  # +1 for identity init
        beta = gb_per_point[:, C:]
        out = ME.SparseTensor(
            features=gamma * out.F + beta,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )
        out = MF.relu(out + residual)
        return out


class FiLMSparseUNet(nn.Module):
    """SparseUNet with FiLM rate conditioning on all 6 ResBlocks.

    Forward signature: (x, rates) where rates is a (B,) tensor of rate values.
    Per-point conditioning via batch indices from sparse tensor coordinates.
    At init, FiLM starts as identity so behavior matches vanilla SparseUNet.

    rate_repr: "onehot" (7-dim, legacy), "bpp" (1-dim continuous bpp scalar)
    """

    MAX_RATE = 7  # r1-r7 (for onehot backward compat)

    def __init__(
        self,
        in_channels: int = 4,
        max_displacement: float = 5.0,
        film_embed_dim: int = 64,
        rate_repr: str = "onehot",
    ):
        super().__init__()
        self.max_displacement = max_displacement
        self.rate_repr = rate_repr

        # FiLM generator
        if rate_repr == "onehot":
            rate_in_dim = self.MAX_RATE
        else:
            rate_in_dim = 1
        self.film_gen = FiLMGenerator(rate_in_dim, film_embed_dim)

        # Encoder
        self.conv_in = ME.MinkowskiConvolution(
            in_channels, 32, kernel_size=3, stride=1, bias=False, dimension=3
        )
        self.bn_in = ME.MinkowskiInstanceNorm(32)
        self.enc_block1 = FiLMResBlock(32, film_embed_dim)

        self.down1 = ME.MinkowskiConvolution(
            32, 64, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_down1 = ME.MinkowskiInstanceNorm(64)
        self.enc_block2 = FiLMResBlock(64, film_embed_dim)

        self.down2 = ME.MinkowskiConvolution(
            64, 128, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_down2 = ME.MinkowskiInstanceNorm(128)
        self.enc_block3 = FiLMResBlock(128, film_embed_dim)

        # Bottleneck
        self.bottleneck = FiLMResBlock(128, film_embed_dim)

        # Decoder
        self.up2 = ME.MinkowskiConvolutionTranspose(
            128, 64, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_up2 = ME.MinkowskiInstanceNorm(64)
        self.dec_merge2 = ME.MinkowskiConvolution(
            128, 64, kernel_size=1, stride=1, bias=False, dimension=3
        )
        self.bn_merge2 = ME.MinkowskiInstanceNorm(64)
        self.dec_block2 = FiLMResBlock(64, film_embed_dim)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            64, 32, kernel_size=2, stride=2, bias=False, dimension=3
        )
        self.bn_up1 = ME.MinkowskiInstanceNorm(32)
        self.dec_merge1 = ME.MinkowskiConvolution(
            64, 32, kernel_size=1, stride=1, bias=False, dimension=3
        )
        self.bn_merge1 = ME.MinkowskiInstanceNorm(32)
        self.dec_block1 = FiLMResBlock(32, film_embed_dim)

        # Head
        self.conv_out = ME.MinkowskiConvolution(
            32, 3, kernel_size=1, stride=1, bias=True, dimension=3
        )
        self.tanh = ME.MinkowskiTanh()

    def _rate_to_input(self, rates: torch.Tensor) -> torch.Tensor:
        """Convert rate tensor (B,) to model input representation.

        For onehot: rates are N/7.0 values -> 7-dim onehot.
        For bpp: rates are raw bpp scalars -> (B, 1).
        """
        if rates.dim() == 0:
            rates = rates.unsqueeze(0)
        B = rates.shape[0]
        if self.rate_repr == "onehot":
            indices = (rates * self.MAX_RATE).round().long() - 1  # 0-indexed
            onehot = torch.zeros(B, self.MAX_RATE, device=rates.device)
            onehot.scatter_(1, indices.unsqueeze(1), 1.0)
            return onehot  # (B, 7)
        return rates.unsqueeze(1)  # (B, 1) -- bpp or scalar

    def forward(self, x: ME.SparseTensor, rates: torch.Tensor) -> ME.SparseTensor:
        rate_input = self._rate_to_input(rates)  # (B, in_dim)
        rate_embeds = self.film_gen(rate_input)  # (B, embed_dim)

        # Encoder
        e1 = MF.relu(self.bn_in(self.conv_in(x)))
        e1 = self.enc_block1(e1, rate_embeds)

        e2 = MF.relu(self.bn_down1(self.down1(e1)))
        e2 = self.enc_block2(e2, rate_embeds)

        e3 = MF.relu(self.bn_down2(self.down2(e2)))
        e3 = self.enc_block3(e3, rate_embeds)

        # Bottleneck
        b = self.bottleneck(e3, rate_embeds)

        # Decoder level 2
        d2 = MF.relu(self.bn_up2(self.up2(b)))
        d2 = ME.cat(d2, e2)
        d2 = MF.relu(self.bn_merge2(self.dec_merge2(d2)))
        d2 = self.dec_block2(d2, rate_embeds)

        # Decoder level 1
        d1 = MF.relu(self.bn_up1(self.up1(d2)))
        d1 = ME.cat(d1, e1)
        d1 = MF.relu(self.bn_merge1(self.dec_merge1(d1)))
        d1 = self.dec_block1(d1, rate_embeds)

        # Head
        out = self.tanh(self.conv_out(d1))
        out = ME.SparseTensor(
            features=out.F * self.max_displacement,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )

        return out


class FiLMHeadSparseUNet(SparseUNet):
    """SparseUNet with FiLM conditioning at output head only.

    Vanilla backbone learns rate-invariant geometry (WHERE to move points).
    Output scalar scaling learns rate-dependent magnitude (HOW MUCH to move).
    Eliminates gradient interference from mixed-rate batches in shared backbone.

    Forward signature: (x, rates) where rates is a (B,) tensor of rate values.
    rate_repr: "onehot" (7-dim, legacy), "bpp" (1-dim continuous bpp scalar)
    """

    MAX_RATE = 7  # for onehot backward compat

    def __init__(
        self,
        in_channels: int = 4,
        max_displacement: float = 5.0,
        film_embed_dim: int = 64,
        rate_repr: str = "onehot",
    ):
        super().__init__(in_channels=in_channels, max_displacement=max_displacement)
        self.rate_repr = rate_repr

        # Rate -> scalar scale factor
        if rate_repr == "onehot":
            rate_in_dim = self.MAX_RATE
        else:
            rate_in_dim = 1
        self.scale_net = nn.Sequential(
            nn.Linear(rate_in_dim, film_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(film_embed_dim, 1),
        )
        # Init bias so scale starts at 1.0 (identity)
        nn.init.zeros_(self.scale_net[0].weight)
        nn.init.zeros_(self.scale_net[0].bias)
        nn.init.zeros_(self.scale_net[2].weight)
        self.scale_net[2].bias.data.fill_(1.0)

    def _rate_to_input(self, rates: torch.Tensor) -> torch.Tensor:
        if rates.dim() == 0:
            rates = rates.unsqueeze(0)
        B = rates.shape[0]
        if self.rate_repr == "onehot":
            indices = (rates * self.MAX_RATE).round().long() - 1
            onehot = torch.zeros(B, self.MAX_RATE, device=rates.device)
            onehot.scatter_(1, indices.unsqueeze(1), 1.0)
            return onehot
        return rates.unsqueeze(1)

    def forward(self, x: ME.SparseTensor, rates: torch.Tensor) -> ME.SparseTensor:
        # Vanilla backbone forward
        out = super().forward(x)

        # Rate-dependent scalar scaling per sample
        rate_input = self._rate_to_input(rates)  # (B, in_dim)
        scale = self.scale_net(rate_input)  # (B, 1)

        # Apply per-point scaling via batch indices
        batch_indices = out.C[:, 0].long()
        scale_per_point = scale[batch_indices]  # (N, 1)

        out = ME.SparseTensor(
            features=out.F * scale_per_point,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )
        return out
