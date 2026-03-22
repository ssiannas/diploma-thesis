"""Sparse U-Net for point cloud displacement prediction.

3-level encoder-decoder with skip connections. Input: 4-ch features per point
(normalized xyz + curvature). Output: 3-ch displacement vector per point,
bounded by tanh * max_displacement.
"""

import math

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import torch
import torch.nn as nn

FOURIER_L = 4  # number of Fourier frequency bands -> 2*L features
MAX_RATE = 7  # r1-r7 discrete rates (onehot backward compat)


def _get_rate_in_dim(rate_repr: str) -> int:
    """Input dimensionality for each rate representation."""
    if rate_repr == "onehot":
        return MAX_RATE
    if rate_repr == "fourier":
        return 2 * FOURIER_L
    return 1  # "bpp" or any scalar


def _rate_to_input(rates: torch.Tensor, rate_repr: str) -> torch.Tensor:
    """Convert (B,) rate tensor to model input representation.

    onehot: r/7.0 values -> 7-dim one-hot.
    bpp:    log-bpp scalars -> (B, 1).
    fourier: log-bpp scalars -> (B, 2*FOURIER_L) positional encoding.
    """
    if rates.dim() == 0:
        rates = rates.unsqueeze(0)
    B = rates.shape[0]
    if rate_repr == "onehot":
        indices = (rates * MAX_RATE).round().long() - 1
        onehot = torch.zeros(B, MAX_RATE, device=rates.device)
        onehot.scatter_(1, indices.unsqueeze(1), 1.0)
        return onehot
    if rate_repr == "fourier":
        return _fourier_encode(rates)
    return rates.unsqueeze(1)  # (B, 1)


def _fourier_encode(t: torch.Tensor) -> torch.Tensor:
    """Positional encoding of a scalar: [sin(pi*2^k*t), cos(pi*2^k*t)] k=0..L-1.

    t: (B,) log-bpp values -> (B, 2*FOURIER_L) features.
    """
    if t.dim() == 0:
        t = t.unsqueeze(0)
    freqs = 2.0 ** torch.arange(FOURIER_L, dtype=t.dtype, device=t.device)  # (L,)
    args = math.pi * freqs.unsqueeze(0) * t.unsqueeze(1)  # (B, L)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2L)


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

    def forward(
        self, x: ME.SparseTensor, return_pre_head: bool = False
    ) -> ME.SparseTensor:
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

        if return_pre_head:
            return d1  # (N, 32) SparseTensor before conv_out

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
        self.film_gen = FiLMGenerator(_get_rate_in_dim(rate_repr), film_embed_dim)

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

    def forward(self, x: ME.SparseTensor, rates: torch.Tensor) -> ME.SparseTensor:
        rate_input = _rate_to_input(rates, self.rate_repr)  # (B, in_dim)
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


class FiLMHeadSparseUNetV2(SparseUNet):
    """SparseUNet with per-channel FiLM conditioning on pre-head 32-ch features.

    Applies gamma+beta (per channel) on the 32-ch backbone features before the
    final 1x1 conv projection to displacement. The backbone stays rate-invariant
    (no rate-dependent gradients in shared weights); only the FiLM head is
    rate-conditioned.

    This addresses the scalar bottleneck in FiLMHeadSparseUNet: instead of
    uniformly scaling all 32 channels by one number, the head can independently
    modulate each feature channel per rate.

    Forward signature: (x, rates) where rates is a (B,) tensor of log-bpp values.
    """

    def __init__(
        self,
        in_channels: int = 4,
        max_displacement: float = 5.0,
        film_embed_dim: int = 64,
        rate_repr: str = "bpp",
        quantize_output: bool = False,
    ):
        super().__init__(in_channels=in_channels, max_displacement=max_displacement)
        self.rate_repr = rate_repr
        self.quantize_output = quantize_output
        self.rate_encoder = nn.Sequential(
            nn.Linear(_get_rate_in_dim(rate_repr), film_embed_dim),
            nn.ReLU(inplace=True),
        )
        # Outputs gamma (32) + beta (32) for the 32-ch pre-head features
        self.film_head = nn.Linear(film_embed_dim, 64)
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

    def forward(self, x: ME.SparseTensor, rates: torch.Tensor) -> ME.SparseTensor:
        d1 = super().forward(x, return_pre_head=True)  # (N, 32)

        rate_input = _rate_to_input(rates, self.rate_repr)
        rate_embed = self.rate_encoder(rate_input)  # (B, embed_dim)
        gb = self.film_head(rate_embed)  # (B, 64)
        gamma = gb[:, :32] + 1.0  # (B, 32) identity init
        beta = gb[:, 32:]  # (B, 32) zero init

        batch_idx = d1.C[:, 0].long()
        d1_mod = ME.SparseTensor(
            features=gamma[batch_idx] * d1.F + beta[batch_idx],
            coordinate_map_key=d1.coordinate_map_key,
            coordinate_manager=d1.coordinate_manager,
        )

        out = self.tanh(self.conv_out(d1_mod))
        disp = out.F * self.max_displacement
        if self.quantize_output:
            if self.training:
                # Additive uniform noise: unbiased gradient estimator, no magnitude drift
                disp = disp + torch.empty_like(disp).uniform_(-0.5, 0.5)
            else:
                disp = disp.round()
        out = ME.SparseTensor(
            features=disp,
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager,
        )
        return out


class FiLMHeadSparseUNetV3(SparseUNet):
    """Rate-conditioned sparse U-Net with integer-class displacement head.

    Replaces regression with 3 x N_CLASSES-way classification over integer
    offsets {-max_disp, ..., +max_disp}. Training uses soft expectation
    (differentiable) for CD loss; inference uses argmax for exact integer output.

    Rate conditioning via per-channel FiLM on the 32-ch pre-head features,
    identical to V2. The backbone stays rate-invariant.
    """

    def __init__(
        self,
        in_channels: int = 4,
        max_displacement: float = 5.0,
        film_embed_dim: int = 64,
        rate_repr: str = "bpp",
    ):
        super().__init__(in_channels=in_channels, max_displacement=max_displacement)
        self.rate_repr = rate_repr
        max_offset = int(max_displacement)
        self.n_classes = 2 * max_offset + 1  # {-max,...,+max}
        self.register_buffer(
            "offsets", torch.arange(-max_offset, max_offset + 1).float()
        )
        self.rate_encoder = nn.Sequential(
            nn.Linear(_get_rate_in_dim(rate_repr), film_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.film_head = nn.Linear(film_embed_dim, 64)
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)
        # Classification head: 32 -> 3 * n_classes logits; zero init -> uniform softmax
        self.cls_head = nn.Linear(32, 3 * self.n_classes)
        nn.init.zeros_(self.cls_head.weight)
        nn.init.zeros_(self.cls_head.bias)

    def forward(
        self, x: ME.SparseTensor, rates: torch.Tensor, return_logits: bool = False
    ):
        d1 = super().forward(x, return_pre_head=True)  # (N, 32) SparseTensor

        rate_input = _rate_to_input(rates, self.rate_repr)
        rate_embed = self.rate_encoder(rate_input)  # (B, embed_dim)
        gb = self.film_head(rate_embed)  # (B, 64)
        gamma = gb[:, :32] + 1.0  # identity init
        beta = gb[:, 32:]  # zero init

        batch_idx = d1.C[:, 0].long()
        feats_mod = gamma[batch_idx] * d1.F + beta[batch_idx]  # (N, 32)

        logits = self.cls_head(feats_mod).view(-1, 3, self.n_classes)  # (N, 3, C)

        if self.training:
            # Soft expectation: differentiable proxy for CD loss
            probs = torch.softmax(logits, dim=-1)  # (N, 3, C)
            disp = (probs * self.offsets).sum(dim=-1)  # (N, 3)
        else:
            # Hard argmax: exact integer displacement, zero train-test mismatch
            disp = (logits.argmax(dim=-1) - self.n_classes // 2).float()  # (N, 3)

        st = ME.SparseTensor(
            features=disp,
            coordinate_map_key=d1.coordinate_map_key,
            coordinate_manager=d1.coordinate_manager,
        )
        if return_logits:
            return st, logits  # logits available regardless of training mode
        return st


class OccupancySparseUNet(SparseUNet):
    """Rate-conditioned sparse U-Net for voxel occupancy prediction.

    Input: 4-ch [x_norm, y_norm, z_norm, is_occupied]
        Candidates = decoded voxels (is_occupied=1) + their empty 26-neighbors (=0).
    Output: 1-ch occupancy logit per candidate (BCEWithLogits externally).

    Identical backbone to FiLMHeadSparseUNetV2/V3. Warm-start with strict=False;
    conv_in and conv_out are reinitialized (channel count / semantics change).
    """

    def __init__(
        self,
        in_channels: int = 4,
        film_embed_dim: int = 64,
        rate_repr: str = "bpp",
    ):
        super().__init__(in_channels=in_channels, max_displacement=1.0)
        self.rate_repr = rate_repr
        # 1-ch occupancy logit replaces 3-ch displacement head
        self.conv_out = ME.MinkowskiConvolution(
            32, 1, kernel_size=1, stride=1, bias=True, dimension=3
        )
        self.rate_encoder = nn.Sequential(
            nn.Linear(_get_rate_in_dim(rate_repr), film_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.film_head = nn.Linear(film_embed_dim, 64)
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

    def forward(self, x: ME.SparseTensor, rates: torch.Tensor) -> ME.SparseTensor:
        d1 = super().forward(x, return_pre_head=True)  # (N, 32) SparseTensor

        rate_input = _rate_to_input(rates, self.rate_repr)
        rate_embed = self.rate_encoder(rate_input)  # (B, embed_dim)
        gb = self.film_head(rate_embed)  # (B, 64)
        gamma = gb[:, :32] + 1.0
        beta = gb[:, 32:]

        batch_idx = d1.C[:, 0].long()
        feats_mod = gamma[batch_idx] * d1.F + beta[batch_idx]

        d1_mod = ME.SparseTensor(
            features=feats_mod,
            coordinate_map_key=d1.coordinate_map_key,
            coordinate_manager=d1.coordinate_manager,
        )
        return self.conv_out(d1_mod)  # 1-ch logit SparseTensor


class FiLMHeadSparseUNet(SparseUNet):
    """SparseUNet with FiLM conditioning at output head only.

    Vanilla backbone learns rate-invariant geometry (WHERE to move points).
    Output scalar scaling learns rate-dependent magnitude (HOW MUCH to move).
    Eliminates gradient interference from mixed-rate batches in shared backbone.

    Forward signature: (x, rates) where rates is a (B,) tensor of rate values.
    rate_repr: "onehot" (7-dim, legacy), "bpp" (1-dim continuous bpp scalar)
    """

    def __init__(
        self,
        in_channels: int = 4,
        max_displacement: float = 5.0,
        film_embed_dim: int = 64,
        rate_repr: str = "onehot",
    ):
        super().__init__(in_channels=in_channels, max_displacement=max_displacement)
        self.rate_repr = rate_repr
        self.scale_net = nn.Sequential(
            nn.Linear(_get_rate_in_dim(rate_repr), film_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(film_embed_dim, 1),
        )
        # Init bias so scale starts at 1.0 (identity)
        nn.init.zeros_(self.scale_net[0].weight)
        nn.init.zeros_(self.scale_net[0].bias)
        nn.init.zeros_(self.scale_net[2].weight)
        self.scale_net[2].bias.data.fill_(1.0)

    def forward(self, x: ME.SparseTensor, rates: torch.Tensor) -> ME.SparseTensor:
        out = super().forward(x)
        rate_input = _rate_to_input(rates, self.rate_repr)  # (B, in_dim)
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


def build_model(model_type: str, in_channels: int, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_type: One of the keys in _MODEL_REGISTRY.
        in_channels: Input feature channels (3 for xyz-only, 4 for xyz+curvature
                     or xyz+is_occupied for occupancy models).
        **kwargs: Forwarded to the model constructor. Common keys:
            max_displacement (float), film_embed_dim (int), rate_repr (str),
            quantize_output (bool).

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type={model_type!r}. Valid: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[model_type](in_channels, **kwargs)


def _build_unet(in_channels: int, **kwargs) -> SparseUNet:
    return SparseUNet(
        in_channels=in_channels,
        max_displacement=kwargs.get("max_displacement", 5.0),
    )


def _build_film(in_channels: int, **kwargs) -> FiLMSparseUNet:
    return FiLMSparseUNet(
        in_channels=in_channels,
        max_displacement=kwargs.get("max_displacement", 5.0),
        film_embed_dim=kwargs.get("film_embed_dim", 64),
        rate_repr=kwargs.get("rate_repr", "bpp"),
    )


def _build_film_head(in_channels: int, **kwargs) -> FiLMHeadSparseUNet:
    return FiLMHeadSparseUNet(
        in_channels=in_channels,
        max_displacement=kwargs.get("max_displacement", 5.0),
        film_embed_dim=kwargs.get("film_embed_dim", 64),
        rate_repr=kwargs.get("rate_repr", "onehot"),
    )


def _build_film_head_v2(in_channels: int, **kwargs) -> FiLMHeadSparseUNetV2:
    return FiLMHeadSparseUNetV2(
        in_channels=in_channels,
        max_displacement=kwargs.get("max_displacement", 5.0),
        film_embed_dim=kwargs.get("film_embed_dim", 64),
        rate_repr=kwargs.get("rate_repr", "bpp"),
        quantize_output=kwargs.get("quantize_output", False),
    )


def _build_film_head_v3(in_channels: int, **kwargs) -> FiLMHeadSparseUNetV3:
    return FiLMHeadSparseUNetV3(
        in_channels=in_channels,
        max_displacement=kwargs.get("max_displacement", 5.0),
        film_embed_dim=kwargs.get("film_embed_dim", 64),
        rate_repr=kwargs.get("rate_repr", "bpp"),
    )


def _build_occupancy(in_channels: int, **kwargs) -> OccupancySparseUNet:
    return OccupancySparseUNet(
        in_channels=in_channels,
        film_embed_dim=kwargs.get("film_embed_dim", 64),
        rate_repr=kwargs.get("rate_repr", "bpp"),
    )


_MODEL_REGISTRY = {
    "unet": _build_unet,
    "film": _build_film,
    "film_head": _build_film_head,
    "film_head_v2": _build_film_head_v2,
    "film_head_v3": _build_film_head_v3,
    "occupancy": _build_occupancy,
}
