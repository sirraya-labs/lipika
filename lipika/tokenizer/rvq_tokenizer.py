"""
lipika/tokenizer/rvq_tokenizer.py

Residual Vector Quantizer with w2v-BERT 2.0 semantic distillation on CB1.
Single unified tokenizer (~21Hz, 10 codebooks) replacing the dual-tokenizer approach.

Key design: CB1 captures semantic content via auxiliary prediction head regressing
w2v-BERT 2.0 activations. CB2-CB10 capture acoustic detail (prosody, timbre).
This disentanglement is what allows the Dual-AR to work: Slow AR models CB1,
Fast AR models CB2-CB10 conditioned on CB1.

References:
  - Fish Audio S2: semantic distillation via w2v-BERT auxiliary head
  - EnCodec / Mimi: RVQ codec design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class RVQConfig:
    # Audio
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 2100          # ~21Hz frame rate (44100 / 2100 = 21)
    n_mels: int = 128

    # RVQ
    n_codebooks: int = 10
    codebook_size: int = 2048
    codebook_dim: int = 256
    commitment_cost: float = 0.25

    # Encoder / Decoder
    encoder_channels: int = 512
    encoder_depth: int = 6
    decoder_channels: int = 512
    decoder_depth: int = 6

    # Semantic distillation (CB1)
    w2v_bert_dim: int = 1024        # w2v-BERT 2.0 hidden size
    semantic_proj_dim: int = 256

    # Script-family adapter
    n_script_families: int = 12     # Devanagari, Tamil, Telugu, etc.
    script_embed_dim: int = 64


class CausalConv1d(nn.Module):
    """Causal convolution — no future context leakage."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, 3, dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class VectorQuantizer(nn.Module):
    """
    Single codebook VQ with straight-through estimator.
    Used as a building block for RVQ.
    """
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z: (B, T, D) continuous latents
        Returns:
            z_q: (B, T, D) quantized
            indices: (B, T) codebook indices
            loss: commitment + embedding losses
        """
        B, T, D = z.shape
        flat_z = z.reshape(-1, D)                           # (B*T, D)

        # L2 distances to all codebook entries
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1).reshape(B, T)              # (B, T)
        z_q = self.embedding(indices)                       # (B, T, D)

        # Losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss  = F.mse_loss(z_q, z.detach())
        loss = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through gradient
        z_q = z + (z_q - z).detach()
        return z_q, indices, loss


class ResidualVectorQuantizer(nn.Module):
    """
    10-codebook RVQ. CB1 has a semantic distillation head.
    """
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        self.codebooks = nn.ModuleList([
            VectorQuantizer(cfg.codebook_size, cfg.codebook_dim, cfg.commitment_cost)
            for _ in range(cfg.n_codebooks)
        ])

        # Semantic distillation head on CB1 only
        # Projects quantized CB1 latent → w2v-BERT feature space
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.codebook_dim, cfg.semantic_proj_dim),
            nn.GELU(),
            nn.Linear(cfg.semantic_proj_dim, cfg.w2v_bert_dim),
        )

    def forward(
        self,
        z: Tensor,
        w2v_targets: Optional[Tensor] = None,   # (B, T, w2v_bert_dim) from frozen w2v-BERT 2.0
    ) -> dict:
        """
        Args:
            z: (B, T, D) encoder output
            w2v_targets: targets for semantic distillation (training only)
        Returns dict with:
            z_q: final quantized representation
            all_codes: (B, T, n_codebooks) integer codes
            vq_loss: total VQ commitment loss
            semantic_loss: CB1 semantic distillation loss (0 if no targets)
        """
        residual = z
        z_q_total = torch.zeros_like(z)
        all_codes = []
        total_vq_loss = 0.0
        semantic_loss = torch.tensor(0.0, device=z.device)

        for i, codebook in enumerate(self.codebooks):
            z_q_i, indices_i, loss_i = codebook(residual)

            # Semantic distillation: only CB1
            if i == 0 and w2v_targets is not None:
                predicted = self.semantic_head(z_q_i)   # (B, T, w2v_bert_dim)
                semantic_loss = F.mse_loss(predicted, w2v_targets)

            z_q_total = z_q_total + z_q_i
            residual = residual - z_q_i.detach()
            all_codes.append(indices_i)
            total_vq_loss = total_vq_loss + loss_i

        all_codes = torch.stack(all_codes, dim=-1)      # (B, T, n_codebooks)

        return {
            "z_q": z_q_total,
            "all_codes": all_codes,
            "cb1_codes": all_codes[..., 0],             # semantic tokens for Slow AR
            "vq_loss": total_vq_loss,
            "semantic_loss": semantic_loss,
        }

    def decode_from_codes(self, codes: Tensor) -> Tensor:
        """
        codes: (B, T, n_codebooks)  integer indices
        Returns reconstructed latent (B, T, D)
        """
        z_q = torch.zeros(*codes.shape[:2], self.cfg.codebook_dim, device=codes.device)
        for i, codebook in enumerate(self.codebooks):
            z_q = z_q + codebook.embedding(codes[..., i])
        return z_q


class AudioEncoder(nn.Module):
    """
    Convolutional encoder: waveform → latent sequence at ~21Hz.
    Causal convolutions for streaming support.
    """
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        # Strided downsampling to reach ~21Hz from 44.1kHz
        # 44100 / (4 * 4 * 5 * 5 * ~2.5) ≈ 21Hz — exact strides tuned to hop_length
        self.stem = CausalConv1d(1, cfg.encoder_channels, kernel=7)

        self.downsample = nn.Sequential(
            nn.ELU(),
            CausalConv1d(cfg.encoder_channels, cfg.encoder_channels, kernel=4, dilation=1),
            nn.ELU(),
            # Progressive downsampling via strided convs
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 5, stride=5, padding=2),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 5, stride=5, padding=2),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 3, stride=2, padding=1),
        )

        self.blocks = nn.Sequential(*[
            ResBlock(cfg.encoder_channels, dilation=2**i)
            for i in range(cfg.encoder_depth)
        ])

        self.proj = nn.Linear(cfg.encoder_channels, cfg.codebook_dim)

    def forward(self, waveform: Tensor) -> Tensor:
        """waveform: (B, 1, samples) → (B, T, codebook_dim)"""
        x = self.stem(waveform)
        x = self.downsample(x)
        x = self.blocks(x)
        return self.proj(x.transpose(1, 2))             # (B, T, D)


class AudioDecoder(nn.Module):
    """
    Lightweight causal ConvNet decoder: quantized latent → waveform.
    No diffusion model needed — key latency advantage.
    """
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.codebook_dim, cfg.decoder_channels)

        self.upsample = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose1d(cfg.decoder_channels, cfg.decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(cfg.decoder_channels, cfg.decoder_channels, 5, stride=5, padding=2, output_padding=4),
            nn.ELU(),
            nn.ConvTranspose1d(cfg.decoder_channels, cfg.decoder_channels, 5, stride=5, padding=2, output_padding=4),
            nn.ELU(),
            nn.ConvTranspose1d(cfg.decoder_channels, cfg.decoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(cfg.decoder_channels, cfg.decoder_channels, 4, stride=2, padding=1),
        )

        self.blocks = nn.Sequential(*[
            ResBlock(cfg.decoder_channels, dilation=2**i)
            for i in range(cfg.decoder_depth)
        ])

        self.out = nn.Conv1d(cfg.decoder_channels, 1, kernel_size=7, padding=3)

    def forward(self, z_q: Tensor) -> Tensor:
        """z_q: (B, T, D) → waveform (B, 1, samples)"""
        x = self.proj(z_q).transpose(1, 2)     # (B, D, T)
        x = self.upsample(x)
        x = self.blocks(x)
        return torch.tanh(self.out(x))          # (B, 1, samples) in [-1, 1]


class LipikaRVQTokenizer(nn.Module):
    """
    Full RVQ tokenizer: Encoder + RVQ + Decoder.
    The fundamental codec underlying the entire Lipika system.

    Training losses:
        L_recon    = multi-scale spectral reconstruction
        L_vq       = VQ commitment (inside RVQ)
        L_semantic = CB1 → w2v-BERT 2.0 feature regression
        L_adv      = GAN discriminator (not included here, see tokenizer_trainer.py)
    """
    def __init__(self, cfg: Optional[RVQConfig] = None):
        super().__init__()
        self.cfg = cfg or RVQConfig()
        self.encoder = AudioEncoder(self.cfg)
        self.rvq     = ResidualVectorQuantizer(self.cfg)
        self.decoder = AudioDecoder(self.cfg)

    def encode(self, waveform: Tensor, w2v_targets: Optional[Tensor] = None) -> dict:
        z = self.encoder(waveform)
        return self.rvq(z, w2v_targets)

    def decode(self, codes: Tensor) -> Tensor:
        """codes: (B, T, n_codebooks) → waveform"""
        z_q = self.rvq.decode_from_codes(codes)
        return self.decoder(z_q)

    def forward(self, waveform: Tensor, w2v_targets: Optional[Tensor] = None) -> dict:
        encoded = self.encode(waveform, w2v_targets)
        reconstructed = self.decoder(encoded["z_q"])

        # Multi-scale spectral loss
        recon_loss = self._multi_scale_spectral_loss(waveform, reconstructed)

        return {
            **encoded,
            "reconstructed": reconstructed,
            "recon_loss": recon_loss,
            "total_loss": (
                recon_loss
                + encoded["vq_loss"]
                + 10.0 * encoded["semantic_loss"]   # semantic distillation weighted heavily
            ),
        }

    def _multi_scale_spectral_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """STFT loss at multiple scales for perceptual quality."""
        loss = 0.0
        for n_fft in [512, 1024, 2048]:
            window = torch.hann_window(n_fft, device=x.device)
            S_real  = torch.stft(x.squeeze(1),     n_fft, return_complex=True, window=window).abs()
            S_recon = torch.stft(x_hat.squeeze(1), n_fft, return_complex=True, window=window).abs()
            loss = loss + F.l1_loss(S_recon.log1p(), S_real.log1p())
        return loss / 3

    @property
    def frame_rate(self) -> float:
        return self.cfg.sample_rate / self.cfg.hop_length
