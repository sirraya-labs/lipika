#!/usr/bin/env python3
"""
LIPIKA TOKENIZER - Production Version (CPU Optimized)
GAN Discriminator with CPU compatibility
"""

import os
import sys
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Any
from enum import IntEnum
from collections import defaultdict, Counter
import json
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
import numpy as np
from tqdm import tqdm

# ==================== AUDIO IO ====================
try:
    import soundfile as sf
    import librosa
    SF_AVAILABLE = True
except ImportError:
    print("ERROR: soundfile and librosa required. Install with: pip install soundfile librosa")
    sys.exit(1)
# =================================================

# ==================== CONFIGURATION ====================

@dataclass
class RVQConfig:
    """Audio and model configuration - Optimized for CPU"""
    # Audio
    sample_rate: int = 16000  # Reduced for CPU
    n_fft: int = 1024
    hop_length: int = 400      # 16000/400 = 40Hz (good for CPU)
    n_mels: int = 80
    
    # RVQ
    n_codebooks: int = 6       # Reduced from 8 for CPU
    codebook_size: int = 384    # Reduced from 512
    codebook_dim: int = 96      # Reduced from 128
    commitment_cost: float = 0.5
    
    # Encoder / Decoder
    encoder_channels: int = 192  # Reduced from 256
    encoder_depth: int = 5       # Reduced from 6
    decoder_channels: int = 192  # Reduced from 256
    decoder_depth: int = 5       # Reduced from 6
    
    # Semantic distillation
    w2v_bert_dim: int = 768      # Reduced from 1024
    semantic_proj_dim: int = 192
    
    # Script-family adapter
    n_script_families: int = 12
    script_embed_dim: int = 48
    
    # GAN Discriminator - Simplified for CPU
    disc_channels: int = 24      # Reduced from 32
    disc_depth: int = 3          # Reduced from 4


@dataclass
class TrainingConfig:
    """Training configuration - Optimized for CPU"""
    batch_size: int = 2          # Reduced for CPU
    learning_rate: float = 5e-4  # Slightly lower for stability
    num_epochs: int = 20         # Reduced from 50 for quicker testing
    num_workers: int = 0         # No multiprocessing on Windows
    device: str = "cpu"
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./training_data"
    max_duration: float = 2.0    # Shorter samples for CPU
    save_every: int = 100
    log_every: int = 5
    eval_every: int = 50
    
    # Loss weights
    recon_weight: float = 1.0
    vq_weight: float = 0.1
    semantic_weight: float = 5.0  # Reduced from 10
    gan_weight: float = 0.05      # Reduced for stability
    feature_matching_weight: float = 0.5
    
    # GAN training
    disc_update_freq: int = 2     # Update discriminator less frequently
    gp_weight: float = 5.0        # Reduced gradient penalty


# ==================== SCRIPT FAMILY ENUM ====================

class ScriptFamily(IntEnum):
    DEVANAGARI = 0
    BENGALI = 1
    GURMUKHI = 2
    GUJARATI = 3
    ORIYA = 4
    TAMIL = 5
    TELUGU = 6
    KANNADA = 7
    MALAYALAM = 8
    PERSO_ARABIC = 9
    MEITEI = 10
    LATIN_INDIA = 11


# ==================== GAN DISCRIMINATOR (CPU Optimized) ====================

class MultiScaleDiscriminator(nn.Module):
    """
    Simplified multi-scale discriminator for CPU training
    """
    
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        
        # Single scale discriminator for CPU (simpler)
        self.discriminator = self._build_discriminator(cfg.disc_channels)
        
    def _build_discriminator(self, channels: int) -> nn.Module:
        """Build a single discriminator with spectral normalization"""
        layers = []
        
        # Input: (B, 1, T)
        layers.append(spectral_norm(nn.Conv1d(1, channels, 15, stride=2, padding=7)))
        layers.append(nn.LeakyReLU(0.2))
        
        # Downsample progressively
        in_ch = channels
        for i in range(self.cfg.disc_depth):
            out_ch = in_ch * 2
            layers.append(spectral_norm(nn.Conv1d(in_ch, out_ch, 41, stride=2, padding=20)))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(spectral_norm(nn.Conv1d(out_ch, out_ch, 5, stride=1, padding=2)))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch
        
        # Final convolution to single channel
        layers.append(spectral_norm(nn.Conv1d(in_ch, 1, 3, padding=1)))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Returns discriminator output and features for feature matching
        """
        features = []
        y = x
        
        # Collect intermediate features
        for layer in self.discriminator:
            y = layer(y)
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                features.append(y)
        
        return [y], [features]  # Match interface of multi-scale version


# ==================== CODEBOOK HEALTH MONITOR ====================

class CodebookMonitor:
    """Monitor codebook usage during training"""
    
    def __init__(self, n_codebooks: int, codebook_size: int):
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.usage_history = [[] for _ in range(n_codebooks)]
        self.perplexity_history = []
        
    def update(self, all_codes: torch.Tensor):
        """
        all_codes: (B, T, n_codebooks) integer codes
        """
        B, T, n_cb = all_codes.shape
        
        for cb_idx in range(n_cb):
            codes = all_codes[:, :, cb_idx].reshape(-1).cpu().numpy()
            unique_codes = len(np.unique(codes))
            usage_pct = (unique_codes / self.codebook_size) * 100
            self.usage_history[cb_idx].append(usage_pct)
            
            # Calculate perplexity (measure of codebook utilization)
            counter = Counter(codes)
            probs = np.array([counter[i] for i in range(self.codebook_size)]) / len(codes)
            probs = probs[probs > 0]
            if len(probs) > 0:
                perplexity = np.exp(-np.sum(probs * np.log(probs)))
            else:
                perplexity = 0
            self.perplexity_history.append(perplexity)
    
    def get_report(self) -> Dict[str, Any]:
        """Get current codebook health report"""
        avg_usage = [np.mean(h[-50:]) if h else 0 for h in self.usage_history]
        avg_perplexity = np.mean(self.perplexity_history[-50:]) if self.perplexity_history else 0
        
        return {
            "avg_usage_per_codebook": avg_usage,
            "avg_perplexity": avg_perplexity,
            "codebook_collapse_warning": any(u < 20 for u in avg_usage),
        }


# ==================== MEL SPECTROGRAM LOSS ====================

class MelSpectrogramLoss(nn.Module):
    """Mel-spectrogram loss for better perceptual quality"""
    
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Pre-compute mel filterbank
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        
    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectrogram L1 loss
        x, x_hat: (B, 1, T)
        """
        # Compute STFT
        window = torch.hann_window(self.n_fft, device=x.device)
        
        def to_mel(y):
            stft = torch.stft(y.squeeze(1), self.n_fft, self.hop_length,
                             window=window, return_complex=True)
            mag = torch.abs(stft)  # (B, F, T)
            # Apply mel filterbank
            mel = torch.einsum("mf,bf...->bm...", self.mel_basis, mag)
            return torch.log1p(mel)  # Log scale for perception
        
        mel_x = to_mel(x)
        mel_x_hat = to_mel(x_hat)
        
        return F.l1_loss(mel_x_hat, mel_x)


# ==================== SCRIPT ADAPTER ====================

class ScriptFamilyAdapter(nn.Module):
    """Enhanced script-family embedding adapter"""
    
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        self.family_embed = nn.Embedding(cfg.n_script_families, cfg.script_embed_dim)
        
        self.register_buffer(
            "retroflex_bias",
            self._build_retroflex_bias(cfg.n_script_families, cfg.script_embed_dim)
        )
        
        self.proj = nn.Sequential(
            nn.Linear(cfg.script_embed_dim, cfg.encoder_channels),
            nn.SiLU(),
            nn.Linear(cfg.encoder_channels, cfg.encoder_channels),
        )
        
        # AdaLN (Adaptive Layer Normalization)
        self.adaln_scale = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        self.adaln_shift = nn.Linear(cfg.encoder_channels, cfg.encoder_channels)
        
        nn.init.zeros_(self.adaln_scale.weight)
        nn.init.zeros_(self.adaln_shift.weight)
        nn.init.ones_(self.adaln_scale.bias)
        nn.init.zeros_(self.adaln_shift.bias)
    
    def _build_retroflex_bias(self, n_families: int, dim: int) -> torch.Tensor:
        bias = torch.zeros(n_families, dim)
        retroflex_families = [
            ScriptFamily.DEVANAGARI, ScriptFamily.TAMIL, ScriptFamily.TELUGU,
            ScriptFamily.KANNADA, ScriptFamily.MALAYALAM, ScriptFamily.ORIYA,
            ScriptFamily.BENGALI, ScriptFamily.GURMUKHI,
        ]
        for f in retroflex_families:
            bias[f, :8] = 0.5
        return bias
    
    def forward(self, script_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.family_embed(script_ids) + self.retroflex_bias[script_ids]
        projected = self.proj(raw)
        return {
            "embed": projected,
            "scale": self.adaln_scale(projected),
            "shift": self.adaln_shift(projected),
        }


# ==================== CAUSAL CONVOLUTIONS ====================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ==================== VECTOR QUANTIZER ====================

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1 / codebook_size, 1 / codebook_size)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        flat_z = z.reshape(-1, D)
        
        dist = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(1).reshape(B, T)
        z_q = self.embedding(indices)
        
        commitment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        loss = embedding_loss + self.commitment_cost * commitment_loss
        
        z_q = z + (z_q - z).detach()
        return z_q, indices, loss


# ==================== RESIDUAL VECTOR QUANTIZER ====================

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        self.codebooks = nn.ModuleList([
            VectorQuantizer(cfg.codebook_size, cfg.codebook_dim, cfg.commitment_cost)
            for _ in range(cfg.n_codebooks)
        ])
        
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.codebook_dim, cfg.semantic_proj_dim),
            nn.GELU(),
            nn.Linear(cfg.semantic_proj_dim, cfg.w2v_bert_dim),
        )
        
        # Codebook monitor
        self.monitor = CodebookMonitor(cfg.n_codebooks, cfg.codebook_size)
    
    def forward(
        self,
        z: torch.Tensor,
        w2v_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        residual = z
        z_q_total = torch.zeros_like(z)
        all_codes = []
        total_vq_loss = 0.0
        semantic_loss = torch.tensor(0.0, device=z.device)
        
        for i, codebook in enumerate(self.codebooks):
            z_q_i, indices_i, loss_i = codebook(residual)
            
            if i == 0 and w2v_targets is not None:
                predicted = self.semantic_head(z_q_i)
                if predicted.shape[1] != w2v_targets.shape[1]:
                    min_len = min(predicted.shape[1], w2v_targets.shape[1])
                    predicted = predicted[:, :min_len, :]
                    w2v_targets_trimmed = w2v_targets[:, :min_len, :]
                    semantic_loss = F.mse_loss(predicted, w2v_targets_trimmed)
                else:
                    semantic_loss = F.mse_loss(predicted, w2v_targets)
            
            z_q_total = z_q_total + z_q_i
            residual = residual - z_q_i.detach()
            all_codes.append(indices_i)
            total_vq_loss = total_vq_loss + loss_i
        
        all_codes = torch.stack(all_codes, dim=-1)
        
        # Update codebook monitor
        if self.training:
            self.monitor.update(all_codes.detach())
        
        return {
            "z_q": z_q_total,
            "all_codes": all_codes,
            "cb1_codes": all_codes[..., 0],
            "vq_loss": total_vq_loss,
            "semantic_loss": semantic_loss,
        }
    
    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = torch.zeros(
            *codes.shape[:2], self.cfg.codebook_dim, device=codes.device
        )
        for i, codebook in enumerate(self.codebooks):
            z_q = z_q + codebook.embedding(codes[..., i])
        return z_q


# ==================== AUDIO ENCODER ====================

class AudioEncoder(nn.Module):
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        
        self.stem = CausalConv1d(1, cfg.encoder_channels, kernel=7)
        
        # Progressive downsampling
        self.downsample = nn.Sequential(
            nn.ELU(),
            CausalConv1d(cfg.encoder_channels, cfg.encoder_channels, 4),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 5, stride=5, padding=2),
        )
        
        self.blocks = nn.Sequential(*[
            ResBlock(cfg.encoder_channels, dilation=2**i)
            for i in range(cfg.encoder_depth)
        ])
        
        self.proj = nn.Linear(cfg.encoder_channels, cfg.codebook_dim)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.stem(waveform)
        x = self.downsample(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x
    
    def get_frame_count(self, num_samples: int) -> int:
        """Calculate number of frames after encoding"""
        x = num_samples
        x = x // 2
        x = x // 2
        x = x // 5
        return max(1, x)


# ==================== AUDIO DECODER ====================

class AudioDecoder(nn.Module):
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        
        self.proj = nn.Linear(cfg.codebook_dim, cfg.decoder_channels)
        
        self.upsample = nn.Sequential(
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
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.proj(z_q)
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = self.blocks(x)
        x = self.out(x)
        return torch.tanh(x)


# ==================== MAIN TOKENIZER WITH GAN ====================

class LipikaTokenizerWithGAN(nn.Module):
    """
    Complete RVQ tokenizer with GAN discriminator
    """
    
    def __init__(self, cfg: Optional[RVQConfig] = None):
        super().__init__()
        self.cfg = cfg or RVQConfig()
        
        self.encoder = AudioEncoder(self.cfg)
        self.rvq = ResidualVectorQuantizer(self.cfg)
        self.decoder = AudioDecoder(self.cfg)
        self.script_adapter = ScriptFamilyAdapter(self.cfg)
        self.discriminator = MultiScaleDiscriminator(self.cfg)
        
        # Loss functions
        self.mel_loss = MelSpectrogramLoss(
            self.cfg.sample_rate,
            self.cfg.n_fft,
            self.cfg.hop_length,
            self.cfg.n_mels
        )
    
    def forward(
        self,
        waveform: torch.Tensor,
        script_ids: Optional[torch.Tensor] = None,
        w2v_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode
        z = self.encoder(waveform)
        
        # Quantize
        quantized = self.rvq(z, w2v_targets)
        
        # Decode
        reconstructed = self.decoder(quantized["z_q"])
        
        # Losses
        recon_loss = F.l1_loss(reconstructed, waveform)
        mel_loss = self.mel_loss(waveform, reconstructed)
        multi_scale_loss = self._multi_scale_spectral_loss(waveform, reconstructed)
        
        # Combine reconstruction losses
        total_recon = (recon_loss + mel_loss + multi_scale_loss) / 3
        
        return {
            **quantized,
            "reconstructed": reconstructed,
            "recon_loss": total_recon,
            "mel_loss": mel_loss,
            "multi_scale_loss": multi_scale_loss,
            "l1_loss": recon_loss,
            "total_loss": (
                total_recon
                + quantized["vq_loss"]
                + (self.cfg.semantic_weight * quantized["semantic_loss"] if w2v_targets is not None else 0)
            ),
        }
    
    def _multi_scale_spectral_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0.0
        for n_fft in [512, 1024]:
            window = torch.hann_window(n_fft, device=x.device)
            S_real = torch.stft(
                x.squeeze(1), n_fft, return_complex=True, window=window
            ).abs()
            S_recon = torch.stft(
                x_hat.squeeze(1), n_fft, return_complex=True, window=window
            ).abs()
            loss = loss + F.l1_loss(S_recon.log1p(), S_real.log1p())
        return loss / 2
    
    def encode(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(waveform)
        return self.rvq(z)
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.rvq.decode_from_codes(codes)
        return self.decoder(z_q)
    
    @property
    def frame_rate(self) -> float:
        return self.cfg.sample_rate / self.cfg.hop_length


# ==================== GAN TRAINER (CPU Optimized) ====================

class GANTrainer:
    """Handles GAN training without autocast for CPU"""
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: TrainingConfig,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        self.device = torch.device(config.device)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Optimizers
        self.gen_optimizer = optim.AdamW(
            generator.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.5, 0.9)
        )
        
        self.disc_optimizer = optim.AdamW(
            discriminator.parameters(),
            lr=config.learning_rate * 2,
            weight_decay=0.01,
            betas=(0.5, 0.9)
        )
    
    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """WGAN-GP gradient penalty"""
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        alpha = alpha.expand_as(real)
        
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        disc_interpolated, _ = self.discriminator(interpolated)
        
        # Compute gradients
        grad = torch.autograd.grad(
            outputs=disc_interpolated[0],
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated[0]),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        grad = grad.view(batch_size, -1)
        grad_norm = grad.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        
        return gp
    
    def train_step(self, real_waveform: torch.Tensor) -> Dict[str, float]:
        """Single GAN training step on CPU"""
        
        # Train Discriminator
        for _ in range(self.config.disc_update_freq):
            self.disc_optimizer.zero_grad()
            
            # Generate fake audio
            with torch.no_grad():
                outputs = self.generator(real_waveform)
                fake_waveform = outputs["reconstructed"]
            
            # Discriminate
            real_outputs, real_features = self.discriminator(real_waveform)
            fake_outputs, fake_features = self.discriminator(fake_waveform.detach())
            
            # WGAN loss
            disc_loss = torch.mean(fake_outputs[0]) - torch.mean(real_outputs[0])
            
            # Gradient penalty
            gp = self.gradient_penalty(real_waveform, fake_waveform)
            disc_loss = disc_loss + self.config.gp_weight * gp
            
            disc_loss.backward()
            self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        # Generate
        outputs = self.generator(real_waveform)
        fake_waveform = outputs["reconstructed"]
        
        # Discriminate
        fake_outputs, fake_features = self.discriminator(fake_waveform)
        real_outputs, real_features = self.discriminator(real_waveform)
        
        # Adversarial loss
        adv_loss = -torch.mean(fake_outputs[0])
        
        # Feature matching loss
        feat_loss = 0
        for fake_feat, real_feat in zip(fake_features[0], real_features[0]):
            feat_loss += F.l1_loss(fake_feat, real_feat.detach())
        
        # Combine with reconstruction losses
        total_loss = (
            outputs["total_loss"]
            + self.config.gan_weight * adv_loss
            + self.config.feature_matching_weight * feat_loss
        )
        
        total_loss.backward()
        self.gen_optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "adv_loss": adv_loss.item(),
            "feat_loss": feat_loss.item(),
            "disc_loss": disc_loss.item(),
            "recon_loss": outputs["recon_loss"].item(),
            "vq_loss": outputs["vq_loss"].item(),
            "semantic_loss": outputs["semantic_loss"].item(),
        }


# ==================== DATASET ====================

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        encoder: AudioEncoder,
        sample_rate: int = 16000,
        max_duration: float = 2.0,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.encoder = encoder
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = []
        for ext in ['*.wav', '*.flac', '*.ogg', '*.mp3']:
            self.files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        if len(self.files) == 0:
            print(f"No audio files found in {data_dir}")
            print("Creating synthetic sine wave files for testing...")
            self._create_synthetic_data()
        
        print(f"Found {len(self.files)} audio files")
    
    def _create_synthetic_data(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(20):  # More synthetic data
            freq = 200 + i * 30
            duration = 2.0
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            
            # More complex waveforms
            audio = (0.5 * np.sin(2 * np.pi * freq * t) +
                     0.3 * np.sin(2 * np.pi * freq * 2 * t) +
                     0.2 * np.sin(2 * np.pi * freq * 3 * t))
            
            # Add some noise for realism
            audio += 0.01 * np.random.randn(len(audio))
            
            file_path = self.data_dir / f"synthetic_{i}.wav"
            sf.write(file_path, audio, self.sample_rate)
            self.files.append(file_path)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        try:
            audio, sr = sf.read(audio_path)
            audio = torch.from_numpy(audio).float()
            
            if len(audio.shape) > 1:
                audio = audio.mean(dim=-1)
            
            audio = audio.unsqueeze(0)
            
            if sr != self.sample_rate:
                audio = torch.from_numpy(
                    librosa.resample(audio.numpy(), orig_sr=sr, target_sr=self.sample_rate)
                ).float()
            
            if audio.shape[1] > self.max_samples:
                start = random.randint(0, audio.shape[1] - self.max_samples)
                audio = audio[:, start:start + self.max_samples]
            else:
                pad = self.max_samples - audio.shape[1]
                audio = F.pad(audio, (0, pad))
            
            T = self.encoder.get_frame_count(audio.shape[1])
            dummy_semantic = torch.randn(T, 768)  # W2V-BERT dimension
            
            return {
                "waveform": audio,
                "semantic_targets": dummy_semantic,
                "path": str(audio_path),
            }
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.files))


def collate_fn(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])
    
    max_T = max(item["semantic_targets"].size(0) for item in batch)
    dim = batch[0]["semantic_targets"].size(1)
    
    semantic_targets = []
    for item in batch:
        T = item["semantic_targets"].size(0)
        if T < max_T:
            padded = F.pad(item["semantic_targets"], (0, 0, 0, max_T - T))
            semantic_targets.append(padded)
        else:
            semantic_targets.append(item["semantic_targets"])
    
    semantic_targets = torch.stack(semantic_targets)
    
    return {
        "waveform": waveforms,
        "semantic_targets": semantic_targets,
        "paths": [item["path"] for item in batch],
    }


# ==================== MAIN TRAINING LOOP ====================

def main():
    model_cfg = RVQConfig()
    train_cfg = TrainingConfig()
    
    print("\n" + "=" * 70)
    print("LIPIKA - Sovereign TTS Tokenizer (CPU Optimized)")
    print("=" * 70)
    print(f"Sample rate: {model_cfg.sample_rate} Hz")
    print(f"Frame rate: {model_cfg.sample_rate / model_cfg.hop_length:.1f} Hz")
    print(f"Codebooks: {model_cfg.n_codebooks} x {model_cfg.codebook_size}")
    print(f"Encoder channels: {model_cfg.encoder_channels}")
    print(f"GAN Discriminator: Enabled (CPU optimized)")
    print(f"Device: {train_cfg.device}")
    print("=" * 70)
    
    # Create model
    model = LipikaTokenizerWithGAN(model_cfg)
    
    # Create dataset
    dataset = AudioDataset(
        data_dir=train_cfg.data_dir,
        encoder=model.encoder,
        sample_rate=model_cfg.sample_rate,
        max_duration=train_cfg.max_duration,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
    )
    
    # Create GAN trainer
    trainer = GANTrainer(
        generator=model,
        discriminator=model.discriminator,
        config=train_cfg,
    )
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_recon_loss = float('inf')
    
    for epoch in range(train_cfg.num_epochs):
        total_losses = defaultdict(float)
        start_time = time.time()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for batch in pbar:
            waveform = batch["waveform"].to(trainer.device)
            
            losses = trainer.train_step(waveform)
            global_step += 1
            
            for k, v in losses.items():
                total_losses[k] += v
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'recon': f"{losses.get('recon_loss', 0):.4f}",
                'adv': f"{losses.get('adv_loss', 0):.4f}",
            })
            
            # Log codebook health
            if global_step % 50 == 0:
                report = model.rvq.monitor.get_report()
                if report['codebook_collapse_warning']:
                    print(f"\n⚠️ Codebook collapse warning: Usage <20%")
                print(f"Codebook perplexity: {report['avg_perplexity']:.2f}")
            
            # Evaluation
            if global_step % train_cfg.eval_every == 0:
                recon_loss = losses.get('recon_loss', float('inf'))
                if recon_loss < best_recon_loss:
                    best_recon_loss = recon_loss
                    torch.save(model.state_dict(), Path(train_cfg.checkpoint_dir) / "best.pt")
                    print(f"\n*** New best model! recon_loss: {best_recon_loss:.6f}")
        
        # End of epoch
        epoch_time = time.time() - start_time
        avg_losses = {k: v / len(loader) for k, v in total_losses.items()}
        
        print(f"\nEpoch {epoch+1} complete in {epoch_time:.2f}s")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path(train_cfg.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.gen_optimizer.state_dict(),
            'losses': avg_losses,
        }, checkpoint_dir / f"epoch_{epoch+1}.pt")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best reconstruction loss: {best_recon_loss:.6f}")
    
    # Final test
    test_tokenizer(model, model_cfg, train_cfg.device)


def test_tokenizer(model: nn.Module, model_config: RVQConfig, device: str):
    """Test the trained tokenizer"""
    print("\n" + "=" * 70)
    print("TESTING TOKENIZER")
    print("=" * 70)
    
    model.eval()
    
    test_dir = Path("./test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Generate complex test tone
    duration = 2.0
    t = np.linspace(0, duration, int(duration * model_config.sample_rate))
    
    # Sweep frequency for better testing
    test_audio_np = (0.3 * np.sin(2 * np.pi * (220 + 50 * t) * t) +
                     0.2 * np.sin(2 * np.pi * 440 * t) +
                     0.1 * np.sin(2 * np.pi * 880 * t))
    
    orig_path = test_dir / "original.wav"
    sf.write(orig_path, test_audio_np, model_config.sample_rate)
    
    test_audio = torch.from_numpy(test_audio_np).float().unsqueeze(0).unsqueeze(0)
    test_audio = test_audio.to(device)
    
    with torch.no_grad():
        outputs = model(test_audio)
        codes = outputs["all_codes"]
        reconstructed = outputs["reconstructed"]
    
    print(f"Input shape: {test_audio.shape}")
    print(f"Codes shape: {codes.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Reconstruction loss: {outputs['recon_loss'].item():.6f}")
    print(f"Mel loss: {outputs['mel_loss'].item():.6f}")
    
    recon_path = test_dir / "reconstructed.wav"
    reconstructed_np = reconstructed.cpu().squeeze().numpy()
    sf.write(recon_path, reconstructed_np, model_config.sample_rate)
    
    print(f"\nFiles saved:")
    print(f"  Original: {orig_path}")
    print(f"  Reconstructed: {recon_path}")
    
    # Codebook health report
    report = model.rvq.monitor.get_report()
    print(f"\nCodebook Health:")
    print(f"  Avg perplexity: {report['avg_perplexity']:.2f}")
    for i, usage in enumerate(report['avg_usage_per_codebook']):
        print(f"  Codebook {i}: {usage:.1f}% usage")


if __name__ == "__main__":
    main()