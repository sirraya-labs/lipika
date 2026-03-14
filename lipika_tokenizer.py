#!/usr/bin/env python3


import os
import sys
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union
from enum import IntEnum
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ==================== AUDIO IO USING SOUNDFILE ONLY ====================
try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    print("ERROR: soundfile is required. Install with: pip install soundfile")
    sys.exit(1)
# =======================================================================

# ==================== CONFIGURATION ====================

@dataclass
class RVQConfig:
    """Configuration for the RVQ tokenizer"""
    # Audio
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 800     # 16000/800 = 20Hz
    n_mels: int = 80

    # RVQ
    n_codebooks: int = 4
    codebook_size: int = 256
    codebook_dim: int = 64
    commitment_cost: float = 0.25

    # Encoder / Decoder
    encoder_channels: int = 128
    encoder_depth: int = 4
    decoder_channels: int = 128
    decoder_depth: int = 4

    # Semantic distillation (simplified for testing)
    w2v_bert_dim: int = 64   # Match codebook_dim for simplicity
    semantic_proj_dim: int = 64

    # Script-family adapter
    n_script_families: int = 12
    script_embed_dim: int = 32


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "./checkpoints"
    data_dir: str = "./training_data"
    max_duration: float = 2.0
    save_every: int = 500
    log_every: int = 10


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


SCRIPT_RANGES = [
    (0x0900, 0x097F, ScriptFamily.DEVANAGARI),
    (0x0980, 0x09FF, ScriptFamily.BENGALI),
    (0x0A00, 0x0A7F, ScriptFamily.GURMUKHI),
    (0x0A80, 0x0AFF, ScriptFamily.GUJARATI),
    (0x0B00, 0x0B7F, ScriptFamily.ORIYA),
    (0x0B80, 0x0BFF, ScriptFamily.TAMIL),
    (0x0C00, 0x0C7F, ScriptFamily.TELUGU),
    (0x0C80, 0x0CFF, ScriptFamily.KANNADA),
    (0x0D00, 0x0D7F, ScriptFamily.MALAYALAM),
    (0x0600, 0x06FF, ScriptFamily.PERSO_ARABIC),
    (0x0041, 0x007A, ScriptFamily.LATIN_INDIA),
]


def detect_script_family(text: str) -> ScriptFamily:
    counts = {family: 0 for family in ScriptFamily}
    for char in text:
        cp = ord(char)
        for lo, hi, family in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[family] += 1
                break
    dominant = max(counts, key=lambda f: counts[f])
    return dominant if counts[dominant] > 0 else ScriptFamily.LATIN_INDIA


# ==================== SCRIPT ADAPTER ====================

class ScriptFamilyAdapter(nn.Module):
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
        
        # Semantic distillation head on CB1 only
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.codebook_dim, cfg.semantic_proj_dim),
            nn.GELU(),
            nn.Linear(cfg.semantic_proj_dim, cfg.w2v_bert_dim),
        )
    
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
            
            # Semantic distillation on CB1 only
            if i == 0 and w2v_targets is not None:
                predicted = self.semantic_head(z_q_i)
                # Ensure predicted and targets have same temporal dimension
                if predicted.shape[1] != w2v_targets.shape[1]:
                    # Take min length
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
        
        self.downsample = nn.Sequential(
            nn.ELU(),
            CausalConv1d(cfg.encoder_channels, cfg.encoder_channels, 4),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(cfg.encoder_channels, cfg.encoder_channels, 5, stride=5, padding=2),
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


# ==================== MAIN TOKENIZER ====================

class LipikaTokenizer(nn.Module):
    def __init__(self, cfg: Optional[RVQConfig] = None):
        super().__init__()
        self.cfg = cfg or RVQConfig()
        
        self.encoder = AudioEncoder(self.cfg)
        self.rvq = ResidualVectorQuantizer(self.cfg)
        self.decoder = AudioDecoder(self.cfg)
        self.script_adapter = ScriptFamilyAdapter(self.cfg)
    
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
        
        # Multi-scale spectral loss
        recon_loss = self._multi_scale_spectral_loss(waveform, reconstructed)
        
        return {
            **quantized,
            "reconstructed": reconstructed,
            "recon_loss": recon_loss,
            "total_loss": (
                recon_loss
                + quantized["vq_loss"]
                + (10.0 * quantized["semantic_loss"] if w2v_targets is not None else 0.0)
            ),
        }
    
    def _multi_scale_spectral_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0.0
        for n_fft in [512, 1024, 2048]:
            window = torch.hann_window(n_fft, device=x.device)
            S_real = torch.stft(
                x.squeeze(1), n_fft, return_complex=True, window=window
            ).abs()
            S_recon = torch.stft(
                x_hat.squeeze(1), n_fft, return_complex=True, window=window
            ).abs()
            loss = loss + F.l1_loss(S_recon.log1p(), S_real.log1p())
        return loss / 3
    
    def encode(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(waveform)
        return self.rvq(z)
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.rvq.decode_from_codes(codes)
        return self.decoder(z_q)
    
    @property
    def frame_rate(self) -> float:
        return self.cfg.sample_rate / self.cfg.hop_length


# ==================== DATASET ====================

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 2.0,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = []
        for ext in ['*.wav', '*.flac', '*.ogg', '*.aiff']:
            self.files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        if len(self.files) == 0:
            print(f"No audio files found in {data_dir}")
            print("Creating synthetic sine wave files for testing...")
            self._create_synthetic_data()
        
        print(f"Found {len(self.files)} audio files")
    
    def _create_synthetic_data(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(10):
            freq = 200 + i * 50
            duration = 2.0
            t = np.linspace(0, duration, int(duration * self.sample_rate))
            
            audio = (0.5 * np.sin(2 * np.pi * freq * t) +
                     0.3 * np.sin(2 * np.pi * freq * 2 * t) +
                     0.2 * np.sin(2 * np.pi * freq * 3 * t))
            
            file_path = self.data_dir / f"synthetic_{i}.wav"
            sf.write(file_path, audio, self.sample_rate)
            self.files.append(file_path)
    
    def _get_frame_count(self, num_samples: int) -> int:
        """Calculate number of frames after encoding"""
        # After all downsampling layers
        x = num_samples
        # Stem conv (kernel 7) - no stride
        # First conv4 (no stride)
        # First stride2 conv
        x = x // 2
        # Second stride2 conv
        x = x // 2
        # First stride5 conv
        x = x // 5
        # Second stride5 conv
        x = x // 5
        return max(1, x)
    
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
                old_len = audio.shape[1]
                new_len = int(old_len * self.sample_rate / sr)
                audio = F.interpolate(audio.unsqueeze(0), size=new_len, mode='linear').squeeze(0)
            
            if audio.shape[1] > self.max_samples:
                start = random.randint(0, audio.shape[1] - self.max_samples)
                audio = audio[:, start:start + self.max_samples]
            else:
                pad = self.max_samples - audio.shape[1]
                audio = F.pad(audio, (0, pad))
            
            # Calculate correct number of frames after encoding
            T = self._get_frame_count(audio.shape[1])
            
            # Create dummy semantic targets with correct temporal dimension
            dummy_semantic = torch.randn(T, self.cfg.w2v_bert_dim)  # (T, dim)
            
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
    
    # Get max temporal dimension for semantic targets
    max_T = max(item["semantic_targets"].size(0) for item in batch)
    dim = batch[0]["semantic_targets"].size(1)
    
    # Pad semantic targets
    semantic_targets = []
    for item in batch:
        T = item["semantic_targets"].size(0)
        if T < max_T:
            padded = F.pad(item["semantic_targets"], (0, 0, 0, max_T - T))
            semantic_targets.append(padded)
        else:
            semantic_targets.append(item["semantic_targets"])
    
    semantic_targets = torch.stack(semantic_targets)  # (B, T, dim)
    
    return {
        "waveform": waveforms,
        "semantic_targets": semantic_targets,
        "paths": [item["path"] for item in batch],
    }


# ==================== TRAINER ====================

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs * len(train_loader)
        )
        
        self.global_step = 0
        self.epoch = 0
        
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch) -> Dict[str, float]:
        self.model.train()
        
        waveform = batch["waveform"].to(self.device)
        semantic_targets = batch["semantic_targets"].to(self.device)
        
        outputs = self.model(waveform, w2v_targets=semantic_targets)
        
        self.optimizer.zero_grad()
        outputs["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": outputs["total_loss"].item(),
            "recon_loss": outputs["recon_loss"].item(),
            "vq_loss": outputs["vq_loss"].item(),
            "semantic_loss": outputs["semantic_loss"].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    def save_checkpoint(self):
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        
        path = Path(self.config.checkpoint_dir) / f"step_{self.global_step}.pt"
        torch.save(checkpoint, path)
        
        latest_path = Path(self.config.checkpoint_dir) / "latest.pt"
        torch.save(checkpoint, latest_path)
    
    def train(self):
        print("=" * 60)
        print("LIPIKA TOKENIZER TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        print("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            total_losses = defaultdict(float)
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                losses = self.train_step(batch)
                self.global_step += 1
                
                for k, v in losses.items():
                    total_losses[k] += v
                
                pbar.set_postfix({
                    'loss': f"{losses['loss']:.4f}",
                    'recon': f"{losses['recon_loss']:.4f}",
                })
                
                if self.global_step % self.config.log_every == 0:
                    print(f"Step {self.global_step}: {losses}")
                
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()
            
            avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}
            print(f"Epoch {epoch+1} complete: {avg_losses}")
            self.save_checkpoint()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        final_path = Path(self.config.checkpoint_dir) / "final.pt"
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")


# ==================== TESTING ====================

def test_tokenizer(model: nn.Module, config: TrainingConfig):
    print("\n" + "=" * 60)
    print("TESTING TOKENIZER")
    print("=" * 60)
    
    model.eval()
    
    test_dir = Path("./test_audio")
    test_dir.mkdir(exist_ok=True)
    
    duration = 2.0
    t = np.linspace(0, duration, int(duration * config.sample_rate))
    test_audio_np = (0.3 * np.sin(2 * np.pi * 220 * t) +
                     0.2 * np.sin(2 * np.pi * 440 * t) +
                     0.1 * np.sin(2 * np.pi * 880 * t))
    
    orig_path = test_dir / "original.wav"
    sf.write(orig_path, test_audio_np, config.sample_rate)
    
    test_audio = torch.from_numpy(test_audio_np).float().unsqueeze(0).unsqueeze(0)
    test_audio = test_audio.to(config.device)
    
    with torch.no_grad():
        outputs = model(test_audio)
        codes = outputs["all_codes"]
        reconstructed = outputs["reconstructed"]
    
    print(f"Input shape: {test_audio.shape}")
    print(f"Codes shape: {codes.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Reconstruction loss: {outputs['recon_loss'].item():.6f}")
    
    recon_path = test_dir / "reconstructed.wav"
    reconstructed_np = reconstructed.cpu().squeeze().numpy()
    sf.write(recon_path, reconstructed_np, config.sample_rate)
    
    print(f"Files saved:")
    print(f"  Original: {orig_path}")
    print(f"  Reconstructed: {recon_path}")
    print("Listen to both files. They should sound similar.")


# ==================== MAIN ====================

def main():
    model_cfg = RVQConfig()
    train_cfg = TrainingConfig()
    
    print("\n" + "=" * 60)
    print("LIPIKA - Sovereign TTS Tokenizer")
    print("=" * 60)
    print(f"Sample rate: {model_cfg.sample_rate} Hz")
    print(f"Frame rate: {model_cfg.sample_rate / model_cfg.hop_length:.1f} Hz")
    print(f"Codebooks: {model_cfg.n_codebooks} x {model_cfg.codebook_size}")
    print(f"Encoder channels: {model_cfg.encoder_channels}")
    print("=" * 60)
    
    # Store config in dataset for frame calculation
    AudioDataset.cfg = model_cfg
    
    dataset = AudioDataset(
        data_dir=train_cfg.data_dir,
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
    
    model = LipikaTokenizer(model_cfg)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=train_cfg,
    )
    
    trainer.train()
    test_tokenizer(model, train_cfg)
    
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()