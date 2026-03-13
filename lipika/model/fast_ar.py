"""
lipika/model/fast_ar.py

Fast AR: 400M parameter transformer for CB2-CB10 (acoustic tokens).
Conditioned on CB1 tokens from Slow AR.

Key architectural property: this model is structurally isomorphic to
a standard decoder LLM → inherits SGLang/vLLM serving optimizations for free.

Multi-Token Prediction (MTP):
  Given CB1[t], predict CB2[t]...CB10[t] in PARALLEL using a shared
  transformer backbone + 9 parallel output heads.

  Standard AR:   CB1[0]→CB1[1]→...  (sequential, O(T) steps)
  MTP in FastAR: Given CB1[t], output CB2[t]..CB10[t] simultaneously

  This is the key to sub-100ms TTFA: once the Slow AR emits CB1[0],
  the Fast AR immediately produces all 9 acoustic codebooks for frame 0
  in a single forward pass.

Conditioning:
  1. CB1 prefix: the entire CB1 sequence prepended as context
  2. Script-family adapter: AdaLN injection (same as Slow AR)
  3. Speaker embedding: from reference audio or speaker ID lookup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class FastARConfig:
    # CB codebook sizes
    n_codebooks: int        = 10
    codebook_size: int      = 2048      # same as RVQ codebook size
    cb1_vocab_size: int     = 2048      # CB1 from Slow AR

    # Model size (400M)
    hidden_dim: int         = 1024
    n_heads: int            = 16
    n_kv_heads: int         = 4         # GQA
    n_layers: int           = 16
    intermediate_dim: int   = 4096

    # Context
    max_seq_len: int        = 8192

    # Conditioning
    script_embed_dim: int   = 1024
    speaker_embed_dim: int  = 512
    n_speakers: int         = 10000     # for speaker ID lookup

    # MTP: predict CB2-CB10 in parallel
    n_acoustic_codebooks: int = 9       # CB2..CB10

    # Inference
    temperature: float      = 0.8
    top_p: float            = 0.9


class FastARLayer(nn.Module):
    """Standard pre-norm transformer layer (no MoE — Fast AR is dense for speed)."""
    def __init__(self, cfg: FastARConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.hidden_dim // cfg.n_heads

        self.norm1 = nn.RMSNorm(cfg.hidden_dim)
        self.norm2 = nn.RMSNorm(cfg.hidden_dim)

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.hidden_dim,    bias=False)

        self.gate_proj = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(cfg.hidden_dim, cfg.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_dim, cfg.hidden_dim, bias=False)

        # AdaLN for script conditioning
        self.adaln_scale = nn.Linear(cfg.script_embed_dim, cfg.hidden_dim)
        self.adaln_shift = nn.Linear(cfg.script_embed_dim, cfg.hidden_dim)
        nn.init.zeros_(self.adaln_scale.weight)
        nn.init.zeros_(self.adaln_shift.weight)
        nn.init.ones_(self.adaln_scale.bias)
        nn.init.zeros_(self.adaln_shift.bias)

    def forward(
        self,
        x: Tensor,
        script_embed: Optional[Tensor] = None,  # (B, D)
        kv_cache: Optional[dict] = None,
    ) -> tuple[Tensor, Optional[dict]]:
        B, T, D = x.shape

        # Attention with AdaLN
        normed = self.norm1(x)
        if script_embed is not None:
            scale = self.adaln_scale(script_embed).unsqueeze(1)
            shift = self.adaln_shift(script_embed).unsqueeze(1)
            normed = normed * scale + shift

        q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache = {"k": k, "v": v}

        factor = self.n_heads // self.n_kv_heads
        k_exp = k.repeat_interleave(factor, dim=1)
        v_exp = v.repeat_interleave(factor, dim=1)

        attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(B, T, -1))

        # FFN (SwiGLU)
        normed2 = self.norm2(x)
        ffn_out = self.down_proj(F.silu(self.gate_proj(normed2)) * self.up_proj(normed2))
        x = x + ffn_out

        return x, kv_cache


class MultiTokenPredictionHead(nn.Module):
    """
    Parallel prediction heads for CB2..CB10.

    Given CB1[t] as context, predicts all 9 acoustic codebooks simultaneously.
    Each head is a small 2-layer MLP with its own output projection.

    Design choice: separate heads (not shared weights) because each codebook
    captures increasingly fine-grained acoustic detail and benefits from
    independent capacity.
    """
    def __init__(self, cfg: FastARConfig):
        super().__init__()
        self.n_acoustic = cfg.n_acoustic_codebooks
        self.codebook_size = cfg.codebook_size

        # One head per acoustic codebook (CB2..CB10)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.codebook_size),
            )
            for _ in range(cfg.n_acoustic_codebooks)
        ])

        # Cross-codebook conditioning:
        # CB(i+1) prediction is conditioned on CB(i) prediction via a small gating network
        # This captures the residual structure of RVQ (each CB refines the previous)
        self.cb_cond = nn.ModuleList([
            nn.Linear(cfg.codebook_size, cfg.hidden_dim)
            for _ in range(cfg.n_acoustic_codebooks - 1)
        ])

    def forward(self, hidden: Tensor) -> list[Tensor]:
        """
        hidden: (B, T, D) from FastAR backbone
        Returns list of 9 tensors, each (B, T, codebook_size)
        """
        logits_list = []
        h = hidden

        for i, head in enumerate(self.heads):
            logits_i = head(h)                  # (B, T, codebook_size)
            logits_list.append(logits_i)

            # Condition next head on this head's soft prediction (teacher-forced during training)
            if i < self.n_acoustic - 1:
                soft_pred = F.softmax(logits_i.detach(), dim=-1)    # (B, T, vocab)
                h = h + self.cb_cond[i](soft_pred)                  # residual conditioning

        return logits_list


class SpeakerEncoder(nn.Module):
    """
    Speaker conditioning from either:
      1. Speaker ID (lookup table, for known speakers)
      2. Reference audio embedding (d-vector, for zero-shot cloning)

    In both cases, produces a (B, speaker_embed_dim) vector projected to hidden_dim.
    """
    def __init__(self, cfg: FastARConfig):
        super().__init__()
        self.speaker_table = nn.Embedding(cfg.n_speakers, cfg.speaker_embed_dim)
        self.proj = nn.Linear(cfg.speaker_embed_dim, cfg.hidden_dim)

        # For reference audio: lightweight speaker verification network
        # (full SV network trained separately, only embedding used here)
        self.ref_proj = nn.Linear(cfg.speaker_embed_dim, cfg.hidden_dim)

    def from_id(self, speaker_id: Tensor) -> Tensor:
        return self.proj(self.speaker_table(speaker_id))

    def from_embedding(self, d_vector: Tensor) -> Tensor:
        """d_vector: (B, speaker_embed_dim) from external SV model"""
        return self.ref_proj(d_vector)


class FastAR(nn.Module):
    """
    The Fast AR model. Takes CB1 tokens from Slow AR → predicts CB2-CB10.

    In the Dual-AR pipeline:
      Slow AR: text → CB1 tokens (one at a time, ~21 tokens/sec)
      Fast AR: CB1[t] → CB2[t]..CB10[t] (all at once per frame)

    The Fast AR's latency dominates only the FIRST frame.
    After that, it runs in parallel with the Slow AR in a pipeline,
    achieving effective throughput of the Slow AR rate.

    TTFA = Slow AR time for first CB1 token + Fast AR time for first frame
         ≈ 50ms + 20ms = ~70ms on A100
    """
    def __init__(self, cfg: Optional[FastARConfig] = None):
        super().__init__()
        self.cfg = cfg or FastARConfig()

        # CB1 input embedding (receives tokens from Slow AR)
        self.cb1_embed = nn.Embedding(
            self.cfg.cb1_vocab_size + 2,    # +2 for BOS/EOS
            self.cfg.hidden_dim
        )

        # Acoustic token embeddings for teacher forcing (CB2..CB10)
        self.acoustic_embeds = nn.ModuleList([
            nn.Embedding(self.cfg.codebook_size + 2, self.cfg.hidden_dim)
            for _ in range(self.cfg.n_acoustic_codebooks)
        ])

        # Transformer backbone
        self.layers = nn.ModuleList([
            FastARLayer(self.cfg) for _ in range(self.cfg.n_layers)
        ])
        self.norm_out = nn.RMSNorm(self.cfg.hidden_dim)

        # Multi-token prediction head for CB2..CB10
        self.mtp_head = MultiTokenPredictionHead(self.cfg)

        # Speaker conditioning
        self.speaker_encoder = SpeakerEncoder(self.cfg)
        self.speaker_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        cb1_ids: Tensor,                            # (B, T) from Slow AR
        acoustic_ids: Optional[Tensor] = None,      # (B, T, 9) CB2..CB10 targets
        script_embed: Optional[Tensor] = None,      # (B, D) from ScriptFamilyAdapter
        speaker_id: Optional[Tensor] = None,        # (B,) known speaker
        d_vector: Optional[Tensor] = None,          # (B, 512) reference audio embedding
    ) -> dict:
        B, T = cb1_ids.shape

        # Build input: CB1 embeddings + speaker conditioning
        x = self.cb1_embed(cb1_ids)                 # (B, T, D)

        # Speaker conditioning via additive prefix embedding
        if speaker_id is not None:
            spk = self.speaker_encoder.from_id(speaker_id).unsqueeze(1)    # (B, 1, D)
            x = x + spk
        elif d_vector is not None:
            spk = self.speaker_encoder.from_embedding(d_vector).unsqueeze(1)
            x = x + spk

        # Transformer forward
        kv_caches = [None] * self.cfg.n_layers
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, script_embed, kv_caches[i])

        x = self.norm_out(x)

        # MTP: predict CB2..CB10 in parallel for each frame
        logits_list = self.mtp_head(x)              # list of 9 × (B, T, vocab)

        if acoustic_ids is not None:
            # Training: compute cross-entropy for each acoustic codebook
            losses = []
            for i, logits in enumerate(logits_list):
                target = acoustic_ids[:, :, i]      # (B, T)
                loss_i = F.cross_entropy(
                    logits.reshape(-1, self.cfg.codebook_size),
                    target.reshape(-1),
                    ignore_index=-1,
                )
                losses.append(loss_i)

            return {
                "acoustic_logits": logits_list,
                "acoustic_loss": sum(losses) / len(losses),
                "per_cb_losses": losses,
            }

        return {"acoustic_logits": logits_list}

    @torch.inference_mode()
    def predict_frame(
        self,
        cb1_token: Tensor,                          # (B,) single CB1 token for current frame
        script_embed: Optional[Tensor] = None,
        speaker_id: Optional[Tensor] = None,
        d_vector: Optional[Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> tuple[Tensor, list]:
        """
        Single-frame inference: given one CB1 token, predict CB2..CB10.
        Used in the real-time streaming pipeline.

        Returns:
            codes: (B, 9) predicted acoustic codes for CB2..CB10
            kv_caches: updated KV caches
        """
        B = cb1_token.shape[0]
        x = self.cb1_embed(cb1_token.unsqueeze(1))  # (B, 1, D)

        if speaker_id is not None:
            x = x + self.speaker_encoder.from_id(speaker_id).unsqueeze(1)
        elif d_vector is not None:
            x = x + self.speaker_encoder.from_embedding(d_vector).unsqueeze(1)

        if kv_caches is None:
            kv_caches = [None] * self.cfg.n_layers

        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, script_embed, kv_caches[i])

        x = self.norm_out(x[:, -1:])               # last position only

        logits_list = self.mtp_head(x)              # 9 × (B, 1, vocab)

        # Greedy decode (Fast AR uses lower temperature, acoustic details less creative)
        codes = torch.stack([
            l.squeeze(1).argmax(-1) for l in logits_list
        ], dim=-1)                                  # (B, 9)

        return codes, kv_caches
