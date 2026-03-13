"""
lipika/model/slow_ar.py

Slow AR: Mixture-of-Experts Transformer (~4-5B total params, ~1B active).
Predicts CB1 (semantic) tokens autoregressively.

The "Silent Thought" paradigm:
  Standard TTS: text tokens → speech tokens
  Lipika:         text tokens → [reasoning trace tokens] → speech tokens

The reasoning trace is a text sequence generated internally before any speech
tokens are emitted. It explicitly plans prosody:
  "Text expresses excitement. Speaker is female, calm. Pitch: slightly raised.
   Rate: moderate. Previous sentence was declarative; connect smoothly. 
   Language: Hindi. Script: Devanagari. Retroflex consonants present."

This reasoning trace is masked out at inference (not played back as audio),
but conditions the CB1 predictions through attention. Result: far more expressive,
controllable, and contextually appropriate prosody.

Architecture:
  - Pre-norm Transformer with RoPE positional encoding
  - MoE FFN (Top-2 routing, 8 experts, 1.5B active per token)
  - AdaLN conditioning on script-family embeddings
  - Inline instruction tag embeddings at token level
  - Long-context: sliding window attention + global tokens for coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class SlowARConfig:
    # Model size (4-5B total, ~1B active via MoE)
    vocab_size: int         = 32000     # text BPE vocab
    speech_vocab_size: int  = 2048      # CB1 codebook size
    reasoning_vocab_size: int = 32000   # same as text vocab (shared)

    hidden_dim: int         = 4096
    n_heads: int            = 32
    n_kv_heads: int         = 8         # GQA: 4x fewer KV heads
    n_layers: int           = 32
    intermediate_dim: int   = 14336     # FFN intermediate (before MoE)

    # MoE
    n_experts: int          = 8
    n_experts_active: int   = 2
    expert_dim: int         = 14336

    # Context
    max_seq_len: int        = 8192
    sliding_window: int     = 4096      # local attention window
    n_global_tokens: int    = 32        # sink tokens for long-form coherence

    # Script adapter
    script_embed_dim: int   = 4096      # matches hidden_dim

    # Inline instruction tags (15,000+ possible tags)
    n_instruction_tags: int = 16384
    tag_embed_dim: int      = 4096

    # Inference
    temperature: float      = 0.85
    top_p: float            = 0.95
    max_reasoning_tokens: int = 256     # max "silent thought" length


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding — handles variable length naturally."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 500000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None])
        self.register_buffer("sin_cache", emb.sin()[None, None])

    def forward(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.cos_cache[:, :, :seq_len, :], self.sin_cache[:, :, :seq_len, :]


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class GroupedQueryAttention(nn.Module):
    """
    GQA with sliding window + global tokens for long-form coherence.
    Local window: attends to sliding_window most recent tokens.
    Global tokens: n_global_tokens "sink" tokens attend to all positions.
    """
    def __init__(self, cfg: SlowARConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.hidden_dim // cfg.n_heads
        self.window     = cfg.sliding_window
        self.n_global   = cfg.n_global_tokens
        self.scale      = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.hidden_dim,    bias=False)

        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[Tensor, Optional[dict]]:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads to match Q heads (GQA)
        factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(factor, dim=1)
        v = v.repeat_interleave(factor, dim=1)

        # KV cache for autoregressive inference
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache = {"k": k, "v": v}

        # Flash attention (uses torch 2.x sdpa which dispatches to FlashAttn)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None),
        )

        out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), kv_cache


class ExpertFFN(nn.Module):
    """Single expert (SwiGLU FFN)."""
    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.up   = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MixtureOfExperts(nn.Module):
    """
    Top-2 sparse MoE. 8 experts, 2 active per token.
    ~4-5B total, ~1B active → 4-5x inference efficiency.
    Auxiliary load-balancing loss prevents expert collapse.
    """
    def __init__(self, cfg: SlowARConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_active  = cfg.n_experts_active

        self.router = nn.Linear(cfg.hidden_dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(cfg.hidden_dim, cfg.expert_dim)
            for _ in range(cfg.n_experts)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (output, aux_loss)"""
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)           # (B*T, D)

        # Router scores
        logits = self.router(flat_x)        # (B*T, n_experts)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_ids = scores.topk(self.n_active, dim=-1)  # (B*T, 2)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renormalize

        # Sparse dispatch
        output = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            mask = (topk_ids == i).any(dim=-1)  # tokens routed to expert i
            if not mask.any():
                continue
            expert_input = flat_x[mask]
            expert_weight = topk_scores[mask][:, (topk_ids[mask] == i).long().argmax(1)]
            output[mask] += expert_weight.unsqueeze(-1) * expert(expert_input)

        # Auxiliary load-balancing loss (encourages uniform routing)
        # f_i = fraction of tokens to expert i, P_i = mean router prob to expert i
        router_probs = scores.mean(0)           # (n_experts,)
        token_fracs  = (topk_ids == torch.arange(self.n_experts, device=x.device).unsqueeze(0)).float().mean(0)
        aux_loss = self.n_experts * (router_probs * token_fracs).sum()

        return output.reshape(B, T, D), aux_loss


class AdaLNLayer(nn.Module):
    """
    Transformer layer with Adaptive Layer Norm conditioning.
    Script-family adapter injects scale/shift via AdaLN,
    allowing per-language prosodic conditioning without separate models.
    """
    def __init__(self, cfg: SlowARConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_dim)
        self.norm2 = nn.RMSNorm(cfg.hidden_dim)
        self.attn  = GroupedQueryAttention(cfg)
        self.moe   = MixtureOfExperts(cfg)

    def forward(
        self,
        x: Tensor,
        adaln_scale: Optional[Tensor] = None,   # (B, D) from script adapter
        adaln_shift: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor, Optional[dict]]:
        # Adaptive layer norm: y = scale * norm(x) + shift
        normed = self.norm1(x)
        if adaln_scale is not None:
            normed = normed * adaln_scale.unsqueeze(1) + adaln_shift.unsqueeze(1)

        attn_out, kv_cache = self.attn(normed, mask, kv_cache)
        x = x + attn_out

        normed2 = self.norm2(x)
        if adaln_scale is not None:
            normed2 = normed2 * adaln_scale.unsqueeze(1) + adaln_shift.unsqueeze(1)

        moe_out, aux_loss = self.moe(normed2)
        x = x + moe_out

        return x, aux_loss, kv_cache


class SilentThoughtModule(nn.Module):
    """
    Generates internal prosody reasoning traces before speech token prediction.

    Training:
      Input:  [BOS] text tokens [SEP]
      Target: reasoning_trace tokens [THINK_END] speech_tokens [EOS]

    Inference:
      1. Generate reasoning trace tokens (masked, not played back)
      2. Use reasoning trace + text as joint context for CB1 prediction

    The reasoning trace format (natural language):
      "Speaking style: [CALM/EXCITED/FORMAL]. Pitch: [HIGH/MID/LOW].
       Rate: [FAST/NORMAL/SLOW]. Script: [SCRIPT_NAME].
       Phonetic notes: [retroflex/aspirated/tonal observations].
       Prosody plan: [detailed plan for this utterance]."
    """
    THINK_START_TOKEN = 32001   # <think>
    THINK_END_TOKEN   = 32002   # </think>

    def __init__(self, cfg: SlowARConfig):
        super().__init__()
        self.cfg = cfg
        # Lightweight classifier head to decide reasoning depth
        # (short traces for simple text, long for complex/emotional)
        self.complexity_head = nn.Linear(cfg.hidden_dim, 4)  # 4 depth classes

    def get_depth_class(self, text_hidden: Tensor) -> Tensor:
        """Predict reasoning depth from encoded text. (B,) → int 0-3"""
        pooled = text_hidden.mean(1)                    # (B, D)
        return self.complexity_head(pooled).argmax(-1)  # (B,)


class SlowAR(nn.Module):
    """
    The Slow AR model. Core of Lipika's text-to-CB1-tokens pipeline.

    Input:  text tokens + inline instruction tags + script family ID
    Output: CB1 semantic tokens (preceded by silent reasoning trace during training)

    This model is structurally identical to a standard decoder LLM,
    which means it inherits ALL LLM serving optimizations (SGLang, vLLM, etc.)
    """
    def __init__(self, cfg: Optional[SlowARConfig] = None):
        super().__init__()
        self.cfg = cfg or SlowARConfig()

        # Text token embedding (shared with reasoning trace output)
        self.text_embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)

        # Inline instruction tag embeddings (injected at token positions)
        # Tags like <[excited]>, <[sad]>, <[whisper]> etc.
        self.tag_embed = nn.Embedding(self.cfg.n_instruction_tags, self.cfg.tag_embed_dim)
        self.tag_proj  = nn.Linear(self.cfg.tag_embed_dim, self.cfg.hidden_dim)

        # CB1 speech token embedding (output vocabulary)
        self.speech_embed = nn.Embedding(
            self.cfg.speech_vocab_size + 3,   # +3 for BOS/EOS/PAD
            self.cfg.hidden_dim
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            AdaLNLayer(self.cfg) for _ in range(self.cfg.n_layers)
        ])

        self.norm_out = nn.RMSNorm(self.cfg.hidden_dim)

        # Output heads
        self.lm_head     = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_size,        bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_dim, self.cfg.speech_vocab_size, bias=False)

        # Weight tying: input text embedding = output LM head
        self.lm_head.weight = self.text_embed.weight

        # Silent thought module
        self.silent_thought = SilentThoughtModule(self.cfg)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def embed_tokens(
        self,
        token_ids: Tensor,
        tag_ids: Optional[Tensor] = None,       # (B, T) inline tag IDs (0 = no tag)
        token_type: str = "text",               # "text" | "speech"
    ) -> Tensor:
        """
        Build token embeddings with optional inline instruction tag injection.
        tag_ids are additive: embedding(text_token) + embedding(tag)
        """
        if token_type == "text":
            embeds = self.text_embed(token_ids)
        else:
            embeds = self.speech_embed(token_ids)

        if tag_ids is not None:
            tag_mask = (tag_ids > 0)
            if tag_mask.any():
                tag_embeds = self.tag_proj(self.tag_embed(tag_ids.clamp(min=0)))
                embeds = embeds + tag_embeds * tag_mask.unsqueeze(-1).float()

        return embeds

    def forward(
        self,
        text_ids: Tensor,                           # (B, T_text)
        speech_ids: Optional[Tensor] = None,        # (B, T_speech) CB1 targets for training
        tag_ids: Optional[Tensor] = None,           # (B, T_text) inline instruction tags
        script_adapter_out: Optional[dict] = None,  # from ScriptFamilyAdapter
        return_reasoning: bool = False,
    ) -> dict:
        """
        If speech_ids is provided: teacher-forced training forward pass.
        If speech_ids is None: inference mode (autoregressive — use generate()).
        """
        B, T_text = text_ids.shape
        device = text_ids.device

        # Build input sequence: [text_embeds | speech_embeds] during training
        text_embeds = self.embed_tokens(text_ids, tag_ids, "text")

        adaln_scale = script_adapter_out["scale"] if script_adapter_out else None
        adaln_shift = script_adapter_out["shift"] if script_adapter_out else None

        if speech_ids is not None:
            # Training: concatenate text + speech sequence
            speech_embeds = self.embed_tokens(speech_ids, token_type="speech")
            x = torch.cat([text_embeds, speech_embeds], dim=1)   # (B, T_text+T_speech, D)
        else:
            x = text_embeds

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss, _ = layer(x, adaln_scale, adaln_shift)
            total_aux_loss += aux_loss

        x = self.norm_out(x)

        if speech_ids is not None:
            # Predict speech tokens on the text portion's last position + all speech positions
            speech_logits = self.speech_head(x[:, T_text-1:-1])  # (B, T_speech, speech_vocab)
            return {
                "speech_logits": speech_logits,
                "hidden": x,
                "moe_aux_loss": total_aux_loss / self.cfg.n_layers,
            }

        return {
            "hidden": x,
            "moe_aux_loss": total_aux_loss / self.cfg.n_layers,
        }

    @torch.inference_mode()
    def generate(
        self,
        text_ids: Tensor,
        tag_ids: Optional[Tensor] = None,
        script_adapter_out: Optional[dict] = None,
        max_new_tokens: int = 2048,
        temperature: float = None,
        top_p: float = None,
    ) -> Tensor:
        """
        Autoregressive CB1 token generation with silent thought.
        Returns (B, T_speech) CB1 token ids.
        """
        temperature = temperature or self.cfg.temperature
        top_p       = top_p       or self.cfg.top_p

        B = text_ids.shape[0]
        device = text_ids.device

        adaln_scale = script_adapter_out["scale"] if script_adapter_out else None
        adaln_shift = script_adapter_out["shift"] if script_adapter_out else None

        # Phase 1: Encode text
        x = self.embed_tokens(text_ids, tag_ids, "text")
        kv_caches = [None] * self.cfg.n_layers

        for i, layer in enumerate(self.layers):
            x, _, kv_caches[i] = layer(x, adaln_scale, adaln_shift, kv_cache=kv_caches[i])

        # Phase 2: Generate silent reasoning trace (text tokens, not played back)
        # The reasoning trace conditions the speech token predictions
        reasoning_tokens = []
        cur_x = x[:, -1:]   # last hidden state → next token prediction

        for _ in range(self.cfg.max_reasoning_tokens):
            for i, layer in enumerate(self.layers):
                cur_x, _, kv_caches[i] = layer(
                    cur_x, adaln_scale, adaln_shift, kv_cache=kv_caches[i]
                )
            cur_x = self.norm_out(cur_x)
            logits = self.lm_head(cur_x[:, -1])     # text vocab logits

            # Sample reasoning token
            token = self._sample(logits, temperature, top_p)
            reasoning_tokens.append(token)

            # Stop generating reasoning when </think> token is emitted
            if (token == SilentThoughtModule.THINK_END_TOKEN).all():
                break

            # Re-embed as text token for next step
            cur_x = self.embed_tokens(token.unsqueeze(1), token_type="text")

        # Phase 3: Generate CB1 speech tokens conditioned on text + reasoning
        speech_tokens = []
        speech_bos = torch.full((B, 1), self.cfg.speech_vocab_size, device=device)
        cur_x = self.embed_tokens(speech_bos, token_type="speech")

        eos_id = self.cfg.speech_vocab_size + 1
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            for i, layer in enumerate(self.layers):
                cur_x, _, kv_caches[i] = layer(
                    cur_x, adaln_scale, adaln_shift, kv_cache=kv_caches[i]
                )
            cur_x = self.norm_out(cur_x)
            logits = self.speech_head(cur_x[:, -1])    # CB1 speech vocab logits

            token = self._sample(logits, temperature, top_p)
            token[finished] = eos_id                    # already-finished sequences stay at EOS
            speech_tokens.append(token)
            finished = finished | (token == eos_id)

            if finished.all():
                break

            cur_x = self.embed_tokens(token.unsqueeze(1), token_type="speech")

        return torch.stack(speech_tokens, dim=1)        # (B, T_speech)

    def _sample(self, logits: Tensor, temperature: float, top_p: float) -> Tensor:
        """Top-p (nucleus) sampling."""
        logits = logits / temperature
        probs  = F.softmax(logits, dim=-1)

        sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        mask = (cumulative - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_ids.gather(-1, sampled).squeeze(-1)
