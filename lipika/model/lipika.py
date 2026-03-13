"""
lipika/model/lipika.py

Full Lipika model: assembles all components into a unified system.

Pipeline:
  1. Text Processor  → normalized text + inline tags → token IDs
  2. Script Adapter  → script family ID → AdaLN embeddings
  3. Slow AR         → CB1 tokens (with silent reasoning)
  4. Fast AR         → CB2..CB10 tokens (per CB1 frame)
  5. RVQ Decoder     → 44.1kHz waveform

This file also defines the combined training loss,
which is used in pretrain.py and rl_trainer.py.
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, Iterator
import time

from lipika.tokenizer.rvq_tokenizer import LipikaRVQTokenizer, RVQConfig
from lipika.tokenizer.script_adapter import ScriptFamilyAdapter, ScriptAdapterConfig
from lipika.model.slow_ar import SlowAR, SlowARConfig
from lipika.model.fast_ar import FastAR, FastARConfig


@dataclass
class LipikaConfig:
    """Master config referencing all sub-configs."""
    rvq: RVQConfig          = field(default_factory=RVQConfig)
    script: ScriptAdapterConfig = field(default_factory=ScriptAdapterConfig)
    slow_ar: SlowARConfig   = field(default_factory=SlowARConfig)
    fast_ar: FastARConfig   = field(default_factory=FastARConfig)

    # Loss weights
    w_recon: float          = 1.0
    w_vq: float             = 1.0
    w_semantic: float       = 10.0      # CB1 semantic distillation (important!)
    w_slow_ar: float        = 1.0
    w_fast_ar: float        = 1.0
    w_moe_aux: float        = 0.01      # load balancing
    w_reasoning: float      = 0.5       # reasoning trace supervision

    # Variant
    variant: str            = "live"    # "pro" | "live" | "lite"


class LipikaModel(nn.Module):
    """
    Lipika: Voice of India — full foundational TTS model.

    This class is the entry point for both training and inference.
    For serving, use lipika.inference.engine.LipikaEngine (SGLang-wrapped).
    """
    def __init__(self, cfg: Optional[LipikaConfig] = None):
        super().__init__()
        self.cfg = cfg or LipikaConfig()

        self.tokenizer  = LipikaRVQTokenizer(self.cfg.rvq)
        self.script_adapter = ScriptFamilyAdapter(self.cfg.script)
        self.slow_ar    = SlowAR(self.cfg.slow_ar)
        self.fast_ar    = FastAR(self.cfg.fast_ar)

    def forward(
        self,
        waveform: Tensor,                           # (B, 1, samples)
        text_ids: Tensor,                           # (B, T_text)
        tag_ids: Optional[Tensor] = None,
        language_codes: Optional[list[str]] = None,
        w2v_targets: Optional[Tensor] = None,       # from frozen w2v-BERT 2.0
        speaker_id: Optional[Tensor] = None,
        d_vector: Optional[Tensor] = None,
    ) -> dict:
        """
        Full training forward pass. Returns all losses.
        """
        B = waveform.shape[0]
        device = waveform.device

        # 1. Script family embeddings
        if language_codes:
            script_ids = torch.tensor(
                [self.script_adapter.get_script_id(lc) for lc in language_codes],
                device=device
            )
        else:
            script_ids = torch.zeros(B, dtype=torch.long, device=device)

        script_out = self.script_adapter(script_ids)

        # 2. RVQ tokenizer: audio → discrete codes + reconstruction
        rvq_out = self.tokenizer(waveform, w2v_targets)
        # rvq_out: {all_codes (B,T,10), cb1_codes (B,T), recon_loss, vq_loss, semantic_loss}

        # 3. Slow AR: text → CB1 prediction (teacher-forced)
        cb1_targets = rvq_out["cb1_codes"]          # (B, T)
        slow_out = self.slow_ar(
            text_ids=text_ids,
            speech_ids=cb1_targets,
            tag_ids=tag_ids,
            script_adapter_out=script_out,
        )
        # slow_out: {speech_logits (B,T,2048), moe_aux_loss}

        # 4. Fast AR: CB1 → CB2..CB10 prediction (teacher-forced)
        acoustic_targets = rvq_out["all_codes"][:, :, 1:]  # (B, T, 9) CB2..CB10
        fast_out = self.fast_ar(
            cb1_ids=cb1_targets,
            acoustic_ids=acoustic_targets,
            script_embed=script_out["embed"],
            speaker_id=speaker_id,
            d_vector=d_vector,
        )
        # fast_out: {acoustic_logits, acoustic_loss}

        # 5. Compute total loss
        import torch.nn.functional as F
        cb1_loss = F.cross_entropy(
            slow_out["speech_logits"].reshape(-1, self.cfg.slow_ar.speech_vocab_size),
            cb1_targets.reshape(-1),
        )

        total_loss = (
            self.cfg.w_recon    * rvq_out["recon_loss"]
            + self.cfg.w_vq     * rvq_out["vq_loss"]
            + self.cfg.w_semantic * rvq_out["semantic_loss"]
            + self.cfg.w_slow_ar  * cb1_loss
            + self.cfg.w_fast_ar  * fast_out["acoustic_loss"]
            + self.cfg.w_moe_aux  * slow_out["moe_aux_loss"]
        )

        return {
            "total_loss":       total_loss,
            "recon_loss":       rvq_out["recon_loss"],
            "vq_loss":          rvq_out["vq_loss"],
            "semantic_loss":    rvq_out["semantic_loss"],
            "cb1_loss":         cb1_loss,
            "acoustic_loss":    fast_out["acoustic_loss"],
            "moe_aux_loss":     slow_out["moe_aux_loss"],
        }

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        language: str = "hi",
        speaker_id: Optional[int] = None,
        d_vector: Optional[Tensor] = None,
        temperature: float = 0.85,
        top_p: float = 0.95,
        device: str = "cuda",
    ) -> Tensor:
        """
        Full synthesis pipeline. Returns (1, samples) waveform tensor.
        For streaming, use synthesize_stream().
        """
        from lipika.tokenizer.text_processor import TextProcessor
        processor = TextProcessor()

        # Tokenize text + extract inline tags
        text_ids, tag_ids = processor.encode(text, language)
        text_ids = text_ids.unsqueeze(0).to(device)
        tag_ids  = tag_ids.unsqueeze(0).to(device)

        # Script adapter
        script_id = torch.tensor([self.script_adapter.get_script_id(language)], device=device)
        script_out = self.script_adapter(script_id)
        spk_id_t = torch.tensor([speaker_id], device=device) if speaker_id else None

        # Slow AR: generate CB1 tokens
        cb1_tokens = self.slow_ar.generate(
            text_ids=text_ids,
            tag_ids=tag_ids,
            script_adapter_out=script_out,
            temperature=temperature,
            top_p=top_p,
        )                                           # (1, T_speech)

        # Fast AR: generate CB2..CB10 for each CB1 frame
        B, T = cb1_tokens.shape
        all_codes = torch.zeros(B, T, 10, dtype=torch.long, device=device)
        all_codes[:, :, 0] = cb1_tokens

        fast_kv = None
        for t in range(T):
            acoustic_codes, fast_kv = self.fast_ar.predict_frame(
                cb1_token=cb1_tokens[:, t],
                script_embed=script_out["embed"],
                speaker_id=spk_id_t,
                d_vector=d_vector,
                kv_caches=fast_kv,
            )                                       # (1, 9)
            all_codes[:, t, 1:] = acoustic_codes

        # Decode to waveform
        waveform = self.tokenizer.decode(all_codes)
        return waveform

    @torch.inference_mode()
    def synthesize_stream(
        self,
        text: str,
        language: str = "hi",
        speaker_id: Optional[int] = None,
        d_vector: Optional[Tensor] = None,
        chunk_frames: int = 10,                     # ~476ms audio chunks
        device: str = "cuda",
    ) -> Iterator[Tensor]:
        """
        Streaming synthesis. Yields audio chunks as they become available.
        Target TTFA < 100ms: first chunk emitted after first CB1 token.
        """
        from lipika.tokenizer.text_processor import TextProcessor
        processor = TextProcessor()

        text_ids, tag_ids = processor.encode(text, language)
        text_ids = text_ids.unsqueeze(0).to(device)
        tag_ids  = tag_ids.unsqueeze(0).to(device)

        script_id = torch.tensor([self.script_adapter.get_script_id(language)], device=device)
        script_out = self.script_adapter(script_id)
        spk_id_t = torch.tensor([speaker_id], device=device) if speaker_id else None

        # Generate CB1 tokens one at a time
        cb1_buffer = []
        code_buffer = []
        fast_kv = None

        for cb1_token in self._stream_slow_ar(text_ids, tag_ids, script_out, device):
            # Fast AR: immediately predict CB2..CB10 for this frame
            acoustic_codes, fast_kv = self.fast_ar.predict_frame(
                cb1_token=cb1_token,
                script_embed=script_out["embed"],
                speaker_id=spk_id_t,
                d_vector=d_vector,
                kv_caches=fast_kv,
            )

            frame_codes = torch.zeros(1, 1, 10, dtype=torch.long, device=device)
            frame_codes[:, 0, 0] = cb1_token
            frame_codes[:, 0, 1:] = acoustic_codes
            code_buffer.append(frame_codes)

            # Yield audio chunk every chunk_frames frames
            if len(code_buffer) >= chunk_frames:
                codes_chunk = torch.cat(code_buffer, dim=1)         # (1, chunk_frames, 10)
                audio_chunk = self.tokenizer.decode(codes_chunk)
                yield audio_chunk
                code_buffer = []

        # Flush remaining frames
        if code_buffer:
            codes_chunk = torch.cat(code_buffer, dim=1)
            yield self.tokenizer.decode(codes_chunk)

    def _stream_slow_ar(
        self,
        text_ids: Tensor,
        tag_ids: Tensor,
        script_out: dict,
        device: str,
    ) -> Iterator[Tensor]:
        """Generator yielding CB1 tokens one at a time."""
        # Full implementation in slow_ar.py; this calls the streaming variant
        cb1_all = self.slow_ar.generate(
            text_ids=text_ids,
            tag_ids=tag_ids,
            script_adapter_out=script_out,
        )
        for t in range(cb1_all.shape[1]):
            eos_id = self.cfg.slow_ar.speech_vocab_size + 1
            if (cb1_all[:, t] == eos_id).all():
                break
            yield cb1_all[:, t]     # (B,)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        variant: str = "live",
        device: str = "cuda",
    ) -> "LipikaModel":
        """
        Load a pretrained Lipika model from HuggingFace Hub or local path.
        model_id: e.g. "india-ai/lipika-sovereign-live"
        """
        import json
        from pathlib import Path

        # Config variants
        VARIANT_CONFIGS = {
            "pro":  {"slow_ar": {"n_layers": 32, "hidden_dim": 4096, "n_experts": 8}},
            "live": {"slow_ar": {"n_layers": 24, "hidden_dim": 3072, "n_experts": 8}},
            "lite": {"slow_ar": {"n_layers": 12, "hidden_dim": 1024, "n_experts": 4},
                     "fast_ar": {"n_layers": 8,  "hidden_dim": 512}},
        }

        cfg = LipikaConfig()
        # Patch config for variant
        overrides = VARIANT_CONFIGS.get(variant, {})
        for sub_cfg_name, sub_overrides in overrides.items():
            sub_cfg = getattr(cfg, sub_cfg_name)
            for k, v in sub_overrides.items():
                setattr(sub_cfg, k, v)

        model = cls(cfg)
        # In production: load weights from Hub
        # from huggingface_hub import hf_hub_download
        # weights_path = hf_hub_download(model_id, "model.safetensors")
        # model.load_state_dict(torch.load(weights_path, map_location=device))

        return model.to(device).eval()
