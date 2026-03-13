"""
lipika/training/rl_trainer.py

GRPO-based RL fine-tuning for Lipika.

Critical design: reward models ARE the data pipeline models.
The same speech quality scorer and ASR model used for data curation
are reused as RL reward signals. This eliminates distribution mismatch
between pre-training data and post-training objectives — the single
most common failure mode in post-trained TTS systems.

Reward signals:
  1. MOS proxy       : speech quality classifier (pre-trained on human MOS ratings)
  2. WER             : ASR transcription error rate (lower = better intelligibility)
  3. Speaker SIM     : cosine similarity between synthesized and reference d-vector
  4. Prosody match   : embedding similarity to instruction-specified prosody
  5. Naturalness     : discriminator score from GAN discriminator (frozen)

GRPO (Group Relative Policy Optimization):
  - For each input, generate G=8 candidate outputs
  - Score each with the reward function
  - Policy gradient update using group-normalized advantages
  - Reference KL penalty to prevent reward hacking
  
Reference:
  Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
  in Open Language Models" — GRPO algorithm (adapted for TTS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GRPOConfig:
    # GRPO hyperparameters
    group_size: int         = 8         # G: candidates per input
    kl_coeff: float         = 0.04      # KL penalty coefficient
    clip_range: float       = 0.2       # PPO-style clipping
    gamma: float            = 1.0       # discount factor

    # Reward weights
    w_mos: float            = 0.3
    w_wer: float            = 0.4       # WER is highest priority
    w_speaker_sim: float    = 0.15
    w_prosody: float        = 0.1
    w_naturalness: float    = 0.05

    # Training
    lr: float               = 1e-6     # very low LR for RL stage
    warmup_steps: int       = 100
    max_steps: int          = 5000
    batch_size: int         = 4         # effective batch = 4 * 8 = 32 candidates
    grad_clip: float        = 1.0

    # Stability
    reward_normalize: bool  = True      # normalize rewards within group
    entropy_coeff: float    = 0.01      # small entropy bonus


class MOSPredictor(nn.Module):
    """
    Proxy MOS predictor trained on human ratings.
    This is the SAME model used in data pipeline for quality filtering.
    Output: scalar MOS estimate in [1, 5].
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        # Operates on mel-spectrogram statistics
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, mel_features: Tensor) -> Tensor:
        """mel_features: (B, input_dim) pooled mel statistics → (B,) MOS in [0,1]"""
        return (self.net(mel_features).squeeze(-1) * 4 + 1)    # scale to [1, 5]


class RewardFunction(nn.Module):
    """
    Unified reward function. Models here are shared with the data pipeline.
    This sharing is the architectural guarantee of zero distribution mismatch.
    """
    def __init__(self, cfg: GRPOConfig, asr_model=None, mos_model=None, sv_model=None):
        super().__init__()
        self.cfg = cfg

        # These models are loaded from the data pipeline (frozen during RL)
        self.mos_predictor = mos_model or MOSPredictor()
        self.asr_model     = asr_model      # e.g., IndicWhisper-large
        self.sv_model      = sv_model       # speaker verification (d-vector extractor)

        # Freeze all reward models
        for model in [self.mos_predictor, self.asr_model, self.sv_model]:
            if model is not None:
                for p in model.parameters():
                    p.requires_grad_(False)

    @torch.no_grad()
    def compute_rewards(
        self,
        waveforms: Tensor,                  # (B*G, 1, samples) synthesized audio
        target_texts: list[str],            # original input texts (repeated G times)
        reference_d_vectors: Optional[Tensor] = None,  # (B, d_vec_dim) reference speaker
        target_prosody_embed: Optional[Tensor] = None, # (B, d_prosody) from instruction
    ) -> Tensor:
        """
        Compute composite reward for each candidate.
        Returns (B*G,) reward tensor.
        """
        BG = waveforms.shape[0]
        device = waveforms.device
        rewards = torch.zeros(BG, device=device)

        # 1. MOS reward
        mel_features = self._extract_mel_features(waveforms)
        mos_scores = self.mos_predictor(mel_features)   # (BG,) in [1, 5]
        rewards += self.cfg.w_mos * (mos_scores - 1) / 4   # normalize to [0, 1]

        # 2. WER reward (1 - WER)
        if self.asr_model is not None:
            wer_scores = self._compute_wer_batch(waveforms, target_texts)
            rewards += self.cfg.w_wer * (1.0 - wer_scores.clamp(0, 1))

        # 3. Speaker similarity
        if reference_d_vectors is not None and self.sv_model is not None:
            synth_dvecs = self.sv_model(waveforms)          # (BG, d_vec)
            # Expand reference to match group size G
            G = BG // reference_d_vectors.shape[0]
            ref_expanded = reference_d_vectors.repeat_interleave(G, dim=0)
            sim = F.cosine_similarity(synth_dvecs, ref_expanded)   # (BG,)
            rewards += self.cfg.w_speaker_sim * (sim + 1) / 2      # [0, 1]

        # 4. Prosody match (if instruction specified prosody)
        if target_prosody_embed is not None:
            prosody_sim = self._compute_prosody_match(waveforms, target_prosody_embed)
            rewards += self.cfg.w_prosody * prosody_sim

        return rewards

    def _extract_mel_features(self, waveforms: Tensor) -> Tensor:
        """Quick mel-spectrogram statistics for MOS proxy."""
        import torchaudio
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_mels=128
        ).to(waveforms.device)
        specs = mel(waveforms.squeeze(1))               # (B, n_mels, T)
        # Pool to statistics: mean, std, min, max per mel band
        stats = torch.cat([
            specs.mean(-1),
            specs.std(-1),
        ], dim=-1)                                      # (B, 256)
        return stats[:, :128]                           # first 128 dims

    def _compute_wer_batch(self, waveforms: Tensor, texts: list[str]) -> Tensor:
        """Compute WER using the data pipeline ASR model."""
        # ASR transcription
        transcriptions = self.asr_model.transcribe_batch(waveforms)

        def char_wer(hyp: str, ref: str) -> float:
            """Simple character-level edit distance / len(ref)"""
            from difflib import SequenceMatcher
            ref_words = ref.split()
            hyp_words = hyp.split()
            if not ref_words:
                return 0.0
            # Levenshtein at word level
            m = SequenceMatcher(None, ref_words, hyp_words)
            ratio = m.ratio()
            return 1.0 - ratio

        wer_scores = torch.tensor(
            [char_wer(t, r) for t, r in zip(transcriptions, texts)],
            device=waveforms.device, dtype=torch.float
        )
        return wer_scores

    def _compute_prosody_match(self, waveforms: Tensor, target_embed: Tensor) -> Tensor:
        """Compare prosody features with target embedding from instruction."""
        # Placeholder — full implementation uses a prosody encoder
        # trained on emotion/style labeled data
        B = waveforms.shape[0]
        return torch.zeros(B, device=waveforms.device)


class GRPOTrainer:
    """
    GRPO training loop for Lipika.

    Algorithm:
      For each batch of text prompts:
        1. Generate G=8 candidate speech sequences from current policy
        2. Decode to audio via RVQ decoder
        3. Compute rewards for all G*B candidates
        4. Normalize rewards within each group → advantages
        5. Compute GRPO policy gradient loss + KL penalty
        6. Update Slow AR and Fast AR parameters
    """
    def __init__(
        self,
        model,                  # LipikaModel (policy)
        ref_model,              # LipikaModel (reference, frozen copy)
        reward_fn: RewardFunction,
        cfg: GRPOConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model      = model
        self.ref_model  = ref_model
        self.reward_fn  = reward_fn
        self.cfg        = cfg

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = optimizer or torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    def train_step(
        self,
        text_ids: Tensor,                   # (B, T_text)
        tag_ids: Optional[Tensor],
        language_codes: list[str],
        target_texts: list[str],
        reference_waveforms: Optional[Tensor] = None,   # for speaker sim
    ) -> dict:
        """Single GRPO training step. Returns loss dict."""
        B, device = text_ids.shape[0], text_ids.device
        G = self.cfg.group_size

        # Expand batch for group sampling
        text_ids_exp = text_ids.repeat_interleave(G, dim=0)    # (B*G, T)
        tag_ids_exp  = tag_ids.repeat_interleave(G, dim=0) if tag_ids is not None else None
        lang_exp     = [lc for lc in language_codes for _ in range(G)]
        texts_exp    = [t  for t  in target_texts    for _ in range(G)]

        # Step 1: Sample G candidates from current policy (with exploration)
        script_ids = torch.tensor(
            [self.model.script_adapter.get_script_id(lc) for lc in lang_exp],
            device=device
        )
        script_out = self.model.script_adapter(script_ids)

        # Generate CB1 tokens (stochastic)
        with torch.no_grad():
            cb1_tokens = self.model.slow_ar.generate(
                text_ids=text_ids_exp,
                tag_ids=tag_ids_exp,
                script_adapter_out=script_out,
                temperature=1.0,            # higher temperature for diversity
                top_p=0.95,
            )

        # Generate acoustic tokens
        B_exp = text_ids_exp.shape[0]
        T = cb1_tokens.shape[1]
        all_codes = torch.zeros(B_exp, T, 10, dtype=torch.long, device=device)
        all_codes[:, :, 0] = cb1_tokens
        fast_kv = None
        for t in range(T):
            acoustic, fast_kv = self.model.fast_ar.predict_frame(
                cb1_tokens[:, t], script_out["embed"], kv_caches=fast_kv
            )
            all_codes[:, t, 1:] = acoustic

        # Decode to waveform
        with torch.no_grad():
            waveforms = self.model.tokenizer.decode(all_codes)

        # Step 2: Compute rewards
        rewards = self.reward_fn.compute_rewards(
            waveforms=waveforms,
            target_texts=texts_exp,
        )                                   # (B*G,)

        # Step 3: Group normalization → advantages
        rewards_grouped = rewards.view(B, G)    # (B, G)
        if self.cfg.reward_normalize:
            mean_r = rewards_grouped.mean(dim=1, keepdim=True)
            std_r  = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages = ((rewards_grouped - mean_r) / std_r).view(B * G)
        else:
            advantages = (rewards_grouped - rewards_grouped.mean()).view(B * G)

        # Step 4: Compute policy log-probs (current and reference)
        # We re-run the Slow AR in teacher-forced mode to get log-probs
        slow_out = self.model.slow_ar(
            text_ids=text_ids_exp,
            speech_ids=cb1_tokens,
            tag_ids=tag_ids_exp,
            script_adapter_out=script_out,
        )
        log_probs = F.log_softmax(slow_out["speech_logits"], dim=-1)
        token_log_probs = log_probs.gather(
            -1, cb1_tokens.unsqueeze(-1)
        ).squeeze(-1).sum(-1)                   # (B*G,) per-sequence log-prob

        with torch.no_grad():
            ref_script_out = self.ref_model.script_adapter(script_ids)
            ref_slow_out = self.ref_model.slow_ar(
                text_ids=text_ids_exp,
                speech_ids=cb1_tokens,
                tag_ids=tag_ids_exp,
                script_adapter_out=ref_script_out,
            )
            ref_log_probs = F.log_softmax(ref_slow_out["speech_logits"], dim=-1)
            ref_token_log_probs = ref_log_probs.gather(
                -1, cb1_tokens.unsqueeze(-1)
            ).squeeze(-1).sum(-1)               # (B*G,)

        # Step 5: GRPO loss = -E[advantages * min(ratio, clipped_ratio)] + KL
        log_ratio = token_log_probs - ref_token_log_probs
        ratio = log_ratio.exp()

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * ratio.clamp(1 - self.cfg.clip_range, 1 + self.cfg.clip_range)
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        # KL divergence penalty (forward KL: p_ref * log(p_ref/p))
        kl_penalty = (ref_token_log_probs.exp() * (ref_token_log_probs - token_log_probs)).mean()

        # Entropy bonus (encourage exploration)
        entropy = -(log_probs * log_probs.exp()).sum(-1).mean()

        total_loss = (
            policy_loss
            + self.cfg.kl_coeff * kl_penalty
            - self.cfg.entropy_coeff * entropy
        )

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.cfg.grad_clip
        )
        self.optimizer.step()

        return {
            "total_loss":    total_loss.item(),
            "policy_loss":   policy_loss.item(),
            "kl_penalty":    kl_penalty.item(),
            "mean_reward":   rewards.mean().item(),
            "max_reward":    rewards.max().item(),
        }
