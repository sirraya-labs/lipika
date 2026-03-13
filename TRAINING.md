# TRAINING.md — Vani Training Recipe

## Overview: 5-Stage Training Pipeline

```
Stage 0: Data Curation
    ↓
Stage 1: RVQ Tokenizer Training
    ↓
Stage 2: Slow AR Pre-training (text LM + CB1 prediction)
    ↓
Stage 3: Joint End-to-End Pre-training
    ↓
Stage 4: Post-training (Instruction Tuning + GRPO RL)
```

Estimated total compute: ~2,000 A100 GPU-hours per stage for the Live variant.

---

## Stage 0: Data Curation

### Target: 5M+ hours of Indian language speech

Sources:
- Government archives (All India Radio, Doordarshan, Parliament recordings)
- Academic corpora (IndicTTS, IITM, LDC-IL)
- Synthetic augmentation (text + pre-trained open TTS → filtered)
- User-contributed (privacy-preserving, on-device recording)

### Dual-Purpose Pipeline (critical — same models reused as RL rewards)

```python
# These models are trained ONCE and used in BOTH data pipeline and RL
quality_scorer = train_mos_predictor(human_rated_subset)   # ~10k rated samples
asr_model      = finetune_indic_whisper(labeled_data)       # WER measurement
sv_model       = train_speaker_verifier(speaker_labeled)    # d-vector extraction

# Filter data
keep = (
    (quality_scorer(audio) > 3.5)    # MOS > 3.5
    & (asr_model.wer(audio, text) < 0.1)  # WER < 10%
    & (audio_length > 1.0)           # > 1 second
    & (audio_length < 30.0)          # < 30 seconds
    & (snr(audio) > 20)              # SNR > 20dB
)
```

### Language Distribution (target hours)

| Language    | Target Hours | Notes |
|-------------|-------------|-------|
| Hindi       | 1,000,000   | Dominant, highest demand |
| Bengali     | 400,000     | |
| Telugu      | 400,000     | |
| Marathi     | 300,000     | |
| Tamil       | 300,000     | |
| Kannada     | 200,000     | |
| Gujarati    | 200,000     | |
| Malayalam   | 200,000     | |
| Urdu        | 200,000     | |
| Others (14) | 800,000     | ~57k each |
| English-IN  | 500,000     | Hinglish code-switching |
| **Total**   | **4,500,000+** | |

---

## Stage 1: RVQ Tokenizer Training

### Duration: ~500 A100-hours

```bash
./scripts/train_tokenizer.sh \
  --data_path /data/audio_44khz \
  --output_dir /checkpoints/rvq_v1 \
  --batch_size 128 \
  --lr 1e-4 \
  --warmup_steps 5000 \
  --max_steps 500000 \
  --w2v_bert_checkpoint /models/w2v_bert_2_0 \
  --semantic_loss_weight 10.0
```

### Loss components:
1. **L_recon**: Multi-scale STFT reconstruction (scales: 512, 1024, 2048)
2. **L_vq**: VQ commitment loss across all 10 codebooks
3. **L_semantic**: CB1 → w2v-BERT 2.0 regression (weight: 10x)
4. **L_adv**: GAN discriminator (multi-period + multi-scale, added at step 50k)

### Convergence target:
- PESQ > 3.5 (reconstruction quality)
- CB1 phone accuracy > 85% (semantic content)

---

## Stage 2: Slow AR Pre-training

### Phase 2a: Text-only LM warmup (500 A100-hours)
Pre-train the MoE transformer on text only (10B+ tokens of Indian language text).
This initializes strong language understanding before speech.

```bash
torchrun --nproc_per_node=8 vani/training/pretrain.py \
  --stage text_lm \
  --config configs/live.yaml \
  --data_path /data/text_corpus \
  --batch_size 512 \
  --lr 3e-4 \
  --max_steps 100000
```

### Phase 2b: Text + CB1 tokens (1000 A100-hours)
Mixed objective: next text token (40%) + next CB1 speech token (60%).
Reasoning traces injected at 20% of examples.

```bash
torchrun --nproc_per_node=8 vani/training/pretrain.py \
  --stage text_cb1 \
  --config configs/live.yaml \
  --data_path /data/speech_and_text \
  --batch_size 256 \
  --lr 1e-4 \
  --max_steps 300000 \
  --reasoning_trace_fraction 0.2 \
  --resume_from /checkpoints/text_lm/latest
```

---

## Stage 3: Joint End-to-End Pre-training

### Duration: ~1000 A100-hours

Full model: Slow AR + Fast AR + RVQ (tokenizer frozen).

```bash
torchrun --nproc_per_node=16 vani/training/pretrain.py \
  --stage joint \
  --config configs/live.yaml \
  --data_path /data/speech_paired \
  --batch_size 128 \
  --lr 5e-5 \
  --max_steps 500000 \
  --freeze_tokenizer true \
  --resume_from /checkpoints/text_cb1/latest
```

---

## Stage 4: Post-training

### 4a: Instruction Tuning (200 A100-hours)

```python
# Training format with inline instruction tags:
{
    "text": "This <[excited]>discovery<[/excited]> will <[slow]>change everything<[/slow]>.",
    "language": "en-IN",
    "audio": "/data/instruction/sample_001.wav"
}
```

Instruction dataset composition:
- 50k examples with emotion tags (excited, calm, sad, angry, etc.)
- 30k examples with rate/pitch control (fast, slow, high, low)
- 20k examples with style control (formal, casual, whisper, shout)
- 10k examples with code-switching (Hinglish, Tamlish, etc.)

### 4b: GRPO RL Alignment (100 A100-hours)

```bash
python vani/training/rl_trainer.py \
  --config configs/live.yaml \
  --base_model /checkpoints/instruction_tuned/latest \
  --reward_mos_model /checkpoints/mos_predictor \
  --reward_asr_model /checkpoints/indic_whisper \
  --reward_sv_model  /checkpoints/speaker_verifier \
  --group_size 8 \
  --kl_coeff 0.04 \
  --max_steps 5000 \
  --lr 1e-6
```

### 4c: Causal Prosody Fine-tuning (50 A100-hours)

Counterfactual training to disentangle emotion from content:
```python
# "How would this sound if spoken with joy instead of sadness?"
# Train with paired (sad_audio, joyful_audio, same_text) triplets
```

---

## Evaluation Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| MOS (naturalness) | > 4.3 | Human evaluation (500 annotators) |
| WER (Hindi) | < 2% | IndicWhisper-large ASR |
| WER (Tamil) | < 4% | Tamil ASR |
| WER (English-IN) | < 2% | Whisper-large-v3 |
| TTFA (Live) | < 100ms | A100 GPU, batch=1 |
| RTF (Live) | < 0.1 | Faster than real-time |
| Speaker SIM | > 0.85 | d-vector cosine similarity |
| EmergentTTS-Eval | > 90% | Paralinguistics win rate |

---

## Hardware Requirements

### Training (Full Live variant)
- 128x A100 80GB (NVLink + InfiniBand)
- Estimated wall-clock: ~2 weeks end-to-end
- Storage: 500TB for raw audio data

### Inference (Production)
- 1x A100 40GB per Live instance (32 concurrent requests)
- 1x A10G for Lite variant (on-premise edge)
- Edge/mobile: INT4 quantized, 6GB VRAM minimum

---

## Data Sovereignty Notes

All data processing, training, and model storage must occur within Indian data centers.
No raw audio data may be transmitted outside India's borders.
All compute to be provisioned on India AI Mission's sovereign compute cluster.
Model weights licensed under Apache 2.0 — usable globally, built sovereignly.
