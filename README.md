# 🎙️ Lipika — Voice of India

**Sovereign, SOTA Foundational TTS Model for Indian Languages**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/pytorch-2.3+-ee4c2c.svg)](https://pytorch.org/)

Lipika is an open-source, sovereign foundational TTS model designed to be the world's best for Indian languages. It combines the latest 2026 research: asymmetric Dual-AR decoding, RVQ tokenization with semantic distillation, inline token-level instruction control, and a "Silent Thought" reasoning paradigm.

## Architecture Overview

```
Input Text + Inline Style Tags
        │
        ▼
┌─────────────────────────────────────────────┐
│         Slow AR (MoE, 4-5B params)          │
│  ┌──────────────────────────────────────┐   │
│  │  Silent Thought Reasoning Traces     │   │
│  │  (internal prosody planning)         │   │
│  └──────────────────────────────────────┘   │
│  Predicts: CB1 (semantic tokens, ~21Hz)     │
└─────────────────┬───────────────────────────┘
                  │ CB1 tokens
                  ▼
┌─────────────────────────────────────────────┐
│         Fast AR (400M params)               │
│  Predicts: CB2-CB10 (acoustic tokens)       │
│  Multi-Token Prediction (MTP) in parallel   │
└─────────────────┬───────────────────────────┘
                  │ All 10 codebook tokens
                  ▼
┌─────────────────────────────────────────────┐
│    RVQ Decoder (lightweight ConvNet)         │
│    Reconstructs 44.1kHz audio               │
└─────────────────────────────────────────────┘
```

## Key Innovations

1. **Single RVQ Tokenizer** (21Hz, 10 codebooks) with w2v-BERT 2.0 semantic distillation on CB1
2. **Asymmetric Dual-AR**: 4-5B MoE Slow AR + 400M Fast AR — isomorphic to standard LLMs (SGLang-native)
3. **Silent Thought Reasoning**: LLM generates internal prosody monologue before speech tokens
4. **Inline Token-Level Instructions**: `<[excited]>` tags at word/phrase level, not utterance-level
5. **Unified RL Pipeline**: Data curation models == reward models (zero distribution shift)
6. **Long-Form Coherence**: Sliding KV speaker-state across context windows
7. **Script-Family Adapters**: Unicode-block-aware phonetic tokenization for all 22 scheduled languages

## Supported Languages

All 22 Scheduled Indian Languages + English, with Hinglish code-switching support.

Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Urdu, Kannada, Odia, Malayalam, Punjabi, Assamese, Maithili, Sanskrit, Santali, Kashmiri, Nepali, Sindhi, Konkani, Dogri, Manipuri, Bodo

## Quickstart

```bash
pip install lipika-tts
```

```python
from lipika import LipikaModel

model = LipikaModel.from_pretrained("india-ai/lipika-sovereign-live")

# Basic synthesis
audio = model.synthesize("नमस्ते, मैं वाणी हूँ।", language="hi")

# With inline style control
audio = model.synthesize(
    "This <[excited]>discovery<[/excited]> will change everything.",
    speaker="female_calm",
    language="en-IN"
)

# Stream for real-time (< 100ms TTFA)
for chunk in model.stream("आज का मौसम बहुत अच्छा है।"):
    play(chunk)
```

## Model Variants

| Variant | Params | Latency | Use Case |
|---------|--------|---------|----------|
| `lipika-sovereign-pro` | ~6B | ~2s | Highest quality, content creation |
| `lipika-sovereign-live` | ~4.5B | <100ms | Real-time conversational |
| `lipika-sovereign-lite` | ~0.6B | <50ms | On-device, edge deployment |

## Project Structure

```
lipika/
├── lipika/
│   ├── tokenizer/
│   │   ├── rvq_tokenizer.py      # RVQ with semantic distillation
│   │   ├── script_adapter.py     # Script-family adapters (22 languages)
│   │   └── text_processor.py     # Text normalization + G2P
│   ├── model/
│   │   ├── slow_ar.py            # MoE Slow AR (4-5B)
│   │   ├── fast_ar.py            # Fast AR (400M) + MTP head
│   │   ├── reasoning.py          # Silent Thought module
│   │   └── lipika.py               # Full model assembly
│   ├── training/
│   │   ├── tokenizer_trainer.py  # RVQ training
│   │   ├── pretrain.py           # LLM pre-training
│   │   ├── rl_trainer.py         # GRPO alignment
│   │   └── data_pipeline.py      # Dual-purpose data + reward
│   └── inference/
│       ├── engine.py             # SGLang-compatible serving
│       └── streaming.py          # Real-time streaming
├── configs/
│   ├── pro.yaml
│   ├── live.yaml
│   └── lite.yaml
├── scripts/
│   ├── train_tokenizer.sh
│   ├── pretrain.sh
│   └── rl_finetune.sh
└── tests/
```

## Training

See [TRAINING.md](TRAINING.md) for the full multi-stage training recipe.

## Citation

```bibtex
@misc{lipika2026,
  title={Lipika: A Sovereign Foundational TTS Model for Indian Languages},
  author={India AI Mission},
  year={2026},
  url={https://github.com/india-ai/lipika}
}
```

## License

Apache 2.0 — built for India, open to the world.
