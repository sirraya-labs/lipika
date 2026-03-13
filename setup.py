"""setup.py — Lipika TTS package"""
from setuptools import setup, find_packages

setup(
    name="lipika-tts",
    version="0.1.0",
    description="Lipika: Sovereign Foundational TTS for Indian Languages",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="India AI Mission",
    url="https://github.com/india-ai/lipika",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "torchaudio>=2.3.0",
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "einops>=0.7.0",
        "soundfile>=0.12.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "indic-nlp-library>=0.9.0",
    ],
    extras_require={
        "train": [
            "wandb>=0.17.0",
            "deepspeed>=0.14.0",
            "bitsandbytes>=0.43.0",
            "flash-attn>=2.5.0",
        ],
        "serve": [
            "fastapi>=0.111.0",
            "uvicorn>=0.30.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    entry_points={
        "console_scripts": [
            "lipika-synthesize=lipika.inference.cli:main",
            "lipika-serve=lipika.inference.engine:serve_cli",
        ]
    },
)