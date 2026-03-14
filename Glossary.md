# Lipika Tokenizer: Complete Glossary of Terms

## A Comprehensive Dictionary of Every Technical Term Used in the System

---

## A

### AdaLN (Adaptive Layer Normalization)
**Definition**: A normalization technique that applies learned scale and shift parameters to features based on conditioning information.
**In Lipika**: The ScriptFamilyAdapter uses AdaLN to modify encoder features based on which Indian language script is being processed. Instead of standard normalization that makes all features have zero mean and unit variance, AdaLN says "for Hindi, amplify these features; for Tamil, amplify those features."

### Adversarial Loss
**Definition**: A loss function from GANs where a generator tries to fool a discriminator, and the discriminator tries to correctly identify real vs fake samples.
**In Lipika**: The generator (encoder+quantizer+decoder) tries to create audio that fools the discriminator. The discriminator tries to spot which audio is reconstructed. This competition drives both to improve.

### Argmin
**Definition**: "Argument of the minimum" - the index that gives the smallest value in a list.
**In Lipika**: When finding the closest codebook entry, we compute distances to all 1024 entries and take the argmin - the index of the smallest distance. That index becomes our token.

### AudioConfig
**Definition**: A configuration class containing audio processing parameters like sample rate, FFT size, and mel bands.
**In Lipika**: Defines that we use 24kHz sampling, 2048-point FFT, 240-sample hop length, and 128 mel bands - all optimized for Indian language speech.

### AudioDecoder
**Definition**: The component that reconstructs waveforms from quantized latent vectors.
**In Lipika**: Takes the 8 tokens per frame, looks up their codebook vectors, sums them, and progressively upsamples back to 24kHz audio using transposed convolutions.

### AudioEncoder
**Definition**: The component that compresses raw waveforms into compact latent representations.
**In Lipika**: Uses causal convolutions with strides [2,4,5,6] to downsample 24kHz audio to 100Hz latent frames, while expanding from 1 channel to 512 features.

---

## B

### Backpropagation
**Definition**: The algorithm that computes gradients of the loss with respect to all model parameters by applying the chain rule from the output back to the input.
**In Lipika**: After computing the total loss, gradients flow backward through the decoder, quantizer, and encoder, updating all parameters to reduce the loss.

### Batch Size (B)
**Definition**: The number of audio samples processed simultaneously in one forward/backward pass.
**In Lipika**: Typically 8, meaning 8 different audio clips are processed together. This enables efficient GPU utilization and provides diverse examples for codebook updates.

### Bias
**Definition**: A learnable offset added to neural network computations, allowing the network to shift activation thresholds.
**In Lipika**: Used throughout convolutions and linear layers. The ScriptFamilyAdapter's retroflex bias is a special initialization, not a learnable parameter in the normal sense.

### Bottleneck
**Definition**: The narrowest part of an autoencoder where information is most compressed.
**In Lipika**: After the encoder's downsampling blocks (which expand channels to 8192), the bottleneck projects back to 512 dimensions - creating a compact representation that forces the model to keep only the most important information.

---

## C

### Causal Convolution
**Definition**: A convolution that only looks at past and current inputs, never future inputs, by padding only on the left side.
**In Lipika**: Essential for streaming applications where the model can't see future audio. All convolutions in encoder and decoder are causal.

### CausalConv1d
**Definition**: The specific implementation of 1D causal convolution used in Lipika.
**In Lipika**: Implements padding = (kernel_size - 1) * dilation on the left side only, then applies standard convolution. This ensures output at time t depends only on inputs at times ≤ t.

### CausalConvTranspose1d
**Definition**: A transposed convolution with causal properties, removing the non-causal right padding that standard transposed convolutions introduce.
**In Lipika**: Used in the decoder for upsampling while maintaining causality. After the transposed convolution, it trims the extra samples from the right.

### Channels
**Definition**: The number of parallel feature maps in a neural network layer.
**In Lipika**: Starts at 1 (raw audio), expands to 512 in encoder, up to 8192 in intermediate layers, then back to 512, and finally to 1 (reconstructed audio). Each channel learns to detect different acoustic features.

### CheckpointManager
**Definition**: A utility class that saves model state during training and manages rolling deletion of old checkpoints.
**In Lipika**: Saves model, discriminator, optimizers, and schedulers every 5000 steps. Keeps only the last 5 checkpoints to save disk space.

### Codebook
**Definition**: A lookup table containing discrete representations (vectors) that serve as the "vocabulary" of the tokenizer.
**In Lipika**: 8 codebooks, each with 1024 entries of 128-dimensional vectors. Each entry represents a learned acoustic pattern. Token indices (0-1023) point to entries in these codebooks.

### Codebook Dimension
**Definition**: The size of each vector in a codebook.
**In Lipika**: 128 dimensions. This is a trade-off: too small and can't capture enough detail, too large and computation becomes expensive. 128 works well for speech.

### Codebook Size
**Definition**: The number of entries in a codebook.
**In Lipika**: 1024 entries per codebook. 2^10, chosen because it's a power of 2 (efficient for computing) and provides enough variety without being too large.

### CodebookMonitor
**Definition**: A diagnostic tool that tracks codebook utilization and detects potential collapse.
**In Lipika**: Monitors usage percentage and perplexity for each codebook over a window of 100 steps. Raises warnings if any codebook uses less than 20% of its entries.

### CodebookMonitor.WINDOW
**Definition**: The number of training steps over which codebook statistics are averaged.
**In Lipika**: 100 steps. Long enough to smooth out batch-to-batch variations, short enough to detect trends quickly.

### Commitment Cost
**Definition**: A hyperparameter (β) that weighs the commitment loss in VQ-VAE training.
**In Lipika**: 0.25. This term encourages the encoder to output vectors close to the chosen codebook entries, preventing it from "wandering" far from the discrete representation space.

### Commitment Loss
**Definition**: The loss that encourages encoder outputs to stay close to their chosen codebook entries.
**In Lipika**: Computed as MSE between encoder output and chosen codebook vectors (with the codebook vectors detached from gradients). Multiplied by commitment_cost (0.25).

### Compression Ratio
**Definition**: The factor by which the encoder reduces the time dimension.
**In Lipika**: 240x compression. 24,000 samples per second become 100 frames per second (24,000 / 240 = 100).

### Continuous vs Discrete
**Definition**: Continuous values can be any real number; discrete values come from a finite set.
**In Lipika**: Encoder outputs continuous vectors (any 128-dimensional real vector). After quantization, we get discrete tokens (only 0-1023). This is the fundamental transformation Lipika performs.

### Cosine Schedule
**Definition**: A learning rate schedule that decays following a cosine curve from the initial value to a minimum.
**In Lipika**: Learning rate decays from initial (3e-4) to 10% of that over decay_steps, following cos(π × progress/2) shape. Provides smooth annealing.

---

## D

### DDP (Distributed Data Parallel)
**Definition**: PyTorch's distributed training paradigm where each GPU has a copy of the model and processes different batches, synchronizing gradients.
**In Lipika**: Used when training on multiple GPUs. The codebook EMA updates are all-reduced across GPUs so all replicas maintain identical codebooks.

### Dead Code
**Definition**: Codebook entries that are rarely or never used during training.
**In Lipika**: Entries used less than threshold_ema_dead_code (2) times in a batch are considered dead and get reset with random vectors from the current batch.

### Dead Code Reset
**Definition**: The process of reinitializing unused codebook entries with random samples from the current batch.
**In Lipika**: Prevents codebook collapse by giving dead entries a fresh start. Reset entries get their cluster_size set to threshold_dead to prevent immediate re-death.

### Decay (EMA)
**Definition**: The factor controlling how quickly EMA statistics update (higher = slower updates).
**In Lipika**: 0.99 for codebook EMA updates. New statistics contribute 1% weight, old statistics retain 99% weight. This smooths learning and prevents oscillations.

### DecoderBlock
**Definition**: A building block of the AudioDecoder that performs upsampling and residual processing.
**In Lipika**: Contains a transposed convolution for upsampling (with specified stride) followed by three residual blocks with dilations [1,3,9]. Each block halves the channel dimension.

### Dilated Convolution
**Definition**: A convolution where the kernel is applied with gaps (dilation) between samples, increasing receptive field without increasing parameters.
**In Lipika**: Residual blocks use dilations [1,3,9] to achieve large receptive fields (27 samples) with only 3 layers of kernel size 3.

### Discriminator
**Definition**: In GANs, the network that tries to distinguish real from generated samples.
**In Lipika**: MultiScaleMultiPeriodDiscriminator with 3 scale discriminators and 5 period discriminators. Provides adversarial training signal to the generator.

### Distillation
**Definition**: Training a smaller "student" model to mimic the outputs of a larger "teacher" model.
**In Lipika**: The first codebook (student) is trained to predict W2V-BERT features (teacher). This transfers linguistic knowledge from the large teacher to the compact codebook.

### Downsampling
**Definition**: Reducing the temporal resolution of a signal.
**In Lipika**: Encoder uses strided convolutions with strides [2,4,5,6] to progressively reduce 24kHz audio to 100Hz latent frames.

---

## E

### ELU (Exponential Linear Unit)
**Definition**: An activation function: f(x) = x if x>0, else α(exp(x)-1). Smooth for negative values, avoids dead neurons.
**In Lipika**: Used throughout encoder and decoder after convolutions. Preferred over ReLU for its smoothness and non-zero negative values.

### EMA (Exponential Moving Average)
**Definition**: A technique to compute a weighted average where older observations have exponentially decreasing weight.
**In Lipika**: Used for codebook updates instead of gradient descent. cluster_size and embed_avg are updated with EMA, then codebook entries are computed as embed_avg / cluster_size.

### Embedding
**Definition**: A learned vector representation of a discrete symbol.
**In Lipika**: The ScriptFamilyAdapter uses an embedding table (12 scripts × 64 dimensions) to convert script IDs into continuous vectors that can be processed by neural networks.

### EncoderBlock
**Definition**: A building block of the AudioEncoder that performs downsampling and residual processing.
**In Lipika**: Contains three residual blocks with dilations [1,3,9], followed by a strided convolution for downsampling (doubling channels) and average pooling.

### Epsilon (EMA)
**Definition**: A small constant added for numerical stability in EMA calculations.
**In Lipika**: 1e-5. Added to cluster_size during Laplace smoothing to prevent division by zero when normalizing codebook entries.

---

## F

### Feature Matching Loss
**Definition**: A loss that compares intermediate features from the discriminator for real and fake samples.
**In Lipika**: Computes L1 difference between discriminator feature maps for real and reconstructed audio. Encourages the generator to match the discriminator's internal representations, not just fool its final output.

### Feature Extractor (W2V-BERT)
**Definition**: The preprocessing component that converts raw audio into the format expected by W2V-BERT.
**In Lipika**: From HuggingFace's AutoFeatureExtractor. Handles resampling to 16kHz, padding, and normalization before feeding to the W2V-BERT model.

### FFT Size (n_fft)
**Definition**: The number of samples used in each Fourier transform when computing spectrograms.
**In Lipika**: 2048 for mel spectrograms; [256,512,1024,2048] for multi-scale STFT loss. Larger FFT sizes give better frequency resolution, smaller give better time resolution.

### Frame
**Definition**: A single temporal slice of processed audio.
**In Lipika**: At 24kHz with 240x compression, one frame represents 10ms of audio (240 samples). Each frame gets 8 tokens (one from each codebook).

### Frame Rate
**Definition**: The number of frames per second in the latent representation.
**In Lipika**: 100 frames per second (24,000 Hz / 240 compression = 100 Hz).

### Frozen Model
**Definition**: A model whose parameters are not updated during training.
**In Lipika**: The SemanticTeacher (W2V-BERT) is completely frozen. It provides stable targets for distillation without adapting to the student's mistakes.

---

## G

### GAN (Generative Adversarial Network)
**Definition**: A framework where a generator and discriminator are trained adversarially.
**In Lipika**: The generator (encoder+quantizer+decoder) tries to create realistic audio; the discriminator tries to distinguish real from reconstructed. This improves perceptual quality.

### GAN Phase
**Definition**: The period during training when GAN losses are active.
**In Lipika**: Starts after disc_start_step (10,000 steps). Before that, only reconstruction losses train the generator. This warmup prevents the discriminator from overpowering an untrained generator.

### GELU (Gaussian Error Linear Unit)
**Definition**: An activation function that weights inputs by their probability under a Gaussian distribution.
**In Lipika**: Used in the semantic head projection layers. Provides smooth, probabilistic gating.

### Generator
**Definition**: In GANs, the network that generates fake samples to fool the discriminator.
**In Lipika**: The combination of AudioEncoder, ResidualVectorQuantizer, and AudioDecoder. Takes real audio, compresses, quantizes, and reconstructs it.

### Gradient Clipping
**Definition**: Limiting the magnitude of gradients during training to prevent exploding gradients.
**In Lipika**: grad_clip = 1.0. Gradients are scaled down if their norm exceeds 1.0, ensuring stable training.

### Gradient Accumulation Steps
**Definition**: Accumulating gradients over multiple batches before updating weights, simulating larger batch sizes.
**In Lipika**: grad_accum_steps = 1 (no accumulation by default). Can be increased to effectively use larger batches with limited memory.

---

## H

### Hinge Loss
**Definition**: A loss function for GANs that uses a margin: L_D = max(0, 1-D(x)) + max(0, 1+D(G(z))).
**In Lipika**: Used for discriminator training. Creates a margin of 1 between real and fake scores, which stabilizes training compared to vanilla GAN loss.

### Hidden Layer (W2V-BERT)
**Definition**: The intermediate representations in the W2V-BERT model between input and output.
**In Lipika**: Uses layer 6 (0-indexed) of W2V-BERT's 24 layers. Research shows this layer best captures phone-level linguistic features.

### Hop Length
**Definition**: The number of samples between successive STFT frames.
**In Lipika**: 240 samples for mel spectrograms. At 24kHz, this gives 100 frames per second (24,000/240 = 100), matching the encoder's frame rate.

---

## I

### Identity Initialization
**Definition**: Initializing layers to perform identity mapping (output = input) at the start of training.
**In Lipika**: The scale_head is initialized with zero weights and one biases (so scale = 1). The shift_head is initialized with zero weights and zero biases (so shift = 0). This makes the adapter start as identity.

### Indices
**Definition**: The integer tokens produced by quantization, pointing to codebook entries.
**In Lipika**: After quantization, each frame gets 8 indices (0-1023). These are the discrete tokens that can be saved, transmitted, or used by language models.

### Inference
**Definition**: Using a trained model on new data (as opposed to training).
**In Lipika**: Two modes: encode (audio → tokens) and decode (tokens → audio). No gradients computed, teacher not used, codebook updates disabled.

### Input Projection
**Definition**: A linear layer that changes the dimension of encoder output to match codebook dimension.
**In Lipika**: Projects from 512 dimensions (encoder output) to 128 dimensions (codebook dimension) before quantization.

---

## K

### Kernel Size
**Definition**: The width of a convolutional filter - how many time steps it looks at simultaneously.
**In Lipika**: Varies: 7 in stem/out layers, 3 in residual blocks, 2*stride in downsampling layers. Larger kernels capture more context but cost more compute.

---

## L

### L1 Loss
**Definition**: Mean absolute error between predicted and target values.
**In Lipika**: Used for time-domain reconstruction loss (weight 0.1). More robust to outliers than L2 loss.

### Laplace Smoothing
**Definition**: Adding a small constant to counts before normalization to prevent division by zero and smooth estimates.
**In Lipika**: In codebook EMA updates: (cluster_size + epsilon) / (n + K*epsilon) * n. Ensures even rarely used codebooks get reasonable updates.

### Latent Representation (z)
**Definition**: The compressed, continuous representation of audio produced by the encoder.
**In Lipika**: Shape (B, T/240, 512). Each of the 100 frames per second has 512 features capturing different aspects of that 10ms audio segment.

### LayerNorm
**Definition**: A normalization technique that normalizes across the feature dimension for each sample independently.
**In Lipika**: Applied after the encoder's bottleneck and in the semantic head. Stabilizes training by ensuring consistent feature scales.

### Learning Rate
**Definition**: The step size for gradient descent updates - how much to change parameters based on gradients.
**In Lipika**: 3e-4 for generator, 3e-4 for discriminator. These are standard AdamW learning rates that work well for most audio tasks.

### Logits
**Definition**: The raw output scores from a classifier before applying an activation function like sigmoid.
**In Lipika**: Discriminator outputs logits (not probabilities). Positive means "real", negative means "fake". Hinge loss operates on these logits directly.

### Loss Weights
**Definition**: Multipliers for different loss components that balance their contributions to total loss.
**In Lipika**: L1=0.1, Mel=1.0, STFT=1.0, VQ=1.0, Semantic=10.0, Adv=3.0, Feat=3.0. Semantic gets highest weight because linguistic accuracy is critical.

---

## M

### Mel Spectrogram
**Definition**: A spectrogram where frequencies are converted to mel scale, which approximates human pitch perception.
**In Lipika**: Used in MelSpectrogramLoss. 128 mel bands from 0-12kHz. The mel scale has finer resolution at low frequencies (where speech is) and coarser at high frequencies.

### MelSpectrogramLoss
**Definition**: L1 loss between mel spectrograms of real and reconstructed audio.
**In Lipika**: Weight 1.0. More perceptually relevant than raw waveform loss because it emphasizes frequencies humans care about.

### Mixed Precision
**Definition**: Using lower-precision (16-bit) arithmetic for faster computation while maintaining 32-bit master weights.
**In Lipika**: Enabled by default on CUDA GPUs, automatically disabled on CPU/MPS. Uses bfloat16 if available (Ampere+ GPUs), otherwise float16.

### ModelConfig
**Definition**: Configuration class for model architecture parameters like channel sizes and depths.
**In Lipika**: Defines encoder_channels=512, decoder_channels=512, n_script_families=12, mpd_periods=[2,3,5,7,11], etc.

### MPD (Multi-Period Discriminator)
**Definition**: A discriminator that operates on audio reshaped by different periods to capture periodic patterns.
**In Lipika**: 5 discriminators with periods [2,3,5,7,11]. Each reshapes audio into (B, period, T/period) and applies 2D convolutions to detect patterns at that periodicity.

### MSD (Multi-Scale Discriminator)
**Definition**: A discriminator that operates on audio at different temporal resolutions.
**In Lipika**: 3 discriminators on original, 2x downsampled, and 4x downsampled audio. Each captures artifacts at different time scales.

### MultiScaleMultiPeriodDiscriminator
**Definition**: The combined discriminator containing both MSD and MPD components.
**In Lipika**: The full discriminator used for adversarial training. Returns lists of logits and feature maps from all sub-discriminators.

### MultiScaleSTFTLoss
**Definition**: STFT reconstruction loss computed at multiple FFT resolutions.
**In Lipika**: Uses FFT sizes [256,512,1024,2048]. Each resolution captures different time-frequency tradeoffs. Weight 1.0.

---

## N

### n_codebooks
**Definition**: The number of residual quantization layers.
**In Lipika**: 8 codebooks. Each captures increasingly fine details. First captures phonemes, later ones capture pitch, timbre, etc.

### n_fft
**Definition**: See FFT Size.

### n_mels
**Definition**: The number of mel frequency bands.
**In Lipika**: 128 bands. Enough to capture speech formants without being computationally expensive.

### n_script_families
**Definition**: The number of distinct script families Lipika supports.
**In Lipika**: 12, covering all major Indic scripts (Devanagari, Bengali, Tamil, etc.) plus Latin for Indian English.

### Normalization
**Definition**: Rescaling data to have consistent statistical properties.
**In Lipika**: Audio is peak-normalized to [-1, 1]. LayerNorm is applied to features. AdaLN adaptively normalizes based on script.

---

## O

### Output Projection
**Definition**: A linear layer that projects quantized vectors back to encoder dimension for the decoder.
**In Lipika**: Projects from 128 dimensions (codebook dimension) to 512 dimensions (decoder input dimension).

---

## P

### Perplexity
**Definition**: A measure of how uniformly codebook entries are used. exp(entropy). Higher = more uniform.
**In Lipika**: Monitored by CodebookMonitor. Maximum is codebook_size (1024) when all entries equally used. Low values (<100) indicate collapse.

### Pin Memory
**Definition**: A DataLoader setting that locks memory pages for faster GPU transfer.
**In Lipika**: True when using CUDA. Speeds up data transfer from CPU to GPU by using pinned (page-locked) memory.

### Period (Discriminator)
**Definition**: The period used to reshape audio for multi-period discrimination.
**In Lipika**: Periods [2,3,5,7,11]. Each captures different rhythmic patterns. 2 for fast oscillations, 11 for slow prosodic patterns.

---

## Q

### Quantization
**Definition**: The process of mapping continuous values to discrete ones.
**In Lipika**: Continuous encoder outputs (any 128-dim vector) are mapped to the nearest codebook entry (one of 1024 discrete vectors). This creates discrete tokens.

---

## R

### Receptive Field
**Definition**: The amount of input context that influences a particular output.
**In Lipika**: With dilations [1,3,9] in residual blocks, each block sees 27 samples of context. Stacking blocks increases receptive field dramatically.

### Reconstruction Loss
**Definition**: Losses that measure how well the reconstructed audio matches the original.
**In Lipika**: Includes L1 loss (time domain), mel loss (perceptual frequency), and multi-scale STFT loss (spectral accuracy).

### Residual
**Definition**: The difference between a target value and an approximation. What's left to fix.
**In Lipika**: In RVQ, after each codebook, residual = previous - quantized. Each codebook quantizes the current residual, not the original.

### Residual Block
**Definition**: A building block with a skip connection: output = input + F(input).
**In Lipika**: Used in encoder and decoder. The skip connection allows gradients to flow directly through the network, preventing vanishing gradients.

### ResidualVectorQuantizer (RVQ)
**Definition**: A quantization system with multiple codebooks, each quantizing the residual of the previous.
**In Lipika**: 8 codebooks in sequence. First quantizes original, second quantizes residual of first, etc. Final representation is sum of all.

### RETROFLEX_SCRIPTS
**Definition**: The set of script families that contain retroflex consonants.
**In Lipika**: Devanagari, Bengali, Gurmukhi, Oriya, Tamil, Telugu, Kannada, Malayalam. These scripts get a bias in their embeddings to help the network detect high-frequency retroflex sounds.

### Retroflex Bias
**Definition**: An initialization bias given to script embeddings for retroflex-rich languages.
**In Lipika**: The first 8 dimensions of embeddings for RETROFLEX_SCRIPTS are initialized with +0.5. This encourages the network to pay attention to high frequencies from the start.

### RVQConfig
**Definition**: Configuration class for residual vector quantizer parameters.
**In Lipika**: Defines n_codebooks=8, codebook_size=1024, codebook_dim=128, commitment_cost=0.25, ema_decay=0.99, threshold_ema_dead_code=2.

---

## S

### Sample Rate
**Definition**: The number of audio samples per second.
**In Lipika**: 24,000 Hz. Chosen to capture frequencies up to 12kHz (Nyquist limit), which includes retroflex sounds (up to ~10kHz) while being efficient.

### Scale (AdaLN)
**Definition**: The multiplicative factor in adaptive layer normalization.
**In Lipika**: Generated by ScriptFamilyAdapter. Multiplied with encoder features to amplify or attenuate specific feature channels based on script.

### ScriptFamily
**Definition**: Enumeration of Indic script families supported by Lipika.
**In Lipika**: 12 values: DEVANAGARI, BENGALI, GURMUKHI, GUJARATI, ORIYA, TAMIL, TELUGU, KANNADA, MALAYALAM, PERSO_ARABIC, MEITEI, LATIN_INDIA.

### ScriptFamilyAdapter
**Definition**: The component that conditions the encoder on script family using AdaLN.
**In Lipika**: Takes script IDs, looks up embeddings, projects to encoder dimension, generates scale and shift parameters, and applies them to encoder output.

### Semantic Head
**Definition**: A small neural network that projects first codebook outputs to predict teacher features.
**In Lipika**: LayerNorm → Linear(128→256) → GELU → Linear(256→1024). Forces first codebook to encode linguistically meaningful information.

### Semantic Loss
**Definition**: MSE between first codebook's predicted teacher features and actual teacher features.
**In Lipika**: Weight 10.0 - the highest weight of any loss. Critical for ensuring first token represents phonemes, not just acoustics.

### SemanticTeacher
**Definition**: The frozen W2V-BERT model that provides linguistic feature targets for distillation.
**In Lipika**: Uses "facebook/w2v-bert-2.0" model, extracts layer 6 hidden states, resamples input to 16kHz. Completely frozen during training.

### Shift (AdaLN)
**Definition**: The additive term in adaptive layer normalization.
**In Lipika**: Generated by ScriptFamilyAdapter. Added to encoder features to bias activation thresholds based on script.

### Spectral Normalization
**Definition**: A weight normalization technique that constrains the Lipschitz constant of a layer.
**In Lipika**: Applied to all discriminator convolutional layers. Stabilizes GAN training by preventing discriminator from overpowering the generator.

### Stem
**Definition**: The first layer of a neural network that processes raw input.
**In Lipika**: 7x1 causal convolution on raw waveform. Expands from 1 channel to 512 channels, providing initial feature extraction.

### STFT (Short-Time Fourier Transform)
**Definition**: A transform that computes frequency content over short time windows, producing a spectrogram.
**In Lipika**: Used in multi-scale STFT loss and mel spectrogram computation. Windowed with Hann window to reduce spectral leakage.

### Straight-Through Estimator
**Definition**: A technique to pass gradients through non-differentiable operations by replacing the backward pass with an identity function.
**In Lipika**: During quantization, forward pass uses argmin (non-differentiable), but backward pass treats quantizer as identity, allowing gradients to flow from decoder back to encoder.

### Stride
**Definition**: The step size of a convolution - how many samples it moves each time.
**In Lipika**: Encoder strides [2,4,5,6] progressively compress time. Decoder uses same strides in reverse to expand.

---

## T

### Tanh
**Definition**: Hyperbolic tangent activation function that outputs values in [-1, 1].
**In Lipika**: Used as final layer of decoder to ensure reconstructed waveform is in valid audio range (-1 to 1).

### Teacher (Semantic)
**Definition**: See SemanticTeacher.

### Tensor
**Definition**: A multi-dimensional array, the fundamental data structure in PyTorch.
**In Lipika**: All data (audio, latents, tokens, parameters) are tensors. Shapes are annotated as (Batch, Channels, Time) or (Batch, Time, Features).

### Threshold_ema_dead_code
**Definition**: The usage threshold below which codebook entries are considered dead and reset.
**In Lipika**: 2. If a codebook entry is used fewer than 2 times in a batch, it gets reinitialized with a random vector.

### Tokens
**Definition**: Discrete symbols produced by quantization, representing audio content.
**In Lipika**: For each 10ms frame, 8 tokens (0-1023). Total 800 tokens per second of audio. These can be processed by language models for TTS.

### TorchScript
**Definition**: PyTorch's intermediate representation that can be run independently from Python.
**In Lipika**: Models can be exported to TorchScript for production deployment without Python dependencies.

### TrainingConfig
**Definition**: Configuration class for training hyperparameters.
**In Lipika**: Defines batch_size, learning rates, loss weights, checkpointing frequency, etc.

### Transposed Convolution
**Definition**: A convolution that upsamples by inserting zeros between inputs, then convolving.
**In Lipika**: Used in decoder for upsampling. Often called "deconvolution" but technically a transposed convolution.

---

## U

### Upsampling
**Definition**: Increasing the temporal resolution of a signal.
**In Lipika**: Decoder uses transposed convolutions with strides [6,5,4,2] to go from 100Hz latent to 24kHz waveform.

### Usage Percentage
**Definition**: The fraction of codebook entries that have been used at least once.
**In Lipika**: Monitored by CodebookMonitor. Low usage (<20%) indicates codebook collapse where most entries are never selected.

---

## V

### VectorQuantizerEMA
**Definition**: A single vector quantizer with exponential moving average codebook updates.
**In Lipika**: Each codebook in RVQ is a VectorQuantizerEMA. Contains embedding table, EMA statistics, and dead code reset logic.

### VQ Loss
**Definition**: The loss associated with vector quantization, typically including commitment loss.
**In Lipika**: commitment_loss × commitment_cost. No embedding loss because EMA updates handle codebook learning without gradients.

---

## W

### W2V-BERT
**Definition**: A self-supervised speech model combining wav2vec 2.0 and BERT-style masked prediction.
**In Lipika**: Used as SemanticTeacher. Provides linguistic features for distillation. Model "facebook/w2v-bert-2.0" with 24 layers, 1024 hidden dimension.

### Warmup Steps
**Definition**: Initial training steps where learning rate increases linearly from 0 to target.
**In Lipika**: 1000 steps. Prevents large initial updates that could destabilize training.

### Waveform
**Definition**: The raw representation of audio as amplitude over time.
**In Lipika**: Input and output are waveforms at 24kHz, 16-bit, normalized to [-1, 1].

### Weight Decay
**Definition**: L2 regularization that penalizes large weights to prevent overfitting.
**In Lipika**: 1e-2 for AdamW optimizer. Helps generalization by keeping weights small.

---

## Other Symbols and Notations

### B (Batch Size)
**Definition**: The number of samples processed together.
**In Lipika**: Typically 8 during training. All tensor shapes start with B.

### T (Time)
**Definition**: The time dimension in samples or frames.
**In Lipika**: Raw audio: T = samples at 24kHz. Latent: T/240 = frames. Tokens: same as frames.

### (B, C, T) vs (B, T, C)
**Definition**: Different tensor layouts. (B, C, T) is "channels-first" (for conv layers). (B, T, C) is "channels-last" (for transformers/layernorm).
**In Lipika**: Convolutions use (B, C, T). After transpose, layernorm uses (B, T, C).

### [42, 137, 89, ...]
**Definition**: Example token sequence. Each number is an index (0-1023) pointing to a codebook entry.
**In Lipika**: 8 such numbers per frame. Together they uniquely identify the sound in that 10ms segment.

---

## Quick Reference Table

| Term | Value/Range | Purpose |
|------|-------------|---------|
| Sample Rate | 24,000 Hz | Capture up to 12kHz (retroflex sounds) |
| Compression Ratio | 240x | 24kHz → 100Hz latent |
| Frames per second | 100 | Temporal resolution |
| Encoder Channels | 512 → 8192 → 512 | Feature expansion/compression |
| Codebooks | 8 | Residual quantization layers |
| Codebook Size | 1024 | Vocabulary per codebook |
| Codebook Dimension | 128 | Vector size per entry |
| Tokens per second | 800 | 100 frames × 8 codebooks |
| Semantic Loss Weight | 10.0 | Force linguistic encoding |
| EMA Decay | 0.99 | Smooth codebook updates |
| Dead Code Threshold | 2 | Reset rarely used entries |
| Discriminator Periods | [2,3,5,7,11] | Capture rhythmic patterns |
| STFT Scales | [256,512,1024,2048] | Multi-resolution spectral loss |
| Script Families | 12 | All major Indic scripts |
| Commitment Cost | 0.25 | VQ-VAE beta parameter |
| Learning Rate | 3e-4 | AdamW optimizer step size |
| Warmup Steps | 1000 | Gradual LR increase |
| GAN Start Step | 10,000 | Warmup before adversarial training |

---

This glossary covers every technical term used in the Lipika Tokenizer system. Each term is defined in the context of how it's used in this specific implementation, with the values and purposes that make Lipika uniquely suited for Indian language speech processing!