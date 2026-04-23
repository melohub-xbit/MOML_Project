# CNN Architectures — Explained Simply

This document explains the three CNN architectures used in this project. It assumes you know only the very basics: what a CNN is, that it has layers, and that it classifies images.

---

## Quick Refresher: What's Inside a CNN?

A CNN processes an image step by step:

```
Image → [Conv layers extract features] → [FC layer makes decision] → Prediction
```

Each **convolutional (conv) layer** slides small filters across the image to detect patterns:
- Early layers detect simple things: **edges, corners, color blobs**
- Later layers combine those into complex things: **eyes, wheels, textures**

After the conv layers, the extracted features are flattened and passed to a **fully connected (FC) layer** which acts like a traditional classifier — it looks at all the features and picks a class.

Key terms we use:
- **Channels** — how many different filters a layer has. More channels = more patterns detected, but more parameters
- **Parameters** — the learnable numbers inside the network. More parameters = bigger model, slower inference
- **BatchNorm** — a trick that stabilizes training by normalizing values between layers
- **MaxPool** — shrinks the image by half (32×32 → 16×16), keeping only the strongest signals
- **Dropout** — randomly turns off some neurons during training so the model doesn't memorize the data

---

## Architecture 1: Plain CNN

**The straightforward approach.** Stack conv layers one after another, each followed by batch normalization and ReLU activation, then pool to shrink the spatial size.

### How it works

```
Input Image (e.g. 32×32×3)
    │
    ▼
┌─────────────────────────┐
│  Conv 3×3 → BatchNorm → ReLU  │  Layer 1: detect edges, simple textures
│  MaxPool 2×2                    │  (32×32 → 16×16)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Conv 3×3 → BatchNorm → ReLU  │  Layer 2: detect shapes, combinations
│  MaxPool 2×2                    │  (16×16 → 8×8)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  AdaptiveAvgPool → Flatten     │  Collapse spatial dims to a single vector
│  FC → ReLU → Dropout → FC     │  Classify based on all extracted features
└─────────────────────────┘
    │
    ▼
  Prediction (10 classes)
```

### What's a Conv 3×3?

A 3×3 convolution slides a tiny 3×3 window across the image. At each position, it multiplies the 9 pixel values by 9 learnable weights and sums them up. This produces one number that tells "how strongly does this patch match the pattern this filter learned?"

Each filter detects a different pattern. If you have 16 filters (channels), you get 16 different "pattern maps" as output.

### Channel doubling

In our implementation, channels **double** at each layer (up to the 3rd layer):
- Layer 1: `num_channels` (e.g. 16)
- Layer 2: `num_channels × 2` (e.g. 32)
- Layer 3: `num_channels × 4` (e.g. 64)
- Layer 4: `num_channels × 4` (stays at 64 — capped)

This is standard practice: as the image gets spatially smaller (via pooling), we increase the number of channels to preserve information capacity.

### Trade-off profile
- **Moderate accuracy** for a given parameter count
- **Moderate inference speed**
- The "default baseline" that the other two architectures improve upon

---

## Architecture 2: Residual CNN (Skip Connections)

**The problem Plain CNN has:** When you stack many layers, gradients (the learning signals) can become very small as they travel backward through the network. This makes deep networks hard to train — they might actually perform *worse* than shallow ones.

**The fix:** Add a **shortcut** (skip connection) that lets the input bypass the convolutional operations and get added directly to the output.

### How it works

```
Input ──────────────────────────────┐
  │                                 │
  ▼                                 │
┌───────────────────────┐           │
│ Conv 3×3 → BN → ReLU │           │  "Learn the residual"
│ Conv 3×3 → BN         │           │  (what to ADD to the input)
└───────────┬───────────┘           │
            │                       │
            │     ┌─────────────────┤
            │     │ 1×1 Conv → BN   │  Project input to match dimensions
            │     └────────┬────────┘
            │              │
            ▼              ▼
          [ output    +   skip  ]    ← Element-wise addition
                │
                ▼
              ReLU
                │
            MaxPool 2×2
```

### Why "residual"?

Instead of learning "what is the output?", the conv layers learn "what should I **change** about the input?" This is the **residual** — the difference between input and desired output.

If the best thing to do is nothing (identity), the conv layers can just learn to output zeros and the skip connection passes the input through unchanged. This makes it much easier for the network to be deep without losing information.

### The 1×1 projection

The skip connection uses a **1×1 convolution** — a convolution with a 1×1 filter. This doesn't look at spatial neighbors at all; it just linearly combines channels at each pixel position. We use it to change the number of channels so the skip path's output can be added to the main path's output (they must have the same shape to add).

### Trade-off profile
- **Higher accuracy** than Plain CNN at the same depth (skip connections help training)
- **More parameters** than Plain CNN (the skip projection adds weights)
- **Slightly slower** inference (two conv operations per block + skip)
- Best for pushing the **accuracy frontier** of the Pareto front

---

## Architecture 3: Depthwise-Separable CNN

**The idea:** A standard convolution is expensive because it simultaneously handles:
1. **Spatial** mixing (looking at neighboring pixels)
2. **Channel** mixing (combining information across channels)

Depthwise-separable convolutions **split this into two cheaper steps**.

### Standard Conv vs Depthwise-Separable

**Standard 3×3 Conv** (what Plain CNN uses):
- One filter looks at a 3×3 area across ALL input channels simultaneously
- For 32 input channels → 64 output channels: `32 × 64 × 3 × 3 = 18,432 parameters`

**Depthwise-Separable Conv** (two steps):

```
Step 1: Depthwise Conv (spatial only)
   - Each channel gets its OWN separate 3×3 filter
   - 32 channels → 32 separate 3×3 filters
   - Parameters: 32 × 3 × 3 = 288

Step 2: Pointwise Conv (channel mixing only)
   - A 1×1 convolution combines channels
   - Parameters: 32 × 64 × 1 × 1 = 2,048

Total: 288 + 2,048 = 2,336 parameters  (vs 18,432 — that's 8× fewer!)
```

### How it works

```
Input (e.g. 16×16 × 32 channels)
    │
    ▼
┌──────────────────────────────────┐
│  Depthwise Conv 3×3              │  Each channel filtered independently
│  (groups=in_channels)            │  Detects spatial patterns per channel
│  BatchNorm → ReLU                │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Pointwise Conv 1×1              │  Mix channel information
│  BatchNorm → ReLU                │  Change number of channels
└──────────────┬───────────────────┘
               │
               ▼
           MaxPool 2×2
```

### What does `groups=in_channels` mean?

In PyTorch, a normal `Conv2d` has `groups=1` — all input channels are connected to all output channels. When you set `groups=in_channels`, each input channel is processed by its own private filter with no cross-channel interaction. That's the "depthwise" part.

### Trade-off profile
- **Fewest parameters** by far (8× fewer than Plain CNN for same channel width)
- **Fastest inference** (fewer multiply-add operations)
- **Somewhat lower accuracy** (the factorization loses some representational power)
- Best for pushing the **compactness/speed frontier** of the Pareto front

---

## All Three Compared

Here are actual numbers from our sanity check (3 epochs, 5K training samples, CIFAR-10):

| Architecture | Params | Inference | Accuracy | Good at |
|---|---|---|---|---|
| **Plain CNN** | 7,946 | 0.51 ms | 35.2% | Balanced baseline |
| **Residual CNN** | 88,234 | 3.21 ms | 45.2% | Pushing accuracy higher |
| **DepthSep CNN** | 32,234 | 2.27 ms | 42.2% | Small + fast models |

> These numbers are from tiny configs (2 layers, 16–32 channels, only 3 epochs). The MOO optimizer will explore much larger configs and more epochs, pushing accuracy well above 80%.

### Why use all three in MOO?

Each architecture occupies a **different region** of the objective space:

```
                    Accuracy
                       ▲
                       │        ★ Residual (high acc, more params)
                       │
                       │   ★ DepthSep (decent acc, very few params)
                       │
                       │  ★ Plain (baseline)
                       │
                       └──────────────────────► Model Size (params)
```

By including all three as a search variable, the optimizer can discover the **full Pareto front** — from ultra-compact fast models (DepthSep, few channels) all the way to high-accuracy models (Residual, many channels). This produces a more diverse and interesting set of trade-off solutions.

---

## Shared Design Choices

All three architectures share these components:

1. **Stem layer** (Residual & DepthSep only): A standard conv that projects the input image channels (1 or 3) up to `num_channels`. This is needed because depthwise conv requires >1 channels, and residual needs consistent channel counts.

2. **Channel doubling**: Channels increase as `num_channels → 2× → 4×` across layers. Spatial dimensions shrink (via pooling), channel count grows — standard CNN practice.

3. **AdaptiveAvgPool2d(1)**: After all conv layers, this collapses whatever spatial size remains down to 1×1. This means the architecture works for any input resolution without changing the FC layer size.

4. **Classifier head**: `Flatten → FC → ReLU → Dropout → FC(num_classes)`. Same for all three architectures. The dropout here is the `dropout_rate` from the search space.
