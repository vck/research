# From the Ground Up: Derivatives → Backprop → Neural Networks

A complete, layered learning roadmap — from the simplest autograd engine to advanced neural architectures.

Each section only uses what came before. No prior deep learning knowledge assumed.

---

## The Three Pillars

| Pillar | What | Where |
|--------|------|-------|
| **Micrograd** | The simplest possible autograd engine | [`notebook/`](notebook/) |
| **The Lecture** | Full mathematical derivation (10 parts) | [`book/`](book/), [`slides/`](slides/) |
| **The Roadmap** | Where to go from here | This document |

---

## 🗺️ Learning Roadmap

```
Micrograd Engine          Core Math              Modern Architectures
     │                      │                        │
     ▼                      ▼                        ▼
 ┌──────────┐        ┌──────────────┐       ┌─────────────────┐
 │  Stage 1  │        │  Stage 2     │       │  Stage 3        │
 │  Build it  │──────│  Understand  │───────│  Scale it       │
 │  from scratch│     │  the math    │       │  & generalize   │
 └──────────┘        └──────────────┘       └─────────────────┘
```

---

## Stage 1: Build the Engine (Micrograd)

**Goal:** Understand backprop by writing it yourself, one operation at a time.

```
notebook/micrograd.ipynb
```

### What you'll build

| Step | Concept | Code |
|------|---------|------|
| 1 | A single `Value` class | `data`, `grad`, `_backward`, `_prev` |
| 2 | Addition (`a + b`) | `∂c/∂a = 1, ∂c/∂b = 1` |
| 3 | Multiplication (`a * b`) | `∂c/∂a = b, ∂c/∂b = a` |
| 4 | Power (`a ** n`) | Chain rule in one line |
| 5 | Activation functions | `tanh`, `relu`, `exp`, `log` |
| 6 | A neuron | `tanh(w·x + b)` |
| 7 | A layer | `nout` neurons |
| 8 | An MLP | Stack layers |
| 9 | Training loop | Forward → Backward → Step |
| 10 | Classify digits | Image → MLP → Loss |

### Key realization

> Every operation knows its own local derivative. The `_backward` function encodes it. The `+=` handles branching. The topological sort handles depth. **Backprop is not magic — it's arithmetic.**

---

## Stage 2: Understand the Math

**Goal:** Know *why* every step works, not just *how*.

```
book/book.tex  →  book/book.pdf
slides/lecture.tex  →  slides/lecture.pdf
```

### The 10 chapters (in order)

| Chapter | Core Idea | Key Equation |
|---------|-----------|--------------|
| **1. Derivatives** | Which way to move `x` to make `f(x)` smaller? | `x ← x − η·f'(x)` |
| **2. Chain Rule** | How derivatives compose through a path | `df/dx = (df/dv)·(dv/du)·(du/dx)` |
| **3. Gradient Descent** | Step opposite to the derivative, repeat | `θ ← θ − η·dL/dθ` |
| **4. Gradients** | Many inputs → a vector of partial derivatives | `∇L = [∂L/∂θ₁, …, ∂L/∂θₙ]` |
| **5. Autograd Engine** | Local derivatives as code | `_backward()` per operation |
| **6. Backprop** | Chain rule at scale — one pass, all gradients | Reverse topological walk |
| **7. Neural Networks** | Composing neurons into networks | `h = tanh(Wx + b)` |
| **8. The Loss** | Turning "wrong" into a number | `∂loss/∂zᵢ = softmax(z)ᵢ − 1[i=t]` |
| **9. Training Process** | Everything together | Forward → Backward → Step |
| **10. Application** | Image classification from scratch | 3301 parameters, 92.5% test acc |

### Exercises (in book)

1. Trace a neuron forward/backward by hand
2. Break `+=` and see gradients fail
3. Add ReLU and compare to tanh
4. Remove gradient zeroing and watch divergence
5. Try MSE + tanh on classification
6. Print the computation DAG
7. Swap in NumPy for speed
8. Add a convolution and see accuracy jump

---

## Stage 3: Scale Up

**Goal:** Go from the tiny MLP to modern architectures. Each step adds one new differentiable op.

### The progression

```
MLP (micrograd)
    │
    ▼  + Convolution op
CNN (conv + pooling + ReLU + softmax)
    │
    ▼  + Recurrence
RNN / LSTM / GRU
    │
    ▼  + Attention
Transformer (self-attention, layer norm, positional encoding)
    │
    ▼  + Memory + External read/write
Neural Turing Machine / Neural Processor (NTP)
```

### Detailed roadmap

#### 3.1 Convolutional Neural Networks
- **New op:** 3×3 conv, stride, padding
- **New backward:** Transposed convolution (the adjoint of conv)
- **Why:** Spatial structure, parameter sharing, translation invariance
- **Exercise:** Add conv to micrograd. Classify digits better.

#### 3.2 Pooling & Deep CNNs
- **Max pooling, average pooling** as differentiable downsample operations
- **Architecture:** LeNet → AlexNet → VGG → ResNet (skip connections)
- **Key insight:** Skip connections solve vanishing gradients

#### 3.3 Recurrent Networks
- **New op:** Hidden state recurrence `hₜ = f(hₜ₋₁, xₜ)`
- **Backward through time (BPTT):** Unroll the recurrence, apply chain rule across time steps
- **Problems:** Vanishing/exploding gradients
- **Solutions:** LSTM (gates), GRU (simplified gates)

#### 3.4 Attention Mechanisms
- **New op:** `Attention(Q, K, V) = softmax(QKᵀ/√d)V`
- **Backward:** Matrix operations — each term differentiates cleanly
- **Key insight:** Attention replaces recurrence entirely

#### 3.5 Transformers
- **Components:** Multi-head attention, positional encoding, layer normalization, feed-forward blocks
- **Architecture:** Encoder-decoder (original) or decoder-only (GPT)
- **Scaling law:** More parameters + more data = better (up to a point)

#### 3.6 Neural Turing Machine / Neural Processor (NTP)
- **New idea:** Neural network + external memory matrix
- **Operations:** Read, write, erase — all differentiable
- **Backward:** Gradient through memory operations (content-based addressing)
- **Capability:** Algorithmic generalization — learn to sort, copy, associate
- **Reference:** Graves et al., *Hybrid Computing using a Neural Network with a External Memory* (2014)

---

## Quick Reference: Derivatives of Every Op You'll Meet

| Operation | Forward | Local derivative (backward) |
|-----------|---------|---------------------------|
| `a + b` | `a+b` | `∂L/∂a += 1·∂L/∂out, ∂L/∂b += 1·∂L/∂out` |
| `a * b` | `a·b` | `∂L/∂a += b·∂L/∂out, ∂L/∂b += a·∂L/∂out` |
| `a ** n` | `aⁿ` | `∂L/∂a += n·aⁿ⁻¹·∂L/∂out` |
| `tanh(a)` | `tanh(a)` | `∂L/∂a += (1−tanh²(a))·∂L/∂out` |
| `relu(a)` | `max(0,a)` | `∂L/∂a += (a>0)·∂L/∂out` |
| `exp(a)` | `eᵃ` | `∂L/∂a += eᵃ·∂L/∂out` |
| `log(a)` | `ln(a)` | `∂L/∂a += (1/a)·∂L/∂out` |
| `conv(x, w)` | feature map | Transposed convolution |
| `matmul(a, b)` | `a·b` | `∂L/∂a += ∂L/∂out · bᵀ, ∂L/∂b += aᵀ · ∂L/∂out` |
| `softmax(z)_i` | `eᶻⁱ/Σⱼeᶻʲ` | `∂L/∂zᵢ = softmax(z)ᵢ − 1[i=t]` |

---

## File Map

```
research/
├── book/
│   ├── book.tex          Full LaTeX mini-book (10 chapters)
│   └── book.pdf          Compiled PDF (33 pages)
├── slides/
│   ├── lecture.tex       Beamer presentation (10 sections)
│   └── lecture.pdf       Compiled PDF (33 slides)
├── notebook/
│   └── micrograd.ipynb   Jupyter notebook — build the engine
├── .gitignore
└── README.md             This file
```

---

## Prerequisites

- **Calculus:** What a derivative is (Chapter 1 of the book)
- **Linear algebra:** Vectors, matrices, dot products
- **Python:** Loops, functions, classes (the notebook walks you through it)
- **That's it.** No deep learning prerequisites needed.

---

## The Core Loop (reminder)

```
FORWARD:  Build the computation graph
BACKWARD: Walk it in reverse, accumulate gradients
STEP:     θ ← θ − η · ∂L/∂θ
```

Every architecture above — CNNs, RNNs, Transformers, NTPs — is just this loop with new differentiable operations. **The engine never changes.**

---

*"That's backprop from the ground up."*
