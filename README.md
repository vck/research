# From the Ground Up: Derivatives → Backprop → Neural Networks

A complete, layered learning roadmap — from the simplest autograd engine to advanced neural architectures including Neural Turing Machines.

Each section only uses what came before. No prior deep learning knowledge assumed.

---

## 📚 The Book

**[`book/book.pdf`](book/book.pdf)** — Full LaTeX mini-book, 51 pages across 19 chapters.

### The 19 Chapters

| # | Chapter | Core Idea |
|---|---------|-----------|
| 1 | Derivatives | Which way to move `x` to make `f(x)` smaller? |
| 2 | The Chain Rule | How derivatives compose through a path |
| 3 | Gradient Descent | Step opposite to the derivative, repeat |
| 4 | Gradients (Many Inputs) | Vector of partial derivatives |
| 5 | The Autograd Engine | Local derivatives as code |
| 6 | Backprop | Chain rule at scale — one pass, all gradients |
| 7 | Neural Networks | Composing neurons into networks |
| 8 | The Loss | Turning "wrong" into a number |
| 9 | The Training Process | Forward → Backward → Step |
| 10 | Putting It All to Work | Digit classification from scratch |
| **11** | **Vanishing & Exploding Gradients** | Why deep networks fail, and how to fix it |
| **12** | **Weight Initialization** | Xavier & He from variance preservation |
| **13** | **Normalization** | Batch norm, layer norm |
| **14** | **Regularization** | Dropout, weight decay, data augmentation |
| **15** | **Optimization** | Momentum, RMSProp, Adam, LR schedules |
| **16** | **Convolutional Networks** | Conv op, transposed conv backward, ResNets |
| **17** | **Recurrent Networks** | LSTM, GRU, BPTT |
| **18** | **Attention & Transformers** | Self-attention, multi-head, architecture |
| **19** | **Neural Turing Machine & NTP** | External memory, algorithmic generalization |

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

### Stage 1: Build the Engine (Micrograd)

**Goal:** Understand backprop by writing it yourself, one operation at a time.

```
notebook/micrograd.ipynb
```

Build: `Value` class → addition → multiplication → activation → neuron → layer → MLP → training loop → digit classification.

### Stage 2: Understand the Math

**Goal:** Know *why* every step works, not just *how*.

```
book/book.tex  →  book/book.pdf
slides/lecture.tex  →  slides/lecture.pdf
```

### Stage 3: Scale Up

**Goal:** Go from the tiny MLP to modern architectures. Each step adds one new differentiable op.

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

| Architecture | New Op | New Backward |
|-------------|--------|-------------|
| MLP | Linear + activation | Chain rule |
| CNN | Convolution | Transposed convolution |
| RNN | Recurrence | Backprop through time |
| LSTM/GRU | Gated cell | Gate derivatives |
| Transformer | Self-attention | Matrix ops + softmax |
| NTM | Read/write memory | Content-based addressing |

---

## 📖 Slides

**[`slides/lecture.pdf`](slides/lecture.pdf)** — Beamer presentation covering all 10 core chapters.

---

## 🔬 Quick Reference: Derivatives of Every Op

| Operation | Forward | Local derivative (backward) |
|-----------|---------|---------------------------|
| `a + b` | `a+b` | `∂L/∂a += 1·∂L/∂out` |
| `a * b` | `a·b` | `∂L/∂a += b·∂L/∂out` |
| `a ** n` | `aⁿ` | `∂L/∂a += n·aⁿ⁻¹·∂L/∂out` |
| `tanh(a)` | `tanh(a)` | `∂L/∂a += (1−tanh²(a))·∂L/∂out` |
| `relu(a)` | `max(0,a)` | `∂L/∂a += (a>0)·∂L/∂out` |
| `exp(a)` | `eᵃ` | `∂L/∂a += eᵃ·∂L/∂out` |
| `log(a)` | `ln(a)` | `∂L/∂a += (1/a)·∂L/∂out` |
| `conv(x, w)` | feature map | Transposed convolution |
| `matmul(a, b)` | `a·b` | `∂L/∂a += ∂L/∂out · bᵀ` |
| `softmax(z)_i` | `eᶻⁱ/Σⱼeᶻʲ` | `∂L/∂zᵢ = softmax(z)ᵢ − 1[i=t]` |

---

## 📂 File Map

```
research/
├── book/
│   ├── book.tex          Full LaTeX book (19 chapters)
│   └── book.pdf          Compiled PDF (51 pages)
├── slides/
│   ├── lecture.tex       Beamer presentation (10 sections)
│   └── lecture.pdf       Compiled PDF
├── notebook/
│   └── micrograd.ipynb   Jupyter notebook — build the engine
├── .gitignore
└── README.md             This file
```

---

## Prerequisites

- **Calculus:** What a derivative is (Chapter 1)
- **Linear algebra:** Vectors, matrices, dot products
- **Python:** Loops, functions, classes
- **That's it.** No deep learning prerequisites needed.

---

## The Core Loop

```
FORWARD:  Build the computation graph
BACKWARD: Walk it in reverse, accumulate gradients
STEP:     θ ← θ − η · ∂L/∂θ
```

Every architecture above — CNNs, RNNs, Transformers, NTPs — is just this loop with new differentiable operations. **The engine never changes.**

---

*"That's backprop from the ground up."*
