# Fashion-MNIST Classification: Architecture Benchmarking & Regularization

An empirical deep learning benchmarking study evaluating 7 custom Convolutional Neural Network (CNN) architectures on the **Fashion-MNIST** dataset in PyTorch—spanning kernel geometry, parameter scaling (up to 520K parameters), spatial regularization, data augmentation, and compiler acceleration.

---

## 📌 Project Overview

This project systematically investigates the architectural and optimization factors governing visual representation on Fashion-MNIST:
1. **Kernel Geometry & Hierarchy**: Comparing shallow $3\times3$ vs $5\times5$ filters against deep VGG-style stacked $3\times3$ convolution blocks.
2. **Channel Capacity & Parameter Scaling**: Evaluating parameter scaling from 10K parameters up to 520K parameters across 7 iterative designs.
3. **Regularization**: Mitigating overfitting in deeper networks via 2D spatial channel dropout (`nn.Dropout2d(p=0.2)`) and dense classification dropout (`nn.Dropout(p=0.5)`).
4. **Data Augmentation**: Incorporating random horizontal flipping and color jitter via `torchvision.transforms.v2`.
5. **Training Systems Optimization**: Leveraging `torch.compile` graph compilation, deterministic seeded splits, and OpenMP/MKL CPU thread tuning.

To ensure statistical rigor, all architectures were benchmarked over **5 independent runs with unique random seeds** for 10 epochs.

---

## 📊 Benchmark Results

Evaluated across 5 independently seeded runs for 10 epochs each. Performance is reported as **Mean Test Accuracy $\pm$ Sample Standard Deviation ($\sigma$)** on the 10,000 test images:

| Architecture | Paradigm / Structure | Parameters | Regularization | Test Accuracy (10 Epochs) | Std Dev ($\sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`MyCNNModel_v1_0`** | 2x Conv ($3\times3$, 32 ch), 1 FC | ~10.5K | None | 87.45% | 0.0037 |
| **`MyCNNModel_v1_1`** | 2x Conv ($3\times3$, $32 \to 64$ ch), 1 FC | ~21.1K | None | 88.61% | 0.0033 |
| **`MyCNNModel_v2_0`** | 2x Conv ($5\times5$, 32 ch), 1 FC | ~26.0K | None | 87.51% | 0.0065 |
| **`MyCNNModel_v2_1`** | 2x Conv ($5\times5$, $32 \to 64$ ch), 1 FC | ~52.1K | None | 88.46% | 0.0040 |
| **`MyCNNModel_v3`** | Hybrid ($5\times5 \to 3\times3$), 2 FC | 27,882 | None | 89.13% | 0.0008 |
| **`MyCNNModel_v4`** | VGG-style 4x Conv ($3\times3$, up to 128 ch) | 135,402 | Spatial + Dense Dropout | 90.58% | 0.0034 |
| **`MyCNNModel_v5`** 🏆 | High-capacity VGG 4x Conv ($3\times3$, up to 256 ch) | **520,330** | Spatial + Dense Dropout | **91.75%** | **0.0023** |

---

## 🔍 Key Empirical Insights

1. **VGG-Style Deep Stacking Superiority**: Transitioning from 2 shallow convolution layers to 4 stacked $3\times3$ convolutions (`v4` and `v5`) produced the most significant performance leap, breaking the $90\%$ barrier ($90.58\%$ and $91.75\%$).
2. **Channel Capacity & Parameter Scaling**: Scaling capacity from `v4` (135K params) to `v5` (520K params) delivered a $+1.17\%$ test accuracy boost while maintaining high stability ($\sigma = 0.0023$).
3. **Effective Regularization**: Using spatial feature dropout (`nn.Dropout2d(0.2)`) in convolutional feature maps combined with dense dropout (`nn.Dropout(0.5)`) prevented over-parameterized models from overfitting augmented data.
4. **Consistent Generalization**: Standard deviations across 5 distinct random initializations remained under $0.4\%$ for all top-performing models, confirming repeatable convergence.

---

## 🛠️ Tech Stack & Engineering Practices

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) (`torch`, `torchvision`)
- **Compilation Acceleration**: `torch.compile()` for JIT graph optimization and kernel fusion.
- **Data Augmentation**: `torchvision.transforms.v2` pipeline (`RandomHorizontalFlip`, `ColorJitter`, `ToImage`, `ToDtype`, `Normalize`).
- **Validation Discipline**: 90/10 split on training data (54,000 train / 6,000 validation / 10,000 test) using seeded `torch.Generator`.
- **Hardware & Multi-threading**: Tuned multi-threading via `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, and `torch.set_num_threads(8)` with multi-worker `DataLoader(num_workers=2)`.

---

## 📂 Project Structure

```text
├── dataset/             # Fashion-MNIST dataset files
├── downloadDataset.py   # Script to download and verify Fashion-MNIST dataset
├── MyCNNModel.py        # PyTorch model definitions (v1_0, v1_1, v2_0, v2_1, v3, v4, v5)
├── runAndEval.py        # Benchmarking pipeline with torch.compile, validation split & seed harness
├── newExptLog.txt       # Full logs and benchmark summary across 5 iterations (v1_0 - v5)
├── exptLog.txt          # Initial experimental logs (v1_0 - v3 across 1, 5, 10 epochs)
└── README.md            # Project documentation and benchmark analysis
```

---

## 🚀 Getting Started

### 1. Installation
Install PyTorch and runtime dependencies:
```bash
pip install torch torchvision numpy tqdm
```

### 2. Download Dataset
Ensure the Fashion-MNIST dataset is ready:
```bash
python downloadDataset.py
```

### 3. Run Benchmark Suite
Execute the multi-seed benchmark across all 7 architectures:
```bash
python runAndEval.py
```
