# Fashion-MNIST Classification with Custom CNN Architectures

An empirical benchmarking and deep learning exploration project evaluating custom Convolutional Neural Network (CNN) architectures on the **Fashion-MNIST** dataset using PyTorch.

---

## 📌 Project Overview

This project investigates the impact of architectural variations—such as kernel dimensions ($3\times3$ vs $5\times5$), feature map channel scaling ($32 \to 32$ vs $32 \to 64$), and Multi-Layer Perceptron (MLP) classification heads—on classification accuracy, convergence speed, and model stability.

To ensure statistical rigor and reproducibility, models were evaluated across **5 independent runs with unique random seeds** over multiple training regimes (1, 5, and 10 epochs).

---

## 📊 Summary of Benchmark Results

All models were evaluated across 5 random seeded iterations. Performance is reported as **Mean Accuracy $\pm$ Sample Standard Deviation** on the 10,000 test set images:

| Architecture | Conv Kernels | Feature Channels | Fully Connected Layers | 1 Epoch | 5 Epochs | 10 Epochs |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`MyCNNModel_v1_0`** | $3\times3$, $3\times3$ | $32 \to 32$ | $128 \to 10$ | $82.55\% \pm 0.86\%$ | $87.51\% \pm 0.14\%$ | $88.28\% \pm 0.31\%$ |
| **`MyCNNModel_v1_1`** | $3\times3$, $3\times3$ | $32 \to 64$ | $256 \to 10$ | $84.65\% \pm 0.67\%$ | $88.83\% \pm 0.36\%$ | $89.50\% \pm 0.25\%$ |
| **`MyCNNModel_v2_0`** | $5\times5$, $5\times5$ | $32 \to 32$ | $32 \to 10$ | $83.17\% \pm 0.33\%$ | $87.44\% \pm 0.30\%$ | $88.41\% \pm 0.58\%$ |
| **`MyCNNModel_v2_1`** | $5\times5$, $5\times5$ | $32 \to 64$ | $64 \to 10$ | $84.93\% \pm 0.83\%$ | $88.25\% \pm 0.27\%$ | $88.69\% \pm 0.55\%$ |
| **`MyCNNModel_v3`** 🏆 | $5\times5$, $3\times3$ | $32 \to 64$ | $256 \to 32 \to 10$ | **$84.12\% \pm 0.49\%$** | **$89.13\% \pm 0.35\%$** | **$90.24\% \pm 0.25\%$** |

### Key Takeaways
1. **Hierarchical Receptive Field (`v3`)**: Combining a $5\times5$ initial receptive field with a subsequent $3\times3$ layer and a 2-stage MLP classification head achieved the highest overall accuracy (**90.24%** at 10 epochs).
2. **Channel Capacity**: Expanding intermediate channels from 32 to 64 consistently improved representation capacity, yielding a $+1.22\%$ gain between `v1_0` and `v1_1` at 10 epochs.
3. **Statistical Stability**: All models demonstrated reliable convergence with standard deviations below $0.6\%$ across 5 independent seeds.

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.13
- **Framework**: [PyTorch](https://pytorch.org/) (`torch-2.14.1+cpu` `torchvision-0.29.1+cpu`)
- **Transforms**: `torchvision.transforms.v2`
- **Numerical Computing**: `numpy`
- **Progress Tracking**: `tqdm`

---

## 📂 Project Structure

```text
├── dataset/             # Fashion-MNIST dataset files (downloaded or local)
├── downloadDataset.py   # Script to download and verify Fashion-MNIST dataset
├── MyCNNModel.py        # PyTorch model definitions (v1_0, v1_1, v2_0, v2_1, v3)
├── runAndEval.py        # End-to-end multi-seed benchmark and evaluation harness
├── exptLog.txt          # Detailed logs and final benchmark summary across all runs
└── README.md            # Project documentation and findings
```

---

## 🚀 Getting Started

### 1. Installation
Ensure PyTorch and supporting packages are installed:
```bash
pip install torch torchvision numpy tqdm
```

### 2. Download Dataset
Download and extract Fashion-MNIST into `./dataset`:
```bash
python downloadDataset.py
```

### 3. Run Experiments
Execute the multi-seed benchmark across all 5 architectures and epoch variations:
```bash
python runAndEval.py
```

---

## 🔬 Experimental Methodology

- **Input Preprocessing**: Images are scaled to $[0, 1]$ and normalized with mean $0.5$ and standard deviation $0.5$.
- **Optimization**: Cross-Entropy Loss (`nn.CrossEntropyLoss`) with the Adam optimizer (`torch.optim.Adam`).
- **Batch Size**: 64.
- **Reproducibility**: Seeding controls applied across `torch`, `numpy`, `random`, and deterministic cuDNN backends (`torch.backends.cudnn.deterministic = True`).
