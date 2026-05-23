# deep-vqa-framework

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub release](https://img.shields.io/github/v/release/autentisitet/deep-vqa-framework?include_prereleases)](https://github.com/autentisitet/deep-vqa-framework/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-blue)](https://github.com/autentisitet/deep-vqa-framework)
[![Version](https://img.shields.io/badge/version-0.9.1--beta-blue.svg)](https://github.com/autentisitet/deep-vqa-framework)

**A Unified Deep Learning Framework for Image Quality Assessment (IQA) and Video Quality Assessment (VQA).**

This framework provides an end-to-end solution for training, evaluating, and deploying quality assessment models. It features a unified architecture that seamlessly handles both image and video inputs, multi-dataset support, cross-validation pipelines, and production-ready inference APIs.

> [!NOTE]
> This framework is primarily tested on AutoDL cloud GPU instances.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Architecture & Design Decisions](#architecture-decisions)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Configuration Guide](#configuration-guide)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgments)

---

## System Requirements <a id="system-requirements"></a>

### Hardware Requirements

| Component | Minimum | Recommended |
| ----------- | --------- | ------------- |
| **GPU** | 8GB VRAM (IQA only) | 24GB+ VRAM (VQA training) |
| **RAM** | 16GB | 32GB+ |
| **Disk Space** | 50GB | 200GB+ (including datasets) |
| **CPU** | 4 cores | 8+ cores |

### Software Requirements

| Component | Version | Notes |
| ----------- | --------- | ------- |
| **OS** | Linux (Ubuntu 20.04+) / Windows 10+ / macOS 12+ | Linux recommended for training |
| **Python** | 3.10 - 3.12 | 3.12+ not fully tested |
| **CUDA** | 11.8 / 12.1 | Required for GPU training |
| **PyTorch** | 2.0+ | 2.5+ recommended for better AMP support |
| **cuDNN** | 8.7+ | Included with PyTorch |

### Storage Breakdown

The framework expects datasets in the following structure:

| Dataset | Size (Compressed) | Size (Extracted) |
| --------- | ------------------- | ------------------- |
| TID2013 | ~500MB | ~3GB |
| KoNViD-1k | ~9GB | ~10GB |
| T2VQA-DB | ~45GB | ~50GB+ |
| **Total** | **~55GB** | **~63GB+** |

### Additional Space

| Item | Estimated Size |
| ------ | ---------------- |
| Python environment (uv/venv) | ~5GB |
| Model checkpoints (5-fold) | ~10GB |
| Training logs & plots | ~2GB |
| **Grand Total** | **~80-100GB** |

> [!NOTE]
> - Use `uv` for faster installation and smaller dependency footprint
> - Symbolic links (see below) do not consume additional disk space
> - `quarantine/` directory may grow if files are isolated; run `scripts/cache_clean.sh` regularly

---

## Architecture & Design Decisions <a id="architecture-decisions"></a>

### Unified IQA/VQA Architecture

The framework implements a dimension-aware routing system that automatically switches between image (4D tensors) and video (5D tensors) processing modes.

**Key Design Decisions:**

| Decision | Implementation | Rationale |
| :--- | :--- | :--- |
| **Unified Model** | Single `IQAVQANet` handles both 4D and 5D inputs | Eliminates duplicate code, ensures consistent quality metrics |
| **Flexible Backbones** | Swin-T / ResNet50 with automatic feature adaptation | Balances accuracy vs. memory consumption |
| **Temporal Fusion** | Transformer encoder for video frame aggregation | Captures long-range dependencies between frames |
| **Hybrid Loss** | MSE (70%) + Rank Loss (30%) | Optimizes both absolute prediction and relative ordering |
| **Multi-Dataset Support** | YAML-based configuration with factory pattern | Easy addition of new datasets without code changes |
| **Path Abstraction** | DSL-based `PathManager` with YAML routing | Eliminates hardcoded paths, supports symbolic links |
| **Lazy Asset Resolution** | `CaseInsensitiveAssetResolver` with pre-built index | O(1) file lookup, case-insensitive matching |

---

## Model Architecture <a id="model-architecture"></a>

### IQAVQANet: Unified Quality Assessment Network

```python
# Architecture overview
Input (4D: [B,3,H,W] or 5D: [B,F,3,H,W])
    ↓
Backbone (Swin-T / ResNet50)
    ↓
Spatial Pooling (AdaptiveAvgPool2d)
    ↓
[Temporal Fusion] ← TransformerEncoder (only for video)
    ↓
Quality Head (3-layer MLP + Sigmoid)
    ↓
Output: Quality Score (0-1 range)
```

### Supported Configurations

| Backbone | Parameters | IQA | VQA | Memory (per sample) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | 25M | ✅ | ✅ | ~2GB (8 frames) |
| **Swin-T** | 28M | ✅ | ✅ | ~4GB (8 frames) |

### Loss Function: Hybrid MSE + Rank Loss

```text
Total Loss = 0.7 × MSE + 0.3 × Rank Loss

- MSE Loss: Absolute prediction accuracy
- Rank Loss: Preserves relative ordering between samples
```

---

## Training Pipeline <a id="training-pipeline"></a>

### Quick Start Training

#### Step 1: Create symbolic links

```bash
cd datasets
ln -s TID2013 tid2013
ln -s KoNViD-1k konvid-1k
ln -s T2VQA-DB t2vqa-db

ls -la
```

#### Step 2: Run training

- **Image Dataset (TID2013)**

```bash
uv run python -m src.main --model resnet_iqa --dataset tid2013
```

- **Video Dataset (KoNViD-1k)**

```bash
uv run python -m src.main --model timeswin_vqa --dataset konvid-1k
```

- **Video Dataset (T2VQA-DB)**

```bash
uv run python -m src.main --model resnet_vqa --dataset t2vqa-db
```

### Configuration Parameters

```yaml
# config/models/resnet_vqa.yaml
preprocessing:
  batch_size: 2          # Reduce if OOM
  num_workers: 4         # Data loading threads
  k_fold: 5              # Cross-validation folds

model:
  backbone: "resnet50"   # or "swin_t"
  num_frames: 8          # Video frames per sample
  transformer_layers: 2  # Temporal fusion depth

train:
  epochs: 50
  lr: 0.0001
  gradient_accumulation_steps: 4  # Effective batch = batch_size × steps
  early_stop:
    enabled: true
    patience: 10
    monitor: "val_srocc"
    mode: "max"
```

### Advanced Options

- **Fast Smoke Test**

```bash
uv run python -m src.main --model resnet_iqa --dataset tid2013 --smoke_test
```

- **Debug Mode (with breakpoints)**

```bash
LOG_LEVEL=DEBUG uv run python -m src.main --model resnet_iqa --dataset tid2013
```

- **Background Training**

```bash
nohup python -m src.main --model resnet_vqa --dataset t2vqa-db > train.log 2>&1 &
tail -f train.log
```

---

## Evaluation & Metrics <a id="evaluation-metrics"></a>

### Core Metrics

| Metric | Full Name | Interpretation |
| :--- | :--- | :--- |
| **PLCC** | Pearson Linear Correlation Coefficient | Linear relationship (accuracy) |
| **SROCC** | Spearman Rank Order Correlation Coefficient | Monotonic relationship (ranking) |
| **KROCC** | Kendall Rank Correlation Coefficient | Ordinal agreement |
| **RMSE** | Root Mean Square Error | Prediction error magnitude |
| **R²** | Coefficient of Determination | Variance explained |

### Output Example

```text
📊 [Epoch 050 | HYPERIQANET] PLCC: 0.9696 | SROCC: 0.9689 | RMSE: 0.0442 | R²: 0.9314
```

### Visualizations

The framework automatically generates:

- **Training History**: Loss curves, PLCC/SROCC progression

- **Residual Analysis**: Scatter plots, error distribution

- **Cross-Model Comparison**: Bar charts for multiple models

Output location: `results/{dataset}/plots/`

---

## Project Structure <a id="project-structure"></a>

```text
deep-vqa-framework/
├── Algorithm-section.png           # Architecture diagram for documentation
├── Makefile                        # Automation commands (training, clean, test)
├── README.md                       # Project documentation (this file)
├── pyproject.toml                  # Python project configuration (uv/pip)
├── uv.lock                         # Lock file for dependency versions
│
├── config/
│   ├── basic.yaml                  # Global defaults (system, training)
│   ├── dataset_config.yaml         # Dataset-specific settings & metadata
│   ├── paths.yaml                  # Path routing & symbolic link rules
│   └── models/                     # Model-specific configurations
│       ├── resnet_iqa.yaml         # ResNet50 for Image QA
│       ├── resnet_vqa.yaml         # ResNet50 for Video QA
│       └── timeswin_vqa.yaml       # TimeSwin transformer for Video QA
│
├── datasets/
│   ├── KoNViD-1k/                  # KoNViD-1k dataset (videos + metadata)
│   │   ├── KoNViD-1k_metadata/     # MOS scores and video annotations
│   │   └── KoNViD-1k_videos/       # Video files (directory only)
│   ├── T2VQA-DB/                   # Text-to-Video QA dataset
│   │   ├── T2VQA-DB/               # Video content (directory only)
│   │   └── info.txt                # Dataset metadata
│   ├── TID2013/                    # TID2013 image quality dataset
│   │   ├── distorted_images/       # Distorted test images (directory only)
│   │   ├── reference_images/       # Reference originals (directory only)
│   │   ├── metrics_values/         # Objective metrics files
│   │   ├── papers/                 # Dataset reference papers
│   │   ├── mos.txt                 # Mean Opinion Scores
│   │   ├── mos_std.txt             # MOS standard deviation
│   │   ├── mos_with_names.txt      # MOS with image filenames
│   │   └── readme                  # Dataset documentation
│   ├── konvid-1k -> KoNViD-1k      # Symlink for case-insensitive access
│   ├── t2vqa-db -> T2VQA-DB        # Symlink for case-insensitive access
│   └── tid2013 -> TID2013          # Symlink for case-insensitive access
│
├── docs/
│   ├── Algorithm-section.mmd       # Mermaid diagram source
│   └── Cloud_Platform_Rental_Guide.md  # Cloud setup instructions
│
├── quarantine/                     # Isolated directory for testing/archiving
│
├── results/
│   ├── scripts_logs/               # Script execution logs (directory)
│   ├── model_outputs/              # Global model checkpoints (directory)
│   ├── train_logs/                 # Global training history (directory)
│   ├── plots/                      # Global visualization outputs
│   │   └── *.png                   # Plot images (filenames only)
│   ├── tid2013_integrity_report.txt  # Dataset validation report
│   ├── TID2013/                    # Image dataset results (directory)
│   ├── tid2013/                    # Symlink to TID2013 (legacy)
│   ├── konvid-1k/                  # Video dataset results (directory)
│   └── t2vqa-db/                   # T2VQA dataset results (directory)
│
├── scripts/
│   ├── archive_result.sh           # Archive old results to quarantine
│   ├── cache_clean.sh              # Clear system cache files, etc
│   ├── download_flag               # Flag file for download completion
│   ├── manage_data.sh              # Dataset download & extraction
│   ├── network_control.sh          # Network proxy configuration, etc
│   ├── plugin.sh                   # Symlink creation
│   ├── setup_env.sh                # Environment initialization
│   └── system_check.sh             # GPU, CUDA, dependency validation, etc
│
├── src/
│   ├── main.py                     # Entry point: training & evaluation
│   ├── core/
│   │   ├── engine.py               # Training/validation loop
│   │   ├── evaluator.py            # Metrics computation (PLCC, SROCC)
│   │   └── trainer.py              # Cross-validation pipeline
│   ├── data/
│   │   ├── data_eda.py             # Exploratory data analysis
│   │   ├── metadata_loader_factory.py  # Factory for dataset parsers
│   │   ├── metadata_loaders.py     # Dataset-specific metadata loaders
│   │   ├── types.py                # Type definitions (enums, dataclasses)
│   │   └── eda/
│   │       ├── integrity.py        # File integrity check (missing/corrupted)
│   │       ├── metrics_plotter.py  # Training curves, residuals, comparison
│   │       ├── split.py            # Stratified K-Fold splitting
│   │       └── statistics.py       # MOS distribution, resolution stats
│   ├── models/
│   │   ├── README.md               # Model architecture documentation
│   │   └── iqavqa_net.py           # Unified IQA/VQA network
│   └── utils/
│       ├── config_loader.py        # YAML config loading & deep merging
│       ├── file_loader.py          # Case-insensitive file path resolver
│       ├── logging_utils.py        # Logging with CSV rotation
│       └── path_manager.py         # Path routing with YAML templates
```

---

## Configuration Guide <a id="configuration-guide"></a>

### Configuration Layering

Configuration files are merged in the following order (later files override earlier ones):

| Layer | File | Purpose |
| ------- | ------ | --------- |
| 1 (Base) | `basic.yaml` | Global defaults |
| 2 (Model) | `models/{model}.yaml` | Model-specific overrides |
| 3 (Dataset) | `dataset_config.yaml` | Dataset-specific settings |

### Memory Optimization for Video Training

```yaml
# If encountering CUDA Out of Memory (OOM)
preprocessing:
  batch_size: 1              # Reduce batch size
  num_workers: 0             # Disable multiprocessing

model:
  num_frames: 4              # Reduce temporal frames
  backbone: "resnet50"       # Use smaller backbone
  transformer_layers: 1      # Reduce transformer depth

train:
  gradient_accumulation_steps: 4  # Simulate larger batch
  amp: true                  # Enable mixed precision
```

---

## Troubleshooting <a id="troubleshooting"></a>

### CUDA Out of Memory

| Symptom | Solution |
| :--- | :--- |
| OOM at first batch | Reduce `batch_size` to 1 |
| OOM after several epochs | Enable `gradient_checkpointing: true` |
| OOM during validation | Reduce `num_frames` to 4 |

### Dataset Not Found

If you encounter `FileNotFoundError` when passing `--dataset xxx`, it means the dataset symlink is missing or incorrect.

```bash
cd datasets
ls -la

ln -s TID2013 tid2013
ln -s KoNViD-1k konvid-1k
ln -s T2VQA-DB t2vqa-db

ls -la
```

### Video Loading Backend (AutoDL Specific)

> [!WARNING]
On AutoDL or similar cloud GPU instances, OpenCV's VideoCapture may fail due to missing system dependencies.

Solution: Use Decord

```bash
uv add decord
```

Decord is pre-configured as the default backend. If Decord is not available, the framework automatically falls back to OpenCV, but on AutoDL this fallback may fail. Always use Decord for video training on AutoDL.

### Slow Training

| Issue | Optimization |
| :--- | :--- |
| Data loading bottleneck | Increase `num_workers: 8` |
| Small batch size | Use `gradient_accumulation_steps` |
| Video decoding slow | Ensure Decord is installed |

---

## 📄 License <a id="license"></a>

- **Framework**: [MIT](LICENSE)
- **Author**: [@autentisitet](https://github.com/autentisitet)
- **Version**: 0.9.1-beta (pre-release)

---

## 🙏 Acknowledgments <a id="acknowledgments"></a>

- PyTorch team for deep learning framework
- Decord developers for efficient video loading
- TID2013, KoNViD-1k, T2VQA-DB dataset providers

---

**Built with ❤️ for the research community**
