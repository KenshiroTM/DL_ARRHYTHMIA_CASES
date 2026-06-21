# Arrhythmia Classification Based on ECG

> Student project in biomedical signal processing and deep learning for automatic cardiac arrhythmia detection using transfer learning and unsupervised anomaly detection (ECG + PPG).

## Overview

This is a student deep-learning project developed as a coursework assignment. It explores automatic classification of cardiac arrhythmias from ECG signals using public datasets **PTB-XL** and **China Physiological Signal Challenge (Arrhythmia)**.

Two supervised tasks are covered:
- **Binary classification** — normal vs. arrhythmia
- **Multiclass classification** — 5 types of arrhythmia

Additionally, an experiment with an **EKG+PPG autoencoder** on the **VitalDB** dataset was conducted for unsupervised anomaly detection. The experiment did not yield expected results and was thoroughly analyzed.

## Key Features

- **Transfer learning** with **ECGFounder** (pretrained on ~10M ECG signals, 150 diagnostic classes)
- **Custom CNN architecture** (CustomECGNet) with multi-scale convolutions and channel attention
- **Database bias control** — independent weight exports for PTB-XL and Arrhythmia to prevent overfitting to a single source
- **Patient deduplication** across merged databases to prevent data leakage
- **Anomaly detection experiment** — standard AE and VAE on EKG+PPG signals
- **Preloaded caching** — processed signals saved to `.parquet` for fast subsequent runs and heavy compression

## Model Architectures

| Model | Type | Features |
|-------|------|----------|
| **ECGFounder** | Transfer learning | Pretrained transformer/CNN, classification head fine-tuning |
| **CustomECGNet** | Custom CNN | Multi-scale convolutions (kernel 3/7/15), Channel Attention (SE-like), residual connections, GELU |
| **Net1D** | Literature adaptation | ResNet-like 1D with Squeeze-and-Excitation, Swish, depthwise separable conv (Hong et al., 2020) |

## Datasets

| Dataset | Purpose | Parameters | Classes |
|---------|---------|------------|---------|
| **PTB-XL** | Training / test (supervised) | 10 s, 500 Hz, 1000/mV, 16-bit | Normal: 7949, Arrhythmia: 5352 |
| **Arrhythmia** | Training / test (supervised) | 10 s, 500 Hz, 1000/mV, 16-bit | 5 classes (see below) |
| **VitalDB** | Autoencoder experiment (EKG+PPG) | Multimodal recordings | ~5829 normal, ~16 AFIB |

Both PTB-XL and Arrhythmia share identical technical parameters (duration, sampling rate, gain, resolution), yet differences in signal energy suggest risk for dataset bias.

### Arrhythmia Dataset — Class Distribution

| Class | Code | Count |
|-------|------|-------|
| Sinus Bradycardia | SBRAD | 8909 |
| Sinus Rhythm | SR | 5908 |
| Supraventricular Tachycardia | STACH | 3223 |
| Atrial Flutter | AFLT | 1478 |
| Sinus Arrhythmia | SARRH | 1234 |

### PTB-XL — Binary Class Distribution

| Class | Count |
|-------|-------|
| Normal (NORM) | 7949 |
| Arrhythmia (ANORM) | 5352 |

### VitalDB — Autoencoder Experiment

| Class | Approx. Count |
|-------|---------------|
| Normal | ~5829 |
| AFIB | ~16 |

## Project Structure

```
.
├── data_visualization/          # Visualizations and domain shift analysis
├── database_processing/         # Dataset building, deduplication, label filtering
│   ├── dataset_builder.py
│   ├── deduplicate_records.py
│   └── preload_data.py
├── databases/                   # Raw data (PTB-XL, Arrhythmia, VitalDB)
├── databases_processed/         # Processed datasets
├── model_training/
│   ├── autoencoder/             # Anomaly detection experiment
│   │   ├── anomaly_small_batch.py
│   │   ├── anomaly_vae_small_batch.py
│   │   └── evaluate_autoencoder.py
│   └── classification/          # Classifier training and evaluation
│       ├── setup_classificator_architecture.py
│       ├── train_classificator.py
│       └── evaluate_classificator.py
├── models/
│   ├── architectures/           # CustomECGNet, Net1D, ECGFounder wrapper
│   ├── config/                  # ModelConfig, DatasetType
│   ├── data_loaders/            # PyTorch loaders
│   └── weights/                 # Checkpoints
├── results/                     # Metrics, plots, confusion matrices
│   ├── ECG_Founder_db1/         # PTB-XL only weights
│   ├── ECG_Founder_db2/         # Arrhythmia only weights
│   └── ...
├── runs/
│   ├── run_anomaly_pipeline.py
│   └── run_classification_pipeline.py
├── config.py                    # Global configuration
├── main.py                      # Entry point
├── download_vitaldb.py          # VitalDB downloader
└── requirements.txt
```

> **Note:** Folders `databases/`, `databases_processed/`, `models/weights/`, and `results/` are excluded from version control via `.gitignore`. You must create them locally and download the datasets.

## Installation

### 1. Create Environment

```bash
# Conda
conda create -n DL_PROJECT python=3.12
conda activate DL_PROJECT

# Or venv
python -m venv DL_PROJECT
source DL_PROJECT/bin/activate  # Linux/Mac
# DL_PROJECT\Scripts\activate  # Windows
```

### 2. Install PyTorch

Select your backend:

```bash
# NVIDIA (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# AMD (ROCm 7.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> PyTorch must be installed first because `torchmetrics` and other packages depend on it.

## Datasets and Model Setup

Create folders:
```bash
python -c "import os; os.makedirs('databases', exist_ok=True); os.makedirs('models', exist_ok=True)"
```

### Downloading Datasets

**PTB-XL & Arrhythmia** from PhysioNet:
```bash
curl -L -o databases/ptb-xl.zip "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip" && \
curl -L -o databases/arrhythmia.zip "https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
```

**VitalDB** (multimodal, autoencoder experiment):
```bash
python download_vitaldb.py
```

**Faster download** (requires `aria2c`):
```bash
aria2c -x 15 -s 15 -o ptb-xl.zip --dir=databases "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip" && \
aria2c -x 15 -s 15 -o arrhythmia.zip --dir=databases "https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
```

**Unpack** (or run the one-liner in the original docs):
```bash
python -c "import zipfile, os; [zipfile.ZipFile(os.path.join('databases', f)).extractall('databases/') or os.remove(os.path.join('databases', f)) for f in os.listdir('databases/') if f.endswith('.zip')]"
```

## Usage

```bash
conda activate DL_PROJECT
python main.py
```

Experiment configuration in `main.py` (`use_multimodal` flag):
- `use_multimodal = False` — binary classification + 5-class arrhythmia exported as separate weights
- `use_multimodal = True` — EKG+PPG autoencoder experiment on VitalDB

### Pipeline Flow

1. **Data Preparation** (`database_processing/`)
   - `dataset_builder.py` — builds and exports all datasets into parquet files (PTB-XL, Arrhythmia, VitalDB)
   - `deduplicate_records.py` — removes overlapping patients across databases
   - `preload_data.py` — preloads all signals from .mat and .dat
   - `count_labels.py` - counts all of the labels which were exported into csv and parquet, useful for class analysis

2. **Classification Training** (`model_training/classification/`)
   - `setup_classificator_architecture.py` — selects architecture and initializes weights for classification pipeline, both ECGFounder and Custom can be initialized
   - `train_classificator.py` — training loop with early stopping (patience=5), class weights, augmentation
   - `evaluate_classificator.py` — F1, confusion matrix, export to `results/<config_name>/`

3. **Database Bias Control** (`results/ECG_Founder_db1`, `db2`)
   - ECGFounder trained separately on PTB-XL and Arrhythmia without merging
   - Comparison with the combined model to verify generalization improvement

4. **Anomaly Detection** (`model_training/autoencoder/`)
   - `anomaly_small_batch` — standard autoencoder (AE) on EKG+PPG
   - `anomaly_vae_small_batch` — variational autoencoder (VAE) with low KL weight
   - `evaluate_autoencoder.py` — ROC-AUC, reconstruction vs anomaly

> **Note:** The pipeline automatically manages weights — if `use_ecg_weights=True` and no trained weights exist, it loads pretrained ECGFounder; if weights exist, it restores them; otherwise it trains from scratch. Configurations are cloned per dataset (`_binary` / `_arrhythmia`), so experiments do not collide.

## Results

| Task | Main Metric | Result | Notes |
|------|-------------|--------|-------|
| Binary classification | Accuracy / F1 | **~80%** | Fine-tuning ECGFounder + custom CNN |
| 5-class arrhythmia classification | Accuracy / F1 | **~95%** | Good class separation; model generalizes |
| Anomaly detection (autoencoder) | ROC-AUC | **51%** | No separation — model did not learn a meaningful representation |

### Experiment Registry

| Experiment | Architecture | Dataset | Goal |
|------------|--------------|---------|------|
| `customECG` | CustomECGNet | Combined (PTB-XL + Arrhythmia) | Baseline |
| `ECGFounde_with_weights` | ECGFounder + fine-tuning | Combined | Transfer learning |
| `ECG_Founder_db1` / `db2` | ECGFounder | Single (PTB-XL / Arrhythmia) | Database bias control |
| `anomaly_small_batch` | Autoencoder (AE) | VitalDB | Unsupervised detection — first iteration |
| `anomaly_vae_small_batch` | Variational AE (VAE) | VitalDB | Unsupervised detection — second iteration with KL |

## Conclusions

- **Transfer learning works:** Fine-tuning ECGFounder on the combined dataset yields 95% accuracy for 5 arrhythmia classes.
- **Database bias is a real problem:** Two independent classificator weight exports are necessary so the model does not overfit to a single database.
- **Binary classification is harder** (~80%) due to class imbalance and subtle differences between "normal" and "arrhythmia".
- **Unsupervised anomaly detection did not work** (ROC-AUC 51%) — the EKG+PPG autoencoder on VitalDB did not learn a separable representation. Likely cause: excessive inter-patient signal variability.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of GPU memory | Reduce batch size or switch to CPU |
| AMD GPU not detected | Ensure ROCm is installed and use the `rocm7.1` PyTorch wheel |
| Missing `databases/` or `models/` folders | These are `.gitignore`d. Create them locally and add your data / weights |
| VitalDB download fails | Check internet connection; the script uses the VitalDB API |
| Windows issues | The project was primarily developed and tested on Linux (EndeavourOS / Arch). It should work on Windows with minimal adjustments, but has not been thoroughly tested. If you encounter path or encoding issues, check Python/Windows compatibility docs |

## Authors

- **[kenshirotm](https://github.com/yourusername)** — Deep learning pipelines, model training, evaluation metrics, experiment design
- **[RaVS02](https://github.com/RaVS02)** — Minimal contributions to this repository; primary work on the hardware acquisition side

> **Related project:** The hardware and acquisition frontend for this ECG system was developed in a companion repository: **[Simple_ECG_AcquisitionPanel](https://github.com/RaVS02/Simple_ECG_AcquisitionPanel)** by RaVS02, which includes the physical PCB design and measurement UI.

*Project developed for a Deep Learning in Biomedical Signal Processing course.*

## License

MIT License.

Links to used datasets:
- https://physionet.org/content/ptb-xl/1.0.3/
- https://physionet.org/content/ecg-arrhythmia/1.0.0/
- https://physionet.org/content/vitaldb/1.0.0/
