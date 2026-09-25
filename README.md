# SSViT-LRLPR

Companion code and reproducibility guide for the manuscript **“A CNN-ViT Framework for Recognizing Low-Resolution License Plates”**, currently **under review**.

This work addresses **Low-Resolution License Plate Recognition (LRLPR)** directly from degraded low-resolution images, without first reconstructing an intermediate high-resolution image. The proposed CNN–ViT framework is designed to preserve fine-grained character structure under real-world degradations such as motion blur, sensor noise, limited resolution, and compression artifacts.

The architecture combines:

- **Restormer-based feature extraction** for dense low-resolution feature refinement;
- the **Surgical Focus Block (SFB)**, a Coordinate Attention adaptation for direction-aware spatial refinement;
- a **ViT decoder** for character-sequence prediction;
- a **normalized cosine classifier** for character classification;
- a **validation-aware Exponential Moving Average (EMA) shadow model** for final inference; and
- **late-stage temporal fusion** when multiple observations of the same license plate are available.

The manuscript evaluates the method on **LRLPR-26**, using license-plate tracklets containing sequential low-resolution observations of the same vehicle. Multi-frame inference combines frame-level evidence using **Bayes Joint Probability (BJP) fusion**.

## Main reported results

Recognition Rate (RR) uses a strict **7/7 exact-match** criterion: a plate is counted as correct only when all seven characters are predicted correctly.

| Input frames | Recognition Rate |
|---:|---:|
| 1 | **60.6%** |
| 3 | **74.2%** |
| 5 | **78.5%** |

At five input frames, the proposed method reaches **78.5% RR**, compared with **73.8%** for the strongest evaluated STR baseline under the same protocol.

The evaluated STR baselines include **SVTRv2, OTE, LISTER, IGTR, CPPD, and MDiff4STR**.

> **Manuscript status:** Under review. Repository documentation and reported results may be updated during the revision process.

A pretrained EMA checkpoint is available from the repository's [GitHub Releases](https://github.com/valfride/SSViT-LRLPR/releases/tag/weights).

## Citation

A formal citation and BibTeX entry will be added after publication. Until then, please refer to the manuscript by its title:

**A CNN-ViT Framework for Recognizing Low-Resolution License Plates**

## Repository layout

```text
SSViT-LRLPR/
├── train_gan.py                 # Main training entry point
├── test.py                      # Validation/test inference and temporal fusion
├── ablation_configs/            # Proposed model and ablation configs
├── baselines_configs/           # Baseline model configs
├── datasets/                    # LMDB dataset reader and wrappers
├── models/                      # Proposed model + baseline bridges
├── train_funcs/                 # Training/validation routines and decoders
└── LMDB-Datasets/               # Expected local dataset link/folder, ignored by git
```

The main training script receives `--config`, `--save`, and an optional `--tag`. It writes results to:

```text
<save>/<CONFIG_STEM>_<tag>/
├── config_snapshot.yaml
├── loss_log.csv
├── student_weights/
│   ├── last.pth
│   └── student_acc_<acc>_ep_<epoch>.pth
└── ghost_weights/               # Present only when use_ema_ghost: true
    ├── last.pth
    └── ghost_acc_<acc>_ep_<epoch>.pth
```

## 1. Environment setup

Python 3.10 is recommended. Create and activate an environment, install a PyTorch build compatible with your CUDA setup, and then install the remaining repository dependencies.

```bash
conda create -n ssvit-lrlpr python=3.10 -y
conda activate ssvit-lrlpr

# Install the PyTorch build appropriate for your system/CUDA version.
pip install torch torchvision torchaudio

# Install the remaining dependencies used by the repository.
pip install -r requirements.txt
```

For CUDA memory fragmentation issues, the following environment variable is useful:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 2. Dataset layout

All current configs expect the LMDB dataset at:

```text
./LMDB-Datasets/competition_dataset_lmdb
```

The LMDB directory must contain a `metadata.pkl` file. The dataset reader filters samples by the `split` field inside the metadata using phases such as `training`, `validation`, and `test`.

A typical local layout is:

```text
SSViT-LRLPR/
└── LMDB-Datasets/
    └── competition_dataset_lmdb/
        ├── data.mdb
        ├── lock.mdb
        └── metadata.pkl
```

If your dataset is stored elsewhere, either create a symbolic link:

```bash
ln -s /path/to/LMDB-Datasets ./LMDB-Datasets
```

or edit `path_split` in each YAML config.

## 3. Training commands

### Single-GPU debug/local training

`train_gan.py` runs in single-GPU debug mode by default when `DEBUG=True` or when `DEBUG` is not set. The script also sets `CUDA_VISIBLE_DEVICES=0` internally when it is not already defined.

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config <CONFIG_PATH> \
  --save ./experiments \
  --tag <RUN_TAG>
```

### Multi-GPU distributed training

For distributed runs, set `DEBUG=False` and launch with `torchrun`:

```bash
DEBUG=False torchrun --nproc_per_node=2 train_gan.py \
  --config <CONFIG_PATH> \
  --save ./experiments \
  --tag <RUN_TAG>
```

Adjust `--nproc_per_node` to the number of GPUs you want to use.

## 4. Proposed model

The main proposed configuration is:

```text
ablation_configs/proposed_config.yaml
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config ablation_configs/proposed_config.yaml \
  --save ./experiments/proposed \
  --tag proposed
```

This config enables the proposed CNN--ViT model (`VSR_CURVATURE`) with SFB and EMA ghost tracking. For final inference, prefer the `ghost_weights/` directory when it exists.

Example validation with five frames and Bayes fusion:

```bash
python3 test.py \
  --config ablation_configs/proposed_config.yaml \
  --checkpoints ./experiments/proposed/proposed_config_proposed/ghost_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

## 5. Baseline model configs

The baseline implementations in this repository are evaluated under the same LRLPR data and temporal-fusion protocol used for the proposed model. The links below point to the original publications and, when provided by the authors, the official implementation.

| Baseline | Venue | Paper | Code | Config | `model_g.name` | `cls_loss` |
|---|---|---|---|---|---:|---:|
| SVTRv2 | ICCV 2025 | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html) | [OpenOCR](https://github.com/Topdu/OpenOCR) | `baselines_configs/SVTRV2_BASELINE.yaml` | `SVTRV2_BASELINE` | `CTC` |
| OTE | CVPR 2024 | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_OTE_Exploring_Accurate_Scene_Text_Recognition_Using_One_Token_CVPR_2024_paper.html) | [OpenOCR](https://github.com/Topdu/OpenOCR) | `baselines_configs/OTE_BASELINE.yaml` | `OTE_BASELINE` | `OTE` |
| LISTER | ICCV 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_LISTER_Neighbor_Decoding_for_Length-Insensitive_Scene_Text_Recognition_ICCV_2023_paper.html) | [OpenOCR](https://github.com/Topdu/OpenOCR) | `baselines_configs/LISTER_BASELINE.yaml` | `LISTER_BASELINE` | `LISTER_INTERNAL` |
| IGTR | TPAMI 2025 | [Paper](https://doi.org/10.1109/TPAMI.2025.3525526) | [OpenOCR](https://github.com/Topdu/OpenOCR) | `baselines_configs/IGTR_BASELINE.yaml` | `IGTR_BASELINE` | `IGTR_INTERNAL` |
| CPPD | TPAMI 2025 | [Paper](https://doi.org/10.1109/TPAMI.2025.3545453) | [OpenOCR](https://github.com/Topdu/OpenOCR) | `baselines_configs/CPPD_BASELINE.yaml` | `CPPD_BASELINE` | `CPPD` |
| MDiff4STR | AAAI 2026 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37370) | [OpenOCR](https://github.com/Topdu/OpenOCR) | `baselines_configs/MDIFF_BASELINE.yaml` | `MDIFF_BASELINE` | `MDIFF_INTERNAL` |

All baseline configs use the same LMDB dataset path by default and validate with `VSR_Sequence_collate_fn`, where `in_images: 5` can be overridden at inference time with `--in_images`.

## 6. How to run each baseline

Create a shared output directory first:

```bash
mkdir -p ./experiments/baselines
```

### 6.1 SVTRv2

Train:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config baselines_configs/SVTRV2_BASELINE.yaml \
  --save ./experiments/baselines \
  --tag svtrv2
```

Validate with `F=5` and Bayes fusion:

```bash
python3 test.py \
  --config baselines_configs/SVTRV2_BASELINE.yaml \
  --checkpoints ./experiments/baselines/SVTRV2_BASELINE_svtrv2/student_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

### 6.2 OTE

Train:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config baselines_configs/OTE_BASELINE.yaml \
  --save ./experiments/baselines \
  --tag ote
```

Validate:

```bash
python3 test.py \
  --config baselines_configs/OTE_BASELINE.yaml \
  --checkpoints ./experiments/baselines/OTE_BASELINE_ote/student_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

### 6.3 LISTER

Train:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config baselines_configs/LISTER_BASELINE.yaml \
  --save ./experiments/baselines \
  --tag lister
```

Validate:

```bash
python3 test.py \
  --config baselines_configs/LISTER_BASELINE.yaml \
  --checkpoints ./experiments/baselines/LISTER_BASELINE_lister/student_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

### 6.4 IGTR

Train:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config baselines_configs/IGTR_BASELINE.yaml \
  --save ./experiments/baselines \
  --tag igtr
```

Validate:

```bash
python3 test.py \
  --config baselines_configs/IGTR_BASELINE.yaml \
  --checkpoints ./experiments/baselines/IGTR_BASELINE_igtr/student_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

### 6.5 CPPD

Train:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config baselines_configs/CPPD_BASELINE.yaml \
  --save ./experiments/baselines \
  --tag cppd
```

Validate:

```bash
python3 test.py \
  --config baselines_configs/CPPD_BASELINE.yaml \
  --checkpoints ./experiments/baselines/CPPD_BASELINE_cppd/student_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

### 6.6 MDiff4STR

Train:

```bash
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py \
  --config baselines_configs/MDIFF_BASELINE.yaml \
  --save ./experiments/baselines \
  --tag mdiff
```

Validate:

```bash
python3 test.py \
  --config baselines_configs/MDIFF_BASELINE.yaml \
  --checkpoints ./experiments/baselines/MDIFF_BASELINE_mdiff/student_weights \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

## 7. Evaluating `F = 1`, `F = 3`, and `F = 5`

Use the same trained checkpoint and change only `--in_images`:

```bash
# Single-frame evaluation
python3 test.py \
  --config <CONFIG_PATH> \
  --checkpoints <CHECKPOINT_DIR> \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 1 \
  --fusion bayes

# Three-frame evaluation
python3 test.py \
  --config <CONFIG_PATH> \
  --checkpoints <CHECKPOINT_DIR> \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 3 \
  --fusion bayes

# Five-frame evaluation
python3 test.py \
  --config <CONFIG_PATH> \
  --checkpoints <CHECKPOINT_DIR> \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode val \
  --in_images 5 \
  --fusion bayes
```

Available temporal fusion options are:

```text
bayes          # product-rule/log-probability fusion
average        # probability averaging
majority       # hard-voting over decoded strings
logit_average  # raw-logit averaging
```

The paper's main multi-frame comparisons use late-stage fusion over the first `F` sequential observations. For reproducing the main reported protocol, use `--fusion bayes` with `F in {1, 3, 5}`.

## 8. Producing a test submission file

For test mode, pass `--mode test` and an output path:

```bash
python3 test.py \
  --config <CONFIG_PATH> \
  --checkpoints <CHECKPOINT_DIR> \
  --split ./LMDB-Datasets/competition_dataset_lmdb \
  --mode test \
  --in_images 5 \
  --fusion bayes \
  --output submission_<model>_F5.txt
```

The output format is:

```text
track_id,predicted_plate;confidence
```

## 9. Suggested reproducibility checklist

Before reporting results, record:

- Git commit SHA.
- Config file path and `config_snapshot.yaml`.
- Dataset split/path used by `--split`.
- Number of frames: `--in_images 1`, `3`, or `5`.
- Fusion rule: usually `bayes` for the paper protocol.
- Checkpoint source: `student_weights/` or `ghost_weights/`.
- Whether SWA (`--swa`) or TTA (`--tta`) was enabled.
- GPU model, CUDA version, PyTorch version, and random seed.

## 10. Quick command summary

```bash
# Proposed model
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config ablation_configs/proposed_config.yaml --save ./experiments/proposed --tag proposed

# Baselines
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config baselines_configs/SVTRV2_BASELINE.yaml --save ./experiments/baselines --tag svtrv2
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config baselines_configs/OTE_BASELINE.yaml    --save ./experiments/baselines --tag ote
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config baselines_configs/LISTER_BASELINE.yaml --save ./experiments/baselines --tag lister
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config baselines_configs/IGTR_BASELINE.yaml   --save ./experiments/baselines --tag igtr
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config baselines_configs/CPPD_BASELINE.yaml   --save ./experiments/baselines --tag cppd
CUDA_VISIBLE_DEVICES=0 DEBUG=True python3 train_gan.py --config baselines_configs/MDIFF_BASELINE.yaml  --save ./experiments/baselines --tag mdiff
```

## 11. Notes

- Do not commit datasets, checkpoints, logs, images, or generated submissions unless explicitly needed. These paths and extensions are already covered by `.gitignore`.
- If a run is interrupted, inspect `student_weights/last.pth` and resume by setting `resume:` in the corresponding YAML config.
- The validation script automatically selects the best `*acc_*.pth` checkpoint from the checkpoint directory; if none exists, it falls back to `last.pth`.
- If using EMA/ghost checkpoints, point `--checkpoints` to `ghost_weights/`; otherwise use `student_weights/`.
