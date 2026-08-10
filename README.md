# CholecSeg8k U-Net Segmentation

## Files
- `config.py` — paths, video-level splits, `CLASS_INFO`/`RAW_TO_CLASS`/`CLASS_NAMES`, hyperparameters
- `dataset.py` — `CholecSegDataset`, raw→class remapping, temporal sub-sampling, Albumentations pipeline, `VideoBalancedBatchSampler`
- `model.py` — 4-layer U-Net (GroupNorm + LeakyReLU, raw logits output)
- `losses.py` — `FocalLoss`, `DiceLoss`, `ComboLoss`
- `metrics.py` — per-class Dice / mIoU with NaN-aware handling of absent classes
- `train.py` — training entrypoint: environment check, auto-resume, training/validation loops, visualization export

## Data flow in this version
- **Code**: cloned from your GitHub repo (not uploaded manually).
- **Dataset**: downloaded straight from Kaggle onto the Colab VM's local disk (`/content/data/archive`) — not stored on Drive.
- **Drive**: mounted once, manually, in its own notebook cell (this codebase never calls `drive.mount()`). Drive is used only to persist `checkpoints/` and `visualizations/` as training progresses, so nothing is lost if the runtime disconnects.

---

## Colab notebook cells (run in this order)

### 1. Mount Google Drive (manual — outside the code)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install required packages
```python
!pip install -q albumentations opencv-python-headless tqdm matplotlib kaggle
```
`torch`/`torchvision` are already preinstalled on Colab GPU runtimes — no need to reinstall unless you want a specific version.

### 3. Clone your code from GitHub
```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
```
(Replace with your actual repo URL/path — this should be the folder containing `config.py`, `dataset.py`, `model.py`, `losses.py`, `metrics.py`, `train.py`.)

### 4. Configure Kaggle credentials and download the dataset
Upload your `kaggle.json` API token (from kaggle.com → Account → Create New API Token) first, e.g. via the Colab file-upload widget, then:
```python
import os
os.makedirs('/root/.kaggle', exist_ok=True)
!cp kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

!mkdir -p /content/data
!kaggle datasets download -d newslab/cholecseg8k -p /content/data --unzip
```
This should leave you with `/content/data/archive/video01/...` etc. If the unzip produces a different top-level folder name, either rename it or update `config.DATA_ROOT` / `config.LOCAL_DATA_ROOT` to match. `config.KAGGLE_DATASET_SLUG` documents the expected slug — update it there too if you're using a different Kaggle dataset listing.

### 5. Run training
```python
!python train.py
```
`train.py` will:
- verify Drive is mounted and the dataset exists locally (raises a clear error otherwise — it does **not** mount Drive or download data for you),
- auto-resume from `checkpoints/checkpoint_latest.pth` on Drive if present,
- save `checkpoint_latest.pth` (every epoch) and `checkpoint_best.pth` (on new best val Dice) to Drive,
- save 3-way visualization grids per epoch to `.../surgical-image-segmentation/visualizations/epoch_{N}/` on Drive.

Because the local Colab VM disk (`/content/data`) is wiped on every new runtime, re-running from a fresh session means repeating step 4 (Kaggle download) — but checkpoints on Drive mean training itself picks up right where it left off.

---

## Notes
- No GPU/torch in the environment used to generate this code, so files were syntax-checked (`py_compile`) but not executed end-to-end. Run `python model.py` first thing in Colab to confirm the `UNet` shape check (`(N, 3, 288, 512) -> (N, 13, 288, 512)`) passes on your runtime.
- `ElasticTransform`'s `alpha_affine` argument is deprecated/removed in newer Albumentations versions — if you hit a `TypeError`, pin `albumentations<=1.3.1` or drop that kwarg.
- Double-check `config.KAGGLE_DATASET_SLUG` and the extracted folder structure match — Kaggle dataset zips don't always extract to exactly `archive/videoXX/...`.
