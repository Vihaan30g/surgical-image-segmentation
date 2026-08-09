# CholecSeg8k U-Net Segmentation

## Files
- `config.py` — paths, video-level splits, `CLASS_INFO`/`RAW_TO_CLASS`/`CLASS_NAMES`, hyperparameters
- `dataset.py` — `CholecSegDataset`, raw→class remapping, temporal sub-sampling, Albumentations pipeline, `VideoBalancedBatchSampler`
- `model.py` — 4-layer U-Net (GroupNorm + LeakyReLU, raw logits output)
- `losses.py` — `FocalLoss`, `DiceLoss`, `ComboLoss`
- `metrics.py` — per-class Dice / mIoU with NaN-aware handling of absent classes
- `train.py` — Colab entrypoint: Drive mount, auto-resume, training/validation loops, visualization export

## Colab setup
```python
!pip install albumentations opencv-python-headless tqdm matplotlib -q
```
Upload the six `.py` files to your Colab working directory (or clone from Drive), make sure
`/content/drive/MyDrive/surgical-image-segmentation/data/archive/videoXX/...` matches the
expected layout, then run:
```python
!python train.py
```
Re-running the same cell/session will automatically resume from
`checkpoints/checkpoint_latest.pth` if present.

## Notes
- Environment used to generate this code has no GPU/torch, so files were syntax-checked
  (`py_compile`) but not executed end-to-end. Run `python model.py` first in Colab to confirm
  the `UNet` shape check (`(N, 3, 288, 512) -> (N, 13, 288, 512)`) passes on your runtime.
- `ElasticTransform`'s `alpha_affine` argument is deprecated/removed in newer Albumentations
  versions — if you hit a `TypeError`, pin `albumentations<=1.3.1` or drop that kwarg.
