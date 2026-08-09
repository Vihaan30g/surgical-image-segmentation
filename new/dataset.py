"""
dataset.py
----------
CholecSegDataset: loads RGB frames + watershed ground-truth masks from the
CholecSeg8k directory layout, remaps raw watershed pixel values to
contiguous class IDs, applies temporal sub-sampling, and runs an
Albumentations augmentation pipeline.

VideoBalancedSampler / VideoBalancedBatchSampler: a custom batch sampler
that guarantees each batch is drawn from up to BATCH_SIZE *different*
source videos, avoiding highly-correlated consecutive frames in one batch.
"""

import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# --------------------------------------------------------------------------
# Fast vectorized raw-value -> class-id remapping
# --------------------------------------------------------------------------
def _build_lookup_table(raw_to_class: dict, table_size: int = 256) -> np.ndarray:
    """
    Builds a 256-entry lookup table so remapping is a single vectorized
    numpy indexing operation instead of a per-pixel Python loop.
    Any raw value not present in raw_to_class maps to class 0 (background)
    as a safe fallback.
    """
    lut = np.zeros(table_size, dtype=np.uint8)
    for raw_value, class_id in raw_to_class.items():
        lut[raw_value] = class_id
    return lut


_LUT = _build_lookup_table(config.RAW_TO_CLASS)


def remap_watershed_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    raw_mask: single-channel (H, W) array containing raw watershed pixel
    values (e.g. 80, 17, 33, ...).
    Returns: (H, W) array of contiguous class ids in [0, NUM_CLASSES - 1].
    """
    return _LUT[raw_mask]


# --------------------------------------------------------------------------
# Augmentation pipelines
# --------------------------------------------------------------------------
def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
            # --- Spatial transforms (applied identically to image + mask) ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomRotate90(p=1.0),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=45,
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=1, sigma=50, alpha_affine=50,
                        border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                        p=1.0,
                    ),
                    A.GridDistortion(p=1.0),
                ],
                p=0.3,
            ),
            # --- Pixel-level transforms (image only; Albumentations does
            #     not apply these to the 'mask' target automatically) ---
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # --- Normalization to [0, 1] (min-max), then to tensor ---
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_eval_transforms() -> A.Compose:
    """Used for validation/test: resize + normalize only, no augmentation."""
    return A.Compose(
        [
            A.Resize(height=config.IMG_HEIGHT, width=config.IMG_WIDTH),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class CholecSegDataset(Dataset):
    """
    Args:
        root_dir: path to the 'archive' directory containing videoXX folders.
        video_list: list of video folder names to include (e.g. TRAIN_VIDEOS).
        frame_step: sub-sampling step size (2 for train, 4 for val/test).
        transforms: an Albumentations Compose pipeline.
    """

    def __init__(self, root_dir: str, video_list: list, frame_step: int,
                 transforms: A.Compose = None):
        self.root_dir = root_dir
        self.video_list = video_list
        self.frame_step = frame_step
        self.transforms = transforms

        # samples: list of dicts {image_path, mask_path, video_name, frame_id}
        self.samples = []
        # video_to_indices: for the balanced sampler
        self.video_to_indices = defaultdict(list)

        self._index_dataset()

    def _index_dataset(self):
        for video_name in self.video_list:
            video_path = os.path.join(self.root_dir, video_name)
            if not os.path.isdir(video_path):
                print(f"[CholecSegDataset] WARNING: missing video dir {video_path}")
                continue

            clip_dirs = sorted(
                d for d in os.listdir(video_path)
                if os.path.isdir(os.path.join(video_path, d))
            )

            for clip_dir in clip_dirs:
                clip_path = os.path.join(video_path, clip_dir)
                frame_files = sorted(
                    f for f in os.listdir(clip_path) if f.endswith(config.IMAGE_SUFFIX)
                )

                # Apply temporal sub-sampling within this clip
                for idx, fname in enumerate(frame_files):
                    if idx % self.frame_step != 0:
                        continue

                    frame_prefix = fname[: -len(config.IMAGE_SUFFIX)]  # "frame_{N}"
                    image_path = os.path.join(clip_path, fname)
                    mask_path = os.path.join(
                        clip_path, frame_prefix + config.WATERSHED_MASK_SUFFIX
                    )

                    if not os.path.isfile(mask_path):
                        continue  # skip frames without ground-truth mask

                    sample_idx = len(self.samples)
                    self.samples.append(
                        {
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "video_name": video_name,
                            "frame_id": frame_prefix,
                        }
                    )
                    self.video_to_indices[video_name].append(sample_idx)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples indexed from {self.root_dir} for videos {self.video_list}. "
                "Check that the directory structure and file suffixes match config.py."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]

        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask_bgr = cv2.imread(record["mask_path"], cv2.IMREAD_COLOR)
        # All 3 channels are identical; take channel 0.
        raw_mask = raw_mask_bgr[:, :, 0]
        class_mask = remap_watershed_mask(raw_mask).astype(np.int64)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=class_mask)
            image = augmented["image"]  # (3, H, W) float tensor
            class_mask = augmented["mask"]  # (H, W) tensor
            if not torch.is_tensor(class_mask):
                class_mask = torch.from_numpy(class_mask)
            class_mask = class_mask.long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            class_mask = torch.from_numpy(class_mask).long()

        return {
            "image": image,
            "mask": class_mask,
            "video_name": record["video_name"],
            "frame_id": record["frame_id"],
            "index": index,
        }


# --------------------------------------------------------------------------
# VideoBalancedSampler
# --------------------------------------------------------------------------
class VideoBalancedBatchSampler(Sampler):
    """
    Yields batches of indices such that each batch draws from up to
    `batch_size` *different* source videos (one sample per video per batch
    whenever possible), reducing intra-batch temporal correlation.

    Strategy: maintain a per-video queue of shuffled sample indices. Each
    batch round-robins across videos, popping one index from each until the
    batch is full. Videos are reshuffled/replenished once exhausted (for
    training) so every epoch still covers the full dataset roughly once.
    """

    def __init__(self, dataset: CholecSegDataset, batch_size: int = config.BATCH_SIZE,
                 shuffle: bool = True, drop_last: bool = False, seed: int = config.SEED):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.video_names = [v for v in dataset.video_to_indices if len(dataset.video_to_indices[v]) > 0]
        if len(self.video_names) == 0:
            raise RuntimeError("VideoBalancedBatchSampler: dataset has no indexed videos.")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _make_video_queues(self, rng: random.Random):
        queues = {}
        for video_name, indices in self.dataset.video_to_indices.items():
            idx_copy = list(indices)
            if self.shuffle:
                rng.shuffle(idx_copy)
            queues[video_name] = idx_copy
        return queues

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        video_order = list(self.video_names)
        if self.shuffle:
            rng.shuffle(video_order)

        queues = self._make_video_queues(rng)
        total_remaining = sum(len(q) for q in queues.values())

        while total_remaining > 0:
            batch = []
            videos_this_round = list(video_order)
            if self.shuffle:
                rng.shuffle(videos_this_round)

            for video_name in videos_this_round:
                if len(batch) >= self.batch_size:
                    break
                queue = queues[video_name]
                if len(queue) == 0:
                    continue
                batch.append(queue.pop())
                total_remaining -= 1

            if len(batch) == 0:
                break  # all queues exhausted

            if len(batch) < self.batch_size and self.drop_last:
                break

            yield batch

    def __len__(self):
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def build_dataloader(dataset: CholecSegDataset, batch_size: int = config.BATCH_SIZE,
                      shuffle: bool = True, drop_last: bool = False,
                      num_workers: int = config.NUM_WORKERS):
    """Convenience wrapper that plugs VideoBalancedBatchSampler into a DataLoader."""
    from torch.utils.data import DataLoader

    batch_sampler = VideoBalancedBatchSampler(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
