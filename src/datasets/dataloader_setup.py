import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Sampler

from src.datasets.dataset_splitter import create_splits
from src.datasets.cholecseg_dataset import CholecSegDataset
from src.transforms.segmentation_transforms import (
    train_transforms,
    val_transforms
)


# =========================================
# VIDEO BALANCED SAMPLER
# =========================================
#
# Problem with plain shuffle=True:
#   The DataLoader just randomly reorders all
#   frame indices. A batch of 8 could still be
#   8 frames from the same clip — just shuffled.
#   They'd be near-identical images, and the
#   model would learn the "flavor" of that clip.
#
# Solution — VideoBalancedSampler:
#   For each sample in an epoch:
#     1. Pick a video at random (uniform over
#        all videos, regardless of how many
#        frames each video has).
#     2. Pick a random frame from that video.
#   Result: every batch of N samples draws from
#   ~N different videos. The model cannot rely
#   on video-specific cues — it must learn the
#   universal anatomy.
#
# Note on epoch length:
#   We define one epoch as iterating over all
#   training frames once (total_samples =
#   total frames after striding). The sampler
#   draws that many samples with replacement
#   from the cross-video distribution.

class VideoBalancedSampler(Sampler):


    def __init__(self, dataset):
        """
        dataset : CholecSegDataset instance.
                  Must have a .video_ids attribute
                  (list of video names, parallel
                  to dataset image_paths).
        """

        self.dataset = dataset

        # =====================================
        # BUILD VIDEO → INDEX REGISTRY
        # =====================================
        #
        # video_to_indices["video01"] = [0, 1, 2, ...]
        # video_to_indices["video09"] = [45, 46, ...]
        # ...

        self.video_to_indices = defaultdict(list)

        for idx, video_id in enumerate(
            dataset.video_ids
        ):
            self.video_to_indices[video_id].append(idx)

        self.video_names = list(
            self.video_to_indices.keys()
        )

        self.num_videos = len(self.video_names)

        # Epoch length = total number of frames
        # after temporal striding. We iterate
        # this many steps per epoch.
        self.total_samples = len(dataset)


    def __iter__(self):
        """
        At each step:
          1. Pick a video uniformly at random.
          2. Pick a frame from that video
             uniformly at random.
        Repeat total_samples times.
        """

        indices = []

        for _ in range(self.total_samples):

            # pick a random video
            video = random.choice(self.video_names)

            # pick a random frame from that video
            idx = random.choice(
                self.video_to_indices[video]
            )

            indices.append(idx)

        return iter(indices)


    def __len__(self):

        return self.total_samples


# =========================================
# CREATE DATALOADERS
# =========================================

def create_dataloaders(batch_size=8):
    """
    Returns (train_loader, val_loader, test_loader).

    train_loader uses VideoBalancedSampler to
    ensure each batch contains frames from
    multiple different videos.

    val_loader and test_loader use standard
    sequential loading (no shuffle) since we
    already applied temporal striding at the
    dataset_splitter level.
    """

    # =====================================
    # GET SPLITS
    # =====================================

    splits = create_splits()

    train_images, train_masks, train_vids = (
        splits["train"]
    )

    val_images, val_masks, val_vids = (
        splits["val"]
    )

    test_images, test_masks, test_vids = (
        splits["test"]
    )


    # =====================================
    # CREATE DATASETS
    # =====================================

    train_dataset = CholecSegDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        video_ids=train_vids,
        transforms=train_transforms
    )

    val_dataset = CholecSegDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        video_ids=val_vids,
        transforms=val_transforms
    )

    test_dataset = CholecSegDataset(
        image_paths=test_images,
        mask_paths=test_masks,
        video_ids=test_vids,
        transforms=val_transforms
    )


    # =====================================
    # BUILD SAMPLER FOR TRAINING
    # =====================================

    train_sampler = VideoBalancedSampler(
        train_dataset
    )


    # =====================================
    # CREATE DATALOADERS
    # =====================================
    #
    # train_loader:
    #   - Uses VideoBalancedSampler (custom).
    #   - shuffle MUST be False when a custom
    #     sampler is provided — PyTorch raises
    #     an error otherwise. The sampler
    #     handles all randomness.
    #   - drop_last=True: if the last batch is
    #     smaller than batch_size, drop it.
    #     Avoids batch-size-1 edge cases that
    #     can destabilize GroupNorm.
    #
    # val_loader / test_loader:
    #   - Plain sequential, no sampler needed.
    #   - shuffle=False always for evaluation.

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        sampler=train_sampler,

        shuffle=False,       # sampler handles this

        num_workers=2,

        pin_memory=True,

        drop_last=True
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True,

        drop_last=False
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True,

        drop_last=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# =========================================
# DEBUG INFO
# =========================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        create_dataloaders(batch_size=8)
    )

    print("\n===== DATALOADER SUMMARY =====\n")

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    print("\n===== FIRST TRAINING BATCH =====\n")

    images, masks = next(iter(train_loader))

    print(f"Image shape : {images.shape}")
    print(f"Mask  shape : {masks.shape}")
    print(f"Image dtype : {images.dtype}")
    print(f"Mask  dtype : {masks.dtype}")
    print(f"Mask unique : {torch.unique(masks)}")
