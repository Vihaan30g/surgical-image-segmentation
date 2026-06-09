
from torch.utils.data import DataLoader

from src.datasets.dataset_splitter import create_splits

from src.datasets.cholecseg_dataset import CholecSegDataset

from src.transforms.segmentation_transforms import (
    train_transforms,
    val_transforms
)


# =========================================
# CREATE DATALOADERS
# =========================================

def create_dataloaders(batch_size=4):


    # =====================================
    # GET SPLITS
    # =====================================

    splits = create_splits()


    train_images, train_masks = splits["train"]

    val_images, val_masks = splits["val"]

    test_images, test_masks = splits["test"]


    # =====================================
    # CREATE DATASETS
    # =====================================

    train_dataset = CholecSegDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        transforms=train_transforms
    )


    val_dataset = CholecSegDataset(
        image_paths=val_images,
        mask_paths=val_masks,
        transforms=val_transforms
    )


    test_dataset = CholecSegDataset(
        image_paths=test_images,
        mask_paths=test_masks,
        transforms=val_transforms
    )


    # =====================================
    # CREATE DATALOADERS
    # =====================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=4,

        pin_memory=True
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True
    )


    return (
        train_loader,
        val_loader,
        test_loader
    )

