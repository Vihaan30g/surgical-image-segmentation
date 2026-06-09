
import albumentations as A

from albumentations.pytorch import ToTensorV2


# TRAIN TRANSFORMS
train_transforms = A.Compose([

    A.Resize(
        height=256,
        width=256
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.RandomRotate90(
        p=0.5
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


# VALIDATION TRANSFORMS
val_transforms = A.Compose([

    A.Resize(
        height=256,
        width=256
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


