from pathlib import Path

import cv2
import numpy as np

from torch.utils.data import Dataset

from configs.class_mapping import RAW_TO_CLASS


class CholecSegDataset(Dataset):


    def __init__(
        self,
        image_paths,
        mask_paths,
        transforms=None
    ):

        self.image_paths = image_paths

        self.mask_paths = mask_paths

        self.transforms = transforms


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):


        # =====================================
        # LOAD RGB IMAGE
        # =====================================

        image = cv2.imread(
            str(self.image_paths[idx])
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # =====================================
        # LOAD WATERSHED MASK
        # =====================================

        mask = cv2.imread(
            str(self.mask_paths[idx])
        )

        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2RGB
        )

        # all channels identical
        # so extract first channel

        mask = mask[:, :, 0]


        # =====================================
        # CONVERT RAW VALUES TO CLASS IDS
        # =====================================

        class_mask = np.zeros_like(mask)

        for raw_value, class_id in RAW_TO_CLASS.items():

            class_mask[mask == raw_value] = class_id


        # =====================================
        # APPLY TRANSFORMS
        # =====================================

        if self.transforms is not None:

            transformed = self.transforms(
                image=image,
                mask=class_mask
            )

            image = transformed["image"]

            class_mask = transformed["mask"]


        return image, class_mask

