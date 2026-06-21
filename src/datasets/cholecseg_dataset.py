import cv2
import numpy as np

from torch.utils.data import Dataset

from configs.class_mapping import RAW_TO_CLASS


class CholecSegDataset(Dataset):


    def __init__(
        self,
        image_paths,
        mask_paths,
        video_ids,
        transforms=None
    ):
        """
        image_paths : list of Path objects, one per frame
        mask_paths  : list of Path objects, parallel to image_paths
        video_ids   : list of str, e.g. "video01", parallel to image_paths.
                      Needed by VideoBalancedSampler to know which video
                      each index belongs to.
        transforms  : albumentations Compose pipeline
        """

        self.image_paths = image_paths

        self.mask_paths = mask_paths

        self.video_ids = video_ids

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

        # All three channels are identical in
        # watershed masks — extract channel 0.

        mask = mask[:, :, 0]


        # =====================================
        # REMAP RAW VALUES → CLASS IDS
        # =====================================
        #
        # Watershed masks store raw pixel values
        # like 80, 17, 33 — not sequential IDs.
        # RAW_TO_CLASS maps them to 0–12.

        class_mask = np.zeros_like(mask)

        for raw_value, class_id in RAW_TO_CLASS.items():

            class_mask[mask == raw_value] = class_id


        # =====================================
        # APPLY TRANSFORMS
        # =====================================
        #
        # Albumentations applies spatial ops
        # (flip, rotate, elastic) identically to
        # both image and mask. Color ops only
        # apply to the image.

        if self.transforms is not None:

            transformed = self.transforms(
                image=image,
                mask=class_mask
            )

            image = transformed["image"]

            class_mask = transformed["mask"]


        return image, class_mask
