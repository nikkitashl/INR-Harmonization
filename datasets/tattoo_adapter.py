import cv2 as cv
import numpy as np
import os


class TattooAdapter:
    def __init__(self, root_path, img_names_file, images_folder, mask_folder, augmentations):
        self._augmentations = augmentations
        self._image_pathes = []
        self._masks_pathes = []
        with open(img_names_file) as reader:
            for line in reader:
                self._image_pathes.append(os.path.join(root_path, images_folder, line.strip()))
                self._masks_pathes.append(os.path.join(root_path, mask_folder, line.strip()))

    def __len__(self):
        return len(self._image_pathes)

    def _read_img(self, path):
        return cv.imread(path)

    def __getitem__(self, idx):
        img = self._read_img(self._image_pathes[idx])
        mask = self._read_img(self._masks_pathes[idx])

        img_after = self._augmentations(image=img)["image"]
        return np.where(mask > 0, img_after, img), mask, img, self._image_pathes[idx]
