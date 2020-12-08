import os
from PIL import Image
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from PrecipitationForecastRadar.dataset.utils_torch import SegDatasetBase


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


class VOCSegmentation(SegDatasetBase):
    CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor']

    def __init__(self, root: str = 'data', mode: str = "train", transform=None,
                 augmentations: bool = False,
                 **kwargs) -> None:
        super().__init__(root, mode, transform, **kwargs)

        base_dir_path = os.path.join(root, "VOC2012")
        image_dir_path = os.path.join(base_dir_path, "JPEGImages")
        mask_dir_path = os.path.join(base_dir_path, "SegmentationClass")

        splits_dir_path = os.path.join(base_dir_path, "ImageSets", "Segmentation")
        if mode == "train":
            split_file_path = os.path.join(splits_dir_path, "train.txt")
        elif mode in ("val", "test", "demo"):
            split_file_path = os.path.join(splits_dir_path, "val.txt")
        elif mode == 'trainval':
            split_file_path = os.path.join(splits_dir_path, "trainval.txt")
        else:
            raise RuntimeError("Unknown dataset splitting mode")

        self.images = []
        self.masks = []
        with open(os.path.join(split_file_path), "r") as lines:
            for line in lines:
                image_file_path = os.path.join(image_dir_path, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(image_file_path)
                self.images.append(image_file_path)
                mask_file_path = os.path.join(mask_dir_path, line.rstrip('\n') + ".png")
                assert os.path.isfile(mask_file_path)
                self.masks.append(mask_file_path)

        assert (len(self.images) == len(self.masks))

        self.augmentations = augmentations

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        # Apply augmentations to both input and target
        if self.augmentations:
            img, target = self.apply_augmentations(img, target)

        # Convert the RGB image to a tensor
        toTensorTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], )
        ])
        img = toTensorTransform(img)
        # Convert target to long tensor
        target = torch.from_numpy(np.array(target)).long()
        target[target == 255] = 0

        return img, target

    def __len__(self):
        return len(self.images)

    def apply_augmentations(self, img, target):
        # Horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            target = TF.hflip(target)
        # Random Rotation (clockwise and counter clockwise)
        if random.random() > 0.5:
            degrees = 10
            if random.random() > 0.5:
                degrees *= -1
            img = TF.rotate(img, degrees)
            target = TF.rotate(target, degrees)
        # Brighten or darken image (only applied to input image)
        if random.random() > 0.5:
            brightness = 1.2
            if random.random() > 0.5:
                brightness -= 0.4
            img = TF.adjust_brightness(img, brightness)
        return img, target


if __name__ == '__main__':
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    voc_segmentation = VOCSegmentation(transform=transformations)

    test = voc_segmentation.__getitem__(1)
    print()
