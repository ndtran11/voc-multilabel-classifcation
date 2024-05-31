import torch
import numpy as np
from PIL import Image
import os
import random


def mosaic_2_2(im1: Image, im2: Image, im3: Image, im4: Image) -> Image:

    board = Image.new('RGB', (im1.width + im2.width, im1.height + im3.height))
    board.paste(im1, (0, 0))
    board.paste(im2, (im1.width, 0))
    board.paste(im3, (0, im1.height))
    board.paste(im4, (im1.width, im1.height))

    return board

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, image_files, labels, n_classes, transform, mosaic = .25):

        self.root = root
        self.image_files = image_files

        self.labels = [np.array(e) for e in labels] if labels is not None else None 

        self.transform = transform
        self.n_classes = n_classes
        self.mosaic = mosaic

        assert labels is None or len(image_files) == len(labels)

    def __len__(self):
        return len(self.image_files)

    def mosaic_roll_dice(self) -> bool:
        return random.random() < self.mosaic

    def __getitem__(self, idx):


        if not self.mosaic_roll_dice():
            image_file = self.image_files[idx]

            path = os.path.join(self.root, image_file) \
                if self.root is not None else image_file

            image = Image.open(path).convert("RGB")
            label_set = np.zeros(self.n_classes)

            if self.labels is not None:
                label = self.labels[idx]
                label_set[label] = 1

        else:
            image_file = self.image_files[idx]
            another_3_images = random.choices(range(len(self)), k = 3)

            if self.root is not None:
                images = [Image.open(os.path.join(self.root, image_file)).convert("RGB") 
                        for image_file in [image_file] + [
                            self.image_files[i] for i in another_3_images]]
            else:
                images = [Image.open(image_file).convert("RGB") 
                        for image_file in [image_file] + [
                            self.image_files[i] for i in another_3_images]]

            image = mosaic_2_2(*images)
            label_set = np.zeros(self.n_classes)

            if self.labels is not None:
                label = self.labels[idx]
                label_set[label] = 1

                for i in another_3_images:
                    label = self.labels[i]
                    label_set[label] = 1

        transforms_img = self.transform(image)
        return transforms_img, torch.tensor(label_set, dtype=torch.float32) 