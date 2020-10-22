import os

from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    def __init__(self, input_dir, label_dir, image_size):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.image_size = image_size

        self.input_images = os.listdir(input_dir)
        self.label_images = os.listdir(label_dir)

    @classmethod
    def preprocess(cls, pil_img, image_size):
        w, h = pil_img.size
        if not image_size == w:
            pil_img = pil_img.resize((image_size, image_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = (img_trans / 127.5) - 1

        return img_trans

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        image_name = self.input_images[index]
        input_file = glob(self.input_dir + image_name)
        label_file = glob(self.label_dir + image_name)

        input_image = Image.open(input_file[0])
        label_image = Image.open(label_file[0])

        input_image = self.preprocess(input_image, self.image_size)
        label_image = self.preprocess(label_image, self.image_size)

        return {
            "input_image": torch.from_numpy(input_image).type(torch.FloatTensor),
            "label_image": torch.from_numpy(label_image).type(torch.FloatTensor),
        }
