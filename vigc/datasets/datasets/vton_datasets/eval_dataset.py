import os
import PIL.Image
import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms


class VtonFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path_info):
        self.path_info = path_info
        self.gen_images = [_["gen_image_path"] for _ in path_info]
        self.ori_images = [_["ori_image_path"] for _ in path_info]

    def __len__(self):
        return len(self.path_info)

    def __getitem__(self, idx):
        gen_image_path = self.gen_images[idx]
        ori_image_path = self.ori_images[idx]

        gen_image_name = os.path.splitext(os.path.basename(gen_image_path))[0]
        ori_image_name = os.path.splitext(os.path.basename(ori_image_path))[0]
        assert gen_image_name == ori_image_name
        gen_image = PIL.Image.open(gen_image_path).convert("RGB")
        ori_image = PIL.Image.open(ori_image_path).convert("RGB")

        size = gen_image.size[::-1]  # height, width
        ori_image_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        gen_image_transform = transforms.Compose([transforms.ToTensor])
        gen_image = gen_image_transform(gen_image)
        ori_image = ori_image_transform(ori_image)

        return gen_image, ori_image
