from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np


class SemanticSegmentationDataset(Dataset):
    def __init__(
            self,
            ann_path,
            media_dir,
            processor=None,
    ):
        super().__init__()
        self.media_dir = media_dir
        if isinstance(ann_path, str):
            ann_path = [ann_path]
        self.ann_path = ann_path
        all_lines = []
        for p_ in self.ann_path:
            with open(p_, 'r') as f:
                lines = f.readlines()
                all_lines.extend(lines)
        all_data = [line.strip().split() for line in all_lines if line.strip()]
        self.inner_dataset = []
        for item in all_data:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(image_path))[0]
            this_item = {
                "image": os.path.join(media_dir, image_path),
                "label": os.path.join(media_dir, label_path),
                "name": name
            }
            self.inner_dataset.append(this_item)
        self.processor = processor

    def __len__(self):
        return len(self.inner_dataset)

    def load_label(self, label_path):
        raise NotImplementedError

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        return image

    def __getitem__(self, index):
        sample = self.inner_dataset[index]
        image_path, label_path, name = sample["image"], sample["label"], sample["name"]
        image = self.load_image(image_path)
        label = self.load_label(label_path)

        if self.processor is not None:
            data = self.processor(image=image, mask=label)
            image = data['image']
            label = data['mask']

        img = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        msk = torch.from_numpy(np.transpose(label, (2, 0, 1)))

        return {
            "image": img,
            "mask": msk,
            "name": name
        }

    def collater(self, batch):
        images = [_["image"] for _ in batch]
        masks = [_["mask"] for _ in batch]
        names = [_["name"] for _ in batch]
        return {
            "image": torch.stack(images, dim=0),
            "mask": torch.stack(masks, dim=0),
            "name": names
        }
