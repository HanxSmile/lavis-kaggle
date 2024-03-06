from torch.utils.data import Dataset
import torch
import logging
import json
import random

try:
    from datatoolchain import storage
except ImportError:
    logging.warning(
        "You should install datatoolchain first by 'pip install -q -U shopee-aip-datasets -i https://pypi.shopee.io/'")
    storage = None
from PIL import Image
from io import BytesIO


class MultiClassDrugDataset(Dataset):

    def __init__(self, ann_path, label_map, image_processor, text_processor, split="train"):

        super().__init__()
        assert split in ("train", "eval")
        self.split = split
        self.train_data = self.load_annotations(ann_path)
        self.image_processor = image_processor
        self.text_processor = text_processor
        self._sto = storage.DatasetStorage.create_cache_storage()
        self.num_classes = len(label_map)
        self.label_map = label_map

    def load_annotations(self, ann_path):
        if isinstance(ann_path, str):
            ann_path = [ann_path]

        all_data = []
        for file in ann_path:
            with open(file) as f:
                data = json.load(f)
            all_data.extend(data)
        return all_data

    def read_image(self, image_file):
        if "," in image_file:
            if self.split == "train":
                image_file = random.choice(image_file.split(","))  # 随机取一张图像
            else:
                image_file = image_file.split(",")[0]  # 取第一张图像
        obj = self._sto.get_content(image_file.strip("/") + '_tn')
        img = Image.open(BytesIO(obj))
        return img

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        row = self.train_data[index]
        try:
            image = self.read_image(row["images"])
        except Exception:
            return self[(index + 1) % len(self)]
        text = row["name"]
        label_index = int(self.label_map.get(row["label"]))
        item_id = str(row["item_id"])

        label = torch.zeros([self.num_classes])
        label[label_index] = 1.0

        try:
            image = self.image_processor(image)
        except Exception:
            return self[(index + 1) % len(self)]
        text = self.text_processor(text)

        return {
            "text": text,
            "image": image,
            "label": label.float(),
            "id": item_id,
        }

    def collater(self, batch):
        # eeg_image_list, spec_image_list, label_list = [], [], []
        text_list, label_list, id_list, image_list = [], [], [], []
        for sample in batch:
            text_list.append(sample["text"])
            label_list.append(sample["label"])
            id_list.append(sample["id"])
            image_list.append(sample["image"])

        return {
            "text": text_list,
            "label": torch.stack(label_list, dim=0),  # [b, c]
            "id": id_list,
            "image": torch.cat(image_list, dim=0),  # [b, c, h, w]
        }