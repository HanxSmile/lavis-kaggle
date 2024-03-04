from torch.utils.data import Dataset
import torch
import logging
import json

try:
    from datatoolchain import storage
except ImportError:
    logging.warning(
        "You should install datatoolchain first by 'pip install -q -U shopee-aip-datasets -i https://pypi.shopee.io/'")
    storage = None
from PIL import Image
from io import BytesIO


class BinaryClassDrugDataset(Dataset):

    def __init__(self, ann_path, image_processor, text_processor):

        super().__init__()
        self.train_data = self.load_annotations(ann_path)
        self.image_processor = image_processor
        self.text_processor = text_processor
        self._sto = storage.DatasetStorage.create_cache_storage()

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
            image_file = image_file.split(",")[0]  # 只取第一张图片
        obj = self._sto.get_content(image_file.strip("/") + '_tn')
        img = Image.open(BytesIO(obj))
        return img

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        row = self.train_data[index]
        try:
            image = self.read_image(row["image"])
        except Exception:
            return self[(index + 1) % len(self)]
        text = row["text"]
        label = int(row["label"])
        item_id = str(row["item_id"])

        try:
            image = self.image_processor(image)
        except Exception:
            return self[(index + 1) % len(self)]
        text = self.text_processor(text)

        return {
            "text": text,
            "image": image,
            "label": label,
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
            "label": torch.FloatTensor(label_list),  # [b]
            "id": id_list,
            "image": torch.cat(image_list, dim=0),  # [b, c, h, w]
        }
