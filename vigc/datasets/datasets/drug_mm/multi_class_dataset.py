from torch.utils.data import Dataset
import pandas as pd
import torch
import logging

try:
    from datatoolchain import storage
except ImportError:
    logging.warning(
        "You should install datatoolchain first by 'pip install -q -U shopee-aip-datasets -i https://pypi.shopee.io/'")
    storage = None
from PIL import Image
from io import BytesIO


class MultiClassDrugDataset(Dataset):

    def __init__(self, ann_path, num_classes, image_processor, text_processor):

        super().__init__()
        self.train_csv = pd.read_csv(ann_path)
        self.image_processor = image_processor
        self.text_processor = text_processor
        self._sto = storage.DatasetStorage.create_cache_storage()
        self.num_classes = num_classes

    def read_image(self, image_file):
        obj = self._sto.get_content(image_file + '_tn')
        img = Image.open(BytesIO(obj))
        return img

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        row = self.train_csv.iloc[index]
        try:
            image = self.read_image(row.image)
        except Exception:
            return self[(index + 1) % len(self)]
        text = row.text
        label_index = int(row.label)
        label = torch.zeros([self.num_classes])
        label[label_index] = 1.0

        image = self.image_processor(image)
        text = self.text_processor(text)

        return {
            "text": text,
            "image": image,
            "label": label.float(),
            "id": index,
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
            "image": torch.cat(image_list, dim=0)  # [b, c, h, w]
        }
