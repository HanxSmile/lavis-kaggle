import torch
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import json


class Im2MkdownDataset(Dataset):

    def __init__(self, ann_path, media_dir, processor):
        super().__init__()
        self.media_dir = media_dir
        self.vis_processor = processor
        self.inner_dataset = self.init_samples(ann_path)

    def __len__(self):
        return len(self.inner_dataset)

    def init_samples(self, ann_path):
        samples = []
        with open(ann_path, 'r') as f:
            data = json.load(f)
        for ann in data:
            item = {
                "image": osp.join(self.media_dir, ann['image']),
                "text": osp.join(self.media_dir, ann['text']),
            }
            samples.append(item)
        return samples

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        image = Image.open(ann['image']).convert('RGB')
        with open(ann['text'], 'r', encoding="utf-8") as f:
            text = f.read().strip()
        try:
            image = self.vis_processor(image)
        except Exception as e:
            print(f"Exception {e} while processing image {ann['image']}")
            return self[(index + 1) % len(self)]
        if image is None:
            print(f"'{ann['image']}' is empty")
            return self[(index + 1) % len(self)]
        return {"image": image, "text_input": text, "id": index}

    def collater(self, samples):
        image_list, question_list, id_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            id_list.append(sample["id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "id": id_list
        }


class Im2MkdownTestDataset(Dataset):

    def __init__(self, ann_path, media_dir, processor):
        super().__init__()
        self.media_dir = media_dir
        self.vis_processor = processor
        self.inner_dataset = self.init_samples(ann_path)

    def __len__(self):
        return len(self.inner_dataset)

    def init_samples(self, ann_path):
        samples = []
        with open(ann_path, 'r') as f:
            data = json.load(f)
        all_images = sorted(list(data.keys()))

        for image_name in all_images:
            item = {
                "image": osp.join(self.media_dir, image_name),
                "id": image_name,
            }
            samples.append(item)
        return samples

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        image = Image.open(ann['image']).convert('RGB')
        id_ = ann["id"]
        try:
            image = self.vis_processor(image)
        except Exception as e:
            print(f"Exception {e} while processing image {ann['image']}")
            return self[(index + 1) % len(self)]
        if image is None:
            print(f"'{ann['image']}' is empty")
            return self[(index + 1) % len(self)]
        return {"image": image, "text_input": "", "id": id_}

    def collater(self, samples):
        image_list, question_list, id_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            id_list.append(sample["id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "id": id_list
        }
