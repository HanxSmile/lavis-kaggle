from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import KFold


class ImageCTRDataset(Dataset):

    def __init__(self, ann_path, media_path, image_processor, split="train", fold_nums=5, fold_idx=0, random_seed=42):

        super().__init__()
        assert split in ("train", "eval")
        self.fold_nums = fold_nums
        self.fold_idx = fold_idx
        self.split = split
        self.random_seed = random_seed
        self.media_path = media_path
        self.inner_dataset = self.load_annotations(ann_path)
        self.image_processor = image_processor

    def load_annotations(self, ann_path):
        df = pd.read_csv(ann_path)
        if self.fold_idx == -1:
            return df
        df['fold'] = -1
        kf = KFold(n_splits=self.fold_nums, shuffle=True, random_state=self.random_seed)
        for fold, (_, val_index) in enumerate(kf.split(df)):
            df.loc[val_index, 'fold'] = fold
        if self.split == "train":
            return df[df['fold'] != self.fold_idx]
        else:
            return df[df['fold'] == self.fold_idx]

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset.iloc[index]
        image_name = ann["file"]
        image_path = os.path.join(self.media_path, image_name)
        label = ann["click"] / ann["show"]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.image_processor(image)
        except Exception:
            print(f"Error occured while processing image '{image_name}'.")
            return self[(index + 1) % len(self)]

        return {
            "image": image,
            "label": label,
            "id": image_name,
        }

    def collater(self, batch):
        label_list, id_list, image_list = [], [], []
        for sample in batch:
            label_list.append(sample["label"])
            id_list.append(sample["id"])
            image_list.append(sample["image"])

        return {
            "label": torch.FloatTensor(label_list),  # [b]
            "id": id_list,
            "image": torch.stack(image_list, dim=0),  # [b, c, h, w]
        }
