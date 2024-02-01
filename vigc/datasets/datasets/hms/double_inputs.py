from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os.path as osp
import numpy as np
from einops import rearrange
import torch


class DoubleInputsDataset(Dataset):

    def __init__(self, data_root, ann_path, fold, fold_idx, random_seed, processor=None, split="train"):
        assert split in ("train", "eval")
        assert fold_idx in list(range(fold))
        super().__init__()
        sgkf = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=random_seed)
        train_csv = pd.read_csv(ann_path)
        self.targets = train_csv.columns[-6:]
        train_csv["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(
                sgkf.split(train_csv, y=train_csv["expert_consensus"], groups=train_csv["patient_id"])):
            train_csv.loc[val_idx, "fold"] = fold_id

        self.data_root = data_root
        self.split = split
        self.fold_idx = fold_idx
        if self.split == "train":
            self.train_csv = train_csv[train_csv.fold != fold_idx]
        else:
            self.train_csv = train_csv[train_csv.fold == fold_idx]
        self.processor = processor

    def __len__(self):
        return len(self.train_csv)

    def _process_spec_data(self, img):
        img = img.T  # [400, 300]
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        # STANDARDIZE PER IMAGE
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)

        return img

    def __getitem__(self, index):
        row = self.train_csv.iloc[index]
        eeg_id, spec_id, sub_id = row.eeg_id, row.spectrogram_id, row.eeg_sub_id

        eeg_file_path = osp.join(self.data_root, "eeg", f"{eeg_id}-{sub_id}.npy")
        spec_file_path = osp.join(self.data_root, "kaggle", f"{spec_id}-{sub_id}.npy")

        eeg_spec_data = np.load(eeg_file_path)  # [128, 256, 4, 4]
        kaggle_spec_data = self._process_spec_data(np.load(spec_file_path))  # [400, 300]

        eeg_spec_data = torch.FloatTensor(eeg_spec_data)
        kaggle_spec_data = torch.FloatTensor(kaggle_spec_data)

        eeg_spec_data = rearrange(eeg_spec_data, "h w m n -> h w (m n)")  # [128, 256, 4, 4] -> [128, 256, 16]
        kaggle_spec_data = kaggle_spec_data[:, :, None]  # [400, 300, 1]
        label = torch.FloatTensor(row[self.targets])  # [6]
        label = label / label.sum()

        if self.processor is not None:
            eeg_spec_data = torch.FloatTensor(self.processor(eeg_spec_data.numpy()))
            kaggle_spec_data = torch.FloatTensor(self.processor(kaggle_spec_data.numpy()))

        return {
            "eeg_image": eeg_spec_data.permute(2, 0, 1),  # [16, 128, 256]
            "spec_image": kaggle_spec_data.permute(2, 0, 1),  # [1, 400, 300]
            "label": label  # [6]
        }

    def collater(self, batch):
        eeg_image_list, spec_image_list, label_list = [], [], []
        for sample in batch:
            eeg_image_list.append(sample["eeg_image"])
            spec_image_list.append(sample["spec_image"])
            label_list.append(sample["label"])

        return {
            "eeg_image": torch.stack(eeg_image_list, dim=0),
            "spec_image": torch.stack(spec_image_list, dim=0),
            "label": torch.stack(label_list, dim=0)
        }
