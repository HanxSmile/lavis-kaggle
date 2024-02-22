from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import torch


class ImageHMSDataset(Dataset):

    def __init__(self, data_root, ann_path, fold, fold_idx, random_seed, processor=None, split="train"):
        assert split in ("train", "eval")
        assert fold_idx in list(range(fold))
        assert isinstance(random_seed, int)
        super().__init__()
        sgkf = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=random_seed)
        train_csv = pd.read_csv(ann_path)
        self.targets = train_csv.columns[-6:]
        train_csv = self.preprocess_csv(train_csv)
        self.eeg_specs = np.load(data_root.eeg, allow_pickle=True).item()
        self.specs = np.load(data_root.spec, allow_pickle=True).item()

        train_csv["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(
                sgkf.split(train_csv, y=train_csv["expert_consensus"], groups=train_csv["patient_id"])):
            train_csv.loc[val_idx, "fold"] = fold_id

        self.mode = split
        self.fold_idx = fold_idx
        if split == "train":
            self.data = train_csv[train_csv.fold != fold_idx]
        else:
            self.data = train_csv[train_csv.fold == fold_idx]
        self.augment = processor

    def preprocess_csv(self, train_csv):
        targets = train_csv.columns[-6:]
        train_df = train_csv.groupby("eeg_id")[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
            {'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
        train_df.columns = ["spec_id", "min"]
        train_df["max"] = train_csv.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
            {'spectrogram_label_offset_seconds': 'max'})
        train_df["patient_id"] = train_csv.groupby("eeg_id")[["patient_id"]].agg("first")

        tmp = train_csv.groupby("eeg_id")[targets].agg("sum")
        for t in targets:
            train_df[t] = tmp[t].values

        y_data = train_df[targets].values
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train_df[targets] = y_data

        train_df["target"] = train_csv.groupby("eeg_id")[["expert_consensus"]].agg("first")
        train_df = train_df.reset_index()
        return train_df

    def __len__(self):
        return len(self.data)

    def _generate_data(self, row):
        X = np.zeros((128, 256, 8), dtype="float32")
        r = int((row['min'] + row['max']) // 4)
        for k in range(4):
            # EXTRACT 300 ROWS OF SPECTROGRAM
            img = self.specs[row.spec_id][r:r + 300, k * 100:(k + 1) * 100].T

            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            ep = 1e-6
            m = np.nanmean(img.flatten())
            s = np.nanstd(img.flatten())
            img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)

            # CROP TO 256 TIME STEPS
            X[14:-14, :, k] = img[:, 22:-22] / 2.0

        img = self.eeg_specs[row.eeg_id]
        X[:, :, 4:] = img
        y = row[self.targets]
        return X, y

    def __getitem__(self, index):
        row = self.data.iloc[index]
        eeg_id = row.eeg_id
        image, label = self._generate_data(row)
        if self.augment is not None:
            image = self.augment(image)

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),  # [6]
            "eeg_id": eeg_id,
        }

    def collater(self, batch):
        image_list, label_list, eeg_id_list = [], [], []
        for sample in batch:
            image_list.append(sample["image"])
            label_list.append(sample["label"])
            eeg_id_list.append(sample["eeg_id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "label": torch.stack(label_list, dim=0),
            "eeg_id": eeg_id_list,
        }
