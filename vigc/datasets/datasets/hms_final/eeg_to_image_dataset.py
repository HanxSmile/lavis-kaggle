from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import torch
import torch.nn.functional as F

from scipy.signal import butter, lfilter


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    return mu_x  # quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


class EEG2ImageDataset(Dataset):

    def __init__(self, data_root, ann_path, fold, fold_idx, random_seed, processor=None, split="train"):
        assert split in ("train", "eval")
        assert fold_idx in list(range(fold))
        assert isinstance(random_seed, int)
        super().__init__()
        sgkf = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=random_seed)
        train_csv = pd.read_csv(ann_path)
        self.targets = train_csv.columns[-6:]
        train_csv = self.preprocess_csv(train_csv)

        self.eegs = np.load(data_root, allow_pickle=True).item()

        # self.specs = np.load(data_root.spec, allow_pickle=True).item()
        # self.low_resource = low_resource
        # if low_resource:
        #     self.eeg_specs = {
        #         int(k): f"{osp.join(data_root.eeg.replace('eeg_specs.npy', 'EEG_Spectrograms'), f'{k}.npy')}"
        #         for k in train_csv.eeg_id.values}
        # else:
        #     self.eeg_specs = np.load(data_root.eeg, allow_pickle=True).item()

        train_csv["fold"] = -1
        for fold_id, (_, val_idx) in enumerate(
                sgkf.split(train_csv, y=train_csv["target"], groups=train_csv["patient_id"])):
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

    def __data_generation(self, row):
        eeg_features = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
        feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}

        X = np.zeros((10_000, 8), dtype='float32')
        data = self.eegs[row.eeg_id]
        if self.mode == "train":
            if np.random.random() < 0.5:
                data = data[::-1, :]

        # === Feature engineering ===
        X[:, 0] = data[:, feature_to_index['Fp1']] - data[:, feature_to_index['T3']]
        X[:, 1] = data[:, feature_to_index['T3']] - data[:, feature_to_index['O1']]

        X[:, 2] = data[:, feature_to_index['Fp1']] - data[:, feature_to_index['C3']]
        X[:, 3] = data[:, feature_to_index['C3']] - data[:, feature_to_index['O1']]

        X[:, 4] = data[:, feature_to_index['Fp2']] - data[:, feature_to_index['C4']]
        X[:, 5] = data[:, feature_to_index['C4']] - data[:, feature_to_index['O2']]

        X[:, 6] = data[:, feature_to_index['Fp2']] - data[:, feature_to_index['T4']]
        X[:, 7] = data[:, feature_to_index['T4']] - data[:, feature_to_index['O2']]

        # === Standarize ===
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # === Butter Low-pass Filter ===
        X = butter_lowpass_filter(X)
        X = torch.from_numpy(X).permute(1, 0).unsqueeze(1).unsqueeze(-1)  # [8, 1, 10000, 1]
        X = F.unfold(X, kernel_size=(256, 1), stride=10000 // 256 - 1)

        y = row[self.targets].values.astype("float")
        return X, y

    def __getitem__(self, index):
        row = self.data.iloc[index]
        eeg_id = row.eeg_id
        image, label = self.__data_generation(row)

        return {
            "image": image,
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
