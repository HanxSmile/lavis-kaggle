from scipy.signal import butter, lfilter
import numpy as np
import torch
import torch.nn.functional as F
import random
from einops import rearrange


def quantize_data(data, classes):
    def mu_law_encoding(data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
        return mu_x

    mu_x = mu_law_encoding(data, classes)
    return mu_x


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def torch_nan_mean(x, dim=0, keepdim=False):
    return torch.sum(torch.where(torch.isnan(x), torch.zeros_like(x), x), dim=dim, keepdim=keepdim) / torch.sum(
        torch.where(torch.isnan(x), torch.zeros_like(x), torch.ones_like(x)), dim=dim, keepdim=keepdim)


def torch_nan_std(x, dim=0, keepdim=False):
    d = x - torch_nan_mean(x, dim=dim, keepdim=True)
    return torch.sqrt(torch_nan_mean(d * d, dim=dim, keepdim=keepdim))


def normalize(x):
    # x.shape = [frame, (group, k, 2)]
    x = rearrange(x, "f (g k c) -> g (f k) c", k=84, c=2)
    x = x.contiguous()
    m = torch_nan_mean(x, dim=1)  # g c
    s = torch_nan_std(x, dim=1)
    x = (x - m[:, None]) / s[:, None]
    x = rearrange(x, "g (f k) c -> f (g k c)", k=84)
    return x


class EEGDataset(torch.utils.data.Dataset):
    TARGETS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
    FEATURE_NUMS = 8

    def __init__(self, data, base_divide=None, eegs=None, augmentations=None, sample_num=5000, x_std_flag=True,
                 v_std_flag=True, x_divide=None, v_divide=None, v_flag=False, training=True):

        self.data = data
        self.eegs = eegs
        self.augmentations = augmentations
        self.sample_num = sample_num
        self.x_divide = x_divide or [50, 20, 10]
        self.x_std_flag = x_std_flag
        self.v_std_flag = v_std_flag
        self.v_divide = v_divide
        self.v_flag = v_flag
        self.training = training
        self.base_divide = base_divide or {"value": 200, "by_divide": True}

    def __len__(self):
        return len(self.data)

    def preprocess_eeg(self, data):
        data = np.clip(data, -1024, 1024)
        data = np.nan_to_num(data, nan=0) / 32.0

        data = butter_lowpass_filter(data)
        # data = quantize_data(data, 1)
        return data

    def local_std(self, x, divide: int):
        frame_num = int(x.shape[0])
        kernel_size = max(frame_num // divide // 2 * 2 + 1, 3)
        padded_x = F.pad(x, (0, 0, kernel_size // 2, kernel_size // 2), "constant", np.nan)
        unfold_x = padded_x.unfold(0, kernel_size, 1)  # [f, (k c), kernel_size]
        x_local_std = x - torch_nan_mean(unfold_x, dim=2, keepdim=False)
        return x_local_std

    def local_mean(self, x, value: int, by_divide=True):
        if by_divide:
            frame_num = int(x.shape[0])
            kernel_size = max(frame_num // value // 2 * 2 + 1, 3)
        else:
            kernel_size = value
        padded_x = F.pad(x, (0, 0, kernel_size // 2, kernel_size // 2), "constant", np.nan)
        unfold_x = padded_x.unfold(0, kernel_size, 1)  # [f, (k c), kernel_size]
        x_local_mean = torch_nan_mean(unfold_x, dim=2, keepdim=False)
        return x_local_mean

    def calculate_velocity(self, x):
        padded_x1 = F.pad(x, (0, 0, 0, 1), "constant", np.nan)
        velocity_x1 = padded_x1[1:] - x

        padded_x2 = F.pad(x, (0, 0, 1, 0), "constant", np.nan)
        velocity_x2 = x - padded_x2[:-1]

        velocity_x = torch.stack([velocity_x1, velocity_x2], dim=0)
        velocity_x = torch_nan_mean(velocity_x, dim=0)
        return velocity_x

    @property
    def channel_number(self):
        return (len(self.x_divide) + self.x_std_flag + self.v_std_flag + self.v_flag + len(
            self.v_divide) + 1) * self.FEATURE_NUMS

    def __getitem__(self, index):

        row = self.data.iloc[index]
        data = self.eegs[row.eeg_id]

        data = self.preprocess_eeg(data)  # [10000, 8]
        x = torch.from_numpy(data).float()

        frame_num = int(x.shape[0])
        whole_x = [x]

        x_local_mean = self.local_mean(x, self.base_divide)

        if self.x_std_flag:
            x_std = x_local_mean - torch_nan_mean(x, dim=0, keepdim=True)
            whole_x.append(x_std)
        if self.x_divide:
            x_local_std = torch.cat([x_local_mean - self.local_mean(x, divide) for divide in self.x_divide], dim=1)
            whole_x.append(x_local_std)

        v_x = self.calculate_velocity(x)  # [f c]

        if self.v_flag:
            whole_x.append(v_x)

        if self.v_std_flag:
            v_x_std = v_x - torch_nan_mean(v_x, dim=0, keepdim=True)
            whole_x.append(v_x_std)

        if self.v_divide:
            v_local_std = torch.cat([v_x - self.local_mean(v_x, divide) for divide in self.v_divide], dim=1)
            whole_x.append(v_local_std)

        sample_frame = list(range(frame_num)) * (self.sample_num // frame_num) + \
                       random.sample(range(frame_num), k=self.sample_num % frame_num)  # [frame]

        whole_x = normalize(torch.cat(whole_x, dim=1))[sample_frame, :]

        frame_index = torch.FloatTensor(sample_frame).unsqueeze(-1) / frame_num

        x = torch.cat([whole_x, frame_index], dim=1)
        if not self.test:
            label = row[self.TARGETS]
            label = torch.tensor(label).float()
            return x, label
        else:
            return x
