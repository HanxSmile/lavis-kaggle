from torch.utils.data import Dataset as torch_Dataset
import json
import numpy as np
import torch


class RAG_CLS_Dataset(torch_Dataset):
    def __init__(
            self,
            data_root,
    ):
        with open(data_root, 'r') as f:
            self.inner_dataset = json.load(f)

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        query, soft_label = ann['query'], ann['soft_label']

        return {
            "id": str(index),
            "query": query,
            "soft_label": torch.from_numpy(np.array(soft_label)),
        }

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        queries = [_["query"] for _ in batch]
        soft_labels = [_["soft_label"] for _ in batch]

        return {
            "id": ids,
            "query": queries,
            "soft_label": torch.stack(soft_labels),
        }
