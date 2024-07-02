from torch.utils.data import Dataset as torch_Dataset
import os.path as osp
import os
import json


class TextPairDataset(torch_Dataset):
    def __init__(self, data_root, source_key, target_key):
        self.inner_dataset = self.prepare_dataset(data_root, source_key, target_key)

    @staticmethod
    def prepare_dataset(data_root, source_key, target_key):
        all_data_paths = []
        all_data = []
        if isinstance(data_root, str):
            if osp.isdir(data_root):
                all_data_paths = [osp.join(data_root, _) for _ in sorted(os.listdir(data_root)) if _.endswith(".json")]
            elif osp.isfile(data_root) and data_root.endswith(".json"):
                all_data_paths = [data_root]
        else:
            for path in data_root:
                if osp.isdir(path) and path.endswith(".json"):
                    all_data_paths.append(path)

        for path in all_data_paths:
            with open(path, "r") as f:
                data = json.load(f)
            for item in data:
                this_data = {"input": item[source_key], "output": item[target_key]}
                all_data.append(this_data)

        return all_data

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        src_text, tgt_text = ann["input"], ann["output"]
        return {"input": src_text, "output": tgt_text, "id": str(index)}

    def collater(self, batch):
        result = dict()
        result["input"] = [_["input"] for _ in batch]
        result["output"] = [_["output"] for _ in batch]
        result["ids"] = [_["id"] for _ in batch]
        return result
