from torch.utils.data import Dataset as torch_Dataset
import os.path as osp
import os
import json
import random
from vigc.datasets.datasets.translation.utils import preproc


class TextPairDataset(torch_Dataset):
    def __init__(self, data_root, source_key, target_key, switch_lang_flag=False):
        self.inner_dataset = self.prepare_dataset(data_root, source_key, target_key)
        self.switch_lang_flag = switch_lang_flag

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
                this_data = {
                    "input": item[source_key],
                    "output": item[target_key],
                    "source_key": source_key,
                    "target_key": target_key
                }
                all_data.append(this_data)

        return all_data

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        source_text, target_text = ann["input"], ann["output"]
        source_text = preproc(source_text)
        target_text = preproc(target_text)
        source_key, target_key = ann["source_key"], ann["target_key"]
        if source_key == "zh":
            source_text = source_text.replace(" ", "")
        if target_key == "zh":
            target_text = target_text.replace(" ", "")
        return dict(input=source_text, output=target_text, input_key=source_key, output_key=target_key, id=str(index))

    def collater(self, batch):
        inputs = [_["input"] for _ in batch]
        outputs = [_["output"] for _ in batch]
        ids = [_["id"] for _ in batch]
        input_key = [_["input_key"] for _ in batch][0]
        output_key = [_["output_key"] for _ in batch][0]
        if self.switch_lang_flag and random.randint(1, 2) == 1:
            inputs, outputs = outputs, inputs
            input_key, output_key = output_key, input_key
        result = dict(input=inputs, output=outputs, input_key=input_key, output_key=output_key, ids=ids)
        return result
