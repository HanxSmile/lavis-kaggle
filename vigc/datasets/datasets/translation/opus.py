from torch.utils.data import Dataset as torch_Dataset
from datasets import load_from_disk
import random
from vigc.datasets.datasets.translation.utils import preproc


class OpusDataset(torch_Dataset):
    def __init__(self, data_root, source_key, target_key, split="train", switch_lang_flag=False):
        assert split in ["train", "test"]
        self.source_key = source_key
        self.target_key = target_key
        self.inner_dataset = load_from_disk(data_root)[split]
        self.switch_lang_flag = switch_lang_flag

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]["translation"]
        source_text, target_text = ann[self.source_key], ann[self.target_key]
        source_text = preproc(source_text)
        target_text = preproc(target_text)
        source_key, target_key = self.source_key, self.target_key

        return {
            "input": source_text,
            "output": target_text,
            "id": str(index),
            "input_key": source_key,
            "output_key": target_key
        }

    def collater(self, batch):
        inputs = [_["input"] for _ in batch]
        outputs = [_["output"] for _ in batch]
        ids = [_["id"] for _ in batch]
        input_key = [_["input_key"] for _ in inputs][0]
        output_key = [_["output_key"] for _ in outputs][0]
        if self.switch_lang_flag and random.randint(1, 2) == 1:
            inputs, outputs = outputs, inputs
            input_key, output_key = output_key, input_key
        result = dict(input=inputs, output=outputs, input_key=input_key, output_key=output_key, ids=ids)
        return result
