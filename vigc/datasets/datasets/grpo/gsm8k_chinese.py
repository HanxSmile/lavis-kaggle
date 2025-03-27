from torch.utils.data import Dataset as torch_Dataset
import datasets


class Gsm8kChineseDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            split="train",
            system_prompt=None
    ):
        if isinstance(split, str):
            split = [split]
        inner_dataset = datasets.load_from_disk(data_root)
        inner_dataset = datasets.concatenate_datasets([inner_dataset[_] for _ in split])
        self.inner_dataset = inner_dataset
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        item = self.inner_dataset[index]
        text_input, text_output = item["question_zh-cn"].strip(), str(item["answer_only"]).strip()
        sample = {
            "system_input": self.system_prompt,
            "text_input": text_input,
            "text_output": text_output,
            "id": str(index)
        }
        return sample

    def collater(self, features):
        result = {
            "system_input": [_["system_input"] for _ in features],
            "text_input": [_["text_input"] for _ in features],
            "text_output": [_["text_output"] for _ in features],
            "id": [_["id"] for _ in features]
        }
        return result
