from torch.utils.data import Dataset as torch_Dataset
import json


class TranslationJsonDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            source_lang,
            target_lang,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.inner_dataset = self.load_samples(data_root)

    def load_samples(self, data_root):
        results = []
        with open(data_root, 'r') as f:
            data = json.load(f)
        for item in data:
            this_item = {
                "text_input": item[self.source_lang],
                "text_output": item[self.target_lang],
            }
            if "id" in item:
                this_item["id"] = item["id"]
            results.append(this_item)
        return results

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        item = self.inner_dataset[index]
        text_input, text_output = item["text_input"].strip(), item["text_output"].strip()
        system_input = f"You are a helpful assistant. Please translate this Markdown format English document paragraph into Chinese."
        sample = {
            "system_input": system_input,
            "text_input": text_input,
            "text_output": text_output,
            "language": None,
            "id": item["id"] if "id" in item else str(index)
        }
        return sample

    def collater(self, features):
        result = {
            "system_input": [_["system_input"] for _ in features],
            "text_input": [_["text_input"] for _ in features],
            "text_output": [_["text_output"] for _ in features],
            "language": [_["language"] for _ in features],
            "id": [_["id"] for _ in features]
        }
        return result
