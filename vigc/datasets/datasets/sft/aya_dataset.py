from torch.utils.data import Dataset as torch_Dataset
import datasets


class AyaInstructionDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            split="train",
            language=None,
    ):
        if isinstance(split, str):
            split = [split]
        inner_dataset = datasets.load_from_disk(data_root)
        inner_dataset = datasets.concatenate_datasets([inner_dataset[_] for _ in split])
        if language is not None:
            if isinstance(language, str):
                language = [language]
            else:
                language = list(language)
            inner_dataset = inner_dataset.filter(lambda x: x in language, input_columns=["language"])
        self.inner_dataset = inner_dataset
        self.language = language

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        item = self.inner_dataset[index]
        text_input, text_output, language = item["inputs"].strip(), item["targets"].strip(), item["language"].strip()
        system_input = f"You are an expert in {language}, please always response in {language}."
        sample = {
            "system_input": system_input,
            "text_input": text_input,
            "text_output": text_output,
            "language": language,
            "id": str(index)
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
