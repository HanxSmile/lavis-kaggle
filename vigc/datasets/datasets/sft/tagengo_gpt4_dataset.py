from torch.utils.data import Dataset as torch_Dataset
import datasets


class TagengoGPT4Dataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            language=None,
            use_system_prompt=True
    ):

        inner_dataset = datasets.load_dataset(data_root)["train"]
        if language is not None:
            if isinstance(language, str):
                language = [language]
            else:
                language = list(language)
            inner_dataset = inner_dataset.filter(lambda x: x.lower() in [_.lower() for _ in language],
                                                 input_columns=["language"])
        inner_dataset = inner_dataset.filter(lambda x: x[0] is not None, input_columns=["response"])
        self.inner_dataset = inner_dataset
        self.language = language
        self.use_system_prompt = use_system_prompt

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        item = self.inner_dataset[index]
        language = item["language"].strip()
        conversation = item["conversations"]
        text_input, text_output = conversation[0]["value"].strip(), conversation[-1]["value"].strip()
        assert conversation[0]['from'] == "human" and conversation[-1]['from'] == 'gpt'
        system_input = f"You are an expert in {language}, please always response in {language}."
        sample = {
            "system_input": system_input if self.use_system_prompt else None,
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
