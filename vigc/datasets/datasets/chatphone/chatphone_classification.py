from torch.utils.data import Dataset
import pandas as pd
import torch


class ChatPhoneClassificationDataset(Dataset):

    def __init__(self, ann_path, processor=None):

        super().__init__()
        all_dataset = []
        if isinstance(ann_path, str):
            csv = pd.read_csv(ann_path)
            for i, row in csv.iterrows():
                all_dataset.append({"text": row.text, "label": row.label})
        else:

            for ann_ in ann_path:
                csv = pd.read_csv(ann_)
                for i, row in csv.iterrows():
                    all_dataset.append({"text": row.text, "label": row.label})
        self.data = all_dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        text = row["text"]
        label = int(row["label"])

        if self.processor is not None:
            text = self.processor(text)

        return {
            "text": text,
            "label": label,
            "id": index,
            "row": row
        }

    def collater(self, batch):
        # eeg_image_list, spec_image_list, label_list = [], [], []
        text_list, label_list, id_list, row_list = [], [], [], []
        for sample in batch:
            text_list.append(sample["text"])
            label_list.append(sample["label"])
            id_list.append(sample["id"])
            row_list.append(sample["row"])

        return {
            "text": text_list,
            "label": torch.FloatTensor(label_list),
            "id": id_list,
            "row": row_list
        }
