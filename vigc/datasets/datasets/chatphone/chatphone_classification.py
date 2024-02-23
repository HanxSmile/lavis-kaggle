from torch.utils.data import Dataset
import pandas as pd


class ChatPhoneClassificationDataset(Dataset):

    def __init__(self, ann_path, processor=None):

        super().__init__()
        self.train_csv = pd.read_csv(ann_path)
        self.processor = processor

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        row = self.train_csv.iloc[index]
        text = row.text
        label = int(row.label)

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
            "label": label_list,
            "id": id_list,
            "row": row_list
        }
