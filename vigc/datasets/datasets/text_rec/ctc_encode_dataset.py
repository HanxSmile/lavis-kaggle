import os.path as osp
import logging

import torch

from .image_ops import ReadImage
from .label_ops import CTCLabelEncode
from torch.utils.data import Dataset


class SimpleDataSet(Dataset):
    def __init__(
            self,
            media_dir,
            ann_file,
            processor,
            delimiter="\t",
            max_text_length=25,
            character_dict_path=None,
            use_space_char=False,
    ):
        super(SimpleDataSet, self).__init__()
        self.media_dir = media_dir
        self.ann_file = ann_file
        self.processor = processor
        self.delimiter = delimiter
        self.read_image = ReadImage(img_mode="RGB")
        self.read_label = CTCLabelEncode(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char
        )
        self.inner_dataset = self.process_data(ann_file)

    def process_data(self, ann_file):
        with open(ann_file, "rb") as f:
            lines = f.readlines()
        lines = [_.decode("utf-8").strip().split(self.delimiter) for _ in lines]
        result = []
        for line in lines:
            if len(line) != 2:
                continue
            file_name, label = line[0].strip(), line[1].strip()
            image_path = osp.join(self.media_dir, file_name)
            item = {"image_path": image_path, "label": label}
            result.append(item)
        return result

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        img_path, text = ann["image_path"], ann["label"]
        if not osp.isfile(img_path):
            logging.warning("{} does not exist!".format(img_path))
            return self[(index + 1) % len(self.inner_dataset)]
        try:
            image = self.read_image(img_path)
            image = self.processor(image)
            label, label_length = self.read_label(text)
        except Exception as e:
            logging.warning("{} corrupt!".format(img_path))
            return self[(index + 1) % len(self.inner_dataset)]

        res = {
            "id": str(index),
            "gt": text,
            "image": torch.from_numpy(image),
            "image_path": img_path,
            "label": torch.LongTensor(label),
            "label_length": torch.LongTensor([label_length])
        }
        return res

    def collater(self, features):
        ids = [_["id"] for _ in features]
        gts = [_["gt"] for _ in features]
        images = [_["image"] for _ in features]
        labels = [_["label"] for _ in features]
        label_lengths = [_["label_length"] for _ in features]
        image_paths = [_["image_path"] for _ in features]
        result = {
            "id": ids,
            "gt": gts,
            "image": torch.stack(images),
            "image_path": image_paths,
            "label": torch.stack(labels),
            "label_length": torch.cat(label_lengths)
        }
        return result
