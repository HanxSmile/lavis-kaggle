import json
from PIL import Image, ImageFile
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True

from io import BytesIO
from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):

    def __init__(self, vis_processor, text_processor, vis_root, anno_path):

        self.vis_root = vis_root
        if isinstance(anno_path, tuple) or isinstance(anno_path, list):
            anno_path = anno_path[0]
        self.anno_path = anno_path

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.samples = self.init_samples()
        self.reader = self.init_reader()

        print('total {} {} samples'.format(self.__len__(), self.__class__.__name__))

        for idx in range(10):
            self.__getitem__(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplementedError

    def init_samples(self):
        # read annotation from ceph
        if self.anno_path.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            samples = json.loads(client.get(self.anno_path))
        else:
            samples = json.load(open(self.anno_path, 'r'))
        return samples

    def init_reader(self):
        if self.vis_root.startswith('cluster'):
            from petrel_client.client import Client
            client = Client("~/petreloss.conf")
            reader = {'type': 'PetrelReader', 'body': client.get}
        else:
            reader = {'type': 'LocalReader', 'body': Image.open}
        return reader

    def _read_image(self, sample, image_key="image"):
        img_file = sample[image_key]
        image_path = osp.join(self.vis_root, img_file)
        image = self.reader['body'](image_path)
        if isinstance(image, bytes):
            bytes_stream = BytesIO(image)
            image = Image.open(bytes_stream)
        image = image.convert("RGB")
        return image

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            "data_type": "vqa",
        }
