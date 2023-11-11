import re
from bnunicodenormalizer import Normalizer
from torch.utils.data import Dataset

bnorm = Normalizer()
chars_to_ignore_regex = re.compile('[\,\?\.\!\-\;\:\"\—\‘\'\‚\“\”\…]')


def remove_special_characters(sentence):
    sentence = re.sub(chars_to_ignore_regex, '', sentence) + " "
    return sentence


def normalize(sentence):
    _words = [bnorm(word)['normalized'] for word in sentence.split()]
    sentence = " ".join([word for word in _words if word is not None])
    return sentence


class IndicCorp(Dataset):
    def __init__(self, anno_path):
        with open(anno_path) as f:
            raw_sentences = f.readlines()
        raw_sentences = [_.strip() for _ in raw_sentences if _.strip()]
        raw_sentences = raw_sentences[::10]
        self.inner_dataset = raw_sentences

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        s = self.inner_dataset[index]
        return normalize(remove_special_characters(s))

    def collater(self, batch):
        return batch
