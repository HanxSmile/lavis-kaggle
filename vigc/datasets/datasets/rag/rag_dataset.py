from torch.utils.data import Dataset as torch_Dataset
import json
import random
import math


class RAGDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            query_prompt=None,
            passage_prompt=None,
            group_size=4,
            processor=None
    ):
        with open(data_root, 'r') as f:
            self.inner_dataset = json.load(f)

        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt

        self.group_size = group_size
        self.processor = processor

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        query, pos, negs = ann['query'], ann['pos'], ann['neg']
        if isinstance(pos, str):
            pos = [pos]
        if isinstance(negs, str):
            negs = [negs]
        pos = random.choice(pos)
        if len(negs) < self.group_size - 1:
            num = math.ceil((self.group_size - 1) / len(negs))
            negs = random.sample(negs * num, self.group_size - 1)
        else:
            negs = random.sample(negs, self.group_size - 1)

        if self.processor is not None:
            query = self.processor(query)
            pos = self.processor(pos)
            negs = [self.processor(_) for _ in negs]

        if self.query_prompt is not None:
            query = self.query_prompt.format(query=query)
        if self.passage_prompt is not None:
            pos = self.passage_prompt.format(passage=pos)
            negs = [self.passage_prompt.format(passage=_) for _ in negs]

        return {
            "id": str(index),
            "query": query,
            "pos": pos,
            "negs": negs,
        }

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        queries = [_["query"] for _ in batch]
        pos_messages = [_["pos"] for _ in batch]
        neg_messages = [_["negs"] for _ in batch]

        return {
            "id": ids,
            "query": queries,
            "pos_message": pos_messages,
            "neg_message": neg_messages,
        }


class RAGEvalDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            query_prompt=None,
            passage_prompt=None,
            processor=None
    ):
        with open(data_root, 'r') as f:
            data = json.load(f)

        self.all_passages = data["passages"]
        self.all_queries = data["queries"]

        self.inner_dataset = self.all_passages + self.all_queries

        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt

        self.processor = processor

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        if index >= len(self.all_passages):  # query
            index = index - len(self.all_passages)
            ann = self.all_queries[index]
            pos = ann["pos"]
            query = ann["query"]
            if self.processor is not None:
                query = self.processor(query)
            if self.query_prompt is not None:
                query = self.query_prompt.format(query=query)
            text_type = "query"
            result = {
                "id": str(index),
                "text": query,
                "pos": pos,
                "text_type": text_type,
            }
        else:  # passage
            index = index
            passage = self.all_passages[index]
            pos = None
            if self.processor is not None:
                passage = self.processor(passage)
            if self.passage_prompt is not None:
                passage = self.passage_prompt.format(passage=passage)
            text_type = "passage"
            result = {
                "id": str(index),
                "text": passage,
                "pos": pos,
                "text_type": text_type,
            }
        return result

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        texts = [_["text"] for _ in batch]
        pos = [_["pos"] for _ in batch]
        text_types = [_["text_type"] for _ in batch]

        return {
            "id": ids,
            "text": texts,
            "pos_messages": pos,
            "text_type": text_types,
        }
