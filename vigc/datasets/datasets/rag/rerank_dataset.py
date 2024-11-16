from torch.utils.data import Dataset as torch_Dataset
import json
import random
import math


class RerankDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            query_prompt=None,
            passage_prompt=None,
            group_size=4,
            processor=None,
            prompt=None,
    ):
        with open(data_root, 'r') as f:
            self.inner_dataset = json.load(f)

        self.query_prompt = query_prompt or "A: {query}"
        self.passage_prompt = passage_prompt or "B: {passage}"
        self.prompt = prompt or "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

        self.group_size = group_size
        self.processor = processor

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        query, pos, negs = ann['query'], ann['pos'], ann['neg']
        prompt = ann.get('prompt', self.prompt)
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
            "prompt": prompt,
        }

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        queries = [_["query"] for _ in batch]
        pos_messages = [_["pos"] for _ in batch]
        neg_messages = [_["negs"] for _ in batch]
        prompts = [_["prompt"] for _ in batch]

        return {
            "id": ids,
            "query": queries,
            "pos_message": pos_messages,
            "neg_message": neg_messages,
            "prompt": prompts,
        }


class RerankEvalDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            query_prompt=None,
            passage_prompt=None,
            processor=None,
            prompt=None,
    ):
        with open(data_root, 'r') as f:
            all_data = json.load(f)
        self.inner_dataset = self.prepare_ds(all_data)

        self.query_prompt = query_prompt or "A: {query}"
        self.passage_prompt = passage_prompt or "B: {passage}"
        self.prompt = prompt or "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

        self.processor = processor

    def prepare_ds(self, data):
        all_res = []
        for ann in data:
            query, pos, negs = ann['query'], ann['pos'], ann['neg']
            prompt = ann.get('prompt', self.prompt)
            if isinstance(pos, str):
                pos = [pos]
            if isinstance(negs, str):
                negs = [negs]
            for pos_ in pos:
                item = {
                    "query": query,
                    "passage": pos_,
                    "prompt": prompt,
                    "label": True
                }
                all_res.append(item)
            for neg_ in negs:
                item = {
                    "query": query,
                    "passage": neg_,
                    "prompt": prompt,
                    "label": False
                }
                all_res.append(item)
        return all_res

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        query, passage, prompt, label = ann['query'], ann['passage'], ann['prompt'], ann["label"]

        if self.processor is not None:
            query = self.processor(query)
            passage = self.processor(passage)

        if self.query_prompt is not None:
            query = self.query_prompt.format(query=query)
        if self.passage_prompt is not None:
            passage = self.passage_prompt.format(passage=passage)

        return {
            "id": str(index),
            "query": query,
            "passage": passage,
            "prompt": prompt,
            "label": label
        }

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        queries = [_["query"] for _ in batch]
        passages = [_["passage"] for _ in batch]
        prompts = [_["prompt"] for _ in batch]
        labels = [_["label"] for _ in batch]

        return {
            "id": ids,
            "query": queries,
            "passage": passages,
            "prompt": prompts,
            "label": labels
        }
