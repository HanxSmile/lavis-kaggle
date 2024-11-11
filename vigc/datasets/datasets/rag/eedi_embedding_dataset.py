from torch.utils.data import Dataset as torch_Dataset
import json
import random


class EediDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            misconception_root,
            fold_idx,
            group_size=4,
    ):
        with open(data_root, 'r') as f:
            inner_dataset = json.load(f)
        with open(misconception_root, 'r') as f:
            self.misconception_mapping = json.load(f)
        self.misconception_mapping_ids = list(self.misconception_mapping.keys())
        self.inner_dataset = [_ for _ in inner_dataset if _["Fold"] != fold_idx]

        self.group_size = group_size

        self.task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'

    def __len__(self):
        return len(self.inner_dataset)

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def __getitem__(self, index):
        ann = self.inner_dataset[index]

        query = f"###question###:{ann['SubjectName']}-{ann['ConstructName']}-{ann['QuestionText']}\n###Correct Answer###:{ann['CorrectAnswer']}\n###Misconcepte Incorrect answer###:{ann['IncorrectAnswer']}"
        pos = ann['MisconceptionText']
        pos_id = ann["MisconceptionId"]
        neg_ids = random.sample(self.misconception_mapping_ids, self.group_size)
        if pos_id in neg_ids:
            neg_ids.remove(pos_id)
        else:
            neg_ids.pop(0)
        negs = [self.misconception_mapping[_] for _ in neg_ids]
        query = self.get_detailed_instruct(self.task_description, query)
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


class EediEvalDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            misconception_root,
            fold_idx,
    ):
        if isinstance(fold_idx, int):
            fold_idx = [fold_idx]
        else:
            fold_idx = list(fold_idx)
        with open(data_root, 'r') as f:
            eval_data = json.load(f)
        eval_data = [_ for _ in eval_data if _["Fold"] in fold_idx]
        with open(misconception_root, 'r') as f:
            self.misconception_mapping = json.load(f)

        passages = list(self.misconception_mapping.items())
        self.passages_id = {passages[_][0]: _ for _, item in enumerate(passages)}

        self.all_passages = [_[1] for _ in passages]
        self.all_queries = eval_data
        self.inner_dataset = self.all_passages + self.all_queries
        self.task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'

    def __len__(self):
        return len(self.inner_dataset)

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def __getitem__(self, index):
        if index >= len(self.all_passages):  # query
            index = index - len(self.all_passages)
            ann = self.all_queries[index]
            query = f"###question###:{ann['SubjectName']}-{ann['ConstructName']}-{ann['QuestionText']}\n###Correct Answer###:{ann['CorrectAnswer']}\n###Misconcepte Incorrect answer###:{ann['IncorrectAnswer']}"
            pos_id = ann["MisconceptionId"]
            pos = self.passages_id[pos_id]
            pos = [pos]

            query = self.get_detailed_instruct(self.task_description, query)
            original_text = query

            text_type = "query"
            result = {
                "id": str(index),
                "text": query,
                "original_text": original_text,
                "pos": pos,
                "text_type": text_type,
            }
        else:  # passage
            index = index
            passage = self.all_passages[index]
            original_text = passage
            pos = None
            text_type = "passage"
            result = {
                "id": str(index),
                "text": passage,
                "original_text": original_text,
                "pos": pos,
                "text_type": text_type,
            }
        return result

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        texts = [_["text"] for _ in batch]
        pos = [_["pos"] for _ in batch]
        text_types = [_["text_type"] for _ in batch]
        original_texts = [_["original_text"] for _ in batch]

        return {
            "id": ids,
            "text": texts,
            "original_text": original_texts,
            "pos_messages": pos,
            "text_type": text_types,
        }
