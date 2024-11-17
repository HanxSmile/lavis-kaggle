import argparse
from tqdm.auto import tqdm
import pandas as pd

import json
import copy
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModel
from peft import LoraConfig, get_peft_model
import numpy as np
from tqdm.autonotebook import trange


class MistralEmbeddingModel(nn.Module):

    def __init__(
            self,
            model_name: str = "Salesforce/SFR-Embedding-2_R",
            normalized: bool = False,
            sentence_pooling_method: str = 'last',
            query_max_len: int = 32,
            passage_max_len: int = 128,
    ):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="FEATURE_EXTRACTION",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method

        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def last_token_pool(
            self, last_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'last':
            return self.last_token_pool(hidden_state, mask)

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normalized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    @property
    def device(self):
        return list(self.parameters())[0].device

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        texts = samples['text']
        text_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max([self.query_max_len, self.passage_max_len]),
            return_tensors='pt',
        ).to(self.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            embeddings = self.encode(text_input)  # [B, D]
        return embeddings.float()

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    @torch.no_grad()
    def inference(self, df):
        batch_size = 16
        sentences = list(df['query_text'].values)
        pids = list(df['order_index'].values)
        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            embeddings = self.generate({"text": sentences_batch})
            embeddings = embeddings.detach().cpu().numpy().tolist()
            all_embeddings.extend(embeddings)

        all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

        sentence_embeddings = np.concatenate(all_embeddings, axis=0)
        result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--dst-path", required=True, help="path to save to the ranking results.")
    args = parser.parse_args()

    return args


cfg = parse_args()
lora_path = cfg.lora_path
dst_path = cfg.dst_path

path_prefix = "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
model_path = "/kaggle/input/sfr-embedding-mistral/SFR-Embedding-2_R"

model = MistralEmbeddingModel(
    model_name=model_path,
    normalized=True,
    sentence_pooling_method='last',
    query_max_len=512,
    passage_max_len=256,
)

ckpt = torch.load(lora_path)
model.load_state_dict(ckpt, strict=False)
model = model.eval()

task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'

tra = pd.read_csv(f"{path_prefix}/test.csv")
print(tra.shape)
misconception_mapping = pd.read_csv(f"{path_prefix}/misconception_mapping.csv")
if tra.shape[0] < 10:
    misconception_mapping = misconception_mapping.sample(n=5, random_state=2023)

train_data = []
for _, row in tra.iterrows():
    for c in ['A', 'B', 'C', 'D']:
        if c == row['CorrectAnswer']:
            continue
        if f'Answer{c}Text' not in row:
            continue
        real_answer_id = row['CorrectAnswer']
        real_text = row[f'Answer{real_answer_id}Text']
        query_text = f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{row[f'Answer{c}Text']}"
        row['query_text'] = model.get_detailed_instruct(task_description, query_text)
        row['answer_name'] = c
        train_data.append(copy.deepcopy(row))
train_df = pd.DataFrame(train_data)
train_df['order_index'] = list(range(len(train_df)))

train_embeddings = model.inference(train_df)

misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']
doc_embeddings = model.inference(misconception_mapping)

sentence_embeddings = np.concatenate([e.reshape(1, -1) for e in list(doc_embeddings.values())])
index_text_embeddings_index = {index: paper_id for index, paper_id in
                               enumerate(list(doc_embeddings.keys()))}

predicts_test = dict()
for _, row in tqdm(train_df.iterrows()):
    query_id = row['order_index']
    query_em = train_embeddings[query_id].reshape(1, -1)

    cosine_similarity = np.dot(query_em, sentence_embeddings.T).flatten()
    uid = f"{row['QuestionId']}_{row['answer_name']}"
    this_score_info = {}
    for i, score in enumerate(cosine_similarity):
        pid = index_text_embeddings_index[i]
        score = float(score)
        this_item = {str(pid): score}
        this_score_info.update(this_item)
    predicts_test[uid] = this_score_info

with open(dst_path, "w") as f:
    json.dump(predicts_test, f, ensure_ascii=False)
print("Done!")
