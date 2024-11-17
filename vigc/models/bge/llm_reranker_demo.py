import torch
import argparse
import json
import pandas as pd
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset as torch_Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from peft import LoraConfig, get_peft_model


class RerankEvalDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            query_prompt=None,
            passage_prompt=None,
            prompt=None,
    ):

        self.query_prompt = query_prompt or "A: {query}"
        self.passage_prompt = passage_prompt or "B: {passage}"
        self.prompt = prompt or "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

        with open(data_root, 'r') as f:
            all_data = json.load(f)
        self.inner_dataset = self.prepare_ds(all_data)

    def prepare_ds(self, data):
        all_res = []
        for uid, ann in data.items():
            query, passages, passage_ids = ann['query'], ann['passages'], ann['passage_ids']

            for passage, passage_id in zip(passages, passage_ids):
                item = {
                    "query": query,
                    "passage": passage,
                    "passage_id": passage_id,
                    "uid": uid,
                }
                all_res.append(item)
        return all_res

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        query, passage, passage_id, uid = ann['query'], ann['passage'], ann["passage_id"], ann["uid"]

        if self.query_prompt is not None:
            query = self.query_prompt.format(query=query)
        if self.passage_prompt is not None:
            passage = self.passage_prompt.format(passage=passage)

        return {
            "id": str(index),
            "uid": uid,
            "passage_id": passage_id,
            "query": query,
            "passage": passage,
            "prompt": self.prompt,
        }

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        uids = [_["uid"] for _ in batch]
        passage_ids = [_["passage_id"] for _ in batch]
        queries = [_["query"] for _ in batch]
        passages = [_["passage"] for _ in batch]
        prompts = [_["prompt"] for _ in batch]

        return {
            "id": ids,
            "uid": uids,
            "passage_ids": passage_ids,
            "query": queries,
            "passage": passages,
            "prompt": prompts,
        }


class LLMRerankerModel(nn.Module):
    def __init__(
            self,
            model_name: str = "upstage/SOLAR-10.7B-v1.0",
            torch_dtype: Optional[str] = "bf16",
            use_lora: bool = True,
            query_max_len: int = 512,
            passage_max_len: int = 512,
    ):
        super().__init__()
        if torch_dtype == "f16":
            self.compute_type = torch.float16
        elif torch_dtype == "bf16":
            self.compute_type = torch.bfloat16
        else:
            self.compute_type = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.compute_type,
            trust_remote_code=True
        )
        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head"
                ],
                bias="none",
                lora_dropout=0.05,  # Conventional
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        self.tokenizer = tokenizer

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    @property
    def device(self):
        return list(self.parameters())[0].device

    def encode(self, features):
        outputs = self.model(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
            output_hidden_states=True
        )

        logits = outputs.logits
        scores = self.last_logit_pool(logits, features['attention_mask'])
        scores = scores[:, self.yes_loc]
        return scores.contiguous()

    @staticmethod
    def last_logit_pool(
            logits: Tensor,
            attention_mask: Tensor
    ) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return logits[:, -1, :]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = logits.shape[0]
            return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def prepare_inputs(self, query, passage, prompt):

        self.tokenizer.truncation_side = "right"
        query = self.tokenizer(
            query,
            padding="longest",
            truncation=True,
            max_length=self.query_max_len,
            return_tensors='pt',
            add_special_tokens=False
        ).to(self.device)

        passage = self.tokenizer(
            ["\n" + _ for _ in passage],
            padding="longest",
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors='pt',
            add_special_tokens=False
        ).to(self.device)

        self.tokenizer.truncation_side = "left"
        prompt = self.tokenizer(
            ["\n" + _ for _ in prompt],
            padding="longest",
            truncation=True,
            max_length=self.query_max_len,
            return_tensors='pt',
            add_special_tokens=False
        ).to(self.device)

        llm_tokens, _ = self.concat_text_input_output(
            query.input_ids,
            query.attention_mask,
            passage.input_ids,
            passage.attention_mask,
        )

        llm_tokens, _ = self.concat_text_input_output(
            llm_tokens['input_ids'],
            llm_tokens['attention_mask'],
            prompt.input_ids,
            prompt.attention_mask,
        )

        return llm_tokens

    @torch.no_grad()
    def generate(
            self,
            samples,
    ):
        query, passage, prompt = samples['query'], samples['passage'], samples['prompt']
        llm_tokens = self.prepare_inputs(query, passage, prompt)

        with torch.cuda.amp.autocast(dtype=self.compute_type):
            logits = self.encode(llm_tokens)  # [B, D]
        scores = torch.sigmoid(logits.float())
        return scores.detach().cpu()

    @torch.no_grad()
    def inference(
            self,
            dataloader,
    ):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--dst-path", required=True, help="path to save to the final ranking results.")
    parser.add_argument("--recall-path", required=True, help="path to the recall results.")
    args = parser.parse_args()

    return args


cfg = parse_args()
lora_path = cfg.lora_path
dst_path = cfg.dst_path
recall_path = cfg.recall_path

path_prefix = "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
model_path = "/kaggle/input/sfr-embedding-mistral/SFR-Embedding-2_R"
with open("/kaggle/input/data-process/misconception_mapping.json") as f:
    misconception_dic = json.load(f)

with open(recall_path) as f:
    recall_info = json.load(f)
model = LLMRerankerModel(
    model_name="upstage/SOLAR-10.7B-v1.0",
    torch_dtype="bf16",
    use_lora=True,
    query_max_len=512,
    passage_max_len=512,
)

ckpt = torch.load(lora_path)
model.load_state_dict(ckpt, strict=False)
model = model.eval()

tra = pd.read_csv(f"{path_prefix}/test.csv")
print(tra.shape)

train_data = {}
for _, row in tra.iterrows():
    for c in ['A', 'B', 'C', 'D']:
        if c == row['CorrectAnswer']:
            continue
        if f'Answer{c}Text' not in row:
            continue
        real_answer_id = row['CorrectAnswer']
        real_text = row[f'Answer{real_answer_id}Text']
        query_text = f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{row[f'Answer{c}Text']}"
        row['query_text'] = query_text
        row['answer_name'] = c
        uid = f"{row['QuestionId']}_{row['answer_name']}"
        train_data[uid] = {
            "query": query_text,
            "passage_ids": recall_info[uid],
            "passages": [misconception_dic[_] for _ in recall_info[uid]],
        }
