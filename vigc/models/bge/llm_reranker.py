import logging
import torch
import torch.nn as nn
from torch import Tensor
from vigc.common.registry import registry
from vigc.models.base_model import BaseModel
import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
from peft import LoraConfig, get_peft_model


@registry.register_model("llm_reranker")
class LLMRerankerModel(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/bge/llm_reranker.yaml",
    }

    def __init__(
            self,
            model_name: str = "upstage/SOLAR-10.7B-v1.0",
            torch_dtype: Optional[str] = None,
            use_lora: bool = False,
            use_qlora: bool = False,
            use_grad_checkpoint: bool = False,
            query_max_len: int = 256,
            passage_max_len: int = 256,
    ):
        super().__init__()
        if torch_dtype == "f16":
            self.compute_type = torch.float16
        elif torch_dtype == "bf16":
            self.compute_type = torch.bfloat16
        else:
            self.compute_type = torch.float32

        if use_lora:

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # quantization_config=bnb_config,
                torch_dtype=self.compute_type,
                trust_remote_code=True
            )

            for name, param in self.model.named_parameters():
                param.requires_grad = False

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
        elif use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_type,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                trust_remote_code=True
            )

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

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # quantization_config=bnb_config,
                torch_dtype=self.compute_type,
                trust_remote_code=True
            )

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

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.3)

        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

        if use_grad_checkpoint:
            self.model.gradient_checkpointing_enable()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

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

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

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

    def forward(self, samples):
        query, pos_message, neg_message, prompt = \
            samples['query'], samples['pos_message'], samples['neg_message'], samples['prompt']
        assert len(query) == len(pos_message) == len(neg_message) == len(prompt)
        assert isinstance(pos_message[0], str) and isinstance(neg_message[0], list) \
               and isinstance(query[0], str) and isinstance(prompt[0], str)
        group_size = len(samples["neg_message"][0]) + 1
        batch_size = len(query)
        all_message = []
        all_queries = []
        all_prompts = []

        for query_, prompt_, pos_m, neg_m_lst in zip(query, prompt, pos_message, neg_message):
            assert len(neg_m_lst) == group_size - 1
            all_message.append(pos_m)
            all_message.extend(neg_m_lst)
            all_queries.extend([query_] * group_size)
            all_prompts.extend([prompt_] * group_size)

        llm_tokens = self.prepare_inputs(all_queries, all_message, all_prompts)

        with self.maybe_autocast(self.compute_type):
            logits = self.encode(llm_tokens).view(batch_size, group_size)
            target = torch.zeros(batch_size, device=self.device, dtype=torch.long)
            loss = self.compute_loss(logits, target)

        return {"loss": loss}

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", False)

        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert finetune_path is not None, "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
            logging.info(f"Loaded finetuned model '{finetune_path}'.")

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        query, passage, prompt = samples['query'], samples['passage'], samples['prompt']
        llm_tokens = self.prepare_inputs(query, passage, prompt)

        with self.maybe_autocast(self.compute_type):
            logits = self.encode(llm_tokens)  # [B, D]
        scores = torch.sigmoid(logits.float())
        return scores.detach().cpu()

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get("model_name", "upstage/SOLAR-10.7B-v1.0")
        torch_dtype = cfg.get("torch_dtype", "bf16")
        use_lora = cfg.get("use_lora", False)
        use_qlora = cfg.get("use_qlora", False)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        query_max_len = cfg.get("query_max_len", 256)
        passage_max_len = cfg.get("passage_max_len", 256)

        model = cls(
            model_name=model_name,
            use_grad_checkpoint=use_grad_checkpoint,
            query_max_len=query_max_len,
            passage_max_len=passage_max_len,
            use_lora=use_lora,
            use_qlora=use_qlora,
            torch_dtype=torch_dtype,
        )
        model.load_checkpoint_from_config(cfg)
        return model
