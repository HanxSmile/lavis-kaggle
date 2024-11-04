"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base


@registry.register_model("qwen2_instruct")
class Qwen2Instruct(Blip2Base):
    """
    Qwen2 Instruct model.
    Supported model types:
        - default
    Usage:
        >>> from vigc.models import load_model
        >>> model = load_model("qwen2_instruct", "default")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/qwen/qwen2_instruct.yaml",
    }

    def __init__(
            self,
            model_name="Qwen/Qwen2.5-3B-Instruct",
            use_grad_checkpoint=False,
            compute_type="bf16",
            max_txt_len=128,
            max_output_txt_len=256,
    ):
        super().__init__()
        assert compute_type in ["bf16", "f16"]
        self.compute_type = torch.float16 if compute_type == "f16" else torch.bfloat16
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name, use_fast=False, truncation_side="left")
        self.model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=self.compute_type)
        if use_grad_checkpoint:
            self.model.gradient_checkpointing_enable()
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len

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

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        system_inputs = samples["system_input"]
        text_inputs = samples["text_input"]
        text_outputs = samples["text_output"]
        model_input_templates = [
            [{'role': "system", "content": sys_in}, {'role': 'user', 'content': txt_in}] if sys_in else [
                {'role': 'user', 'content': txt_in}] for
            sys_in, txt_in in zip(system_inputs, text_inputs)]
        text_inputs = [
            self.tokenizer.apply_chat_template(
                _,
                tokenize=False,
                add_generation_prompt=True) for _ in model_input_templates
        ]
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        text_input_tokens = self.tokenizer(
            text_inputs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        self.tokenizer.truncation_side = "right"
        text_output_tokens = self.tokenizer(
            [_ + self.tokenizer.eos_token for _ in text_outputs],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        with self.maybe_autocast(self.compute_type):
            outputs = self.model(
                input_ids=llm_tokens['input_ids'],
                attention_mask=llm_tokens['attention_mask'],
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            do_sample=True,
            num_beams=5,
            max_new_tokens=256,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            temperature=0.7,
    ):
        system_inputs = samples["system_input"]
        text_inputs = samples["text_input"]
        model_input_templates = [
            [{'role': "system", "content": sys_in}, {'role': 'user', 'content': txt_in}] if sys_in else [
                {'role': 'user', 'content': txt_in}] for
            sys_in, txt_in in zip(system_inputs, text_inputs)]
        text_inputs = [
            self.tokenizer.apply_chat_template(
                _,
                tokenize=False,
                add_generation_prompt=True) for _ in model_input_templates
        ]

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        text_input_tokens = self.tokenizer(
            text_inputs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(self.device)

        with self.maybe_autocast(self.compute_type):
            generated_ids = self.model.generate(
                input_ids=text_input_tokens.input_ids,
                attention_mask=text_input_tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(text_input_tokens.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return response

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        compute_type = cfg.get("compute_type", "bf16")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        model = cls(
            model_name=model_name,
            use_grad_checkpoint=use_grad_checkpoint,
            compute_type=compute_type,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
        )

        model.load_checkpoint_from_config(cfg)

        return model
