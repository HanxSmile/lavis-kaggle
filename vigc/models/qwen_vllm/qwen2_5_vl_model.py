import torch
from transformers import Qwen2Tokenizer
from vigc.models.qwen_vllm.qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'visual']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


@registry.register_model("qwen2_5_vl_instruct")
class Qwen2_5VLInstruct(Blip2Base):
    """
    Qwen2.5 VL Instruct model.
    Supported model types:
        - default
    Usage:
        >>> from vigc.models import load_model
        >>> model = load_model("qwen2_5_vl_instruct", "default")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/vllm/qwen2_5_vl_instruct.yaml",
    }

    def __init__(
            self,
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            use_grad_checkpoint=False,
            compute_type="bf16",
            max_txt_len=128,
            max_output_txt_len=256,
            q_lora_flag=False,
            load_in_8bit=True,
            lora_flag=False,
    ):
        if q_lora_flag:
            lora_flag = True
        super().__init__()
        assert compute_type in ["bf16", "f16"]
        self.compute_type = torch.float16 if compute_type == "f16" else torch.bfloat16
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name, use_fast=False, truncation_side="left")

        if q_lora_flag:
            load_in_nbit_cfg = {"load_in_8bit": True} if load_in_8bit else {"load_in_4bit": True}
            bnb_model_from_pretrained_args = dict()
            bnb_model_from_pretrained_args.update(dict(
                # device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    llm_int4_skip_modules=["llm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_skip_modules=["lm_head"],
                    bnb_4bit_compute_dtype=self.compute_type,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",  # {'fp4', 'nf4'}
                    **load_in_nbit_cfg
                )
            ))
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                **bnb_model_from_pretrained_args)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=self.compute_type)
        if lora_flag or q_lora_flag:
            for name, param in model.named_parameters():
                param.requires_grad = False
        if lora_flag:
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=8,
                bias="none",
                target_modules=find_all_linear_names(model),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model

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
        image_inputs = samples["image_input"]
        system_inputs = samples["system_input"]
        text_inputs = samples["text_input"]
        text_outputs = samples["text_output"]
        model_input_templates = [
            [
                {'role': "system", "content": sys_in},
                {
                    'role': 'user',
                    'content': [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": txt_in}
                    ]
                }
            ] if sys_in else [
                {
                    'role': 'user',
                    'content': [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": txt_in}
                    ]
                }
            ] for
            sys_in, txt_in in zip(system_inputs, text_inputs)]
        text_inputs = [
            self.processor.apply_chat_template(
                _,
                tokenize=False,
                add_generation_prompt=True) for _ in model_input_templates
        ]
        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.truncation_side = "left"
        text_input_tokens = self.processor(
            text=text_inputs,
            images=image_inputs,
            padding="longest",
            truncation=True,
            # max_length=self.max_txt_len,
            # padding_side="right",
            # truncation_side="left",
            return_tensors="pt",
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
                pixel_values=text_input_tokens["pixel_values"],
                image_grid_thw=text_input_tokens["image_grid_thw"],
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
        image_inputs = samples["image_input"]
        system_inputs = samples["system_input"]
        text_inputs = samples["text_input"]
        model_input_templates = [
            [
                {'role': "system", "content": sys_in},
                {
                    'role': 'user',
                    'content': [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": txt_in}
                    ]
                }
            ] if sys_in else [
                {
                    'role': 'user',
                    'content': [
                        {"type": "image", "image": ""},
                        {"type": "text", "text": txt_in}
                    ]
                }
            ] for
            sys_in, txt_in in zip(system_inputs, text_inputs)]
        text_inputs = [
            self.processor.apply_chat_template(
                _,
                tokenize=False,
                add_generation_prompt=True) for _ in model_input_templates
        ]

        self.processor.tokenizer.padding_side = "right"
        self.processor.tokenizer.truncation_side = "left"
        text_input_tokens = self.processor(
            text=text_inputs,
            images=image_inputs,
            padding="longest",
            truncation=True,
            # max_length=self.max_txt_len,
            # padding_side="right",
            # truncation_side="left",
            return_tensors="pt",
        ).to(self.device)

        with self.maybe_autocast(self.compute_type):
            generated_ids = self.model.generate(
                **text_input_tokens,
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

        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        return response

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        compute_type = cfg.get("compute_type", "bf16")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)
        q_lora_flag = cfg.get("q_lora_flag", False)
        load_in_8bit = cfg.get("load_in_8bit", True)
        lora_flag = cfg.get("lora_flag", False)

        model = cls(
            model_name=model_name,
            use_grad_checkpoint=use_grad_checkpoint,
            compute_type=compute_type,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            q_lora_flag=q_lora_flag,
            lora_flag=lora_flag,
            load_in_8bit=load_in_8bit,
        )

        model.load_checkpoint_from_config(cfg)

        return model
