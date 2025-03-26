"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base, disabled_train


@registry.register_model("qwen2_grpo")
class Qwen2GRPO(Blip2Base):
    """
    Qwen2 GRPO model.
    Supported model types:
        - default
    Usage:
        >>> from vigc.models import load_model
        >>> model = load_model("qwen2_grpo", "default")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/qwen/qwen2_grpo.yaml",
    }

    def __init__(
            self,
            model_name="Qwen/Qwen2.5-3B-Instruct",
            use_grad_checkpoint=False,
            compute_type="bf16",
            max_txt_len=128,
            max_output_txt_len=256,
            # compute loss
            epsilon=0.2,
            beta=0.04,
            # generate cfg
            generate_cfg=None,
            num_generations=8,
    ):
        super().__init__()
        assert compute_type in ["bf16", "f16"]
        self.compute_type = torch.float16 if compute_type == "f16" else torch.bfloat16
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_name, use_fast=False, truncation_side="left")
        self.model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=self.compute_type)
        if use_grad_checkpoint:
            self.model.gradient_checkpointing_enable()
        self.ref_model = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=self.compute_type)
        for name, param in self.ref_model.named_parameters():
            param.requires_grad = False
        self.ref_model = self.ref_model.eval()
        self.ref_model.train = disabled_train
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.epsilon = epsilon
        self.beta = beta
        self.generate_cfg = generate_cfg
        self.num_generations = num_generations

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

    def forward(self, *, prepare_inputs_flag=False, samples=None, reward_funcs=None, llm_inputs=None, advantages=None,
                rewards=None, old_per_token_logps=None, ref_per_token_logps=None):
        if not prepare_inputs_flag:
            loss = self.compute_loss(llm_inputs, advantages, old_per_token_logps, ref_per_token_logps)
            return {"loss": loss, "rewards": rewards.mean()}
        else:
            return self.prepare_grpo_inputs(samples, reward_funcs)

    def _prepare_llm_inputs(self, samples, responses):
        num_generations = len(responses[0])
        results = list()
        for idx in range(num_generations):
            text_outputs = [_[idx] for _ in responses]
            text_inputs, system_inputs = samples["text_input"], samples["text_output"]
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

            results.append((llm_tokens, targets))
        return results

    def _get_per_token_logps(self, model, model_inputs):
        if not isinstance(model_inputs, list):
            model_inputs = [model_inputs]
        results = []
        for llm_tokens, sft_targets in model_inputs:
            with self.maybe_autocast(self.compute_type):
                logits = model(
                    input_ids=llm_tokens['input_ids'],
                    attention_mask=llm_tokens['attention_mask'],
                    return_dict=True,
                ).logits / self.generate_cfg.temperature

            logits = logits[:, :-1, :]
            sft_targets = sft_targets[:, 1:, :]
            labels = sft_targets.clone()
            labels[labels == -100] = 0  # dummy token
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            results.append((per_token_logps, sft_targets))
        if len(results) == 1:
            return results[0]
        return results  # [G, B]

    def prepare_grpo_inputs(
            self,
            samples,
            reward_funcs,
    ):
        responses = self.generate_group_responses(
            samples, self.num_generations, self.generate_cfg.max_new_tokens, self.generate_cfg.top_k,
            self.generate_cfg.top_p, self.generate_cfg.repetition_penalty, self.generate_cfg.temperature
        )  # [B, G]
        advantages, rewards = self.compute_advantages(samples, responses, reward_funcs)
        advantages = list(advantages.permute(1, 0))  # [G, B]
        rewards = list(rewards.permute(1, 0))
        llm_inputs = self._prepare_llm_inputs(samples, responses)  # [G, B]
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(self.model, llm_inputs)  # [G, B]
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, llm_inputs)  # [G, B]

        grpo_inputs = {
            "llm_inputs": llm_inputs,
            "advantages": advantages,
            "rewards": rewards,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
        return grpo_inputs

    def compute_loss(self, llm_inputs, advantages, old_per_token_logps, ref_per_token_logps):
        """
        Only compute one batch
        llm_inputs: [input_ids, attention_mask]
            - shape: [B, L]
        advantages: [input_ids, attention_mask]
            - shape: [B]
        old_per_token_logps:
            - shape: [B, L]
        ref_per_token_logps:
            - shape: [B, L]
        """

        per_token_logps, labels = self._get_per_token_logps(self.model, llm_inputs)
        old_per_token_logps = old_per_token_logps[0]
        ref_per_token_logps = ref_per_token_logps[0]
        per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        completion_mask = torch.ones_like(labels)
        completion_mask[labels == -100] = 0
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss

    def compute_advantages(self, samples, responses, reward_funcs):
        all_rewards = list()
        for sample, response_lst in zip(samples, responses):
            model_input = sample["text_input"]
            model_gt = sample["text_output"]
            this_reward_lst = [0 for _ in response_lst]
            for reward_func in reward_funcs:
                reward_lst = [reward_func(model_input, model_gt, _) for _ in response_lst]
                this_reward_lst = [a + b for a, b in zip(this_reward_lst, reward_lst)]
            all_rewards.append(this_reward_lst)
        all_rewards = torch.FloatTensor(all_rewards).to(self.device)  # [B, G]
        mean_grouped_rewards = all_rewards.mean(dim=1, keepdim=True)
        std_grouped_rewards = all_rewards.std(dim=1, keepdim=True)
        advantages = (all_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        return advantages, all_rewards  # [B, G]

    @torch.no_grad()
    def generate_group_responses(
            self,
            samples,
            num_generations=8,
            max_new_tokens=256,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.05,
            temperature=0.7,
    ):
        system_inputs, text_inputs = samples["system_input"], samples["text_input"]
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
        all_responses = []
        for i in range(num_generations):
            with self.maybe_autocast(self.compute_type):
                generated_ids = self.model.generate(
                    input_ids=text_input_tokens.input_ids,
                    attention_mask=text_input_tokens.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(text_input_tokens.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_responses.append(response)
        all_responses = list(zip(*all_responses))  # [B, G]
        return all_responses

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
        system_inputs, text_inputs = samples["system_input"], samples["text_input"]

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
        # compute loss
        epsilon = cfg.get("epsilon", 0.2)
        beta = cfg.get("beta", 0.04)
        # generate cfg
        generate_cfg = cfg.generate_cfg
        num_generations = cfg.get("num_generations", 8)

        model = cls(
            model_name=model_name,
            use_grad_checkpoint=use_grad_checkpoint,
            compute_type=compute_type,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            epsilon=epsilon,
            beta=beta,
            generate_cfg=generate_cfg,
            num_generations=num_generations,
        )

        model.load_checkpoint_from_config(cfg)

        return model
