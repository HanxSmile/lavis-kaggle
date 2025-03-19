import torch
import torch.nn as nn
from ftfy import fix_text
from transformers import DonutSwinConfig, VisionEncoderDecoderConfig
from transformers import AutoModel, VisionEncoderDecoderModel, AutoImageProcessor
from .model import VariableDonutSwinConfig, VariableDonutSwinModel
from .processor import VariableDonutProcessor, VariableDonutImageProcessor


class DonutEncoderDecoder(nn.Module):

    def __init__(self, model_name, num_tokens, pad_token_id, bos_token_id, eos_token_id):
        super().__init__()
        config = VisionEncoderDecoderConfig.from_pretrained(model_name)
        encoder_config = vars(config.encoder)
        encoder = DonutSwinConfig(**encoder_config)
        config.encoder = encoder
        self.config = config

        AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name, config=self.config)

        self.model.config.decoder_start_token_id = bos_token_id
        self.model.config.pad_token_id = pad_token_id
        self.model.config.eos_token_id = eos_token_id
        self.model.decoder.resize_token_embeddings(num_tokens)
        self.pad_token_id = pad_token_id

    def forward(self, pixel_values, decoder_input_ids, decoder_attention_mask, **kwargs):
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        labels = decoder_input_ids * 1
        labels = labels.masked_fill(labels == self.pad_token_id, -100)

        loss = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids[:, :-1],
            decoder_attention_mask=decoder_attention_mask[:, :-1],
            labels=labels[:, 1:]
        ).loss
        return loss

    @torch.no_grad()
    def generate(self, pixel_values, temperature, max_new_tokens, decoder_start_token_id, do_sample, top_p,
                 **kwargs):

        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        outputs = self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p
        )
        return outputs[:, 1:]


class DonutTokenizer:
    def __init__(self, path):
        AutoImageProcessor.register(VariableDonutSwinConfig, VariableDonutImageProcessor)
        processor = VariableDonutProcessor.from_pretrained(path)
        processor.train = False
        self.tokenizer = processor.tokenizer
        self.max_seq_len = 784
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, texts):
        text_inputs = self.tokenizer(
            texts,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len,
        )
        return text_inputs

    @staticmethod
    def post_process(text):
        text = fix_text(text)
        return text

    def token2str(self, tokens) -> list:
        generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        generated_text = [self.post_process(text) for text in generated_text]
        return generated_text

    def detokenize(self, tokens):
        toks = [self.tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in ([self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]):
                    del toks[b][i]
        return toks
