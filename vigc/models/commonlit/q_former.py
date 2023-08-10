import logging
import torch
from vigc.common.registry import registry
from vigc.models.blip2_models.blip2 import Blip2Base
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn


@registry.register_model("commonlit_q_former")
class CommonLitQFormer(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "",
    }

    def __init__(
            self,
            encoder_model_name,
            max_input_txt_len=256,
            max_output_txt_len=128,
    ):
        super().__init__()
        self.encoder_config = AutoConfig.from_pretrained(encoder_model_name)
        self.encoder_config.hidden_dropout = 0.
        self.encoder_config.hidden_dropout_prob = 0.
        self.encoder_config.attention_dropout = 0.
        self.encoder_config.attention_probs_dropout_prob = 0.
        logging.info(self.encoder_config)

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.decoder_tokenizer = self.init_tokenizer()

        self.encoder = AutoModel.from_pretrained(encoder_model_name, config=self.encoder_config)
        self.decoder, self.query_tokens = self.init_Qformer(2, self.encoder_config.hidden_size)
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.decoder.cls = None

        self.max_input_txt_len = max_input_txt_len
        self.max_output_txt_len = max_output_txt_len

        self.content_head = nn.Linear(self.decoder.config.hidden_size, 1)
        self.wording_head = nn.Linear(self.decoder.config.hidden_size, 1)

        self.criterion = nn.SmoothL1Loss(reduction='mean')

    def forward(self, samples):
        text_input = samples["text_input"]
        text_output = samples["text_output"]
        labels = samples["label"]  # [batch, 2]

        text_input = self.encoder_tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=self.max_input_txt_len,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.query_tokens.device)

        encoder_output = self.encoder(input_ids=text_input.input_ids,
                                      attention_mask=text_input.attention_mask).last_hidden_state

        query_tokens = self.query_tokens.expand(len(text_input), -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(query_tokens.device)

        text_output = self.decoder_tokenizer(
            text_output,
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
            return_length="pt",
            add_special_tokens=False
        ).to(self.query_tokens.device)

        Qformer_atts = torch.cat([query_atts, text_output.attention_mask], dim=1)
        query_output = self.decoder.bert(
            text_output.input_ids,
            attention_mask=Qformer_atts,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=text_input.attention_mask,
            return_dict=True
        ).last_hidden_state

        logits = query_output[:, :2, :]
        content_out = self.wording_head(logits[:, 0])
        wording_out = self.content_head(logits[:, 1])

        output = torch.cat([content_out, wording_out], dim=-1)  # [b, 2]
        loss = self.criterion(output, labels)

        return {"loss": loss}

    @torch.no_grad()
    def generate(self):
        pass

    @classmethod
    def from_config(cls, cfg):
        pass
