from transformers import AutomaticSpeechRecognitionPipeline


class WhisperPipeline(AutomaticSpeechRecognitionPipeline):

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        encoder = self.model.get_encoder()
        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # we need to pass `processed.get("attention_mask")` here since audio encoder
        # attention mask  length is different from expected text decoder `encoder_attention_mask` length
        # `generate` magic to create the mask automatically won't work, we basically need to help
        # it here.
        attention_mask = model_inputs.pop("attention_mask", None)
        tokens = self.model.generate(
            encoder_outputs=encoder(inputs, attention_mask=attention_mask),
            attention_mask=attention_mask,
            temperature=0,
        )

        out = {"tokens": tokens}

        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}
