from transformers import AutomaticSpeechRecognitionPipeline


class MoEAutomaticSpeechRecognitionPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(
            self,
            feature_extractor,
            *,
            decoder=None,
            moe_list=None,
            **kwargs,
    ):
        super().__init__(feature_extractor=feature_extractor, decoder=decoder, **kwargs)
        self.moe_list = moe_list

    @staticmethod
    def rescale_stride(stride, ratio):
        """
        Rescales the stride values from audio space to tokens/logits space.

        (160_000, 16_000, 16_000) -> (2000, 200, 200) for instance.
        """
        # Shape is [B, SEQ] for tokens
        # [B, SEQ, V] for logits

        new_strides = []
        for input_n, left, right in stride:
            token_n = int(round(input_n * ratio))
            left = int(round(left / input_n * token_n))
            right = int(round(right / input_n * token_n))
            new_stride = (token_n, left, right)
            new_strides.append(new_stride)

        return new_strides

    def _forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}
        if return_timestamps and self.type == "seq2seq_whisper":
            generate_kwargs["return_timestamps"] = return_timestamps
        is_last = model_inputs.pop("is_last")

        stride = model_inputs.pop("stride", None)
        input_values = model_inputs.pop("input_values")
        attention_mask = model_inputs.pop("attention_mask", None)

        logits = self.moe_list.extract_logits(input_values, attention_mask)

        if self.type == "ctc_with_lm":
            out = {"logits": logits}
        else:
            out = {"tokens": logits.argmax(dim=-1)}
        if stride is not None:
            # Send stride to `postprocess`.
            # it needs to be handled there where
            # the pieces are to be concatenated.
            ratio = 1 / self.model.config.inputs_to_logits_ratio
            if isinstance(stride, tuple):
                out["stride"] = self.rescale_stride([stride], ratio)[0]
            else:
                out["stride"] = self.rescale_stride(stride, ratio)
        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}
