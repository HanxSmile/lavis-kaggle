from itertools import groupby
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch.nn as nn
import sys


class Wav2Vec2Timestamp(nn.Module):
    SAMPLE_RATE = 16000

    def __init__(self, lan=None, model_dir=None, logfile=sys.stderr):
        super().__init__()

        self.logfile = logfile
        self.transcribe_kargs = {}
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(self.device)
        self.sample_rate = self.processor.feature_extractor.sampling_rate

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def transcribe(self, audio, init_prompt=""):
        if isinstance(audio, str):
            array, sr = librosa.load(audio, sr=None)
            audio, sample_rate = librosa.resample(array, orig_sr=sr, target_sr=self.sample_rate), self.sample_rate
        else:
            sample_rate = self.sample_rate
        features = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)

        logits = self.model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        print(transcription)

        words = [w for w in transcription.split(' ') if len(w) > 0]
        predicted_ids = pred_ids[0].tolist()
        duration_sec = input_values.shape[1] / self.sample_rate

        ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
        # remove entries which are just "padding" (i.e. no characers are recognized)
        ids_w_time = [i for i in ids_w_time if i[1] != self.processor.tokenizer.pad_token_id]
        # now split the ids into groups of ids where each group represents a word
        split_ids_w_time = [list(group) for k, group
                            in groupby(ids_w_time, lambda x: x[1] == self.processor.tokenizer.word_delimiter_token_id)
                            if not k]

        assert len(split_ids_w_time) == len(
            words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong

        word_start_times = []
        word_end_times = []
        for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
            _times = [_time for _time, _id in cur_ids_w_time]
            word_start_times.append(min(_times))
            word_end_times.append(max(_times))

        segments = [dict(start=s, end=e, word=w) for w, s, e in zip(words, word_start_times, word_end_times)]
        return segments

    def ts_words(self, segments):
        return [(_["start"], _["end"], _["word"]) for _ in segments]

    def segments_end_ts(self, res):
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


if __name__ == "__main__":
    model = Wav2Vec2Timestamp(model_dir="/mnt/data/hanxiao/models/audio/experiment/wav2vec2-large-xlsr-persian-v3")

    audio_path = "/mnt/data/hanxiao/models/audio/hx-asr/audio.mp3"

    print(model.transcribe(audio_path))
