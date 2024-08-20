from torch.utils.data import Dataset as torch_Dataset
import torch
from typing import Dict, List, Union
import numpy as np
import os
import os.path as osp
from transformers.feature_extraction_utils import BatchFeature
from .romanize import uromanize
import librosa


class CustumVitsTTSTrain(torch_Dataset):
    def __init__(
            self,
            data_root,
            processor,
            tokenizer,
            max_duration_in_seconds=20,
            min_duration_in_seconds=1.0,
            max_tokens_length=500,
            do_lower_case=False,
            filter_on_speaker_id=None,
            do_normalize=False,
            uroman_path=None,
            language=None
    ):
        self.sampling_rate = processor.sampling_rate
        self.max_input_length = max_duration_in_seconds * self.sampling_rate
        self.min_input_length = min_duration_in_seconds * self.sampling_rate

        inner_dataset = self.prepare_samples(data_root)

        self.speaker_id_dict = dict()
        self.new_num_speakers = 0

        if filter_on_speaker_id is not None:
            inner_dataset = [_ for _ in inner_dataset if _["speaker_id"] == filter_on_speaker_id]
        all_speaker_ids = sorted(list(set([_["speaker_id"] for _ in inner_dataset])))
        self.speaker_id_dict = {
            speaker_id: i for (i, speaker_id) in enumerate(all_speaker_ids)
        }
        self.new_num_speakers = len(self.speaker_id_dict)

        self.processor = processor
        self.tokenizer = tokenizer
        self.language = language
        self.max_tokens_length = max_tokens_length
        self.inner_dataset = inner_dataset
        self.do_normalize = do_normalize
        self.do_lower_case = do_lower_case
        self.is_uroman = tokenizer.is_uroman
        self.uroman_path = uroman_path
        if self.is_uroman:
            assert uroman_path is not None
            assert os.path.exists(uroman_path)

    def prepare_samples(self, data_root):
        label_root = osp.join(data_root, "transcript.txt")
        with open(label_root) as f:
            data = f.readlines()
        result = []
        for line in data:
            line = line.strip()
            if not line:
                continue
            try:
                audio_name, speaker_id, text = line.split("|")
            except Exception as e:
                continue
            text = text.strip()
            if not text:
                continue
            audio_path = osp.join(data_root, audio_name)
            item = {
                "audio_path": audio_path,
                "speaker_id": speaker_id,
                "text": text
            }
            result.append(item)
        return result

    def __len__(self):
        return len(self.inner_dataset)

    def _parse_ann_info(self, index):
        sample = self.inner_dataset[index]
        audio_path, speaker_id, text = sample["audio_path"], sample["speaker_id"], sample["text"]

        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=self.sampling_rate), self.sampling_rate
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }

        return audio, text, speaker_id

    def is_valid(self, input_values):
        input_length = len(input_values)
        return self.min_input_length < input_length < self.max_input_length

    def __getitem__(self, index):
        audio, text, raw_speaker_id = self._parse_ann_info(index)

        if not self.is_valid(audio["array"]):
            return self[(index + 1) % len(self)]  # filter too long or too short audio

        audio_inputs = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
            do_normalize=self.do_normalize
        )

        labels = audio_inputs.get("input_features")[0]
        input_str = text.lower() if self.do_lower_case else text
        if self.is_uroman:
            input_str = uromanize(input_str, self.uroman_path)
        string_inputs = self.tokenizer(input_str, return_attention_mask=False).input_ids
        if len(string_inputs) > self.max_tokens_length:
            return self[(index + 1) % len(self.inner_dataset)]
        input_ids = string_inputs[: self.max_tokens_length]
        waveform_input_length = len(audio["array"])
        tokens_input_length = len(input_ids)
        waveform = audio["array"]
        mel_scaled_input_features = audio_inputs.get("mel_scaled_input_features")[0]
        speaker_id = None
        if self.new_num_speakers > 1:
            speaker_id = self.speaker_id_dict[raw_speaker_id]
        res = {
            "labels": labels,
            "input_ids": input_ids,
            "tokens_input_length": tokens_input_length,
            "waveform": waveform,
            "mel_scaled_input_features": mel_scaled_input_features,
            "waveform_input_length": waveform_input_length,
            "id": index
        }
        if speaker_id is not None:
            res["speaker_id"] = speaker_id

        return res

    def pad_waveform(self, raw_speech):
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
                isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # convert into correct format for padding

        padded_inputs = self.processor.pad(
            batched_speech,
            padding=True,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_features"]

        return padded_inputs

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = "input_ids"
        input_ids = [{model_input_name: feature[model_input_name]} for feature in features]

        # pad input tokens
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", return_attention_mask=True)

        # pad waveform
        waveforms = [np.array(feature["waveform"]) for feature in features]
        batch["waveform"] = self.pad_waveform(waveforms)

        # pad spectrogram
        label_features = [np.array(feature["labels"]) for feature in features]
        labels_batch = self.processor.pad(
            {"input_features": [i.T for i in label_features]}, return_tensors="pt", return_attention_mask=True
        )

        labels = labels_batch["input_features"].transpose(1, 2)
        batch["labels"] = labels
        batch["labels_attention_mask"] = labels_batch["attention_mask"]

        # pad mel spectrogram
        mel_scaled_input_features = {
            "input_features": [np.array(feature["mel_scaled_input_features"]).squeeze().T for feature in features]
        }
        mel_scaled_input_features = self.processor.pad(
            mel_scaled_input_features, return_tensors="pt", return_attention_mask=True
        )["input_features"].transpose(1, 2)

        batch["mel_scaled_input_features"] = mel_scaled_input_features
        batch["speaker_id"] = (
            torch.tensor([feature["speaker_id"] for feature in features]) if "speaker_id" in features[0] else None
        )
        batch["ids"] = [_["id"] for _ in features]

        result = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            waveform=batch["waveform"],
            labels=batch["labels"],
            labels_attention_mask=batch["labels_attention_mask"],
            mel_scaled_input_features=batch["mel_scaled_input_features"],
            speaker_id=batch["speaker_id"],
            ids=batch["ids"],
        )

        return result
