from torch.utils.data import Dataset as torch_Dataset
import datasets
import torch
from typing import Dict, List, Union
from datasets import concatenate_datasets
from datasets.features import Audio
import numpy as np
import os
from transformers.feature_extraction_utils import BatchFeature
from .romanize import uromanize
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler


class VitsTTSTrain(torch_Dataset):
    def __init__(
            self,
            data_root,
            processor,
            tokenizer,
            audio_column_name,
            text_column_name,
            max_duration_in_seconds=20,
            min_duration_in_seconds=1.0,
            max_tokens_length=500,
            do_lower_case=False,
            speaker_id_column_name=None,
            filter_on_speaker_id=None,
            do_normalize=False,
            num_workers=4,
            uroman_path=None,
            split="train",
            language=None
    ):
        if isinstance(split, str):
            split = [split]
        sampling_rate = processor.sampling_rate
        inner_dataset = datasets.load_from_disk(data_root)
        inner_dataset = concatenate_datasets([inner_dataset[_] for _ in split])
        inner_dataset = inner_dataset.cast_column(audio_column_name, Audio(sampling_rate=sampling_rate))
        max_input_length = max_duration_in_seconds * sampling_rate
        min_input_length = min_duration_in_seconds * sampling_rate

        def _is_audio_in_length_range(length, text):
            length_ = len(length["array"])
            return length_ > min_input_length and length_ < max_input_length and text is not None

        inner_dataset = inner_dataset.filter(
            _is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=[audio_column_name, text_column_name]
        )

        self.speaker_id_dict = dict()
        self.new_num_speakers = 0
        if speaker_id_column_name is not None:
            if filter_on_speaker_id is not None:
                inner_dataset = inner_dataset.filter(
                    lambda speaker_id: (speaker_id == filter_on_speaker_id),
                    num_proc=num_workers,
                    input_columns=[speaker_id_column_name],
                )
            self.speaker_id_dict = {
                speaker_id: i for (i, speaker_id) in enumerate(sorted(list(set(inner_dataset[speaker_id_column_name]))))
            }
            self.new_num_speakers = len(self.speaker_id_dict)

        self.processor = processor
        self.tokenizer = tokenizer
        self.language = language
        self.max_tokens_length = max_tokens_length
        self.audio_column_name = audio_column_name
        self.text_column_name = text_column_name
        self.speaker_id_column_name = speaker_id_column_name
        self.inner_dataset = inner_dataset
        self.do_normalize = do_normalize
        self.do_lower_case = do_lower_case
        self.is_uroman = tokenizer.is_uroman
        self.uroman_path = uroman_path
        if self.is_uroman:
            assert uroman_path is not None
            assert os.path.exists(uroman_path)

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        sample = self.inner_dataset[index]
        audio = sample[self.audio_column_name]
        text = sample[self.text_column_name]
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
        string_inputs = self.tokenizer(input_str, return_attention_mask=False)
        if len(string_inputs) > self.max_tokens_length:
            return self[(index + 1) % len(self.inner_dataset)]
        input_ids = string_inputs[: self.max_tokens_length]
        waveform_input_length = len(audio["array"])
        tokens_input_length = len(input_ids)
        waveform = audio["array"]
        mel_scaled_input_features = audio_inputs.get("mel_scaled_input_features")[0]
        speaker_id = 0
        if self.new_num_speakers > 0:
            speaker_id = self.speaker_id_dict.get(sample[self.speaker_id_column_name], 0)

        return {
            "labels": labels,
            "input_ids": input_ids,
            "tokens_input_length": tokens_input_length,
            "waveform": waveform,
            "mel_scaled_input_features": mel_scaled_input_features,
            "speaker_id": speaker_id,
            "waveform_input_length": waveform_input_length,
            "id": index
        }

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

        return batch
    #
    # def get_sampler(self, batch_size):
    #     return DistributedLengthGroupedSampler(
    #         batch_size=batch_size,
    #         dataset=self,
    #         lengths=self.lengths
    #     )
