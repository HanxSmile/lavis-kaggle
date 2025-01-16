import json
from torch.utils.data import Dataset as torch_Dataset
import datasets
import torch
from typing import Dict, List, Union
from datasets import Audio
import numpy as np
from .normalize import normalize

MIN_SECS = 0.1
MAX_SECS = 30
TARGET_SR = 16_000


class FleursClipDataset(torch_Dataset):
    def __init__(
            self,
            data_root,
            ann_path,
            clip_duration,
            processor,
            transform=None,
            pre_normalize=False,
            max_label_length=448,
            split="train",
            language=None,
            using_space_sep=True
    ):
        if isinstance(split, str):
            split = [split]
        hf_dataset = datasets.load_from_disk(data_root)
        hf_dataset = {_: hf_dataset[_].cast_column("audio", Audio(sampling_rate=TARGET_SR)) for _ in split}
        self.using_space_sep = using_space_sep
        self.split = split
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.transform = transform
        self.max_label_length = max_label_length
        self.pre_normalize = pre_normalize
        self.language = language
        self.clip_duration = clip_duration
        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)
        self.inner_dataset = self.prepare_clips(annotations, clip_duration)

    def prepare_clips(self, annotations, clip_duration):
        all_results = []
        for ann in annotations:
            split = ann["split"]
            dataset_index = ann["dataset_index"]
            sep = " " if self.using_space_sep else ""
            if split not in self.split:
                continue
            segments = ann["segments"]
            all_words = []
            for segment in segments:
                all_words += segment["words"]
            this_sample = []
            for word in all_words:
                word_start, word_end, word_text = word["start"], word["end"], word["word"]
                if len(this_sample) == 0:
                    this_sample.append(word)
                    continue
                segment_start = this_sample[0]["start"]
                this_duration = word_end - segment_start
                this_sample.append(word)
                if this_duration <= clip_duration:
                    continue
                segment_start = this_sample[0]["start"]
                segment_end = this_sample[-1]["end"]
                text = sep.join([_["word"] for _ in this_sample])
                this_item = {
                    "start": segment_start,
                    "end": segment_end,
                    "text": text,
                    "duration": this_duration,
                    "split": split,
                    "dataset_index": dataset_index,
                    "sep": sep,
                }
                all_results.append(this_item)
                this_sample = []
        return all_results

    def __len__(self):
        return len(self.inner_dataset)

    def segment_audio(self, audio, start, end):
        return audio[int(start * TARGET_SR):int(end * TARGET_SR)]

    def _parse_ann_info(self, index):
        ann = self.inner_dataset[index]
        split, dataset_index = ann["split"], ann["dataset_index"]
        sample = self.hf_dataset[split][dataset_index]
        audio = sample["audio"]["array"].astype(np.float32)
        text = ann["text"]
        segment_start, segment_end = ann["start"], ann["end"]
        segment_audio = self.segment_audio(audio, segment_start, segment_end)
        sentence = normalize(text, self.language) if self.pre_normalize else text
        audio = {
            "array": segment_audio,
            "sampling_rate": TARGET_SR,
        }
        return audio, sentence, ann, str(index)

    def is_valid(self, input_values):
        input_length = len(input_values)
        input_secs = input_length / TARGET_SR
        return MAX_SECS > input_secs > MIN_SECS

    def transform_array(self, audio):
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        if self.transform is not None:
            audio["array"] = self.transform(audio["array"], sample_rate=audio["sampling_rate"])
        return audio

    def __getitem__(self, index):
        audio, sentence, raw_ann, ann_id = self._parse_ann_info(index)
        audio = self.transform_array(audio)

        if not self.is_valid(audio["array"]):
            return self[(index + 1) % len(self)]  # filter too long or too short audio

        labels = self.processor.tokenizer(sentence, truncation=True, max_length=self.max_label_length).input_ids

        return {"audio": audio, "labels": labels, "sentence": sentence, "id": ann_id, "raw_info": raw_ann}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        audio_arrays = [_["audio"]["array"] for _ in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]
        batch = self.processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True
        )
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        result = {}
        result["input_features"] = batch["input_features"]
        result["attention_mask"] = batch["attention_mask"]
        result["labels"] = labels
        result["sentences"] = [_["sentence"] for _ in features]
        result["ids"] = [_["id"] for _ in features]
        result["raw_info"] = [_["raw_info"] for _ in features]
        result["raw_audios"] = [_["audio"] for _ in features]
        return result
