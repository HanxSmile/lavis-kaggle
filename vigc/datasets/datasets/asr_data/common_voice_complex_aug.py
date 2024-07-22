from .common_voice import CommonVoiceTrain
import numpy as np
import random
from .normalize import normalize


class CommonVoiceSplit(CommonVoiceTrain):

    def __init__(self, data_root, processor, transform=None, pre_normalize=False, max_label_length=448, split="train",
                 split_nums: int = 2, language=None):
        self.split_nums = split_nums
        super().__init__(data_root, processor, transform, pre_normalize, max_label_length, split=split,
                         language=language)

    def transform_array(self, audio):
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        array = audio["array"]
        array_lst = np.array_split(array, self.split_nums)
        if self.transform is not None:
            for i, array_ in enumerate(array_lst):
                array_lst[i] = self.transform(array_, sample_rate=audio["sampling_rate"])
            array = np.concatenate(array_lst, axis=0)
            audio["array"] = array
        return audio


class CommonVoiceConcat(CommonVoiceTrain):
    def __init__(self, data_root, processor, transform=None, pre_normalize=False, max_label_length=448, split="train",
                 concat_nums: int = 2, language=None):
        self.concat_nums = concat_nums
        super().__init__(data_root, processor, transform, pre_normalize, max_label_length, split=split,
                         language=language)

    def _sample_ann_array(self):
        other_index = random.choice(range(len(self)))
        other_ann = self.inner_dataset[other_index]
        sentence = normalize(other_ann["sentence"], self.language) if self.pre_normalize else other_ann["sentence"]
        return other_ann["audio"], sentence, str(other_index)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset[index]
        audio, sentence = ann["audio"], ann["sentence"]
        sentence = normalize(ann["sentence"], self.language) if self.pre_normalize else ann["sentence"]

        audio_lst = [audio]
        sentence_lst = [sentence]
        for i in range(self.concat_nums - 1):
            other_audio, other_sentence, _ = self._sample_ann_array()
            audio_lst.append(other_audio)
            sentence_lst.append(other_sentence)

        return audio_lst, " ".join(sentence_lst), str(index)

    def transform_array(self, audio):
        array_lst = [_["array"] for _ in audio]
        array_lst[0] = np.trim_zeros(array_lst[0], "f")
        array_lst[-1] = np.trim_zeros(array_lst[-1], "b")

        sampling_rate = audio[0]["sampling_rate"]
        if self.transform is not None:
            for i, array in enumerate(array_lst):
                array_lst[i] = self.transform(array, sample_rate=audio[i]["sampling_rate"])

        return {"array": np.concatenate(array_lst, axis=0), "path": audio[0]["path"], "sampling_rate": sampling_rate}


class CommonVoiceSplitAndConcat(CommonVoiceConcat):
    def __init__(
            self,
            data_root,
            processor,
            transform=None,
            pre_normalize=False,
            max_label_length=448,
            split="train",
            split_nums: int = 2,
            concat_nums: int = 2,
            language=None
    ):
        self.concat_nums = concat_nums
        self.split_nums = split_nums
        super().__init__(data_root, processor, transform, pre_normalize, max_label_length, split, concat_nums,
                         language=language)

    def transform_single_array(self, audio):
        # audio["array"] = np.trim_zeros(audio["array"], "fb")
        array = audio["array"]
        array_lst = np.array_split(array, self.split_nums)
        if self.transform is not None:
            for i, array_ in enumerate(array_lst):
                array_lst[i] = self.transform(array_, sample_rate=audio["sampling_rate"])
            array = np.concatenate(array_lst, axis=0)
            audio["array"] = array
        return audio

    def transform_array(self, audio):
        audio = [self.transform_single_array(_) for _ in audio]
        array_lst = [_["array"] for _ in audio]
        array_lst[0] = np.trim_zeros(array_lst[0], "f")
        array_lst[-1] = np.trim_zeros(array_lst[-1], "b")

        sampling_rate = audio[0]["sampling_rate"]

        return {"array": np.concatenate(array_lst, axis=0), "path": audio[0]["path"], "sampling_rate": sampling_rate}
