from .bengali_wav2vec_asr import Wav2VecBase, MIN_SECS, MAX_SECS
import pandas as pd
import numpy as np
import os
import random


class Wav2VecFilteredDataset(Wav2VecBase):
    def __init__(self, anno_path, data_root, ratio, processor, transform=None):
        self.media_root = data_root
        data = pd.read_csv(anno_path)
        data = data.loc[(data.input_sec > MIN_SECS) & (data.input_sec < MAX_SECS)]
        data = data.reset_index(drop=True)
        data["audio"] = self.media_root + os.sep + data["id"] + ".mp3"
        inner_dataset = data.head(int(len(data) * ratio))
        super().__init__(inner_dataset, processor, transform)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = ann.audio
        audio = self.read_and_resample_audio(audio_path, return_dict=True)
        return audio, ann.sentence, ann.id


class Wav2VecFilteredSegAugDataset(Wav2VecFilteredDataset):
    def __init__(self, anno_path, data_root, ratio, processor, transform=None, seg_nums: int = 3):
        self.seg_nums = seg_nums
        super().__init__(anno_path, data_root, ratio, processor, transform)

    def transform_array(self, audio):
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        array = audio["array"]
        array_lst = np.array_split(array, self.seg_nums)
        if self.transform is not None:
            for i, array_ in enumerate(array_lst):
                array_lst[i] = self.transform(array_, sample_rate=audio["sampling_rate"])
            array = np.concatenate(array_lst, axis=0)
            audio["array"] = array
        return audio


class Wav2VecFilteredConcatAugDataset(Wav2VecFilteredDataset):
    def __init__(self, anno_path, data_root, ratio, processor, transform=None, seg_nums=2):
        self.seg_nums = seg_nums
        super().__init__(anno_path, data_root, ratio, processor, transform)

    def _sample_ann_array(self):
        other_index = random.choice(range(len(self)))
        other_ann = self.inner_dataset.iloc[other_index]
        other_audio_path = other_ann.audio
        other_array, _ = self.read_and_resample_audio(other_audio_path)
        return other_array, other_ann.sentence

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]

        audio_path = ann.audio
        array, sr = self.read_and_resample_audio(audio_path)

        array_lst = [array]
        sentence_lst = [ann.sentence]
        for i in range(self.seg_nums - 1):
            other_array, other_sentence = self._sample_ann_array()
            array_lst.append(other_array)
            sentence_lst.append(other_sentence)

        audio = {
            "path": audio_path,
            "array": array_lst,
            "sampling_rate": sr,
        }
        return audio, " ".join(sentence_lst), ann.id

    def transform_array(self, audio):
        array_lst = audio["array"]
        array_lst[0] = np.trim_zeros(array_lst[0], "f")
        array_lst[-1] = np.trim_zeros(array_lst[-1], "b")

        sampling_rate = audio["sampling_rate"]
        if self.transform is not None:
            for i, array in enumerate(array_lst):
                array_lst[i] = self.transform(array, sample_rate=sampling_rate)

        return {"array": np.concatenate(array_lst, axis=0), "path": audio["path"], "sampling_rate": sampling_rate}
