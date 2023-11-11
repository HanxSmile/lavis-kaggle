from .bengali_wav2vec_asr import Wav2VecBengaliASR, MIN_SECS, MAX_SECS, TARGET_SR
import random
import numpy as np


class Wav2VecConcatAugASR(Wav2VecBengaliASR):
    def __init__(self, processor, data_root, split: str, transform=None, split_style="default", fold_idx=None,
                 fold_nums=None, seed=None, sample_nums=None, seg_nums=2):
        self.seg_nums = seg_nums
        super().__init__(processor, data_root, split, transform, split_style, fold_idx, fold_nums, seed, sample_nums)

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

    def is_valid(self, input_values):
        if self.split != "train":
            return True
        input_length = len(input_values)
        input_secs = input_length / TARGET_SR
        return self.seg_nums * MAX_SECS > input_secs > self.seg_nums * MIN_SECS
