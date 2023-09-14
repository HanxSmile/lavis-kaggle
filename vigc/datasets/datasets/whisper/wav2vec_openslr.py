from .bengali_wav2vec_asr import Wav2VecBengaliOpenSLR, MAX_SECS, MIN_SECS, TARGET_SR
import numpy as np
import random
import os.path as osp


class OpenSLRSegAugDataset(Wav2VecBengaliOpenSLR):
    def __init__(self, ann_file, data_root, processor, transform=None, seg_nums: int = 3):
        self.seg_nums = seg_nums
        super().__init__(ann_file, data_root, processor, transform)

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


class OpenSLRConcatAugDataset(Wav2VecBengaliOpenSLR):
    def __init__(self, ann_file, data_root, processor, transform=None, seg_nums: int = 2):
        self.seg_nums = seg_nums
        super().__init__(ann_file, data_root, processor, transform)

    def _sample_ann_array(self):
        other_index = random.choice(range(len(self)))
        other_ann = self.inner_dataset.iloc[other_index]
        other_ann_id = other_ann.id
        other_audio_path = osp.join(self.media_root, other_ann_id[:2], other_ann_id + ".flac")
        other_array, _ = self.read_and_resample_audio(other_audio_path)
        return other_array, other_ann.sentence

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        ann_id = ann.id
        audio_path = osp.join(self.media_root, ann_id[:2], ann_id + ".flac")
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
        input_length = len(input_values)
        input_secs = input_length / TARGET_SR
        return self.seg_nums * MAX_SECS > input_secs > self.seg_nums * MIN_SECS


class OpenSLRConcatSegAugDataset(OpenSLRConcatAugDataset):
    def __init__(self, ann_file, data_root, processor, transform=None, concat_seg_nums=2, split_seg_nums=3):
        self.concat_seg_nums = concat_seg_nums
        self.split_seg_nums = split_seg_nums
        super().__init__(ann_file, data_root, processor, transform, concat_seg_nums)

    def transform_array(self, audio):
        array_lst = audio["array"]
        array_lst[0] = np.trim_zeros(array_lst[0], "f")
        array_lst[-1] = np.trim_zeros(array_lst[-1], "b")

        sampling_rate = audio["sampling_rate"]

        for i, array in enumerate(array_lst):
            array_lst[i] = self._transform(array, sample_rate=sampling_rate)

        return {"array": np.concatenate(array_lst, axis=0), "path": audio["path"], "sampling_rate": sampling_rate}

    def _transform(self, array, sample_rate):
        array_lst = np.array_split(array, self.split_seg_nums)
        if self.transform is not None:
            for i, array_ in enumerate(array_lst):
                array_lst[i] = self.transform(array_, sample_rate=sample_rate)
            array = np.concatenate(array_lst, axis=0)
        return array
