from .bengali_wav2vec_asr import Wav2VecBengaliASR
import numpy as np


class Wav2VecSegAugASR(Wav2VecBengaliASR):
    def __init__(self, processor, data_root, split: str, transform=None, split_style="default", fold_idx=None,
                 fold_nums=None, seed=None, sample_nums=None, seg_nums: int = 3):
        self.seg_nums = seg_nums
        super().__init__(processor, data_root, split, transform, split_style, fold_idx, fold_nums, seed, sample_nums)

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
