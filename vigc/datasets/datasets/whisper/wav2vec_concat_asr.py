from .bengali_wav2vec_asr import Wav2VecBengaliASR, MIN_SECS, MAX_SECS, TARGET_SR
import librosa
import random
import numpy as np


class Wav2VecConcatAugASR(Wav2VecBengaliASR):
    def __init__(self, processor, data_root, split: str, transform=None, split_style="default", fold_idx=None,
                 fold_nums=None, seed=None, sample_nums=None):
        super().__init__(processor, data_root, split, transform, split_style, fold_idx, fold_nums, seed, sample_nums)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]

        audio_path = ann.audio
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR), TARGET_SR

        other_index = random.choice(range(len(self)))
        other_ann = self.inner_dataset.iloc[other_index]
        other_audio_path = other_ann.audio
        other_array, other_sr = librosa.load(other_audio_path, sr=None)
        other_array = librosa.resample(other_array, orig_sr=other_sr, target_sr=TARGET_SR)

        audio = {
            "path": audio_path,
            "array": array,
            "other_array": other_array,
            "sampling_rate": sr,
        }
        return audio, " ".join([ann.sentence, other_ann.sentence]), ann.id

    def transform_array(self, audio):
        array = np.trim_zeros(audio["array"], "f")
        other_array = np.trim_zeros(audio["other_array"], "b")
        sampling_rate = audio["sampling_rate"]
        if self.transform is not None:
            array = self.transform(array, sample_rate=sampling_rate)
            other_array = self.transform(other_array, sample_rate=sampling_rate)

        array = np.concatenate([array, other_array], axis=0)
        return {"array": array, "path": audio["path"], "sampling_rate": sampling_rate}

    def is_valid(self, input_values):
        if self.split != "train":
            return True
        input_length = len(input_values)
        input_secs = input_length / TARGET_SR
        return 2 * MAX_SECS > input_secs > 2 * MIN_SECS
