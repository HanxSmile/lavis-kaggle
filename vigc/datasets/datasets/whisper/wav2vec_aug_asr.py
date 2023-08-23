from .bengali_wav2vec_asr import Wav2VecBengaliASR, trim_silence, normalize, remove_special_characters
import librosa
import numpy as np


class Wav2VecSegAugASR(Wav2VecBengaliASR):
    def __init__(self, processor, data_root, split: str, transform=None, split_style="default", fold_idx=None,
                 fold_nums=None, seed=None, sample_nums=None, seg_nums: int = 3):
        super().__init__(processor, data_root, split, transform, split_style, fold_idx, fold_nums, seed, sample_nums)
        self.seg_nums = seg_nums

    def __getitem__(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = ann.audio
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        array = np.trim_zeros(array, "fb")
        array_lst = np.array_split(array, self.seg_nums)
        if self.transform is not None:
            for i, array_ in enumerate(array_lst):
                array_lst[i] = self.transform(array_, sample_rate=sr)
            array = np.concatenate(array_lst, axis=0)
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        input_values = self.processor.feature_extractor(audio["array"], sampling_rate=16_000).input_values[0]
        input_values = trim_silence(input_values)
        input_length = len(input_values)
        input_secs = input_length / 16_000
        if input_secs <= 1 or input_secs >= 10:
            return self[(index + 1) % len(self)]  # filter too long or too short audio
        sentence = normalize(remove_special_characters(ann.sentence))
        labels = self.processor.tokenizer(sentence).input_ids

        return {"input_values": input_values, "labels": labels, "sentence": sentence, "id": ann.id,
                "input_length": input_length, "audio": audio}
