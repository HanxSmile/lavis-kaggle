from .bengali_wav2vec_asr import Wav2VecBase
import os.path as osp
import pandas as pd
import os


class Wav2VecWholeDataset(Wav2VecBase):
    def __init__(self, processor, data_root):
        self.media_root = osp.join(data_root, "train_mp3s")
        self.anno_path = osp.join(data_root, "train.csv")

        inner_dataset = pd.read_csv(self.anno_path)
        inner_dataset["audio"] = self.media_root + os.sep + inner_dataset["id"] + ".mp3"

        super().__init__(inner_dataset, processor)

    def _parse_ann_info(self, index):
        ann = self.inner_dataset.iloc[index]
        audio_path = ann.audio
        audio = self.read_and_resample_audio(audio_path, return_dict=True)
        return audio, ann.sentence, ann.id

    def is_valid(self, input_values):
        return True
