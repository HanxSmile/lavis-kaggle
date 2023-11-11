from accelerate import Accelerator, DistributedDataParallelKwargs, notebook_launcher
import pandas as pd
import torch.nn as nn
import logging
import torch
from torch.utils.data import Dataset as torch_Dataset, DataLoader
import librosa
from typing import Dict, List, Union, Any
from dataclasses import dataclass
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, pipeline
from bnunicodenormalizer import Normalizer
import os.path as osp
import os
import json
from tqdm.auto import tqdm
import numpy as np

# Helper
bnorm = Normalizer()


def normalize(sen):
    _words = [bnorm(word)['normalized'] for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


def dari(sentence):
    try:
        if sentence[-1] != "।":
            sentence += "।"
    except:
        print(sentence)
    return sentence


## Config
@dataclass
class InferenceConfig:
    # model
    model_name = ""
    processor_name = ""
    post_process_flag = True
    ckpt = ""

    # data
    data_root = "/kaggle/input/bengaliai-speech/test_mp3s"
    batch_size = 8
    num_workers = 2

    # runtime
    mixed_precision = "fp16"
    output_dir = "/kaggle/working/"
    num_processes = 2


config = InferenceConfig()


## Model
class BengaliWav2Vec(nn.Module):

    def __init__(
            self,
            model_name="Sameen53/cv_bn_bestModel_1",
            processor_name="arijitx/wav2vec2-xls-r-300m-bengali",
            post_process_flag=True
    ):
        super().__init__()
        self.post_process_flag = post_process_flag
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.config.ctc_zero_infinity = True
        self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(processor_name)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def load_checkpoint(self, ckpt, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        checkpoint = torch.load(ckpt, map_location="cpu")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info(f"Missing keys exist when loading '{ckpt}'.")
        logging.info("load checkpoint from %s" % ckpt)

    @torch.no_grad()
    def generate(
            self,
            samples,
            **kwargs
    ):
        inputs = samples["raw_audios"]
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            chunk_length_s=10,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            decoder=self.processor.decoder,
            device=self.device
        )

        transcription = pipe(inputs, batch_size=8)
        transcription = [_["text"] for _ in transcription]
        if self.post_process_flag:
            transcription = [dari(normalize(_)) for _ in transcription]
        return transcription


## Dataset
class BengaliASR(torch_Dataset):
    def __init__(self, data_root):
        all_data = [(osp.join(data_root, _), _.replace(".mp3", "")) for _ in os.listdir(data_root) if
                    _.endswith(".mp3")]
        self.all_files = [_[0] for _ in all_data]
        self.all_ids = [_[1] for _ in all_data]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        audio_path = self.all_files[index]
        audio_id = self.all_ids[index]
        array, sr = librosa.load(audio_path, sr=None)
        array, sr = librosa.resample(array, orig_sr=sr, target_sr=16_000), 16_000
        audio = {
            "path": audio_path,
            "array": array,
            "sampling_rate": sr
        }
        audio["array"] = np.trim_zeros(audio["array"], "fb")
        return {"id": audio_id, "audio": audio}

    def collater(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, Any]:
        result = {}
        result["ids"] = [_["id"] for _ in features]
        result["raw_audios"] = [_["audio"] for _ in features]
        return result


## Factory
class Factory:

    @classmethod
    def prepare_model(cls, cfg: InferenceConfig):
        model = BengaliWav2Vec(
            model_name=cfg.model_name,
            processor_name=cfg.processor_name,
            post_process_flag=cfg.post_process_flag
        )
        model.load_checkpoint(cfg.ckpt)
        return model

    @classmethod
    def prepare_dataset(cls, cfg: InferenceConfig):
        dataset = BengaliASR(cfg.data_root)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collater,
            drop_last=False,
        )
        return loader


# Inference Loop

def inference_loop(cfg: InferenceConfig):
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, kwargs_handlers=[ddp_kwargs])

    dataloader = Factory.prepare_dataset(cfg)
    model = Factory.prepare_model(cfg)

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False

    model, dataloader = accelerator.prepare(model, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    all_results = {}
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        ids = batch["ids"]
        with torch.no_grad():
            preds = unwrapped_model.generate(batch)
        for id_, pred in zip(ids, preds):
            all_results[id_] = pred
    rank = accelerator.local_process_index
    dst_path = osp.join(cfg.output_dir, f"rank_{rank}.json")
    with open(dst_path, "w") as f:
        json.dump(all_results, f)
    accelerator.print("Done inference.")


notebook_launcher(inference_loop, (config,), num_processes=config.num_processes)

##  Submit
all_json = [osp.join(config.output_dir, _) for _ in os.listdir(config.output_dir) if
            _.endswith(".json") and _.startswith("rank_")]
final_result = {}
for file in all_json:
    with open(file) as f:
        this_res = json.load(f)
    final_result.update(this_res)
ids, preds = [], []
for k, v in final_result.items():
    ids.append(k)
    if len(v) == 0:
        v = '।'
    preds.append(v)
submission = pd.DataFrame({"id": ids, "sentence": preds})
submission.to_csv("submission.csv", index=False)
