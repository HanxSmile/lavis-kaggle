import torch
import whisperx
from datasets import load_from_disk
import numpy as np
from whisperx.asr import WhisperModel
import argparse
from tqdm.auto import tqdm
import json


def load_transcribe_model(
        model_path,
        language,
        device="cuda",
        compute_type="float16",
        num_workers=4,
        cpu_threads=4
):
    base_model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
        cpu_threads=cpu_threads,
        download_root=None,
    )
    model = whisperx.load_model(
        "large-v2",
        language=language,
        model=base_model,
        device=device,
        compute_type=compute_type
    )
    return model


def load_alignment_model(
        model_path,
        language,
        device="cuda"
):
    model_a, metadata = whisperx.load_align_model(
        model_name=model_path,
        language_code=language,
        device=device,
    )
    return model_a, metadata


def align(
        model_asr,
        model_align,
        metadata,
        audio,
        text,
        device="cuda"
):
    asr_results = model_asr.transcribe(audio, batch_size=16)
    segments = asr_results["segments"]
    segment_start = segments[0]["start"]
    segment_end = segments[-1]["end"]
    segment = [{"text": text.strip(), "start": segment_start, "end": segment_end}]
    align_results = whisperx.align(segment, model_align, metadata, audio, device, return_char_alignments=False)
    return align_results


def parse_args():
    parser = argparse.ArgumentParser(description="clip audios")

    parser.add_argument("--asr-model", required=True, type=str, help="path to asr model")
    parser.add_argument("--align-model", required=True, type=str, help="path to alignment model.")
    parser.add_argument("--dst-path", required=True, type=str, help="path to save the results.")
    parser.add_argument("--language", required=True, type=str, help="the language of the audio.")
    parser.add_argument("--dataset", required=True, type=str, help="the path of dataset.")
    parser.add_argument("--split", required=True, type=str, default="train", help="the split of dataset.")
    parser.add_argument("--use-space", action="store_true", default=False, help="whether to use spaces.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_model = load_transcribe_model(args.asr_model, language=args.language, device=device)
    align_model, metadata = load_alignment_model(args.align_model, language=args.language, device=device)

    dataset = load_from_disk(args.dataset)
    splits = args.split.split("+")
    sep = " " if args.use_space else ""
    all_results = []
    for split in splits:
        ds = dataset[split]
        for i, sample in tqdm(enumerate(ds), total=len(ds)):
            audio = sample["audio"]["array"].astype(np.float32)
            text = sample["raw_transcription"]
            align_result = align(asr_model, align_model, metadata, audio, text, device=device)
            align_result["split"] = split
            align_result["sep"] = sep
            align_result["dataset_index"] = i
            all_results.append(align_result)
    with open(args.dst_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f)
    print("Done!")
