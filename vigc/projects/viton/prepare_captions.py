import argparse
import os
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image


def prepare_caption(model, processor, device, image, garment_flag=False):
    image = Image.open(image).convert("RGB")
    if not garment_flag:
        prompt = "Write a description for the photo."
    else:
        prompt = "Write a description for the garment in the photo."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)
    return generated_text


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare captions for Viton")
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path or repo name of InstructBlip. "
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help=(
            "Which split to use. "
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    model = InstructBlipForConditionalGeneration.from_pretrained(args.repo_path)
    processor = InstructBlipProcessor.from_pretrained(args.repo_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for sub_folder in ['upper_body', 'lower_body', 'dresses']:
        assert os.path.exists(os.path.join(args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."
        if args.split == 'train':
            pair_txt = os.path.join(args.data_root_path, sub_folder, 'train_pairs.txt')
        else:
            pair_txt = os.path.join(args.data_root_path, sub_folder, 'test_pairs_paired.txt')
        assert os.path.exists(pair_txt), f"File {pair_txt} does not exist."

        output_dir = os.path.join(args.data_root_path, sub_folder, 'captions')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        for image_name in tqdm(os.listdir(os.path.join(args.data_root_path, sub_folder, "images")),
                               desc=f"Processing {sub_folder}"):
            image_path = os.path.join(args.data_root_path, sub_folder, "images", image_name)
            garment_flag = image_name.endswith("_1.jpg")
            assert image_name.endswith("_1.jpg") or image_name.endswith("_0.jpg")
            dst_path = os.path.join(args.data_root_path, sub_folder, "captions", image_name.replace(".jpg", ".txt"))

            caption = prepare_caption(model, processor, device, image_path, garment_flag)
            with open(dst_path, "w") as f:
                f.write(caption.strip())


if __name__ == "__main__":
    args = parse_args()
    main(args)
