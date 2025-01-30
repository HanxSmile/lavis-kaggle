import argparse
import os
from tqdm import tqdm
import numpy as np
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from scipy.ndimage import binary_dilation
from PIL import Image


def prepare_cloth_mask(model, image):
    image = Image.open(image).convert("RGB")
    result = model.predict(image)
    mask = binary_dilation(
        result.mask[0],
        structure=np.ones((3, 3), dtype=bool),
        iterations=2)
    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare captions for Viton")
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="Path to the dataset to evaluate."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                "clothes": "clothes",
                "upper body clothes": "clothes",
                "lower body clothes": "clothes",
            }
        )
    )

    for sub_folder in ['upper_body', 'lower_body', 'dresses']:
        assert os.path.exists(os.path.join(args.data_root_path, sub_folder)), f"Folder {sub_folder} does not exist."

        output_dir = os.path.join(args.data_root_path, sub_folder, 'cloth_mask')
        all_images = os.listdir(os.path.join(args.data_root_path, sub_folder, "images"))
        all_images = [_ for _ in all_images if _.endswith("_1.jpg")]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        for image_name in tqdm(all_images, desc=f"Processing {sub_folder}"):
            image_path = os.path.join(args.data_root_path, sub_folder, "images", image_name)
            assert image_name.endswith("_1.jpg")
            dst_path = os.path.join(args.data_root_path, sub_folder, "cloth_mask",
                                    image_name.replace("_1.jpg", "_6.jpg"))

            cloth_mask = prepare_cloth_mask(base_model, image_path)
            cloth_mask = Image.fromarray(cloth_mask)
            cloth_mask.save(dst_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
