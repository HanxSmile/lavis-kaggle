import os.path as osp
from typing import Tuple, Literal
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import AutoProcessor
from PIL import Image


class DressCodeDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: Literal['train', 'test'],
            order: Literal['paired', 'unpaired'] = 'paired',
            category: Tuple[str] = ('dresses', 'upper_body', 'lower_body'),
            size: Tuple[int, int] = (512, 384),
            clip_vit_path: str = "openai/clip-vit-large-patch14",
            offset=None,
    ):

        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.height = size[0]
        self.width = size[1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.order = order
        self.inner_dataset = self.prepare_data()
        self.vit_image_processor = AutoProcessor.from_pretrained(clip_vit_path)
        self.offset = offset

    def prepare_data(self):
        results = []
        for c in self.category:
            assert c in ['dresses', 'upper_body', 'lower_body']
            dataroot = osp.join(self.dataroot, c)
            if self.phase == 'train':
                filename = osp.join(dataroot, f"{self.phase}_pairs.txt")
            else:
                filename = osp.join(dataroot, f"{self.phase}_pairs_{self.order}.txt")
            image_dir = osp.join(dataroot, "images")
            agnostic_mask_dir = osp.join(dataroot, "agnostic_masks")
            cloth_mask_dir = osp.join(dataroot, "cloth_mask")
            caption_dir = osp.join(dataroot, "captions")
            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_caption_name = im_name.replace(".jpg", ".txt")
                    c_caption_name = c_name.replace(".jpg", ".txt")
                    c_mask_name = c_name.replace("_1.jpg", "_6.jpg")
                    agnostic_mask_name = im_name.replace(".jpg", ".png")
                    item = {
                        "image_name": im_name,
                        "image": osp.join(image_dir, im_name),
                        "cloth": osp.join(image_dir, c_name),
                        "cloth_mask": osp.join(cloth_mask_dir, c_mask_name),
                        "agnostic_mask": osp.join(agnostic_mask_dir, agnostic_mask_name),
                        "image_caption": osp.join(caption_dir, im_caption_name),
                        "garment_caption": osp.join(caption_dir, c_caption_name),
                        "category": c,
                    }
                    results.append(item)
            return results

    def __getitem__(self, index):
        ann = self.inner_dataset[index]

        # Clothing image
        cloth = Image.open(ann["cloth"]).convert("RGB")
        cloth_mask = Image.open(ann["cloth_mask"]).convert("L")
        # # Mask out the background
        # cloth = Image.composite(ImageOps.invert(mask.convert('L')), cloth, ImageOps.invert(mask.convert('L')))
        cloth = cloth.resize((self.width, self.height))
        cloth_mask = cloth_mask.resize((self.width, self.height))
        cloth_mask = torch.from_numpy((np.array(cloth_mask) > 127).astype(np.float32))
        cloth_vit = self.vit_image_processor(images=cloth, return_tensors="pt").pixel_values
        cloth = self.transform(cloth)  # [-1,1]

        image = Image.open(ann["image"]).convert("RGB")
        agnostic_mask = Image.open(ann["agnostic_mask"]).convert("L")
        agnostic_mask = agnostic_mask.resize((self.width, self.height))
        image = image.resize((self.width, self.height))
        agnostic_mask = torch.from_numpy((np.array(agnostic_mask) > 127).astype(np.float32))
        image_vit = self.vit_image_processor(images=image, return_tensors="pt").pixel_values
        image = self.transform(image)

        with open(ann["image_caption"], 'r') as f:
            image_caption = f.read().strip()
        with open(ann["garment_caption"], 'r') as f:
            garment_caption = f.read().strip()

        result = dict(
            vton_image=image[None],
            vton_vit_image=image_vit,
            vton_mask_image=agnostic_mask[None, None],
            vton_caption=image_caption,
            vton_instruction="Describe the garment the model is wearing in the photo.",
            garm_image=cloth[None],
            garm_vit_image=cloth_vit,
            garm_mask_image=cloth_mask[None, None],
            garm_caption=garment_caption,
            garm_instruction="Describe the garment in the photo.",
            category=ann["category"],
            image_path=ann["image"],
            image_name=ann["image_name"],
            dataset_name="dresscode",
            order=self.order,
            id=f"dresscode_{self.order}_{ann['category']}_{ann['image_name']}"
        )

        return result

    def __len__(self):
        if self.offset is None:
            return len(self.inner_dataset)
        else:
            return min(self.offset, len(self.inner_dataset))

    def collater(self, samples):
        vton_images = [_["vton_image"] for _ in samples]
        vton_vit_images = [_["vton_vit_image"] for _ in samples]
        vton_mask_images = [_["vton_mask_image"] for _ in samples]
        vton_captions = [_["vton_caption"] for _ in samples]
        vton_instructions = [_["vton_instruction"] for _ in samples]
        garm_images = [_["garm_image"] for _ in samples]
        garm_vit_images = [_["garm_vit_image"] for _ in samples]
        garm_mask_images = [_["garm_mask_image"] for _ in samples]
        garm_captions = [_["garm_caption"] for _ in samples]
        garm_instructions = [_["garm_instruction"] for _ in samples]
        categories = [_["category"] for _ in samples]
        image_paths = [_["image_path"] for _ in samples]
        image_names = [_["image_name"] for _ in samples]
        dataset_names = [_["dataset_name"] for _ in samples]
        orders = [_["order"] for _ in samples]
        ids = [_["id"] for _ in samples]

        result = {
            "vton_image": torch.cat(vton_images),
            "vton_vit_image": torch.cat(vton_vit_images),
            "vton_mask_image": torch.cat(vton_mask_images),
            "vton_caption": vton_captions,
            "vton_instruction": vton_instructions,
            "garm_image": torch.cat(garm_images),
            "garm_vit_image": torch.cat(garm_vit_images),
            "garm_mask_image": torch.cat(garm_mask_images),
            "garm_caption": garm_captions,
            "garm_instruction": garm_instructions,
            "category": categories,
            "image_path": image_paths,
            "image_name": image_names,
            "dataset_name": dataset_names,
            "order": orders,
            "id": ids,
        }
        return result
