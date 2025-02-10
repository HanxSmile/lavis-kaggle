import random
import os.path as osp
from typing import Tuple, Literal
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import AutoProcessor
from PIL import Image, ImageOps


class DressCodeDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: Literal['train', 'test'],
            order: Literal['paired', 'unpaired'] = 'paired',
            category: Tuple[str] = ('dresses', 'upper_body', 'lower_body'),
            size: Tuple[int, int] = (512, 384),
            clip_vit_path: str = "openai/clip-vit-large-patch14",
            cloth_background_whitening: bool = False,
            offset=None,
            cloth_mask_augmentation_ratio: float = 1.0,
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
        self.cloth_background_whitening = cloth_background_whitening
        self.offset = offset
        assert cloth_mask_augmentation_ratio >= 1.0
        self.cloth_mask_augmentation_ratio = cloth_mask_augmentation_ratio if phase == "train" else 1.0

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

    @staticmethod
    def get_bounding_box(mask_image):
        mask_array = np.array(mask_image)
        horizontal_sum = np.sum(mask_array != 0, axis=0)

        vertical_sum = np.sum(mask_array != 0, axis=1)

        left = np.argmax(horizontal_sum > 0)
        right = np.argmax(horizontal_sum[::-1] > 0)
        right = mask_array.shape[1] - right - 1  # 翻转后重新计算右边界

        top = np.argmax(vertical_sum > 0)
        bottom = np.argmax(vertical_sum[::-1] > 0)
        bottom = mask_array.shape[0] - bottom - 1  # 翻转后重新计算下边界
        return left, right, top, bottom  # x1, x2, y1, y2

    def get_coarse_mask(self, mask_image):
        mask_array = np.array(mask_image)
        left, right, top, bottom = self.get_bounding_box(mask_image)
        scale_x = random.uniform(1, self.cloth_mask_augmentation_ratio)
        scale_y = random.uniform(1, self.cloth_mask_augmentation_ratio)

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        new_width = (right - left) * scale_x
        new_height = (bottom - top) * scale_y

        new_left = int(center_x - new_width / 2)
        new_right = int(center_x + new_width / 2)
        new_top = int(center_y - new_height / 2)
        new_bottom = int(center_y + new_height / 2)

        new_left = max(0, new_left)
        new_right = min(mask_array.shape[1], new_right)
        new_top = max(0, new_top)
        new_bottom = min(mask_array.shape[0], new_bottom)

        new_mask = np.zeros_like(mask_array)
        new_mask[new_top:new_bottom, new_left:new_right] = 255
        new_mask_image = Image.fromarray(new_mask)
        return new_mask_image

    def __getitem__(self, index):
        ann = self.inner_dataset[index]

        # Clothing image
        cloth = Image.open(ann["cloth"]).convert("RGB")
        cloth_vit = self.vit_image_processor(images=cloth, return_tensors="pt").pixel_values
        cloth_mask_fine = Image.open(ann["cloth_mask"]).convert("L")
        if self.cloth_background_whitening:
            # Mask out the background
            cloth = Image.composite(
                ImageOps.invert(cloth_mask_fine), cloth,
                ImageOps.invert(cloth_mask_fine.convert('L'))
            )
        # TODO：process garment cloth
        cloth = cloth.resize((self.width, self.height))
        cloth_mask_fine = cloth_mask_fine.resize((self.width, self.height))
        cloth_mask_coarse = self.get_coarse_mask(cloth_mask_fine)

        cloth_mask_fine = torch.from_numpy((np.array(cloth_mask_fine) > 127).astype(np.float32))
        cloth_mask_coarse = torch.from_numpy((np.array(cloth_mask_coarse) > 127).astype(np.float32))
        cloth = self.transform(cloth)  # [-1,1]
        # Model Image
        image = Image.open(ann["image"]).convert("RGB")
        agnostic_mask = Image.open(ann["agnostic_mask"]).convert("L")
        agnostic_mask = agnostic_mask.resize((self.width, self.height))
        image = image.resize((self.width, self.height))
        agnostic_mask = torch.from_numpy((np.array(agnostic_mask) > 127).astype(np.float32))
        image_vit = self.vit_image_processor(images=image, return_tensors="pt").pixel_values
        image = self.transform(image)
        agnostic_vton_image = image * (1 - agnostic_mask[None])

        with open(ann["image_caption"], 'r') as f:
            image_caption = f.read().strip()
        with open(ann["garment_caption"], 'r') as f:
            garment_caption = f.read().strip()

        result = dict(
            agnostic_vton_image=agnostic_vton_image[None],
            vton_image=image[None],
            vton_vit_image=image_vit,
            vton_mask_image=agnostic_mask[None, None],
            vton_caption=image_caption,
            vton_instruction="Describe the garment the model is wearing in the photo.",
            garm_image=cloth[None],
            garm_vit_image=cloth_vit,
            garm_mask_image=cloth_mask_coarse[None, None],
            garm_fine_mask_image=cloth_mask_fine[None, None],
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
        agnostic_vton_image = [_["agnostic_vton_image"] for _ in samples]
        vton_images = [_["vton_image"] for _ in samples]
        vton_vit_images = [_["vton_vit_image"] for _ in samples]
        vton_mask_images = [_["vton_mask_image"] for _ in samples]
        vton_captions = [_["vton_caption"] for _ in samples]
        vton_instructions = [_["vton_instruction"] for _ in samples]
        garm_images = [_["garm_image"] for _ in samples]
        garm_vit_images = [_["garm_vit_image"] for _ in samples]
        garm_mask_images = [_["garm_mask_image"] for _ in samples]
        garm_fine_mask_images = [_["garm_fine_mask_image"] for _ in samples]
        garm_captions = [_["garm_caption"] for _ in samples]
        garm_instructions = [_["garm_instruction"] for _ in samples]
        categories = [_["category"] for _ in samples]
        image_paths = [_["image_path"] for _ in samples]
        image_names = [_["image_name"] for _ in samples]
        dataset_names = [_["dataset_name"] for _ in samples]
        orders = [_["order"] for _ in samples]
        ids = [_["id"] for _ in samples]

        result = {
            "agnostic_vton_image": torch.cat(agnostic_vton_image),
            "vton_image": torch.cat(vton_images),
            "vton_vit_image": torch.cat(vton_vit_images),
            "vton_mask_image": torch.cat(vton_mask_images),
            "vton_caption": vton_captions,
            "vton_instruction": vton_instructions,
            "garm_image": torch.cat(garm_images),
            "garm_vit_image": torch.cat(garm_vit_images),
            "garm_mask_image": torch.cat(garm_mask_images),
            "garm_fine_mask_image": torch.cat(garm_fine_mask_images),
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
