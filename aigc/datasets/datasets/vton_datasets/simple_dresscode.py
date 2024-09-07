# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import os
from typing import Tuple, Literal
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import AutoProcessor
from PIL import Image, ImageDraw, ImageOps
from numpy.linalg import lstsq

from aigc.datasets.datasets.vton_datasets.utils.labelmap import label_map


class DressCodeDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: Literal['train', 'test'],
            radius=5,
            order: Literal['paired', 'unpaired'] = 'paired',
            outputlist: Tuple[str] = ('c_name', 'im_name', 'cloth', 'image', 'im_mask', 'inpaint_mask', 'category'),
            category: Tuple[str] = ('dresses', 'upper_body', 'lower_body'),
            size: Tuple[int, int] = (512, 384),
            clip_vit_path: str = "openai/clip-vit-large-patch14",
            offset=None,
    ):

        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = category
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []

        possible_outputs = ['c_name', 'im_name', 'cloth', 'image', 'im_mask', 'inpaint_mask', 'category']

        assert all(x in possible_outputs for x in outputlist)

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")

            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()

                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.vit_image_processor = AutoProcessor.from_pretrained(clip_vit_path)
        self.offset = offset

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = dataroot.split('/')[-1]

        if "cloth" in self.outputlist:  # In-shop clothing image
            # Clothing image
            cloth = Image.open(os.path.join(dataroot, 'images', c_name))
            # mask = Image.open(os.path.join(dataroot, 'masks', c_name.replace(".jpg", ".png")))
            #
            # # Mask out the background
            # cloth = Image.composite(ImageOps.invert(mask.convert('L')), cloth, ImageOps.invert(mask.convert('L')))
            cloth = cloth.resize((self.width, self.height))
            cloth_vit = self.vit_image_processor(images=cloth, return_tensors="pt").pixel_values
            cloth = self.transform(cloth)  # [-1,1]

        if "image" in self.outputlist or "im_head" in self.outputlist or "im_cloth" in self.outputlist:
            # Person image
            image = Image.open(os.path.join(dataroot, 'images', im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if (
                "im_pose" in self.outputlist or
                "parser_mask" in self.outputlist or
                "im_mask" in self.outputlist or
                "parse_mask_total" in self.outputlist or
                "parse_array" in self.outputlist or
                "pose_map" in self.outputlist or
                "parse_array" in self.outputlist or
                "shape" in self.outputlist or
                "im_head" in self.outputlist
        ):
            # Label Map
            parse_name = im_name.replace('_0.jpg', '_4.png')
            im_parse = Image.open(os.path.join(dataroot, 'label_maps', parse_name))
            im_parse = im_parse.resize((self.width, self.height), Image.Resampling.NEAREST)
            parse_array = np.array(im_parse)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                                (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["hat"]).astype(np.float32) + \
                                (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                                (parse_array == label_map["scarf"]).astype(np.float32) + \
                                (parse_array == label_map["bag"]).astype(np.float32)

            parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

            if dataroot.split('/')[-1] == 'dresses':

                parse_mask = (parse_array == 7).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)
                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

            elif dataroot.split('/')[-1] == 'upper_body':

                parse_mask = (parse_array == 4).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                     (parse_array == label_map["pants"]).astype(np.float32)

                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
            elif dataroot.split('/')[-1] == 'lower_body':

                parse_mask = (parse_array == 6).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                     (parse_array == 14).astype(np.float32) + \
                                     (parse_array == 15).astype(np.float32)
                parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
            else:
                raise NotImplementedError

            parse_head = torch.from_numpy(parse_head)  # [0,1]

            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            parse_mask = parse_mask.cpu().numpy()

            # Load pose points
            pose_name = im_name.replace('_0.jpg', '_2.json')
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 4))

            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
                point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body' or dataroot.split('/')[
                -1] == 'lower_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
                    shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
                    elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
                    elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
                    wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
                    wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
                    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)

                if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                    parse_mask += im_arms
                    parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    points = []
                    points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0))
                    points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0))
                    x_coords, y_coords = zip(*points)
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, c = lstsq(A, y_coords, rcond=None)[0]
                    for i in range(parse_array.shape[1]):
                        y = i * m + c
                        parse_head_2[int(y - 20 * (self.height / 512.0)):, i] = 0

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)

            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            inpaint_mask = inpaint_mask.unsqueeze(0)

        result = dict(
            vton_image=image[None],
            vton_mask_image=im_mask[None],
            garm_image=cloth[None],
            garm_vit_image=cloth_vit,
            inpaint_mask=inpaint_mask[None].float(),
            category=category,
            image_path=os.path.join(dataroot, 'images', im_name),
            image_name=im_name,
            dataset_name="dresscode",
            paired=self.order,
            id=f"dresscode_{self.order}_{category}_{im_name}"
        )

        return result

    def __len__(self):
        if self.offset is None:
            return len(self.c_names)
        else:
            return min(self.offset, len(self.c_names))

    def collater(self, samples):
        vton_images, garm_images, gt_images, garm_vit_images, mask_images, captions = [], [], [], [], [], []
        image_paths, paireds, ids, dataset_names, image_names = [], [], [], [], []
        for input_sample in samples:
            vton_images.append(input_sample["vton_mask_image"])
            garm_images.append(input_sample["garm_image"])
            gt_images.append(input_sample["vton_image"])
            garm_vit_images.append(input_sample["garm_vit_image"])
            mask_images.append(input_sample["inpaint_mask"])
            captions.append(input_sample["category"])
            image_paths.append(input_sample["image_path"])
            image_names.append(input_sample["image_name"])
            paireds.append(input_sample["paired"])
            ids.append(input_sample["id"])
            dataset_names.append(input_sample["dataset_name"])

        return {
            "vton_images": torch.cat(vton_images),
            "garm_images": torch.cat(garm_images),
            "gt_images": torch.cat(gt_images),
            "garm_vit_images": torch.cat(garm_vit_images),
            "mask_images": torch.cat(mask_images),
            "captions": captions,
            "category": captions,
            "paired": paireds,
            "dataset_name": dataset_names,
            "id": ids,
            "image_path": image_paths,
            "image_name": image_names
        }
