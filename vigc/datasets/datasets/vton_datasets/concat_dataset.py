import torch
from torch.utils.data import Dataset
from vigc.datasets.datasets.vton_datasets.simple_dresscode import DressCodeDataset
from vigc.datasets.datasets.vton_datasets.simple_vitonhd import VitonHDDataset


class VtonConcatTestDataset(Dataset):

    def __init__(self, dataset_root_info, orders, clip_vit_path, datasets, size, offset=None):
        if isinstance(orders, str):
            orders = [orders]
        if isinstance(datasets, str):
            datasets = [datasets]
        all_datasets = list()
        for order in orders:
            for dataset in datasets:
                if dataset == "dresscode":
                    dataset_cls = DressCodeDataset
                else:
                    dataset_cls = VitonHDDataset
                all_datasets.append(dataset_cls(
                    dataroot_path=dataset_root_info[dataset],
                    order=order,
                    size=size,
                    phase="test",
                    clip_vit_path=clip_vit_path,
                    offset=offset,
                ))
        self.inner_datasets = all_datasets

    def __len__(self):
        return sum([len(_) for _ in self.inner_datasets])

    def __getitem__(self, index):
        for i, dataset in enumerate(self.inner_datasets):
            if index < len(dataset):
                return dataset[index]
            else:
                index -= len(dataset)

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
