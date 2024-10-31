from .base_ds import SemanticSegmentationDataset
import numpy as np
from PIL import Image


class BDD(SemanticSegmentationDataset):
    def __init__(
            self,
            ann_path,
            media_dir,
            processor=None,
    ):
        super().__init__(ann_path, media_dir, processor)

        self.color_list = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ]
        self.classes = [
            "road", "sidewalk", "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person",
            "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
        ]

    def load_label(self, label_path):
        color_map = Image.open(label_path).convert('RGB')
        color_map = np.array(color_map)
        label_shape = [color_map.shape[0], color_map.shape[1], len(self.classes)]
        label = np.zeros(label_shape)
        for i, v in enumerate(self.color_list):
            label[:, :, i][(color_map == v).sum(2) == 3] = 1

        return label.astype(np.float32)

