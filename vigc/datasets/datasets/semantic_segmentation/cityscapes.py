from .base_ds import SemanticSegmentationDataset
import cv2
import numpy as np


class CityScapes(SemanticSegmentationDataset):
    def __init__(
            self,
            ann_path,
            media_dir,
            processor=None,
    ):
        super().__init__(ann_path, media_dir, processor)

        self.label_mapping = {
            7: 0, 8: 1, 11: 2, 12: 3,
            13: 4, 17: 5, 19: 6, 20: 7,
            21: 8, 22: 9, 23: 10, 24: 11,
            25: 12, 26: 13, 27: 14, 28: 15,
            31: 16, 32: 17, 33: 18
        }
        self.classes = [
            "road", "sidewalk", "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person",
            "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
        ]

    def load_label(self, label_path):
        color_map = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_shape = [color_map.shape[0], color_map.shape[1], len(self.classes)]
        label = np.zeros(label_shape)
        for k, v in self.label_mapping.items():
            label[:, :, v][color_map == k] = 1

        return label.astype(np.float32)
