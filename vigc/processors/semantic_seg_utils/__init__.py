import cv2
import numpy as np
from typing import Optional, List, Tuple, Union


class Resize:

    def __init__(
            self,
            img_size=None,
            ratio_range=(0.75, 1.5),
    ):
        if img_size is None:
            self.img_size = None
        else:
            if isinstance(img_size, list):
                self.img_size = img_size
            else:
                self.img_size = [img_size, img_size]
        self.ratio_range = ratio_range

    @staticmethod
    def random_sample_ratio(img_size, ratio_range):
        height, width = img_size

        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(height * ratio), int(width * ratio)
        return scale

    def _resize(self, image, mask):
        """Resize images with ``results['scale']``."""
        if self.img_size is None:
            height, width = image.shape[:2]
            mask_height, mask_width = mask.shape[:2]
            assert mask_height == height and mask_width == width
        else:
            height, width = self.img_size
        new_height, new_width = self.random_sample_ratio((height, width), ratio_range=self.ratio_range)
        new_image = cv2.resize(image, (new_width, new_height), cv2.INTER_LINEAR)
        new_mask = cv2.resize(mask, (new_width, new_height), cv2.INTER_NEAREST)
        return new_image, new_mask

    def __call__(self, image, mask):
        image, mask = self._resize(image, mask)
        return {"image": image, "mask": mask}


class RandomCrop:
    """
    Random crop the image & seg.
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, image, mask):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        crop_bbox = self.get_crop_bbox(image)

        # crop the image
        image = self.crop(image, crop_bbox)
        mask = self.crop(mask, crop_bbox)

        return {"image": image, "mask": mask}


class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(
            self,
            size=None,
            pad_val=0,
            seg_pad_val=-100.0):
        self.size = size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, image):
        """Pad images according to ``self.size``."""

        padded_img = self.impad(
            image, shape=self.size, pad_val=self.pad_val)

        return padded_img

    @staticmethod
    def impad(img: np.ndarray,
              *,
              shape: Optional[Tuple[int, int]] = None,
              padding: Union[int, tuple, None] = None,
              pad_val: Union[float, List] = 0,
              padding_mode: str = 'constant') -> np.ndarray:
        """Pad the given image to a certain shape or pad on all sides with
        specified padding mode and padding value.

        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w). Default: None.
            padding (int or tuple[int]): Padding on each border. If a single int is
                provided this is used to pad all borders. If tuple of length 2 is
                provided this is the padding on left/right and top/bottom
                respectively. If a tuple of length 4 is provided this is the
                padding for the left, top, right and bottom borders respectively.
                Default: None. Note that `shape` and `padding` can not be both
                set.
            pad_val (Number | Sequence[Number]): Values to be filled in padding
                areas when padding_mode is 'constant'. Default: 0.
            padding_mode (str): Type of padding. Should be: constant, edge,
                reflect or symmetric. Default: constant.

                - constant: pads with a constant value, this value is specified
                  with pad_val.
                - edge: pads with the last value at the edge of the image.
                - reflect: pads with reflection of image without repeating the last
                  value on the edge. For example, padding [1, 2, 3, 4] with 2
                  elements on both sides in reflect mode will result in
                  [3, 2, 1, 2, 3, 4, 3, 2].
                - symmetric: pads with reflection of image repeating the last value
                  on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
                  both sides in symmetric mode will result in
                  [2, 1, 1, 2, 3, 4, 4, 3]

        Returns:
            ndarray: The padded image.
        """

        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            width = max(shape[1] - img.shape[1], 0)
            height = max(shape[0] - img.shape[0], 0)
            padding = (0, 0, width, height)

        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, (int, float)):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, (int, float)):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                             f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)

        return img

    def _pad_seg(self, mask):
        """Pad masks according to ``results['pad_shape']``."""

        mask = self.impad(
            mask,
            shape=self.size,
            pad_val=self.seg_pad_val)
        return mask

    def __call__(self, image, mask):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        image = self._pad_img(image)
        mask = self._pad_seg(mask)
        return {'image': image, 'mask': mask}
