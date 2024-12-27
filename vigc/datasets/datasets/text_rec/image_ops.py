import six
import cv2
import numpy as np


class ReadImage(object):
    """ read image to np.uint8 array """

    def __init__(
            self,
            img_mode='RGB',
            channel_first=False,
            ignore_orientation=False,
            **kwargs
    ):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, img_path):
        with open(img_path, 'rb') as f:
            img = f.read()
        if six.PY2:
            assert type(img) is str and len(img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


if __name__ == '__main__':
    img_path = "/Users/hanxiao/Downloads/images.jpeg"
    decode = ReadImage(img_mode="BGR")
    img = decode(img_path)
    from PIL import Image

    print(img.max())
    img = Image.fromarray(img)
    img.show()
