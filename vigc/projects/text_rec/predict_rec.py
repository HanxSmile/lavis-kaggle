import math
import cv2
import numpy as np
from vigc.datasets.datasets.text_rec.image_ops import ReadImage
from vigc.models.text_rec.postprocess.rec_postprocess import CTCLabelDecode
from .onnx_engine import ONNXEngine


class TextRecognizer(ONNXEngine):
    def __init__(self, model_path, input_shape, use_gpu=False, batch_size=1, character_dict_path=None,
                 use_space_char=False):
        super().__init__(model_path, use_gpu)
        self.read_img = ReadImage(img_mode="RGB")
        self.batch_size = batch_size
        self.rec_image_shape = [int(_) for _ in input_shape]
        self.postprocess = CTCLabelDecode(character_dict_path=character_dict_path, use_space_char=use_space_char)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        if not isinstance(img_list, list):
            img_list = [img_list]
        img_list = [self.read_img(_) for _ in img_list]
        img_num = len(img_list)

        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.batch_size
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            preds = self.run(norm_img_batch)
            rec_result = self.postprocess(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res


if __name__ == '__main__':
    model_path = "/mnt/data/wangyesong01/output/text_rec/crnn.onnx"
    input_shape = [3, 32, 320]
    use_gpu = False
    batch_size = 1
    character_dict_path = "/mnt/data/xuyang/datasets/ocr_russian/russian_characters.txt"
    use_space_char = True
    model = TextRecognizer(
        model_path=model_path,
        use_space_char=use_space_char,
        character_dict_path=character_dict_path,
        use_gpu=use_gpu,
        batch_size=batch_size,
        input_shape=input_shape,
    )

    result = model("/mnt/data/xuyang/datasets/ocr_russian/lines_w25/59027_a.png")
    print(result)
