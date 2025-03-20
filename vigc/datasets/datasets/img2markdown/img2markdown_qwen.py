from .img2markdown import Im2MkdownDataset


class QwenIm2MkdownDataset(Im2MkdownDataset):

    def collater(self, samples):
        image_list, question_list, id_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            id_list.append(sample["id"])

        return {
            "image": image_list,
            "text_input": question_list,
            "id": id_list
        }
