from .img2markdown import Im2MkdownDataset
from PIL import Image


class QwenIm2MkdownDataset(Im2MkdownDataset):
    SYSTEM_INPUT = "You are a helpful assistant."
    USER_INPUT = "Please automatically translate this English document image into Chinese and convert the output into Markdown format."

    def __getitem__(self, index):
        ann = self.inner_dataset[index]
        image = Image.open(ann['image']).convert('RGB')
        with open(ann['text'], 'r', encoding="utf-8") as f:
            text = f.read().strip()
        try:
            image = self.vis_processor(image)
        except Exception as e:
            print(f"Exception {e} while processing image {ann['image']}")
            return self[(index + 1) % len(self)]
        if image is None:
            print(f"'{ann['image']}' is empty")
            return self[(index + 1) % len(self)]
        return {"image": image, "text_output": text, "id": index}

    def collater(self, samples):
        image_list, output_list, id_list = [], [], []
        system_input_list, input_list = [], []
        for sample in samples:
            image_list.append(sample["image"])
            system_input_list.append(self.SYSTEM_INPUT)
            input_list.append(self.USER_INPUT)
            output_list.append(sample["text_output"])
            id_list.append(sample["id"])

        return {
            "image_input": image_list,
            "system_input": system_input_list,
            "text_input": input_list,
            "text_output": output_list,
            "id": id_list
        }
