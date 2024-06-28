import torch
from transformers import MarianMTModel, AutoTokenizer
import datasets
import re
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
import os.path as osp
import json
from tqdm.auto import tqdm



class Translator:

    def __init__(self, model_path, device_id=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path)
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def translate(self, text):
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        translated_outputs = self.model.generate(**model_inputs)
        return [self.tokenizer.decode(_, skip_special_tokens=True) for _ in translated_outputs]


mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]


def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


replace_nonprint = get_non_printing_char_replacer(" ")


def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean


if __name__ == '__main__':
    data_root = "/mnt/data/hanxiao/dataset/nlp/opus-100-zh-en"
    dst_root = "/mnt/data/hanxiao/dataset/nlp/opus-100-zh-en-json"
    dataset = datasets.load_from_disk(data_root)
    all_data = []
    corpus_id = 0
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    batch_size = 256
    model = Translator("/mnt/data/hanxiao/models/nlp/mt_zh2en/en_multi_models/opus-mt-en-el", device_id=5)
    pbar = tqdm(train_dataset.iter(batch_size=batch_size), total=len(train_dataset) // batch_size)
    for batch in pbar:
        batch = batch["translation"]
        en_text = [preproc(_["en"]) for _ in batch]
        zh_text = [preproc(_["zh"]) for _ in batch]
        target_text = model.translate(en_text)
        target_text = [preproc(_) for _ in target_text]
        data_item = [{"zh": zh_t, "el": el_t} for zh_t, el_t in zip(zh_text, target_text)]
        all_data.extend(data_item)
        if len(all_data) >= 1e5:
            with open(osp.join(dst_root, f"corpus-{corpus_id}.json"), "wb") as f:
                json.dump(all_data, f, ensure_ascii=False)
            corpus_id += 1
            all_data = []
