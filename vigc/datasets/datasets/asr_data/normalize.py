import re
import unicodedata
from khoshnevis import Normalizer

PersianNormalizer = Normalizer()


def normalize(text, language=None):
    if language == "fa":
        text = PersianNormalizer.normalize(text=text, zwnj="\u200c", clean_url=True, remove_emoji=True)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = "".join([unicodedata.normalize('NFC', _) for _ in text])
    return text


if __name__ == '__main__':
    text = """
    خود بیماری‌های عفونی
    """
    print(normalize(text, language="fa"))
