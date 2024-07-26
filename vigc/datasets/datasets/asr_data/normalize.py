import re
import unicodedata
from khoshnevis import Normalizer

PersianNormalizer = Normalizer()


def normalize(text, language=None):
    if language == "fa":
        text = PersianNormalizer.normalize(text=text, zwnj="\u200c", clean_url=True, remove_emoji=True)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(
        r"\s{2,}",
        " ",
        text,
    )
    text = "".join([unicodedata.normalize('NFC', _) for _ in text])
    return text.strip()


if __name__ == '__main__':
    text = """
    甘納豆はどう？カステラも、パンもあるよ、などと言って騒ぎますので、
    """
    print(normalize(text, language="ja"))
