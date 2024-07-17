import re
import unicodedata


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = "".join([unicodedata.normalize('NFC', _) for _ in text])
    return text


if __name__ == '__main__':
    text = "Bu süreç bir anlamda, bir el arabasını tepeye çıkarmak gibidir. Çekirdeği tekrar yarar ve sonra o enerjinin bir kısmını serbest bırakır."
    print(normalize(text))
