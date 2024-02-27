import re
import unicodedata

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

import yaml
import unidecode

NORMALIZED_PATTERNS = (
    (r'^\(([a-zA-Z0-9])\)$', r'\1'),
    (r'^\[([a-zA-Z0-9])\]$', r'\1')
)

NORMALIZED_REGEXS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in NORMALIZED_PATTERNS]

MAPPING_FILE = 'vigc/datasets/builders/chatphone/chatphone_processor/resources/mapping.yaml'


def get_mapping():
    with open(MAPPING_FILE) as f:
        mapping = yaml.load(f, Loader=yaml.SafeLoader)
    return mapping


MAPPING = get_mapping()


def normalize_map(text, mapping=MAPPING):
    output = [mapping.get(c, c) for c in text]

    return ''.join(output)


def normalize(text):
    def normalize_character(c):
        norm = unidecode.unidecode(c)
        for regex, repl in NORMALIZED_REGEXS:
            if regex.match(norm):
                s = regex.sub(repl, norm)
                if len(s) == 1:
                    return s

        return c

    # Normalize using Python built-in unicodedata.
    norm = unicodedata.normalize('NFKD', text)
    # Remove accent characters.
    norm = ''.join(c for c in norm if not unicodedata.combining(c))
    # Normalize using mapping.
    norm = normalize_map(norm)
    # Normalize using unidecode.
    norm = ''.join(normalize_character(c) for c in norm)

    return norm
