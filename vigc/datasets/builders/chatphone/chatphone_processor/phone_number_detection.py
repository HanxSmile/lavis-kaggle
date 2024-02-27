import re
from functools import reduce
from .fancy_text import normalize, MAPPING
from .resources.strange_numbers import STRANGE_NUMBERS_MAPPING, STRANGE_NUMBERS_REPLACE, OTHER_MAPPING
from enum import Enum

MAPPING.update(STRANGE_NUMBERS_MAPPING)
MAPPING.update(OTHER_MAPPING)

SEP_LIST = ["-", "/", "+", "*", ".", "(", ")", "_", r"\s", "[", "]", "=", "'", '"', "—", ",", "?", "!", "&", "%"]


class MsgType(Enum):
    GAME = 0
    NORMAL_PN = 1
    COMPLEX_PN = 2
    NO_PN = 3


class PhoneNumberDetection:
    ID_REGEX = r"(62|0|o|O|x|nol)?(811|812|813|814|815|816|817|818|819|821|822|823|828|831|832|833|838|851|852|853|855|856|857|858|859|8681|877|878|879|881|882|883|884|885|886|887|888|889|895|896|897|898|899|8ii|8ll|8i2|8l2|8i3|8l3|8i5|8l5|8i6|8l6|8i8|8l8|82i|82l|83i|83l|85i|85l|88i|8i9|8zz)[0-9]{6,9}"
    SECOND_STAGE_REGEX = r"""(([0-9](-|/|\+|\*|\.|\(|\)|_|\s|\[|\]|=|'|"|—|/|,|\?|!|&|%)*){6,})"""
    SEP_REGEX = r"""(-|/|\+|\*|\.|\(|\)|_|\s|\[|\]|=|'|"|—|/|,)*"""
    SPECIAL_NUMBER_MAPPING = {
        "0": ("o", "x", "O", "X", "@", "Q", "D"),
        "1": ("i", "l", "I"),
        "2": ("z", "Z"),
        "8": ("B", "&"),
        "6": ("b",),
        "5": ("s", "S"),
        "9": ("q",)
    }
    SPECIAL_ZEROS = {
        "0": ("KOSONG", "Kosong", "kosong", "Nol", "NOL",),
        "o": ("nol",),
        "q": ("sembilan", "Sembilan", "SEMBILAN"),
        "B": ("delapan", "Delapan", "DELAPAN"),
        "7": ("tujuh", "Tujuh", "TUJUH"),
        "b": ("enam", "Enam", "ENAM"),
        "S": ("lima", "Lima", "LIMA"),
        "4": ("empat", "Empat", "EMPAT"),
        "3": ("tiga", "Tiga", "TIGA"),
        "Z": ("dua", "Dua", "DUA"),
        "l": ("satu", "Satu", "SATU"),
    }
    ALL_SPECIAL_NUMBER_SRC = reduce(lambda x, y: x + y, SPECIAL_NUMBER_MAPPING.values())
    VALID_STRING = (
        "https://games.shopee.co.id/",
    )
    SHOPEE_ID_PATTERN = re.compile(r"https:\/\/shopee\.co\.id\/product\/\d+\/\d+")

    def __init__(self, sub_string_flag=False):
        self.sub_string_flag = sub_string_flag
        self.digit_pattern = re.compile(self.SECOND_STAGE_REGEX)
        self.sep_pattern = re.compile(self.SEP_REGEX)
        self.id_pattern = re.compile(self.ID_REGEX)
        self._msg_type = None

    def pre_stage_judge(self, text):
        for string in self.VALID_STRING:
            if string in text:
                return True
        return False

    def first_stage_extract_digit_list(self, text):
        valid_lst = []
        substring_lst = []
        for res in self.id_pattern.finditer(text):
            match_text = res.group().strip()
            start = res.start()
            end = start + len(match_text)
            if (start == 0 or not (text[start - 1].isdigit())) and (end == len(text) or (not text[end].isdigit())):
                valid_lst.append(match_text)
            else:
                substring_lst.append(match_text)
        return valid_lst, substring_lst

    def second_stage_extract_digit_list(self, text):
        res_lst = []
        for res in self.digit_pattern.finditer(text):
            match_text = res.group().strip()

            # start = res.start()
            # end = start + len(match_text)
            res_lst.append(match_text)
        return res_lst

    def third_stage_extract_digit_list(self, text):
        text = self.sep_pattern.sub("", text)
        match_result, substring = self.first_stage_extract_digit_list(text)
        return match_result + substring

    def process(self, text):
        pre_stage_valid_flag = self.pre_stage_judge(text)
        if pre_stage_valid_flag:
            self._msg_type = MsgType.GAME
            return list()

        text = self.preprocess_text(text)
        first_number_list, substring_list = self.first_stage_extract_digit_list(text)

        if first_number_list:
            self._msg_type = MsgType.NORMAL_PN
            return first_number_list
        for substring in substring_list:
            text = text.replace(substring, "")

        second_number_list = self.second_stage_extract_digit_list(text)
        third_number_list = []
        for number_str in second_number_list:
            third_number_list.extend(self.third_stage_extract_digit_list(number_str))
        if third_number_list:
            self._msg_type = MsgType.COMPLEX_PN
        else:
            self._msg_type = MsgType.NO_PN
        return third_number_list

    def preprocess_text(self, text):
        text = text.strip().replace("\n", " ")
        text = re.sub(self.SHOPEE_ID_PATTERN, "shopee_id", text)
        for d_, s_ in self.SPECIAL_NUMBER_MAPPING.items():
            text = self._process_special_number(text, s_, d_)
        for k, v in STRANGE_NUMBERS_REPLACE.items():
            text = text.replace(k, v)
        text = normalize(text)
        return text

    def _process_special_number(self, text, src, dst):
        for k, v in self.SPECIAL_ZEROS.items():
            for v_ in v:
                text = text.replace(v_, k)
        res = ""
        for i, c in enumerate(text):
            this_letter = c
            if c in src:
                former_letter = None if i == 0 else text[i - 1]
                latter_letter = None if i == len(text) - 1 else text[i + 1]
                former_digit_flag = former_letter is None or (
                    not former_letter.isalpha()) or former_letter in self.ALL_SPECIAL_NUMBER_SRC
                latter_digit_flag = latter_letter is None or (
                    not latter_letter.isalpha()) or latter_letter in self.ALL_SPECIAL_NUMBER_SRC
                if former_digit_flag and latter_digit_flag:
                    this_letter = dst
            res += this_letter

        return res
