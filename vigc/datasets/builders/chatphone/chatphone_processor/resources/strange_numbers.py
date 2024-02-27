_STRANGE_NUMBERS = [
    "０,１,２,３,４,５,６,７,８,９",  # 全形数字
    "⓪,①,②,③,④,⑤,⑥,⑦,⑧,⑨",  # 带圈数字
    "零,一,二,三,四,五,六,七,八,九",  # 中文数字
    "⓪,①,②,③,④,⑤,⑥,⑦,⑧,⑨",  # 带框的数字
    "0️⃣,1️⃣,2️⃣,3️⃣,4️⃣,5️⃣,6️⃣,7️⃣,8️⃣,9️⃣",  # emoji
    "𝟬,𝟭,𝟮,𝟥,𝟰,𝟱,𝟲,𝟳,𝟴,𝟵",  # 数学黑体粗体数字
    "〇,壹,贰,叁,肆,伍,陆,柒,捌,玖",  # 中文大写
    "𝟶,𝟷,𝟸,𝟹,𝟺,𝟻,𝟼,𝟽,𝟾,𝟿",  # 刻字体数字
    "𝟬,𝟭,𝟮,𝟯,𝟰,𝟱,𝟲,𝟳,𝟴,𝟵",  # 双线字体数字
    "𝟢,𝟣,𝟤,𝟥,𝟦,𝟧,𝟨,𝟩,𝟪,𝟫",  # 无衬线字体数字
    "⁰,¹,²,³,⁴,⁵,⁶,⁷,⁸,⁹",  # 顶注
    "₀,₁,₂,₃,₄,₅,₆,₇,₈,₉",  # 底注
    "⓿,➊,➋,➌,➍,➎,➏,➐,➑,➒",  # 闭合的数字
    "0⃣,1⃣,2⃣,3⃣,4⃣,5⃣,6⃣,7⃣,8⃣,9⃣",  # 有效果的emoji
    "0⃞,1⃞,2⃞,3⃞,4⃞,5⃞,6⃞,7⃞,8⃞,9⃞",  # 带框的数字
]

STRANGE_NUMBERS = [_.split(",") for _ in set(_STRANGE_NUMBERS)]
STRANGE_NUMBERS_MAPPING = {}
STRANGE_NUMBERS_REPLACE = {}
for line in STRANGE_NUMBERS:
    this_dic = {k.strip(): str(i) for i, k in enumerate(line) if len(k.strip()) == 1}
    replace_dic = {k.strip(): str(i) for i, k in enumerate(line) if len(k.strip()) > 1}
    STRANGE_NUMBERS_MAPPING.update(this_dic)
    STRANGE_NUMBERS_REPLACE.update(replace_dic)

STRANGE_NUMBERS_MAPPING = {k: v for k, v in STRANGE_NUMBERS_MAPPING.items() if k != v}
STRANGE_NUMBERS_REPLACE = {k: v for k, v in STRANGE_NUMBERS_REPLACE.items() if k != v}

OTHER_MAPPING = {
    "≋": "-",
    "░": "-",
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "』": "]",
    "『": "[",
    "。": ".",
    "，": ",",
    "↊": "2",
    "∠": "7",
    "Ɩ": "1",
    "↋": "3",
    ":": "-",
    "‘": "'",
    "’": "'",
    "‑": "-"
}

if __name__ == '__main__':
    print(STRANGE_NUMBERS_REPLACE)
    print(STRANGE_NUMBERS_MAPPING)
    print("‑" == "-")
