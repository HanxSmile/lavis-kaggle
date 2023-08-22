import gensim
from bnunicodenormalizer import Normalizer

bnorm = Normalizer()


def normalize(sen):
    _words = [bnorm(word)['normalized'] for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


def dari(sentence):
    try:
        if sentence[-1] != "।":
            sentence += "।"
    except:
        print(sentence)
    return sentence


class BengaliSpellCorrection:
    all_prun = (
        '\u200d', ':', '—', '।', '"', '”', '?', '’', '“', '‘', '\t', '\u200c', '/', '\x94', '!', '\x93', '‚', '–', '॥',
        ';', '.', '৷', '…', ',', '-', '|', "'", '\n')

    def __init__(self, w2v_model_path, length_threshold=None):
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path)
        words = model.index_to_key
        w_rank = {}
        for i, word in enumerate(words):
            w_rank[word] = i
        self.WORDS = w_rank
        self.length_threshold = length_threshold

    def P(self, word):
        return - self.WORDS.get(word, 0)

    def correction(self, word):
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        letters = 'ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def correction_sen(self, s):
        len_th = self.length_threshold
        in_ = s.split(" ")
        in_ = [_.strip() for _ in in_]
        out_ = []
        for w in in_:
            start, end = "", ""
            while len(w) >= 1 and w[0] in self.all_prun:
                start += w[0]
                w = w[1:]
            while len(w) >= 1 and w[-1] in self.all_prun:
                end = w[-1] + end
                w = w[:-1]
            if len_th is None or (len_th is not None and len(w) <= len_th):
                w = self.correction(w)
            out_.append(start + w + end)
        res = " ".join(out_)
        return dari(normalize(res))
