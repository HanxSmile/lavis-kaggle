from typing import Dict
from sentencepiece import SentencePieceProcessor
from tqdm import trange
from collections import OrderedDict


class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, src_model: str, tgt_model: str):
        # Get SentencePiece
        self.src_sp = SentencePieceProcessor()
        self.src_sp.Load(src_model)

        self.tgt_sp = SentencePieceProcessor()
        self.tgt_sp.Load(tgt_model)

    def extract(self) -> Dict[str, int]:
        src_sp = self.src_sp
        src_vocab = [src_sp.id_to_piece(index) for index in trange(src_sp.GetPieceSize())]

        tgt_sp = self.tgt_sp
        tgt_vocab = [tgt_sp.id_to_piece(index) for index in trange(tgt_sp.GetPieceSize())]

        assert "</s>" in src_vocab and "<unk>" in src_vocab

        for token in ("<s>", "</s>", "<unk>", "<pad>"):
            if token in src_vocab:
                src_vocab.remove(token)
            if token in tgt_vocab:
                tgt_vocab.remove(token)

        vocab = src_vocab + tgt_vocab
        vocab_dic = OrderedDict()
        for token in vocab:
            vocab_dic[token] = None
        vocab = ["</s>", "<unk>"] + list(vocab_dic.keys()) + ["<pad>"]
        vocab_dic = {token: index for index, token in enumerate(vocab)}

        return vocab_dic
