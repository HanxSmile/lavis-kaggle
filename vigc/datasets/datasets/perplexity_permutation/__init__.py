from torch.utils.data import Dataset as torch_Dataset
from itertools import permutations


class PerplexityPermutationDataset(torch_Dataset):
    def __init__(
            self,
            initial_text,
            permutation_idx,
    ):
        self.words = [_.strip() for _ in initial_text.strip().split(" ") if _.strip()]
        self.permutation_idx = permutation_idx
        self.inner_dataset = list(permutations(permutation_idx))
        self.frozen_index = {idx: idx for idx in range(len(self.words)) if idx not in permutation_idx}

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        index_mapping = {}
        index_mapping.update(self.frozen_index.copy())
        this_permutation = self.inner_dataset[index]
        unfrozen_index = {p1: p2 for p1, p2 in zip(self.permutation_idx, this_permutation)}
        index_mapping.update(unfrozen_index)

        words = [self.words[index_mapping[_]] for _ in range(len(self.words))]
        result = " ".join(words).strip()

        return {
            "id": str(index),
            "text": result,
        }

    def collater(self, batch):
        ids = [_["id"] for _ in batch]
        texts = [_["text"] for _ in batch]

        return {
            "id": ids,
            "text": texts,
        }
