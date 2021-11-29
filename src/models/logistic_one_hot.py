import heapq
import pickle
from collections import defaultdict
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm.auto import tqdm

from src import models, preprocess
from src.models.base import _set_determinism

BANNED = {"megathread", "ama", "thread", ""}
RowTokens = List[List[str]]
RowPairs = List[Tuple[List[int], List[int], List[int]]]


class Logistic1HotModelConfig(models.ModelConfig):
    model_type: models.ModelType = models.ModelType.Logistic1Hot
    lr: float = 1e-5
    weight_decay: float = 3e-5
    epochs: int = 25
    vocab_size: int = 20000
    batch_size: int = 64

    def build(self, device: str) -> List[Tuple[models.Model, float]]:
        name = f"{self.model_type}_{self.subreddit}"

        torch_device = torch.device(device)
        torch.cuda.set_device(device)
        _set_determinism()

        rows, pairs = preprocess.load(self.subreddit)
        row_tokens = _tokenize(rows, name)
        vocab = _build_vocab(self, row_tokens)
        row_pairs = _pair_rows(vocab, rows, row_tokens, pairs, name)
        dataset = Dataset(row_pairs, self.vocab_size)
        def build_model(): return Logistic1HotModel(self.vocab_size).to(device)
        return self.training_scheme(torch_device, self, build_model, dataset, name)


def _parse_word(word: str) -> str:
    return "".join(c if c.isalpha() else "#" for c in filter(str.isalnum, word.lower()))


def _tokenize(rows: pd.DataFrame, name: str) -> RowTokens:
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

    row_tokens = []
    for doc in tqdm(
        nlp.pipe(rows.loc[:, "title"], batch_size=64),
        total=len(rows),
        desc=f"{name}: Tokenizing",
    ):
        tokens = (_parse_word(t.text) for t in doc if not t.is_stop)
        row_tokens.append([t for t in tokens if t not in BANNED])
    return row_tokens


def _build_vocab(config: Logistic1HotModelConfig, row_tokens: RowTokens) -> Set[str]:
    word_freq = defaultdict(int)
    for title in row_tokens:
        for word in title:
            word_freq[word] += 1

    return {
        w: i
        for i, w in enumerate(
            heapq.nlargest(config.vocab_size, word_freq.keys(), word_freq.get)
        )
    }


def _pair_rows(
    vocab: Set[str],
    rows: pd.DataFrame,
    row_tokens: RowTokens,
    pairs: preprocess.Pairs,
    name: str,
) -> RowPairs:
    data = []
    for x, y in tqdm(pairs, desc=f"{name}: Pairing rows"):
        x_sparse = [vocab[w] for w in row_tokens[x] if w in vocab]
        y_sparse = [vocab[w] for w in row_tokens[y] if w in vocab]
        better = [1 if rows.iloc[x]["score"] > rows.iloc[y]["score"] else -1]
        data.append((x_sparse, y_sparse, better))
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: RowPairs, vocab_size: int):
        self._data = data
        self._vocab_size = vocab_size

    def __len__(self):
        return len(self._data)

    def _dense(self, sparse: List[int]) -> torch.Tensor:
        ret = torch.zeros(self._vocab_size, dtype=torch.float64)
        for i in sparse:
            ret[i] = 1.0
        return ret

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_sparse, y_sparse, better = self._data[idx]
        return (
            self._dense(x_sparse),
            self._dense(y_sparse),
            torch.tensor(better, dtype=torch.float64),
        )


class Logistic1HotModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.ff = torch.nn.Linear(vocab_size, 1)

    def forward(self, x):
        return self.ff(x)
