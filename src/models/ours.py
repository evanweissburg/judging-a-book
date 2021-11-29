from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src import models, preprocess
from src.models.base import _set_determinism

RowTokens = List[torch.Tensor]
RowPairs = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class OursModelConfig(models.ModelConfig):
    model_type: models.ModelType = models.ModelType.Ours
    lr: float = 1e-5
    weight_decay: float = 3e-5
    epochs: int = 25
    batch_size: Optional[int] = None

    embedding_size: int = 300
    n_heads: int = 1
    dropout: float = 0.3
    kernel_size = 5

    def build(self, device: str) -> List[Tuple[models.Model, float]]:
        name = f"{self.model_type}_{self.subreddit}"

        torch_device = torch.device(device)
        torch.cuda.set_device(device)
        _set_determinism()

        rows, pairs = preprocess.load(self.subreddit)
        row_tokens, _, max_sentence_len = _tokenize(rows, name)
        row_pairs = _pair_rows(rows, row_tokens, pairs, name)
        dataset = Dataset(row_pairs)
        build_model = lambda: OursModel(self, max_sentence_len).to(device)
        return self.training_scheme(torch_device, self, build_model, dataset, name)


def _tokenize(rows: pd.DataFrame, name: str) -> Tuple[RowTokens, List[str], int]:
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

    row_tokens = []
    title_tokens = []
    max_sentence_len = -1
    for doc in tqdm(
        nlp.pipe(rows.loc[:, "title"], batch_size=64),
        total=len(rows),
        desc=f"{name}: Tokenizing",
    ):
        vec = torch.stack(
            [torch.from_numpy(e.vector) for e in doc],
            dim=0,
        )[:, None, :]
        max_sentence_len = max(max_sentence_len, vec.shape[0])
        row_tokens.append(vec)
        title_tokens.append([e.text for e in doc])
    return row_tokens, title_tokens, max_sentence_len


def _pair_rows(
    rows: pd.DataFrame, row_tokens: RowTokens, pairs: preprocess.Pairs, name: str
) -> RowPairs:
    data = []
    for x, y in tqdm(pairs, desc=f"{name}: Pairing rows"):
        x_tensor = row_tokens[x].clone().float()
        y_tensor = row_tokens[y].clone().float()
        better = torch.tensor(
            [1 if rows.iloc[x]["score"] > rows.iloc[y]["score"] else -1],
            dtype=torch.float32,
        )
        data.append((x_tensor, y_tensor, better))
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class OursModel(torch.nn.Module):
    def __init__(self, config: OursModelConfig, max_sentence_len: int):
        super().__init__()

        CONV_OUT_TARGET_DIM = max_sentence_len - config.kernel_size + 1
        self._max_sentence_len = max_sentence_len

        self.attn = torch.nn.MultiheadAttention(300, config.n_heads, config.dropout)
        self.conv1d = torch.nn.Conv1d(300, 1, config.kernel_size)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(CONV_OUT_TARGET_DIM, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
        )

    def forward(self, title):
        x = self.attn(title, title, title)[0]
        x = x.permute(1, 2, 0)
        x = F.pad(x, pad=(0, self._max_sentence_len - x.shape[2]))
        x = self.conv1d(x)
        return self.linear(x.flatten())
