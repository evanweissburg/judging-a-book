from typing import List, Optional, Tuple

import pandas as pd
import spacy
import torch
from torch._C import device
from tqdm.auto import tqdm

from src import models, preprocess
from src.models.base import _set_determinism

RowTokens = List[torch.Tensor]
RowPairs = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class BiLSTMModelConfig(models.ModelConfig):
    model_type: models.ModelType = models.ModelType.BiLSTM
    lr: float = 0.001
    weight_decay: float = 3e-5
    epochs: int = 25
    batch_size: Optional[int] = None

    def build(self, device: str) -> List[Tuple[models.Model, float]]:
        name = f"{self.model_type}_{self.subreddit}"

        torch_device = torch.device(device)
        torch.cuda.set_device(device)
        _set_determinism()

        rows, pairs = preprocess.load(self.subreddit)
        row_tokens = _tokenize(rows, name)
        row_pairs = _pair_rows(rows, row_tokens, pairs, name)
        dataset = Dataset(row_pairs)

        build_model = lambda: BiLSTMModel().to(torch_device)
        return self.training_scheme(torch_device, self, build_model, dataset, name)


def _tokenize(rows: pd.DataFrame, name: str) -> RowTokens:
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

    row_tokens = []
    for doc in tqdm(
        nlp.pipe(rows.loc[:, "title"], batch_size=64),
        total=len(rows),
        desc=f"{name}: Tokenizing",
    ):
        vec = torch.stack(
            [torch.from_numpy(e.vector) for e in doc],
            dim=0,
        )[:, None, :]
        row_tokens.append(vec)
    return row_tokens


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


class BiLSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(300, 300, bidirectional=True)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(300, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
        )

    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.ff(ht[-1])
