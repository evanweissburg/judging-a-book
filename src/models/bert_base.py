from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer

from src import models, preprocess
from src.models.base import _set_determinism

RowTokens = List[torch.Tensor]
RowPairs = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class BertBaseModelConfig(models.ModelConfig):
    model_type: models.ModelType = models.ModelType.BertBase
    lr: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 35
    batch_size: Optional[int] = 64

    def build(self, device: str) -> List[Tuple[models.Model, float]]:
        name = f"{self.model_type}_{self.subreddit}"

        torch_device = torch.device(device)
        torch.cuda.set_device(device)
        _set_determinism()

        rows, pairs = preprocess.load(self.subreddit)
        row_tokens = _tokenize(rows, torch_device, name)
        row_pairs = _pair_rows(rows, row_tokens, pairs, name)
        dataset = Dataset(row_pairs)
        def build_model(): return BertBaseModel().to(torch_device)
        return self.training_scheme(torch_device, self, build_model, dataset, name)


def _tokenize(rows: pd.DataFrame, device: torch.device, name: str) -> torch.Tensor:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    row_tokens = torch.zeros([len(rows), 768], dtype=torch.float32)
    with torch.no_grad():
        for i, title in enumerate(tqdm(rows.title, desc=f"{name}: Tokenizing (BERT)")):
            tokens = tokenizer.encode(
                title, truncation=True, padding="max_length", return_tensors="pt"
            )
            row_tokens[i] = model(
                tokens.to(device)).last_hidden_state.mean(dim=[0, 1])
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


class BertBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1),
        )

    def forward(self, title):
        return self.ff(title)
