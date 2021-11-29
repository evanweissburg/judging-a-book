import heapq
import math
import pathlib
import pickle
import string
from collections import defaultdict
from typing import List, Tuple

import click
import pandas as pd
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm.auto import tqdm

from src import preprocess
from src.helpers import constants
from src.models import ours
from src.models.base import TrainAll


def generate_attention_weights(name: str) -> List[Tuple[str, torch.Tensor]]:
    device = next(iter(constants.DEVICES))  # Pick any GPU.
    with (constants.MODELS_DIR / f"{name}.model").open("rb") as f:
        wrapper = pickle.load(f)
        model = wrapper.model.to(device)
        config = wrapper.config
        assert isinstance(
            config.training_scheme, TrainAll
        ), "Top k models should be TrainAll!"

    rows, _ = preprocess.load(config.subreddit)
    row_tokens, title_tokens, _ = ours._tokenize(rows, config.subreddit)

    result = []
    model.eval()  # Disable dropout
    with torch.no_grad():
        for title, vec in zip(
            title_tokens, tqdm(row_tokens, desc="Collecting weights")
        ):
            vec = vec.to(device).float()
            vec = model.attn(vec, vec, vec)[1].cpu().numpy()
            result.append((title, vec[0]))  # vec is 1 x n x n
    return result


def get_top_k_ours(name: str, k: int):
    BAD_WORDS = (
        STOP_WORDS | set([e for e in string.punctuation]) | {"‘", "’", "-", "--", "---"}
    )
    attention_weights = generate_attention_weights(name)

    word_to_weight = defaultdict(int)
    word_to_freq = defaultdict(int)

    for title, title_weights in attention_weights:
        for i, word in enumerate(title):
            word = word.lower()
            if word not in BAD_WORDS:
                word_to_weight[word] += sum([weights[i] for weights in title_weights])
                word_to_freq[word] += 1

    word_to_avg_weight = dict()
    for word in word_to_weight:
        if word_to_freq[word] > 25:
            word_to_avg_weight[word] = word_to_weight[word] / word_to_freq[word]

    word_to_avg_weight = list(word_to_avg_weight.items())
    word_to_avg_weight.sort(reverse=True, key=lambda x: x[1])

    return word_to_avg_weight[:k]


@click.command()
@click.argument(
    "models",
    type=str,
    nargs=-1,
)
@click.option(
    "--k",
    type=int,
    required=True,
    help="Number of words.",
)
@click.option(
    "--output_name",
    type=str,
    required=True,
    help="Output path for visualization.",
)
def generate(k: int, output_name: str, models: List[str]):
    output_path = constants.TABLES_DIR / f"{output_name}.csv"
    assert not output_path.exists()

    top_ks = [get_top_k_ours(model_name, k) for model_name in models]
    df = pd.DataFrame(list(zip(*top_ks)), columns=models)
    print(df)
    df.to_csv(output_path)


if __name__ == "__main__":
    generate()
