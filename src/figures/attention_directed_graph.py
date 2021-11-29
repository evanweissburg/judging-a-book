import heapq
import pathlib
import pickle
import string
from collections import defaultdict
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from pydantic import BaseModel
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm.auto import tqdm

from src import preprocess
from src.helpers import constants
from src.models import TrainAll, ours

BAD_WORDS = (
    STOP_WORDS
    | set(string.punctuation)
    | {"‘", '"', "’", "“", "”", "–", "...", "nt", "10", "20"}
)


class ADGConfig(BaseModel):
    num_top_words: int = 20
    num_connections: int = 3
    node_size_range: Tuple[int, int] = (300, 3000)
    edge_size_range: Tuple[float, float] = (0.0, 0.8)  # between zero and one


DEFAULTS = {
    "soccer": ADGConfig(
        num_top_words=30,
    ),
}


def build_config(subreddit: str) -> ADGConfig:
    assert subreddit in DEFAULTS, "No config found. Please add it to DEFAULTS!"
    return DEFAULTS[subreddit]


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


def _compute_graph(weights, config: ADGConfig):
    word_to_words_weight = defaultdict(lambda: defaultdict(int))
    word_to_word_freq = defaultdict(int)

    for title, title_weights in tqdm(weights, desc="Calculating word weights"):
        for i, out_word in enumerate(title):
            out_word = out_word.lower()
            if out_word not in BAD_WORDS:
                for j, in_word in enumerate(title):
                    in_word = in_word.lower()
                    if in_word not in BAD_WORDS:
                        word_to_words_weight[out_word][in_word] += title_weights[i][j]
                word_to_word_freq[out_word] += 1

    word_to_words_avg_weight = defaultdict(dict)
    for out_word in tqdm(word_to_words_weight, desc="Normalizing"):
        for in_word in word_to_words_weight[out_word]:
            word_to_words_avg_weight[out_word][in_word] = (
                word_to_words_weight[out_word][in_word] / word_to_word_freq[out_word]
            )

    # This will be populated soon!
    nodes = []  # (node, frequency)
    edges = []  # (x, y, weight)

    # Generate the graph, solely of words that "make the cut"/
    _node_names = set()

    _top_words = heapq.nlargest(
        config.num_top_words, word_to_word_freq.keys(), key=word_to_word_freq.get
    )

    for out_word in _top_words:
        _connections = heapq.nlargest(
            config.num_connections,
            set(word_to_words_avg_weight[out_word].keys()) - {out_word},
            key=word_to_words_avg_weight[out_word].get,
        )

        for in_word in _connections:
            weight = word_to_words_avg_weight[out_word][in_word]
            edges.append((out_word, in_word, weight))
            _node_names |= {out_word, in_word}

    # Compute all mentioned nodes
    nodes = [(n, word_to_word_freq[n]) for n in _node_names]

    # Normalize nodes:
    _node_weights = np.array([w for n, w in nodes], dtype=np.float32)
    _node_weights = np.interp(
        _node_weights,
        (_node_weights.min(), _node_weights.max()),
        config.node_size_range,
    )
    nodes = [(n, w) for (n, _), w in zip(nodes, _node_weights)]

    # Normalize edges:
    _edge_weights = np.array([w for _, _, w in edges], dtype=np.float32)
    _edge_weights = np.interp(
        _edge_weights,
        (_edge_weights.min(), _edge_weights.max()),
        config.edge_size_range,
    )
    edges = [(a, b, w) for (a, b, _), w in zip(edges, _edge_weights)]

    return nodes, edges


def _visualize(nodes, edges, output_path: str):
    k = 5
    nodes = nodes[k:] + nodes[:k]

    G = nx.DiGraph()
    G.add_nodes_from([n for n, _ in nodes])
    G.add_weighted_edges_from(edges)

    plt.figure(1, figsize=(8, 8), dpi=200)

    nx.draw(
        G,
        nx.circular_layout(G),
        node_size=[w for _, w in nodes],
        edge_color=[(w, w, w) for _, _, w in edges],
        with_labels=True,
        font_size=15,
        alpha=0.9,
        node_color="#B4CDED",
        arrowstyle="->",
    )

    plt.savefig(output_path)


@click.command()
@click.option(
    "--subreddit",
    type=str,
    required=True,
    help="Subreddit we are running on (for config)",
)
@click.option(
    "--model_name",
    type=str,
    required=True,
    help="Model to load for processing title.",
)
@click.option(
    "--output_name",
    type=str,
    required=True,
    help="Output file name for visualization.",
)
def generate(subreddit: str, model_name: str, output_name: str):
    output_path = constants.FIGURES_DIR / output_name
    config = build_config(subreddit)

    weights = generate_attention_weights(model_name)
    nodes, edges = _compute_graph(weights, config)
    _visualize(nodes, edges, output_path)


if __name__ == "__main__":
    generate()
