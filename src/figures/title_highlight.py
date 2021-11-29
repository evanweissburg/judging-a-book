import pathlib
import pickle

import click
import numpy as np
import spacy
import torch

from src.helpers import constants
from src.models import TrainAll
from src.models.ours import OursModel


def _process_title(title: str, model: OursModel):
    # TODO what if sentence is too long?
    if "nlp" not in globals():
        nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

    title_tokens = nlp(title)
    title_vec = torch.stack([torch.from_numpy(e.vector) for e in title_tokens], dim=0)[
        :, None, :
    ].float()

    weights = model.attn(title_vec, title_vec, title_vec)[1].detach().numpy()[0]
    return [e.text for e in title_tokens], weights


def _compute_input_weights(title, weights):
    """Gather attention weights for a single word"""
    n = len(title)
    input_weights = np.zeros(n, dtype=np.float32)

    for output_idx in range(n):
        for input_idx in range(n):
            input_weights[input_idx] += weights[output_idx][input_idx]

    normed_input_weights = np.interp(
        input_weights, (input_weights.min(), input_weights.max()), (0.05, 0.5)
    )

    return normed_input_weights


def _visualize(title_tokens, input_weights, output_path):
    header = """
    <!-- Font -->
    <link href="https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@300&display=swap" rel="stylesheet">

    <!-- Paragraph styling -->
    <style>

    .highlight-sentence {
        font-size:18px;
        font-family: 'Zilla Slab', serif;
        line-height: 150%;
    }

    .bordered {
    border: 0.5px;
    border-style: dashed;
    border-color: rgba(0,0,0,0.2);
    padding: 5px;
    display: inline-block;
    width: 34em;
    }
    </style>
    """

    def style_word(word, weight):
        return (
            f'<span style="background-color:rgba(125,140,196,{weight})">{word}</span>'
        )

    content = '<div class="bordered">{}</div>'.format(
        " ".join(
            style_word(word, weight)
            for word, weight in zip(title_tokens, input_weights)
        )
    )

    html = header + content
    output_path.write_text(html)


@click.command()
@click.option("--title", type=str, required=True, help="Title to visualize")
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
    help="Output path for visualization.",
)
def generate(title: str, model_name: str, output_name: str):
    output_path = constants.FIGURES_DIR / f"{output_name}.html"

    with (constants.MODELS_DIR / f"{model_name}.model").open("rb") as f:
        wrapper = pickle.load(f)
        model = wrapper.model
        config = wrapper.config
        assert isinstance(
            config.training_scheme, TrainAll
        ), "Top k models should be TrainAll!"

    title_tokens, attn_weights = _process_title(title, model)
    input_weights = _compute_input_weights(title_tokens, attn_weights)
    _visualize(title_tokens, input_weights, output_path)


if __name__ == "__main__":
    generate()
