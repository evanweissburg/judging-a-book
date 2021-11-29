import statistics
from collections import defaultdict
from typing import List, Tuple

import click
import pandas as pd

from src.helpers import constants, scheduler
from src.models import (
    BertBaseModelConfig,
    BiLSTMModelConfig,
    GloveMLPModelConfig,
    KFold,
    Logistic1HotModelConfig,
    Model,
    OursModelConfig,
)


def compute_statistics(models: List[Tuple[Model, float]]) -> Tuple[float, float]:
    accuracies = [accuracy for _, accuracy in models]
    mean = statistics.mean(accuracies)
    standard_deviation = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
    return mean, standard_deviation


@click.command()
@click.option(
    "--output_name",
    type=str,
    required=True,
    help="Output path for visualization.",
)
def generate(output_name: str):
    output_path = constants.TABLES_DIR / f"{output_name}.csv"
    assert not output_path.exists()

    subreddits = constants.DEFAULT_SUBREDDITS

    model_types = [
        Logistic1HotModelConfig,
        GloveMLPModelConfig,
        BiLSTMModelConfig,
        BertBaseModelConfig,
        OursModelConfig,
    ]

    model_configs = [
        model_type(subreddit=subreddit, training_scheme=KFold(num_folds=5))
        for subreddit in subreddits
        for model_type in model_types
    ]

    models = scheduler.train(model_configs)

    table = defaultdict(dict)
    for model_type_idx, model_type in enumerate(model_types):
        for subreddit_idx, subreddit in enumerate(subreddits):
            accuracy, st_dev = compute_statistics(
                models[subreddit_idx * len(model_types) + model_type_idx]
            )
            table[model_type][subreddit] = (accuracy, st_dev)

    df = pd.DataFrame(table)
    df.to_csv(output_path)


if __name__ == "__main__":
    generate()
