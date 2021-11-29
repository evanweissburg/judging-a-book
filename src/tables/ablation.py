import statistics
from collections import defaultdict
from typing import List, Tuple

import click
import pandas as pd

from src.helpers import constants, scheduler
from src.models import KFold, Model, OursAblateModelConfig, OursModelConfig


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
@click.option(
    "--subreddit",
    type=str,
    default="travel",
    help="Output path for visualization.",
)
def generate(output_name: str, subreddit: str):
    output_path = constants.TABLES_DIR / f"{output_name}.csv"
    assert not output_path.exists()

    our_config = OursModelConfig(
        subreddit=subreddit, training_scheme=KFold(num_folds=5)
    )
    ablate_config = OursAblateModelConfig(
        subreddit=subreddit, training_scheme=KFold(num_folds=5)
    )

    models = scheduler.train([our_config, ablate_config])

    table = {
        "Our Model": compute_statistics(models[0]),
        "Our Model, without CNN": compute_statistics(models[1]),
    }

    df = pd.DataFrame(table)
    df.to_csv(output_path)


if __name__ == "__main__":
    generate()
