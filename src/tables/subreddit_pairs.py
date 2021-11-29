import click
import pandas as pd

from src import preprocess
from src.helpers import constants


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

    subreddits = []
    for subreddit in constants.DEFAULT_SUBREDDITS:
        _, pairs = preprocess.load(subreddit)
        subreddits.append([subreddit, len(pairs)])

    df = pd.DataFrame(subreddits)
    df.to_csv(output_path)


if __name__ == "__main__":
    generate()
