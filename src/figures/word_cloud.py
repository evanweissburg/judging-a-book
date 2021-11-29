"""
Generate the wordgraphs.
"""

from multiprocessing import Pool
from random import sample
from typing import Counter

import click
import pandas as pd
import spacy
from tqdm.auto import tqdm
from wordcloud import STOPWORDS, WordCloud
from heapq import nlargest

from src.helpers import constants


def gather_scores(subreddit):
    q, n = Counter(), Counter()
    rows = pd.read_csv(
        constants.SUBREDDITS_DIR / "{}.posts".format(subreddit),
        low_memory=False,
    ).sort_values("score")

    if "nlp" not in globals():
        nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    for i, doc in enumerate(nlp.pipe(rows.loc[:, "title"], batch_size=256)):
        for word in doc:
            if word.is_stop or word.is_punct:
                continue
            cleaned = word.lemma_.lower()
            q[cleaned] += rows.score[i]
            n[cleaned] += 1

    norm_factor = sum(q.values())
    for k in q:
        q[k] /= norm_factor

    return q, n


def _visualize(output_name: str):
    subreddits = constants.DEFAULT_SUBREDDITS

    # GATHER WORD SCORES
    Q, N = Counter(), Counter()
    with Pool(processes=8) as p:
        for q, n in tqdm(
            p.imap(gather_scores, subreddits),
            total=len(subreddits),
        ):
            Q.update(q)
            N.update(n)

    word_scores = {w: Q[w] / N[w] for w in Q if N[w] >= 25}
    c1, c2, c3 = pd.Series(word_scores).quantile([0.25, 0.50, 0.75])

    # SPLIT INTO QUARTILES:
    pbar = tqdm(desc="generating word cloud", total=4 * 5)
    for i in range(4):
        l, u = [float("-inf"), c1, c2, c3, float("inf")][i : i + 2]
        quartile = {k: v for k, v in word_scores.items() if l <= v <= u}
        for k in range(5):
            WordCloud(
                width=1000,
                height=1000,
                max_words=150,
                stopwords=STOPWORDS,
                background_color="white",
                colormap="tab20",
            ).generate_from_frequencies(
                # {k: quartile[k] for k in sample(list(quartile), 150)}
                {k: quartile[k] for k in nlargest(150, quartile, key=quartile.get)}
            ).to_file(
                constants.FIGURES_DIR
                / output_name
                / f"wordcloud-q{i+1}-k{k+1}.png"
            )
            pbar.update()


@click.command()
@click.option(
    "--output_name",
    type=str,
    required=True,
    help="Output path for visualization.",
)
def generate(output_name: str):
    (constants.FIGURES_DIR / output_name).mkdir(exist_ok=True)
    _visualize(output_name)


if __name__ == "__main__":
    generate()
