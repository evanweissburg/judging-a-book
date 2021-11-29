import json
import pathlib
import pickle
import shutil
from collections import defaultdict
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Set, Tuple

import click
import pandas as pd
import requests
import zstandard
from tqdm import trange
from tqdm.auto import tqdm

from src.helpers import constants

Pairs = List[Tuple[int, int]]

COLUMNS = ["score", "title", "subreddit", "link", "timestamp"]
COLUMN_TYPES = ["int", "str", "str", "str", "int"]


def _download_url(stem: str) -> str:
    return f"https://files.pushshift.io/reddit/submissions/{stem}.zst"


def _zst_path(stem: str) -> pathlib.Path:
    return constants.RAW_DIR / f"{stem}.zst"


def _txt_path(stem: str) -> pathlib.Path:
    return constants.RAW_DIR / f"{stem}.txt"


def _posts_path(subreddit: str) -> pathlib.Path:
    return constants.SUBREDDITS_DIR / f"{subreddit}.posts"


def _pairs_path(subreddit: str) -> pathlib.Path:
    return constants.SUBREDDITS_DIR / f"{subreddit}.pairs"


def _maybe_build_dirs():
    constants.DATA_DIR.mkdir(parents=True, exist_ok=True)
    constants.SUBREDDITS_DIR.mkdir(parents=True, exist_ok=True)
    constants.RAW_DIR.mkdir(parents=True, exist_ok=True)
    constants.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    constants.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    constants.TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _download_zst_file(stem: str) -> None:
    print(f"Downloading data for {stem}")
    url, zst_path = _download_url(stem), _zst_path(stem)
    with requests.get(url, stream=True) as r:
        assert r.ok, f"Error downloading file from {url}."
        with zst_path.open("wb") as f:
            shutil.copyfileobj(r.raw, f)


def _decompress_txt_file(stem: str):
    print(f"Decompressing data for {stem}")
    zst_path, txt_path = _zst_path(stem), _txt_path(stem)
    with zst_path.open("rb") as ifh, txt_path.open("wb") as ofh:
        zstandard.ZstdDecompressor(max_window_size=2 ** 31).copy_stream(ifh, ofh)
    zst_path.unlink()


def _maybe_download_dataset() -> None:
    stems = [stem for stem in constants.DATASET_PATHS if not (_txt_path(stem)).exists()]
    if not stems:
        return

    with Pool(len(stems)) as pool:
        pool.map(_download_zst_file, stems)

    for stem in stems:
        _decompress_txt_file(stem)


def _gather_stem_subreddit_posts(subreddits: Set[str], stem: str) -> Dict[str, list]:
    stem_subreddit_posts = defaultdict(list)
    for line in _txt_path(stem).open("r"):
        row = json.loads(line)
        if "subreddit" in row and row["subreddit"] in subreddits:
            if row["stickied"]:
                continue  # artificially inflated
            if row["score"] < 2:
                continue  # Match cats and captions
            stem_subreddit_posts[row["subreddit"]].append(
                [
                    row["score"],
                    row["title"],
                    row["subreddit"],
                    row["permalink"],
                    row["created_utc"],
                ]
            )
    return stem_subreddit_posts


def _gather_subreddit_posts(subreddits: List[str]) -> Dict[str, pd.DataFrame]:
    """Preprocess a subreddit, downloading the dataset if necessary."""
    subreddit_posts = defaultdict(list)
    print(f"Preprocessing subreddits {subreddits}")
    with Pool(len(constants.DATASET_PATHS)) as pool:
        for stem_subreddit_posts in pool.starmap(
            _gather_stem_subreddit_posts,
            [(subreddits, stem) for stem in constants.DATASET_PATHS],
        ):
            for subreddit in subreddits:
                subreddit_posts[subreddit].extend(stem_subreddit_posts[subreddit])

    subreddit_dfs = {}
    for subreddit in subreddits:
        print(f"Finalizing subreddit {subreddit}")
        subreddit_dfs[subreddit] = (
            pd.DataFrame(subreddit_posts[subreddit], columns=COLUMNS)
            .astype(dict(zip(COLUMNS, COLUMN_TYPES)))
            .sort_values(by="timestamp")
        )
        subreddit_dfs[subreddit].to_csv(_posts_path(subreddit))

    return subreddit_dfs


def _gather_single_subreddit_pairs(subreddit: str, posts: pd.DataFrame):
    pairs: Pairs = []
    matched = set()
    N = len(posts)

    for i in trange(N, desc=f"Pairing r/{subreddit} posts"):
        if i in matched:
            continue  # skip already paired i's

        i_score = posts.iloc[i]["score"]
        i_ts = posts.iloc[i]["timestamp"]

        for j in range(i + 1, N):
            if posts.iloc[j]["timestamp"] - i_ts > constants.PAIRING_INTERVAL_RANGE_S:
                break  # break if window is too large

            if j in matched:
                continue  # skip already paired j's

            j_score = posts.iloc[j]["score"]
            if i_score <= 2 * j_score and j_score <= 2 * i_score:
                continue  # one must be at least double the other

            if abs(i_score - j_score) < 20:
                continue  # difference must be at least 20

            pairs.append((i, j))
            matched |= {i, j}  # add to matched
            break

    with _pairs_path(subreddit).open("wb") as f:
        pickle.dump(pairs, f)
    return pairs


def _gather_subreddit_pairs(subs_to_posts: Dict[str, pd.DataFrame]) -> Dict[str, Pairs]:
    return {
        subreddit: _gather_single_subreddit_pairs(subreddit, posts)
        for subreddit, posts in subs_to_posts.items()
    }


@click.command(short_help="Preprocess a set of subreddits for future usage.")
@click.argument("subreddits", nargs=-1)
def preprocess(
    subreddits: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Pairs]]:
    subreddits = sorted(
        sub
        for sub in (subreddits or constants.DEFAULT_SUBREDDITS)
        if not (_posts_path(sub).exists() and _pairs_path(sub).exists())
    )
    _maybe_build_dirs()
    _maybe_download_dataset()
    posts = _gather_subreddit_posts(subreddits)
    pairs = _gather_subreddit_pairs(posts)
    return posts, pairs


def verify(subreddit: str):
    """Verify that the given subreddit has been preprocessed."""
    assert (
        _posts_path(subreddit).exists() and _pairs_path(subreddit).exists()
    ), f"{subreddit} has not been preprocessed; run `python -m src.preprocess {subreddit}`"


def load(subreddit: str) -> Tuple[pd.DataFrame, Pairs]:
    verify(subreddit)
    with _posts_path(subreddit).open("r") as f:
        posts = pd.read_csv(f)
    with _pairs_path(subreddit).open("rb") as f:
        pairs = pickle.load(f)
    return posts, pairs


if __name__ == "__main__":
    preprocess()
