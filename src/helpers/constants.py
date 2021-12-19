import getpass
import os
import pathlib

BASE_DIR = pathlib.Path(os.path.dirname(__file__)).parent.parent.absolute()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SUBREDDITS_DIR = DATA_DIR / "subreddits"

MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

DATASET_PATHS = [f"RS_2017-{month:02}" for month in range(1, 13)]
DEFAULT_SUBREDDITS = {
    "Showerthoughts",
    "AskReddit",
    "news",
    "worldnews",
    "relationships",
    "depression",
    "aww",
    "pics",
    "politics",
    "The_Donald",
    "sports",
    "soccer",
    "science",
    "NoStupidQuestions",
    "funny",
    "Jokes",
    "soccer",
    "sports",
}
DEVICES = {f"cuda:{n}" for n in [0, 1, 2, 3, 4, 5, 6, 7]}

PAIRING_INTERVAL_RANGE_S = 60 * 30  # 30 mins, like cats + captions