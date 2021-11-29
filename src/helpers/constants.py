import getpass
import os
import pathlib

user = getpass.getuser()
remote_mode = {"evancw", "arkumar"}

if user in remote_mode:
    BASE_DIR = pathlib.Path(f"/home/data-master/{user}/judging-a-book/")
else:
    BASE_DIR = pathlib.Path(os.path.dirname(__file__)).parent.parent.absolute()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SUBREDDITS_DIR = DATA_DIR / "subreddits"

MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

if user in remote_mode:
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
    DEVICES = {f"cuda:{n}" for n in [1, 2, 3, 6, 7]}

else:
    DATASET_PATHS = [f"RS_2017-{month:02}" for month in range(1, 4)]
    DEFAULT_SUBREDDITS = {"soccer", "sports"}
    DEVICES = {"cpu"}

PAIRING_INTERVAL_RANGE_S = 60 * 30  # 30 mins, like cats + captions
