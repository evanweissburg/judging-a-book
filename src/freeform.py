from src.helpers import constants, scheduler
from src.models import GloveMLPModelConfig, KFold, TrainAll
from src.models.logistic_one_hot import Logistic1HotModelConfig
from src.models.ours import OursModelConfig


def main():
    for model_folds in scheduler.train(
        [
            OursModelConfig(
                subreddit="depression",
                training_scheme=TrainAll(),
                epochs=25,
            ),
        ]
    ):
        model, _ = model_folds[0]
        model.write()


if __name__ == "__main__":
    main()
