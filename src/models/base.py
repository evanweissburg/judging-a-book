import abc
import enum
import operator
import pickle
from functools import reduce
from typing import Callable, Generator, List, Optional, Tuple

import numpy as np
import pydantic
import torch
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm

from src.helpers import constants


class ModelType(str, enum.Enum):
    Logistic1Hot = "Logistic1Hot"
    GloveMLP = "GloveMLP"
    BiLSTM = "BiLSTM"
    BertBase = "BertBase"
    Ours = "Ours"
    OursAblate = "OursAblate"


class Model:
    def __init__(
        self, trained_model: torch.nn.Module, config: pydantic.BaseConfig, name: str
    ) -> None:
        self.model = trained_model
        self.config = config
        self.name = name

    def forward(self, x):
        return self.model.forward(x)

    def write(self) -> None:
        with (constants.MODELS_DIR / f"{self.name}.model").open("wb") as f:
            pickle.dump(self, f)


class TrainingScheme(pydantic.BaseModel, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        device: torch.device,
        config: "ModelConfig",
        build_model: Callable[[], torch.nn.Module],
        dataset: Dataset,
        name: str,
    ) -> List[Tuple[Model, float]]:
        pass


class TrainAll(TrainingScheme):
    def __call__(
        self,
        device: torch.device,
        config: "ModelConfig",
        build_model: Callable[[], torch.nn.Module],
        dataset: Dataset,
        name: str,
    ) -> List[Tuple[Model, float]]:
        trained_model, accuracy = _train_fold(
            build_model(), device, config, dataset, None, name
        )
        return [(Model(trained_model, config, name), accuracy)]


class EightyTwenty(TrainingScheme):
    def __call__(
        self,
        device: torch.device,
        config: "ModelConfig",
        build_model: Callable[[], torch.nn.Module],
        dataset: Dataset,
        name: str,
    ) -> List[Tuple[Model, float]]:
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])
        trained_model, accuracy = _train_fold(
            build_model(), device, config, train_set, test_set, name
        )
        return [(Model(trained_model, config, name), accuracy)]


class KFold(TrainingScheme):
    num_folds: pydantic.conint(ge=1)

    def __call__(
        self,
        device: torch.device,
        config: "ModelConfig",
        build_model: Callable[[], torch.nn.Module],
        dataset: Dataset,
        name: str,
    ) -> List[Tuple[Model, float]]:
        results = []
        for train_set, test_set in self._split_dataset(dataset):
            trained_model, accuracy = _train_fold(
                build_model(), device, config, train_set, test_set, name
            )
            results.append((Model(trained_model, config, name), accuracy))
        return results

    def _split_dataset(
        self,
        dataset: Dataset,
    ) -> Generator[Tuple[Dataset, Dataset], None, None]:
        splits = [len(dataset) // self.num_folds] * self.num_folds
        for i in range(len(dataset) - sum(splits)):
            splits[i] += 1  # use the rows that don't fit mod n
        data_splits = random_split(dataset, splits)

        for fold in range(self.num_folds):
            train_set = reduce(
                operator.add, (t for i, t in enumerate(data_splits) if i != fold)
            )
            test_set = data_splits[fold]
            yield train_set, test_set


class ModelConfig(pydantic.BaseModel):
    model_type: ModelType
    subreddit: str
    lr: float
    weight_decay: float
    epochs: int
    batch_size: Optional[int] = None
    n_dl_workers: int = 0
    training_scheme: TrainingScheme = EightyTwenty()

    @abc.abstractmethod
    def build(self, device: str) -> List[Tuple[Model, float]]:
        pass


######################
# Training Utilities #
######################


def _train_fold(
    model: torch.nn.Module,
    device: torch.device,
    config: ModelConfig,
    train_set: Dataset,
    test_set: Optional[Dataset],
    name: str,
) -> Tuple[torch.nn.Module, float]:
    # Init model, loss, optimizer
    loss = torch.nn.MarginRankingLoss(margin=1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Init train / test data split
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.n_dl_workers,
    )
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.n_dl_workers,
        )

    best_accuracy = float("-inf")
    for epoch in range(config.epochs):
        total_loss = 0
        train_count = 0

        for title1, title2, pop in tqdm(
            train_loader,
            leave=False,
            desc=f"{name}: Training epoch {epoch + 1} of {config.epochs}",
        ):
            optimizer.zero_grad()

            pred1 = model.forward(title1.to(device).float())
            pred2 = model.forward(title2.to(device).float())

            cur_loss = loss(pred1, pred2, pop.to(device).float())
            cur_loss.backward()

            optimizer.step()
            total_loss += cur_loss
            train_count += 1

        if test_set is None:
            tqdm.write(
                f"{name}: Finished epoch {epoch + 1} of {config.epochs} with loss {total_loss / train_count:.4f}"
            )
            continue  # no test set to test with!

        with torch.no_grad():
            correct = 0
            total = 0
            for title1, title2, pop in tqdm(
                test_loader,
                leave=False,
                desc=f"{name}: Testing epoch {epoch + 1} of {config.epochs}",
            ):
                pred1 = model.forward(title1.to(device).float())
                pred2 = model.forward(title2.to(device).float())
                correct += (
                    ((pred1 - pred2).sign() == pop.to(device).sign()).sum().item()
                )
                total += config.batch_size if config.batch_size else 1

        tqdm.write(
            f"{name}: Finished epoch {epoch + 1} of {config.epochs} with loss {total_loss / train_count:.4f} and accuracy {correct / total:.4f}"
        )

        best_accuracy = max(best_accuracy, correct / total)

    if test_set is not None:
        tqdm.write(f"{name}: Best accuracy was {best_accuracy}\n")

    del optimizer, loss, train_loader
    if test_set is not None:
        del test_loader

    return model.to("cpu"), best_accuracy


def _set_determinism():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')
