import multiprocessing
import traceback
from typing import List

from src.helpers import constants
from src.models import ModelConfig
from src.preprocess import load, verify


def _verify_subreddits_preprocessed(configs: List[ModelConfig]):
    for config in configs:
        verify(config.subreddit)


def _train_on_device(
    free_devices: "multiprocessing.Manager.Queue[str]", config: ModelConfig
):
    device = free_devices.get(block=True, timeout=None)

    try:
        model = config.build(device)
    except Exception:
        print('Encountered an error while training!')
        print(config)
        print(traceback.format_exc())
    finally:
        free_devices.put(device)

    return model


def train(configs: List[ModelConfig]):
    _verify_subreddits_preprocessed(configs)

    # https://stackoverflow.com/questions/57657490/can-i-pass-queue-object-in-multiprocessing-pool-starmap-method
    free_devices = multiprocessing.Manager().Queue()
    for device in constants.DEVICES:
        free_devices.put(device)

    with multiprocessing.pool.ThreadPool(processes=len(constants.DEVICES)) as pool:
        models = pool.starmap(
            _train_on_device, [(free_devices, conf) for conf in configs]
        )

    return models
