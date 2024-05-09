import logging
import os
import shutil
import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate.logging import get_logger
from monotonic_align.core import maximum_path_c
from munch import Munch


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_list = f.readlines()
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_list = f.readlines()

    return train_list, val_list


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def _setup_logging(log_dir, logger_name, log_level="DEBUG"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = get_logger(logger_name, log_level)
    logger.setLevel(logging.DEBUG)

    # Create handlers for console and log file
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))

    # Create formatters and add it to handlers
    formatter = logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.logger.addHandler(console_handler)
    logger.logger.addHandler(file_handler)

    return logger


def configure_environment(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Setup logging
    log_dir = config["log_dir"]
    logger = _setup_logging(log_dir, __name__)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    return config, logger, log_dir
