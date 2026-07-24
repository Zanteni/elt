import yaml

import copy


# ============================================================================
# YAML
# ============================================================================

def load_yaml(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path:
            Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        yaml.YAMLError:
            If the YAML syntax is invalid.
    """

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def merge_configs(default_cfg: dict, stage_cfg: dict) -> dict:
    """
    Recursively merge two configuration dictionaries.

    Values in stage_cfg override values in default_cfg.

    Args:
        default_cfg:
            Base configuration.

        stage_cfg:
            Configuration overriding the defaults.

    Returns:
        Merged configuration dictionary.
    """

    merged = copy.deepcopy(default_cfg)

    for key, value in stage_cfg.items():

        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):

            merged[key] = merge_configs(
                merged[key],
                value,
            )

        else:

            merged[key] = value

    return merged

def load_config(
    default_path: str,
    stage_path: str,
) -> dict:
    """
    Load and merge configuration files.

    Args:
        default_path:
            Path to the shared default configuration.

        stage_path:
            Path to the stage-specific configuration.

    Returns:
        Merged configuration dictionary.
    """

    default_cfg = load_yaml(default_path)

    stage_cfg = load_yaml(stage_path)

    return merge_configs(
        default_cfg,
        stage_cfg,
    )

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed:
            Random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def count_parameters(
    model,
    trainable_only: bool = False,
):
    """
    Count model parameters.

    Args:
        model:
            PyTorch model.

        trainable_only:
            If True, count only parameters that require gradients.

    Returns:
        Number of parameters.
    """

    if trainable_only:

        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    return sum(
        p.numel()
        for p in model.parameters()
    )

class AverageMeter:
    """
    Tracks the running average of a scalar quantity.

    Example:
        meter = AverageMeter()

        meter.update(2.0)
        meter.update(4.0)

        print(meter.avg)   # 3.0
    """

    def __init__(self):
        self.reset()


    def reset(self):
        """
        Reset all statistics.
        """

        self.sum = 0.0
        self.count = 0
        self.avg = 0.0


    def update(
        self,
        value: float,
        n: int = 1,
    ):
        """
        Add a new observation.

        Args:
            value:
                Scalar value.

            n:
                Number of samples represented by value.
        """

        self.sum += value * n

        self.count += n

        self.avg = self.sum / self.count