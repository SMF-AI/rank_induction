import json
import random
from typing import List, Tuple

import numpy as np
import torch
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator


def read_json(file_path) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def get_deepzoom_level(desired_mpp: float, slide: OpenSlide) -> int:
    """
    Args:
        desired_mpp (float): 타깃 MPP(microns per pixels)
        slide (OpenSlide): openslide object

    Returns:
        int: DeepZoom level

    Example:
        1) desired_mpp=0.24, slide_mpp=0.24, max_deepzoom_level=17
        => return 17
        2) desired_mpp=0.5, slide_mpp=0.24, max_deepzoom_level=17
        => return 16

    Raise:
        ValueError: desired_mpp is too small
    """

    try:
        slide_mpp: str = slide.properties["openslide.mpp-x"]
        slide_mpp = float(slide_mpp)
    except KeyError:
        raise KeyError("slide_mpp is not found")
    except ValueError:
        raise ValueError(
            f"slide_mpp is not float, passed: {slide_mpp}, type({type(slide_mpp)})"
        )

    up_level = int(np.log2(round(desired_mpp / slide_mpp)))

    if up_level < 0:
        raise ValueError(
            "desired_mpp is too small: "
            f"desired_mpp({desired_mpp}), slide_mpp({slide_mpp})"
        )

    dzg = DeepZoomGenerator(slide)
    max_deepzoom_level = dzg.level_count - 1

    return max_deepzoom_level - up_level


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # GPU 연산의 결정성을 높이기 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)