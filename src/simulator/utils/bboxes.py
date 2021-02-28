from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CenterPoint:
    x: float = 10.0
    y: float = 10.0


def make_anchor_boxes(
    scales: Tuple[float, ...] = (1.0, 1.5, 0.5),
    aspect_ratios: Tuple[float, ...] = (1.0, 1.5, 0.5, 0.1),
    center_point: CenterPoint = CenterPoint(10.0, 10.0),
    num_samples: int = 5000,
    radius: float = 3.0,
):
    """Return anchor boxes for the simulation.

    Args:
        scales (Tuple[float, ...], optional): Scales of BBoxes. Defaults to (1.0, 1.5, 0.5).
        aspect_ratios (Tuple[float, ...], optional): Aspect ratios of BBoxes. Defaults to (1.0, 1.5, 0.5, 0.1).
        center_point (CenterPoint, optional): Offset for BBoxes positions. Defaults to CenterPoint(10.0, 10.0).
        num_samples (int, optional): Number of samples of BBoxes. Defaults to 5000.
        radius (float, optional): [description]. Defaults to 3.0.

    Returns:
        numpy.ndarray: BBoxes; the size of [num_samples, len(scales), len(aspect_ratios), 4]
    """
    # return array of [5000: sample, 7: scale, 7: aspect_ratio, 4: bbox]
    num_scales = len(scales)
    num_aspect_ratios = len(aspect_ratios)

    def rand():
        return np.random.rand(num_samples) - 0.5

    x = radius * np.tile(
        (rand() + center_point.x)[:, None, None, None],
        (1, num_scales, num_aspect_ratios, 1),
    )
    y = radius * np.tile(
        (rand() + center_point.y)[:, None, None, None],
        (1, num_scales, num_aspect_ratios, 1),
    )

    ss, aa = np.meshgrid(scales, aspect_ratios, indexing="ij")
    w = np.tile(
        (ss * aa)[None, :, :, None],
        (num_samples, 1, 1, 1),
    )
    h = np.tile((ss)[None, :, :, None], (num_samples, 1, 1, 1))
    return np.concatenate((x, y, w, h), axis=-1)
