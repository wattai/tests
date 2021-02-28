import numpy as np

from src.simulator.utils import bboxes

if __name__ == "__main__":

    MAX_ITERATION = 100
    error = 0.0

    num_scales = 7
    num_aspect_ratios = 7

    scales = np.random.rand(num_scales)
    aspect_ratios = np.random.rand(num_aspect_ratios)

    boxes = bboxes.make_anchor_boxes()

    # TODO: Simulate
