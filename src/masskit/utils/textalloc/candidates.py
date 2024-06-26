# MIT License

# Copyright (c) 2022 Christoffer Kjellson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def generate_candidates(
    w: float,
    h: float,
    x: float,
    y: float,
    xmindistance: float,
    ymindistance: float,
    xmaxdistance: float,
    ymaxdistance: float,
    nbr_candidates: int,
) -> np.ndarray:
    """Generates 36 candidate boxes

    Args:
        w (float): width of box
        h (float): height of box
        x (float): xmin of box
        y (float): ymin of box
        xmindistance (float): fraction of the x-dimension to use as margins for text bboxes
        ymindistance (float): fraction of the y-dimension to use as margins for text bboxes
        xmaxdistance (float): fraction of the x-dimension to use as max distance for text bboxes
        ymaxdistance (float): fraction of the y-dimension to use as max distance for text bboxes
        nbr_candidates (int): nbr of candidates to use. If <1 or >36 uses all 36

    Returns:
        np.ndarray: candidate boxes array
    """
    candidates = np.array(
        [
            [
                x + xmindistance,
                y + ymindistance,
                x + w + xmindistance,
                y + h + ymindistance,
            ],  # upper right side
            [
                x - w - xmindistance,
                y + ymindistance,
                x - xmindistance,
                y + h + ymindistance,
            ],  # upper left side
            [
                x - w - xmindistance,
                y - h - ymindistance,
                x - xmindistance,
                y - ymindistance,
            ],  # lower left side
            [
                x + xmindistance,
                y - h - ymindistance,
                x + w + xmindistance,
                y - ymindistance,
            ],  # lower right side
            [x - w - xmindistance, y - h / 2, x - xmindistance, y + h / 2],  # left side
            [
                x + xmindistance,
                y - h / 2,
                x + w + xmindistance,
                y + h / 2,
            ],  # right side
            [x - w / 2, y + ymindistance, x + w / 2, y + h + ymindistance],  # above
            [x - w / 2, y - h - ymindistance, x + w / 2, y - ymindistance],  # below
            [
                x - 3 * w / 4,
                y + ymindistance,
                x + w / 4,
                y + h + ymindistance,
            ],  # above left
            [
                x - w / 4,
                y + ymindistance,
                x + 3 * w / 4,
                y + h + ymindistance,
            ],  # above right
            [
                x - 3 * w / 4,
                y - h - ymindistance,
                x + w / 4,
                y - ymindistance,
            ],  # below left
            [
                x - w / 4,
                y - h - ymindistance,
                x + 3 * w / 4,
                y - ymindistance,
            ],  # below right
            # We move all points a bit further from the target
        ]
    )
    if nbr_candidates > candidates.shape[0]:
        candidates2 = np.zeros((nbr_candidates - candidates.shape[0], 4))
        n_gen = candidates2.shape[0]
        for i in range(n_gen):
            frac = i / n_gen
            x_sample = np.random.uniform(
                x - frac * xmaxdistance, x + frac * xmaxdistance
            )
            y_sample = np.random.uniform(
                y - frac * ymaxdistance, y + frac * ymaxdistance
            )
            candidates2[i, :] = [
                x_sample - w / 2,
                y_sample - h / 2,
                x_sample + w / 2,
                y_sample + h / 2,
            ]
        candidates = np.vstack([candidates, candidates2])
    return candidates
