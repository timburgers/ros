from math import pi

import torch


def gaussian(x, mu, sigma):
    """
    Gaussian PDF with broadcasting.
    """
    return torch.exp(-((x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * torch.sqrt(2 * torch.tensor(pi)))
