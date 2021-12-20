import matplotlib.pyplot as plt
import numpy as np
import skdim
import time
import datasets
import traceback
import logging
from utils import random_search


# dataset
size = 500
dataset = datasets.uniform_N(10, size, 0)
ground_truth = np.full(size, 10)
dataset_name = f"{size}_uniform_10"

# model
model_fun = skdim.id.CorrInt
model_name = "corrint"
parameters_ranges = {
        "k1": np.arange(5, 30),
        "k2": np.arange(10, 50)
}

random_search(
        model_fun,
        dataset,
        ground_truth,
        parameters_ranges,
        model_name,
        dataset_name,
        num_iter=10)
