import neptune.new as neptune
import matplotlib.pyplot as plt
from neptune.new.types import File
import numpy as np
import skdim
import time
import datasets
import traceback
import logging


def init_run():
    return neptune.init(
    project="rm360179/LIDL",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ=="
)  # your credentials


def loguniform(low, high):
    llow = np.log(low)
    lhigh = np.log(high)

    return np.exp(np.random.uniform(llow, lhigh))


def random_search(
        model_fun,
        data,
        ground_truth,
        parameters_ranges,
        model_name,
        dataset_name,
        num_iter=20):
    """
    Args:
        model (object)
        data (numpy array)
        ground_truth (numpy array)
        parameters_ranges (dict): key-list 
        model_name (str): for logging
        dataset_name (str): for logging
        num_iter (int): number of runs
    """
    assert ground_truth.ndim == 1
    assert data.shape[0] == ground_truth.shape[0]

    def mse(p, q):
        p, q = np.sort(p), np.sort(q)
        return ((p-q)**2).mean()

    def mae(p, q):
        p, q = np.sort(p), np.sort(q)
        return abs(p-q).mean()

    np.random.seed(0)
    run = init_run()
    run['dataset_name'] = dataset_name
    run['model_name'] = model_name
    run['parameters'] = parameters_ranges

    for i in range(num_iter):
        try:
            start_time = time.time()
            params = dict()
            if i != 0:
                for pname, vals in parameters_ranges.items():
                    pval = np.random.choice(vals)
                    params[pname] = pval

            model = model_fun(**params)
            ldims = model.fit_transform_pw(data)
            model = model_fun(**params)
            gdim = model.fit_transform(data)
            end_time = time.time()


            m1, m2 = mse(ldims, ground_truth), mae(ldims, ground_truth)

            fig, axes = plt.subplots(1, 2)
            fig.set_size_inches(15, 6)
            axes[0].hist(ldims, bins=100)
            axes[0].set_title('prediction')
            axes[1].hist(ground_truth, bins=100)
            axes[1].set_title('ground truth')
            run['histogram'].log(fig)
            run['mse'].log(m1)
            run['mae'].log(m2)
            run['params'].log(params)
            run['time'].log(end_time - start_time)
        except Exception as e:
            logging.error(traceback.format_exc())

    run.stop()

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