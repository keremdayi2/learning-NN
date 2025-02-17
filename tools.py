import time
import typing
import glob 
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

class TimeLogger:
    def __init__(self):
        self.checkpoints = [("", time.time())]
        self.elapsed = []

    def log(self, label : str = ""):
        t = (label, time.time())
        self.checkpoints.append(t)
        e = (label, t[1] - self.checkpoints[-2][1]) # compute elapsed time since last checkpoint
        self.elapsed.append(e)

    def get_results(self):
        labels = set([e[0] for e in self.elapsed])
        
        averages = {}

        for l in labels:
            this_label = list(filter(lambda x: x[0] == l, self.elapsed))
            this_label = [x[1] for x in this_label]
            this_label = np.array(this_label)
            averages[l] = this_label.mean()

        return averages

    # total time elapsed since birth
    def total(self):
        return time.time() - checkpoints[0][1]

# plotter based on a directory
class Plotter:
    def __init__(self, directory : str):
        self.dir = directory

    def _match_dict(self, key: dict, other : dict):
        for k, v in key.items():
            try:
                if other[k] != v: 
                    return False
            except: # if key does not exist
                return False
        return True

    # get experiments that match the subset of params
    # given
    # return: list[dict] containing the full params for matching experiments
    def get_experiments(self, params : dict):
        filenames = glob.glob(f"{self.dir}/*/*.json")

        ret = []

        for file in filenames:
            other_params = {}
            with open(file, "r") as json_file:
                other_params = json.load(json_file)
            
            if not self._match_dict(params, other_params):
                # means there is not a match
                continue

            ret.append(other_params)

        return ret

    def get_series(self, params : dict):
        run_id = params['run_id']

        losses = torch.load(os.path.join(self.dir, run_id, "losses.pt"))
        multi_losses = torch.load(os.path.join(self.dir, run_id, "multi_losses.pt"))

        return losses, multi_losses

    def _plot_single(self, series : torch.tensor):
        plt.grid()
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("MSE", fontsize=14)

        plt.plot(series)
        plt.show()

    def _plot_multi(self, series : torch.tensor):
        plt.grid()
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Component MSE", fontsize=14)

        for i in range(series.shape[1]):
            plt.plot(series[:,i], label= f'Component {i}')

        plt.legend()
        plt.show()


    def plot_series(self, series : torch.tensor):
        if len(series.shape) > 1:
            self._plot_multi(series)
        else:
            self._plot_single(series)
