import time
import typing

import numpy as np

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
