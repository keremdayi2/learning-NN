"""
    Question: What do the loss landscapes of NN training look like
    Hypothesis: The time averaged value of angle(grad, new_model - old_model) stays positive 
    Experiment:
    -   Basic dataset (e.g. MNIST), run optimization 1 with seed, compute the limiting value
    -   Run the same experiment with the same initialization, tracking the variables
        - overlap between gradient and limit 
        - overlap between model and limit
        - overlap between gradient and difference of models
"""

import torch
import math

"""
    Compute the overlap between two given models with the same architecture
"""
def overlap(model1, model2):
    with torch.no_grad():
        ip = 0.0
        n1, n2 = 0.0, 0.0
        d1, d2 = model1.state_dict(), model2.state_dict()
        for p in d1.keys():
            n1 += d1[p].norm() ** 2
            n2 += d2[p].norm() ** 2
            ip += (d1[p] * d2[p]).sum()

        return ip * (math.sqrt(n1 * n2)) 


