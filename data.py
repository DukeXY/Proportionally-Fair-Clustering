import numpy as np

class AlgoData:
    def __init__(self, iterations):
        self.avg = []
        self.errorbar = np.zeros(iterations)

class ExperimentData:

    def __init__(self, kmedian = 0, kcenter = 0, kmeans = 0, alpha = 0, beta = 0):
        self.kmedian = kmedian
        self.kcenter = kcenter
        self.kmeans = kmeans
        self.alpha = alpha
        self.beta = beta