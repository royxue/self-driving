import numpy as np


class Line():
    def __init__(self, n):
        self.n = n
        self.detected = False
        self.last_fits = []
        self.best_fit = np.array([0., 0., 0.])

    def get_fit(self):
        return self.best_fit

    def add_fit(self, fit):
        if len(self.last_fits) == self.n:
            self.last_fits.pop(0)

        self.last_fits.append(fit)

        for i in range(3):
            self.best_fit[i] = np.mean([fit[i] for fit in self.last_fits])

