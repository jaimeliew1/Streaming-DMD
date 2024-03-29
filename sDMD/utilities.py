from collections import deque
import numpy as np


class Stacker:
    """
    Maintains a deque of 1D numpy arrays.
    """

    def __init__(self, n: int, s: int):
        self.stack = deque(s * [np.zeros((n))], maxlen=s)

    def update(self, x):
        x = x.reshape(-1)
        self.stack.appendleft(x)

        return self()

    def __call__(self):
        # returns an (n*s, 1) numpy array of the current stack.
        return np.hstack(self.stack).reshape([-1, 1])


class Delayer:
    def __init__(self, n: int, f: int):
        self.stack = deque((f + 1) * [np.zeros((n, 1))], maxlen=f + 1)

    def update(self, x):
        x = x.reshape(-1)
        self.stack.appendleft(x)
        return self()

    def __call__(self):
        return self.stack[-1].reshape([-1, 1])
