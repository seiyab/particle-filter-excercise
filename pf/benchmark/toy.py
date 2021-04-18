from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from pf.particlefilter import SystemModel as PFSystemModel
from pf.auxiliaryparticlefilter import SystemModel as APFSystemModel

@dataclass
class ToySystem(PFSystemModel, APFSystemModel):
    a: float = 1/20
    b: float = 1/2
    c: float = 25
    d: float = 8
    omega: float = 1.2
    v: float = 10
    w: float = 1

    def likelihood(self, y: np.ndarray, t: int, theta: np.ndarray) -> np.ndarray:
        d = y - self.a * theta * theta
        return np.exp(- 1/2 * (d * d / self.v)) / np.sqrt(2 * np.pi * self.v)

    def observe(self, theta: np.ndarray, t: int, rng: Generator) -> np.ndarray:
        return self.a * theta * theta + rng.normal(scale=np.sqrt(self.v))

    def evolve(self, prev_theta: np.ndarray, t: int, rng: Generator) -> np.ndarray:
        w = rng.normal(scale=np.sqrt(self.w), size=prev_theta.shape)
        return self.likely_evolve(prev_theta, t, rng) + w

    def likely_evolve(self, prev_theta: np.ndarray, t: int, rng: Generator) -> np.ndarray:
        return self.b * prev_theta \
                + self.c * prev_theta / (1 + prev_theta * prev_theta) \
                + self.d * np.cos(self.omega * t) \

