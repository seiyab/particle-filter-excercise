from dataclasses import dataclass
from mypy_extensions import trait
from typing import Callable, Tuple

import numpy as np
from numpy.random import Generator

@dataclass(frozen=True)
class ParticleFilter:
    model: 'SystemModel'
    rng: Generator

    def step(self, t: int, prev_particles: np.ndarray, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.model.evolve(prev_particles, t, self.rng)
        unnormalized_weights = self.model.likelihood(observation, t, samples)
        weights = unnormalized_weights / unnormalized_weights.sum()
        return samples, weights

    def resample(self, samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.rng.choice(samples, size=samples.shape[0], p=weights)

    def run(self, particles: np.ndarray, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sample = particles
        samples = []
        weights = []
        for t, observation in enumerate(observations):
            sample, weight = self.step(t, sample, observation)
            samples.append(sample)
            weights.append(weight)
            sample = self.resample(sample, weight)
        return np.stack(samples), np.stack(weights)

@trait
class SystemModel:
    def likelihood(self, observation: np.ndarray, t: int, random_variable: np.ndarray) -> np.ndarray:
        ...

    def evolve(self, prev: np.ndarray, t: int, rng: Generator) -> np.ndarray:
        ...

