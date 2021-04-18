from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng, Generator

from pf.auxiliaryparticlefilter import AuxiliaryParticleFilter
from pf.particlefilter import ParticleFilter
from pf.benchmark.toy import ToySystem

def toypf():
    rng = default_rng(0)
    trajectory, observation = draw(100, rng)
    model = ToySystem()
    particles = rng.standard_normal(size=(100,))

    particle_filter = ParticleFilter(
            model,
            rng,
            )
    samples, weights = particle_filter.run(particles, observation)
    fig = plot(samples, weights, trajectory)
    fig.savefig('out/particle_filter.png')

    auxiliary_particle_filter = AuxiliaryParticleFilter(
            model,
            rng,
            )
    samples, weights = auxiliary_particle_filter.run(particles, observation)
    fig = plot(samples, weights, trajectory)
    fig.savefig('out/auxiliary_particle_filter.png')



def draw(length: int, rng: Generator) -> Tuple[np.ndarray, np.ndarray]:
    model = ToySystem()
    theta = rng.standard_normal(size=())
    thetas = []
    ys = []
    for t in range(length):
        theta = model.evolve(theta, t, rng)
        y = model.observe(theta, t, rng)
        thetas.append(theta)
        ys.append(y)
    return np.stack(thetas), np.stack(ys)

def plot(samples: np.ndarray, weights: np.ndarray, trajectory: np.ndarray):
    fig, ax = plt.subplots()
    mean = np.average(samples, axis=1, weights=weights)
    std = np.sqrt(np.average((mean[:,np.newaxis] - samples) ** 2, axis=1, weights=weights))
    ax.fill_between(
            np.arange(len(mean)),
            mean - std,
            mean + std,
            color='blue',
            edgecolor='none',
            alpha=0.3,
            )
    ax.plot(mean, color='blue')
    ax.plot(trajectory, color='orange')

    return fig

