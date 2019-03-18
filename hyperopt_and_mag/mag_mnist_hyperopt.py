import os
import subprocess
from types import SimpleNamespace
from hyperopt import hp, fmin, tpe, space_eval
import numpy as np

from mag_mnist import run_experiment

def args(n_layers=1, hidden_units=128, activation='relu', batch_size=32,
        n_epochs=20, lr=0.001, momentum=0.9, experiments_dir='experiments_mnist'):
        
        return SimpleNamespace(
            n_layers=n_layers,
            hidden_units=hidden_units,
            activation=activation,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            momentum=momentum,
            experiments_dir=experiments_dir
        )

def run_experiment_wrapper(d):
    return -1 * run_experiment(SimpleNamespace(**d))

def run_hyperopt():
    space = {
        'n_layers': hp.choice('n_layers', [1, 2, 3]),
        'hidden_units': hp.choice('hidden_units', [16, 32, 64, 128, 256]),
        'activation': hp.choice('activation', ['relu', 'sigmoid']),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256]),
        'n_epochs': 20,
        'lr': hp.qloguniform('lr', np.log(1e-4), np.log(1e-1), q=1e-6),
        'momentum': 0.9,
        'experiments_dir': 'experiments_mnist_hyperopt_v5'
    }
    best = fmin(run_experiment_wrapper, space, algo=tpe.suggest, max_evals=5000)
    print("\n\n\n")
    print(best)
    print(space_eval(space, best))

if __name__ == "__main__":
    run_hyperopt()