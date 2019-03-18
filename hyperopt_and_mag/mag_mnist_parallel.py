import os
import subprocess
from types import SimpleNamespace

from mag_mnist import run_experiment

def args(n_layers=1, hidden_units=128, activation='relu', batch_size=128,
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

if __name__ == "__main__":
    # use subprocess.Popen() to run things in parallel.
    acc = run_experiment(args(lr=1, n_epochs=1))
    print("FINISHED!\n\n{}".format(acc))