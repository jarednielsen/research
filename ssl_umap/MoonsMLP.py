from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class MoonsMLP(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2, 5),
            nn.Linear(5, self.num_classes)
        )
#         self.classifier = nn.Linear(2, self.num_classes)

    def forward(self, x):
        return self.classifier(x)