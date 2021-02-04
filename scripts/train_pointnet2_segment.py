import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

category = 'Airplane'  # Pass in `None` to train on all categories.
path = './data/shape-net'
transform = T.Compose([
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval', transform=transform, pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test', pre_transform=pre_transform)

'''
ShapeNet download fails / dataset has been moved
'''