import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

from models.pointnet2_classify import PointNet2Classify
from scripts.utils import plot_object

def train_epoch():
    model.train()

    epoch_losses = []
    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()

        pred = model(batch.to(DEVICE))
        loss = criterion(pred, batch.y.to(DEVICE))

        loss.backward()
        optimizer.step()    

        epoch_losses.append(loss.cpu().detach().numpy())

    return np.mean(epoch_losses)

@torch.no_grad()
def test():
    model.eval()

    correct = []
    for i_batch, batch in enumerate(dataloader):
        x = batch.pos.reshape((-1,1024,3)).to(DEVICE)
        pred = model(batch.to(DEVICE))
        pred = torch.argmax(pred, dim=-1)
        pred, y = pred.cpu().detach().numpy(), batch.y.cpu().detach().numpy()
        correct.append(np.mean(pred==y))
    return np.mean(correct)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
dataset = ModelNet('./data/model-net', '10', True, transform, pre_transform)
dataloader = DataLoader(dataset, num_workers=2, batch_size=16, shuffle=True)

loss_dict = {}
for lr in (5e-3, ):
    model = PointNet2Classify().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    key = lr
    loss_dict[key] = []

    for epoch in range(1):
        loss = train_epoch()
        loss_dict[key].append(loss)
        acc = test()
        print(f'Key {key} epoch {epoch}, loss {loss: 0.2f}, acc {acc: 0.2f}')
    print()

plt.figure()
for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.title('Losses')
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()        
plt.show()