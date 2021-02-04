import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

'''
PointNet++ implementation using PyG

TODO:
- compare accuracy w/ example!
'''

def create_mlp(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i-1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i])) for i in range(1, len(channels))
    ])

class SetAbstractionLayer(nn.Module):
    def __init__(self, ratio, radius, mlp):
        super(SetAbstractionLayer, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.mlp = mlp

        self.point_conv = gnn.PointConv(mlp)

    def forward(self, x, pos, batch):
        sample_indices = gnn.fps(pos, batch, self.ratio)
        sparse_indices, dense_indices = gnn.radius(pos, pos[sample_indices], self.radius, batch, batch[sample_indices], max_num_neighbors=64)
        edge_index = torch.stack((dense_indices, sparse_indices), dim=0) #TODO/CARE: Indices are propagated internally? Care edge direction: a->b <=> a is in N(b)

        x = self.point_conv(x, (pos, pos[sample_indices]), edge_index)

        return x, pos[sample_indices], batch[sample_indices]     

class GlobalAbstractionLayer(nn.Module):
    def __init__(self, mlp):
        super(GlobalAbstractionLayer, self).__init__()
        self.mlp = mlp

    def forward(self, x, pos, batch):
        x = torch.cat((x, pos), dim=1)
        x = self.mlp(x)
        x = gnn.global_max_pool(x, batch)
        return x
           

class PointNet2Classify(nn.Module):
    def __init__(self):
        super(PointNet2Classify, self).__init__()

        self.set_abstraction0 = SetAbstractionLayer(0.5, 0.2, create_mlp([3, 64, 128]))
        self.set_abstraction1 = SetAbstractionLayer(0.25, 0.4, create_mlp([128 + 3, 128, 128]))
        self.global_abstraction = GlobalAbstractionLayer(create_mlp([128 + 3, 256]))

        self.linear0 = nn.Linear(256, 256)
        self.linear1 = nn.Linear(256, 10)

    def forward(self, data):
        x, pos, batch = None, data.pos, data.batch
        
        x, pos, batch = self.set_abstraction0(x, pos, batch)
        x, pos, batch = self.set_abstraction1(x, pos, batch)
        
        x = self.global_abstraction(x, pos, batch)

        x = F.relu(x)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)

        return x

if __name__=='__main__':
    model = Net().to('cuda')
    pos = torch.rand(100,3)*5
    batch = torch.cat(( 0*torch.ones(50), 1*torch.ones(50))).long()
    out = model(pos.to('cuda'),batch.to('cuda'))
    print(out.shape)

