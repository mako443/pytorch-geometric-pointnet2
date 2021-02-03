import torch
import torch.nn as nn

'''
Naive PointNet module w/o PyG
'''

class PointNet(nn.Module):
    def __init__(self, input_dim, h_dims, g_dims, num_classes):
        super(PointNet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        g_layers = []
        g_layers.append(nn.Linear(h_dims[-1], g_dims[0]))
        for i in range(0, len(g_dims)-1):
            g_layers.append(nn.ReLU())
            g_layers.append(nn.Linear(g_dims[i], g_dims[i+1]))
        g_layers.append(nn.ReLU())
        g_layers.append(nn.Linear(g_dims[-1], num_classes))
        self.g_mlp = nn.Sequential(*g_layers)

        h_layers = []
        h_layers.append(nn.Linear(input_dim, h_dims[0]))
        for i in range(len(h_dims)-1):
            h_layers.append(nn.ReLU())
            h_layers.append(nn.Linear(h_dims[i], h_dims[i+1]))
        self.h_mlp = nn.Sequential(*h_layers)

        # print(self.h_mlp)
        # print(self.g_mlp)

    def forward(self, x):
        assert len(x.shape)==3 and x.shape[-1]==self.input_dim

        x = self.h_mlp(x)
        x, _ = torch.max(x, -2)
        x = self.g_mlp(x)

        return x

if __name__=='__main__':
    model = PointNet(3, [64, 64], [128, 128], 10)
    x = torch.rand(2, 1024, 3)
    out = model(x)
    print(out.shape)
