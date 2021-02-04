import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from models.pointnet2_classify import SetAbstractionLayer, GlobalAbstractionLayer, create_mlp

class FeaturePropagationLayer(nn.Module):
    def __init__(self, k, mlp):
        super(FeaturePropagationLayer, self).__init__()

        self.k = k
        self.mlp = mlp

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x_upsampled = gnn.knn_interpolate(x, pos, pos_skip, batch_x=batch, batch_y=batch_skip, k=self.k) #Interpolate coarse features to dense positions
        
        if x_skip is not None:
            x_upsampled = torch.cat((x_upsampled, x_skip), dim=1) #Concatenate new and old dense features
        
        # print('dims:', x.shape, x_skip.shape, x_upsampled.shape)
        x_upsampled = self.mlp(x_upsampled) #Run MLP

        return x_upsampled, pos_skip, batch_skip

class PointNet2Segment(nn.Module):
    def __init__(self):
        super(PointNet2Segment, self).__init__()

        self.set_abstraction0 = SetAbstractionLayer(0.5, 0.2, create_mlp([3, 64, 128]))
        self.set_abstraction1 = SetAbstractionLayer(0.25, 0.4, create_mlp([128 + 3, 256, 256]))
        # self.global_abstraction = GlobalAbstractionLayer(create_mlp([128 + 3, 256])) #TODO: example uses global layer?!

        self.feature_prop1 = FeaturePropagationLayer(3, create_mlp([256 + 128, 128]))
        self.feature_prop0 = FeaturePropagationLayer(3, create_mlp([128, 64]))

        self.linear0 = nn.Linear(64, 64)
        self.linear1 = nn.Linear(64, 10)

    def forward(self, pos, batch):
        #x_in, pos_in, batch_in = None, data.pos, data.batch # [N0]
        x_in, pos_in, batch_in = None, pos, batch # [N0]
        
        x0, pos0, batch0 = self.set_abstraction0(x_in, pos_in, batch_in) # [N1, 128]
        x1, pos1, batch1 = self.set_abstraction1(x0, pos0, batch0) # [N2, 256]
        
        #x = self.global_abstraction(x1, pos1, batch1)

        x1, pos1, batch1 = self.feature_prop1(x1, pos1, batch1, x0, pos0, batch0) #[N1, 128]
        x_in, pos_in, batch_in = self.feature_prop0(x0, pos0, batch0, x_in, pos_in, batch_in) #[N0, 64]

        x = x_in

        x = F.relu(x)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)

        return x

if __name__ == "__main__":
    model = PointNet2Segment().to('cuda')
    pos = torch.rand(100,3)*5
    batch = torch.cat(( 0*torch.ones(50), 1*torch.ones(50))).long()
    out = model(pos.to('cuda'),batch.to('cuda'))
    print(out.shape)