import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from models.pointnet2_classify import SetAbstractionLayer, GlobalAbstractionLayer
from models.utils import create_mlp

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

'''
Modular PointNet++ for segmentation / point-wise feature extraction
'''
class PointNet2Segment(nn.Module):
    def __init__(self, abstraction_layers, feature_prop_layers, linear_layers):
        super(PointNet2Segment, self).__init__()

        assert len(abstraction_layers)>=len(feature_prop_layers)

        self.abstraction_layers = nn.ModuleList(abstraction_layers)
        self.feature_prop_layers = nn.ModuleList(feature_prop_layers)
        self.linear_layers = nn.ModuleList(linear_layers)

    def forward(self, pos, batch):
        #Perform the downsampling steps
        data_downsampling = [(None, pos, batch), ]
        for i in range(len(self.abstraction_layers)):
            x_out, pos_out, batch_out = self.abstraction_layers[i](*data_downsampling[i])    
            data_downsampling.append((x_out, pos_out, batch_out))

        #Perform the upsampling steps
        data_upsampling = [data_downsampling[-1], ]
        for i in range(len(self.feature_prop_layers)): #FP layers have been added "reverse" / coarse-to-fine order
            idx_down = len(data_downsampling)-1 - i - 1
            x_out, pos_out, batch_out = self.feature_prop_layers[i](*data_upsampling[i], *data_downsampling[idx_down])
            data_upsampling.append((x_out, pos_out, batch_out))

        x, _, _ = data_upsampling[-1]

        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            if i<len(self.linear_layers)-1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x        

if __name__ == "__main__":
    sa_layers = []
    sa_layers.append(SetAbstractionLayer(0.5, 0.2, create_mlp([3,64])))
    sa_layers.append(SetAbstractionLayer(0.25, 0.4, create_mlp([64 + 3, 128])))

    fp_layers = []
    fp_layers.append(FeaturePropagationLayer(3, create_mlp([128 + 64, 64])))

    li_layers = []
    li_layers.append(nn.Linear(64, 64))
    li_layers.append(nn.Linear(64, 10))
    # fp_layers.append(FeaturePropagationLayer(3, create_mlp([64, 10])))

    model = PointNet2Segment(sa_layers, fp_layers, li_layers).to('cuda')
    # model = PointNet2Segment_old().to('cuda')

    pos = torch.rand(100,3)*5
    batch = torch.cat(( 0*torch.ones(50), 1*torch.ones(50))).long()
    out = model(pos.to('cuda'),batch.to('cuda'))
    print(out.shape)

    

# class PointNet2Segment_old(nn.Module):
#     def __init__(self):
#         super(PointNet2Segment_old, self).__init__()

#         self.set_abstraction0 = SetAbstractionLayer(0.5, 0.2, create_mlp([3, 64, 128]))
#         self.set_abstraction1 = SetAbstractionLayer(0.25, 0.4, create_mlp([128 + 3, 256, 256]))
#         # self.global_abstraction = GlobalAbstractionLayer(create_mlp([128 + 3, 256])) #TODO: example uses global layer?!

#         self.feature_prop1 = FeaturePropagationLayer(3, create_mlp([256 + 128, 128]))
#         self.feature_prop0 = FeaturePropagationLayer(3, create_mlp([128, 64]))

#         self.linear0 = nn.Linear(64, 64)
#         self.linear1 = nn.Linear(64, 10)

#     def forward(self, pos, batch):
#         #x_in, pos_in, batch_in = None, data.pos, data.batch # [N0]
#         x_in, pos_in, batch_in = None, pos, batch # [N0]
        
#         x0, pos0, batch0 = self.set_abstraction0(x_in, pos_in, batch_in) # [N1, 128]
#         print(x0.device, pos0.device, batch0.device)
#         x1, pos1, batch1 = self.set_abstraction1(x0, pos0, batch0) # [N2, 256]
#         print(x1.device, pos1.device, batch1.device)
        
#         #x = self.global_abstraction(x1, pos1, batch1)

#         x1, pos1, batch1 = self.feature_prop1(x1, pos1, batch1, x0, pos0, batch0) #[N1, 128]
#         print(x1.device, pos1.device, batch1.device)
#         x_in, pos_in, batch_in = self.feature_prop0(x0, pos0, batch0, x_in, pos_in, batch_in) #[N0, 64]
#         print(x_in.device, pos_in.device, batch_in.device)

#         x = x_in

#         x = F.relu(x)
#         x = self.linear0(x)
#         x = F.relu(x)
#         x = self.linear1(x)

#         return x