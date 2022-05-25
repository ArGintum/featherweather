import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


## ____________________________________________________________________________________________________
## ________________________________  BASELINES  _______________________________________________________


class InterpolationModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        pass
    
    def forward(self, x):
        result = nn.functional.interpolate(x, mode='bicubic', align_corners=False, scale_factor=4)
        return result
        

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = nn.functional.interpolate(x, mode='bicubic', align_corners=False, scale_factor=4)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
## ____________________________________________________________________________________________________
## ________________________________  OUR MODELS _______________________________________________________

def ConvoBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, act_type='relu', norm=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    operands = [conv]
    if act_type:
        act = nn.PReLU(num_parameters=1, init=0.2)
        operands.append(act)
    if norm:
        n = nn.BatchNorm2d(out_channels)
        operands.append(n)
    return nn.Sequential(*operands)
        

def DeconvoBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, output_padding=0, padding=0, act_type='relu', norm=False):
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias, output_padding=output_padding) #nn.utils.parametrizations.spectral_norm
    operands = [deconv]
    if act_type:
        act = nn.PReLU(num_parameters=1, init=0.2)
        operands.append(act)
    if norm:
        n = nn.BatchNorm2d(out_channels)
        operands.append(n)
    return nn.Sequential(*operands)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False
            
class StackedBlock(nn.Module):
    def __init__(self, num_features, act_type, norm):
        super(StackedBlock, self).__init__()
        stride = 4
        padding = 2
        kernel_size = 8
        self.num_groups = 2
        self.num_steps = 4

        self.compress_in = ConvoBlock(2*num_features, num_features, kernel_size=1, act_type=act_type, norm=norm)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvoBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm=norm))
            self.downBlocks.append(ConvoBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm=norm))
            if idx > 0:
                self.uptranBlocks.append(ConvoBlock(num_features*(idx+1), num_features, kernel_size=1, stride=1, act_type=act_type, norm=norm))
                self.downtranBlocks.append(ConvoBlock(num_features*(idx+1), num_features, kernel_size=1, stride=1, act_type=act_type, norm=norm))

        self.compress_out = ConvoBlock(self.num_groups*num_features, num_features, kernel_size=1, act_type=act_type, norm=norm)
        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output
        return output

    def reset_state(self):
        self.should_reset = True
    
hidden_dim = 32

class FeatherNET_base(nn.Module):
    def __init__(self, num_channels=1):
        super(FeatherNET_base, self).__init__()
        self.num_steps = 1
        self.layer1 = ConvoBlock(num_channels, 4 * hidden_dim, 3, padding = 3 // 2)
        self.layer2 = ConvoBlock(4 * hidden_dim, hidden_dim, 1, padding = 1 // 2)        
        self.main_block = StackedBlock(hidden_dim, 'prelu', None)
        self.deconv_out = DeconvoBlock(hidden_dim, hidden_dim, 8, padding = 2, stride=4, output_padding=0, norm=False)
        self.conv_out = ConvoBlock(hidden_dim, num_channels, 3, padding = 3 // 2, act_type=None, norm=False)

    def forward(self, x):
        self._reset_state()
        upscale = nn.functional.interpolate(x, mode='bicubic', align_corners=False, scale_factor=4)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        
        outs = []
        for i in range(self.num_steps):
            h = self.main_block(x_2)
            h = torch.add(upscale, self.conv_out(self.deconv_out(h)))
     
            outs.append(h)
        
        if self.training:
            return outs
        return h
        
    def _reset_state(self):
        self.main_block.reset_state()
        
        

    
class FeatherNET_contextualized(nn.Module):
    def __init__(self, num_channels=1, memory_length=1):
        super(FeatherNET_contextualized, self).__init__()
        self.num_steps = memory_length

        self.layer1 = ConvoBlock(num_channels, 4 * hidden_dim, 3, padding = 3 // 2)
        self.layer2 = ConvoBlock(4 * hidden_dim, hidden_dim, 1, padding = 1 // 2)        
        self.main_block = StackedBlock(hidden_dim, 'prelu', None)
        self.deconv_out = DeconvoBlock(hidden_dim, hidden_dim, 8, padding = 2, stride=4, output_padding=0, norm=False)
        
        dim = 2
        self.fc_1 = nn.Linear(hidden_dim + dim, 2 * hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim + hidden_dim + dim, hidden_dim)
        self.fc_4 = nn.Linear(hidden_dim + hidden_dim + dim, hidden_dim)
        self.fc_5 = nn.Linear(hidden_dim, num_channels)
        self.activ = nn.LeakyReLU()
        self.fc = [self.fc_3, self.fc_4]


    def forward(self, x):
        self._reset_state()
        upscale = nn.functional.interpolate(x, mode='bicubic', align_corners=False, scale_factor=4)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        
        #outs = []
        for i in range(self.num_steps):
            h = self.main_block(x_2[i : x_2.shape[0] - self.num_steps + 1 + i])
            h = self.deconv_out(h)
            
            grid_x, grid_y = torch.meshgrid(torch.arange(h.shape[2]), torch.arange(h.shape[3]), indexing="ij")
            coords = torch.tile(torch.unsqueeze(torch.stack([grid_x, grid_y], 2), 0), (h.shape[0], 1, 1, 1)).to(h.device)
            
            h = torch.moveaxis(h, (1, 2, 3), (3, 1, 2))
            h = torch.cat([h, coords], dim=-1)
            x_tmp = self.fc_1(h)
            x_tmp = self.activ(x_tmp)
            for fc_layer in self.fc:
                x_tmp = fc_layer(torch.cat([x_tmp, h], dim=-1))
                x_tmp = self.activ(x_tmp)

            h = self.fc_5(x_tmp) 
            h = torch.moveaxis(h, (1,2,3), (2, 3, 1))            
            h = torch.add(upscale[i : x_2.shape[0] - self.num_steps + 1 + i], h)
          #  outs.append(h)

        return [[],[],[],h]
        
    def _reset_state(self):
        self.main_block.reset_state()
        
        

##################### LOSS FUNCTIONS DEFINED HERE #################################################

def spectral_loss(output, target):
    fffoutput = (torch.fft.rfft2(output, norm='forward').abs()) ** 2
    fftarget = (torch.fft.rfft2(target, norm='forward').abs()) ** 2
    mse = nn.MSELoss()

    loss =  mse(fffoutput, fftarget) + mse(output, target)
    return loss

def gradient_loss(output, target, lam = 1.0):
    mae = torch.nn.L1Loss()
    loss = mae(output, target)
    jy = output[:,:,1:,:] - output[:,:,:-1,:]
    jx = output[:,:,:,1:] - output[:,:,:,:-1]
    jy_ = target[:,:,1:,:] - target[:,:,:-1,:]
    jx_ = target[:,:,:,1:] - target[:,:,:,:-1]
    loss += lam * mae(jy, jy_)
    loss += lam * mae(jx, jx_)
    return loss
