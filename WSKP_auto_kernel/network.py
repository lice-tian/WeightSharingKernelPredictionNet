def SumKernelParameter(kernel_size, device):
    kernel = np.ones((1, 1, kernel_size, kernel_size), dtype=np.float32)
    kernel = torch.from_numpy(kernel.reshape(1, 1, kernel_size, kernel_size))
   
    kernel = kernel.to(device)
    return torch.nn.Parameter(kernel, requires_grad=False)
    

class repSharingKernelNet(nn.Module):
    def __init__(self, device):
        super(repSharingKernelNet, self).__init__()
        self.device = device
        
        conv1_in_channels = 10
        self.base_depth = 14
                    
        self.kernel_num = 6
        self.kernel_base_size = 3
        self.kernel_size_stride = 2
        
        self.convSs = []
        for i in range(self.kernel_num):
            kernel_size = self.kernel_base_size + i * self.kernel_size_stride
            padding = (kernel_size - 1) // 2
            convS = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
            convS.weight = SumKernelParameter(kernel_size=kernel_size, device=device)
            self.convSs.append(convS)

        self.convs = Convs() # 6 conv layers
        self.softmax = nn.Softmax2d()
      
        
    def forward(self, x_in):
        x_irradiance = x_in[:, 0:3]
        x_albedo = x_in[:, 3:6]
        x_inputs = torch.cat((BMFRGammaCorrection(x_irradiance * x_albedo),
                                            x_in[:, 3:]), axis=1)
        
        x_final_out = self.convs(x_inputs)
        
        x_guidemap = torch.exp(x_final_out[:, :self.kernel_num]) # from 20-NV-AdaptiveSampling
        x_alpha = self.softmax(x_final_out[:, self.kernel_num:])

        B, _, H, W = x_inputs.shape
        # Every channel apply 2d filter with same kernel weight,
        # from https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840
        x_out = 0.0
        for i in range(self.kernel_num):
            x_guidemap_windowsum = self.convSs[i](x_guidemap[:, i:(i+1)])
            x_out += x_alpha[:, i:i+1] * (self.convSs[i]((x_guidemap[:, i:(i+1)] * x_irradiance).view(-1, 1, H, W)).view(B, -1, H, W) / x_guidemap_windowsum)

        x_out = x_out * x_albedo

        return x_out