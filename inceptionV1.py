import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from torchsummary import summary


# 初始化神经网络层参数，确保不同类型的层权重和偏置有合理的初始值，以加速训练并提高稳定性。
def layer_init(m):
    classname = m.__class__.__name__
    classname = classname.lower()
    if classname.find('conv') != -1 or classname.find('linear') != -1:
        gain = nn.init.calculate_gain(classname)
        nn.init.xavier_uniform(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('batchnorm') != -1:
        nn.init.constant(m.weight, 1)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('embedding') != -1:
        # The default initializer in the TensorFlow embedding layer is a truncated normal with mean 0 and
        # standard deviation 1/sqrt(sparse_id_column.length). Here we use a normal truncated with 3 std dev
        num_columns = m.weight.size(1)
        sigma = 1/(num_columns**0.5)
        m.weight.data.normal_(0, sigma).clamp_(-3*sigma, 3*sigma)

class InceptionNaive(nn.Module):
    def __init__(self, in_channels):
        super(InceptionNaive, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 64, kernel_size=1,stride=1)

        self.conv3x3 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1,stride=1)
        
        self.conv5x5 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2,stride=1)
        
        self.pool_proj = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)
        branch3 = self.conv5x5(x)
        branch4 = self.pool_proj(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
def test_InceptionNaive():
    input_data = torch.randn(8,128,256,256)
    model = InceptionNaive(in_channels=128)
    output = model(input_data)
    summary(model,(128,256,256))
    print(output.shape)

class InceptionWithReduction(nn.Module):
    def __init__(self, in_channels):
        super(InceptionWithReduction, self).__init__()

        # 1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 降维的 1x1 卷积 + 3x3 卷积
        self.conv3x3_reduce = nn.Conv2d(in_channels, 96, kernel_size=1)
        self.conv3x3 = nn.Conv2d(96, 128, kernel_size=3, padding=1)

        # 降维的 1x1 卷积 + 5x5 卷积
        self.conv5x5_reduce = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.conv5x5 = nn.Conv2d(16, 32, kernel_size=5, padding=2)

        # 3x3 最大池化 + 1x1 卷积
        self.pool_proj = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_1x1 = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.apply(layer_init)


    def forward(self, x):
        branch1 = self.conv1x1(x)

        branch2 = self.conv3x3_reduce(x)
        branch2 = self.conv3x3(branch2)

        branch3 = self.conv5x5_reduce(x)
        branch3 = self.conv5x5(branch3)

        branch4 = self.pool_proj(x)
        branch4 = self.pool_1x1(branch4)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
def test_InceptionWithReduction():
    input_data = torch.randn(8,128,256,256)
    model = InceptionWithReduction(in_channels=128)
    output = model(input_data)
    summary(model,(128,256,256))
    print(output.shape)
# # 示例
# x = torch.randn(1, 192, 28, 28)  # 假设输入通道数为 192
# model = InceptionWithReduction(in_channels=192)
# output = model(x)
# print(output.shape)







# 局部响应归一化层，现如今已经被批归一化层替代，现代CNN网络已经放弃该网络
class LRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim # 指定在哪个维度拼接数据

        #mixed 'name'_1x1
        self.conv1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0)

        #mixed 'name'_3x3_bottleneck
        self.conv3_1 = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0)
        #mixed 'name'_3x3
        self.conv3_3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1)

        # mixed 'name'_5x5_bottleneck
        self.conv5_1 = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0)
        # mixed 'name'_5x5
        self.conv5_5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1)
        #mixed 'name'_pool_reduce
        self.conv_max_1 = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0)

        #self.apply(layer_init)

    def forward(self, input):

        output1 = F.relu(self.conv1(input))

        output2 = F.relu(self.conv3_1(input))
        output2 = F.relu(self.conv3_3(output2))

        output3 = F.relu(self.conv5_1(input))
        output3 = F.relu(self.conv5_5(output3))

        output4 = F.relu(self.conv_max_1(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)

class Inception_v1(nn.Module):
    def __init__(self, num_classes=1000):
        super(Inception_v1, self).__init__()

        #conv2d0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)

        #conv2d1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        #conv2d2
        self.conv3  = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn3 = LRN(local_size=11, alpha=0.00109999999404, beta=0.5, k=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3a = Inception_base(1, 192, [[64], [96,128], [16, 32], [3, 32]]) #3a
        self.inception_3b = Inception_base(1, 256, [[128], [128,192], [32, 96], [3, 64]]) #3b
        self.max_pool_inc3= nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception_4a = Inception_base(1, 480, [[192], [ 96,204], [16, 48], [3, 64]]) #4a
        self.inception_4b = Inception_base(1, 508, [[160], [112,224], [24, 64], [3, 64]]) #4b
        self.inception_4c = Inception_base(1, 512, [[128], [128,256], [24, 64], [3, 64]]) #4c
        self.inception_4d = Inception_base(1, 512, [[112], [144,288], [32, 64], [3, 64]]) #4d
        self.inception_4e = Inception_base(1, 528, [[256], [160,320], [32,128], [3,128]]) #4e
        self.max_pool_inc4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_5a = Inception_base(1, 832, [[256], [160,320], [48,128], [3,128]]) #5a
        self.inception_5b = Inception_base(1, 832, [[384], [192,384], [48,128], [3,128]]) #5b
        self.avg_pool5 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.dropout_layer = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        #self.apply(layer_init)

    def forward(self, input):

        output = self.max_pool1(F.relu(self.conv1(input)))
        output = self.lrn1(output)

        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = self.max_pool3(self.lrn3(output))

        output = self.inception_3a(output)
        output = self.inception_3b(output)
        output = self.max_pool_inc3(output)

        output = self.inception_4a(output)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        output = self.inception_4e(output)
        output = self.max_pool_inc4(output)

        output = self.inception_5a(output)
        output = self.inception_5b(output)
        output = self.avg_pool5(output)

        output = output.view(-1, 1024)

        if self.fc is not None:
            output = self.dropout_layer(output)
            output = self.fc(output)

        return output

def test_Inception_v1():
    input_data = torch.randn(8,3,1920,1920)
    model = Inception_v1(num_classes=1000)
    output = model(input_data)
    
    print(output.shape)
    summary(model,(3,1920,1920))

def inception_v1_pretrained(path_to_weights='default'):
    model = Inception_v1(num_classes=1000)
    if path_to_weights.lower() == 'default':
        _currDir = os.path.dirname(os.path.abspath(__file__))
        path_to_weights = os.path.join(_currDir, 'inception_v1_weights.pth')
        model.load_state_dict(torch.load(path_to_weights))
    return model
if __name__ == '__main__':
    #test_InceptionNaive()
    #test_InceptionWithReduction()
    test_Inception_v1()
