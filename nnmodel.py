import torch
import torch.nn as nn
import torch.nn.functional as functional

relu = functional.relu
CONV_INIT_STD = 0.1
LINEAR_INIT_STD = 0.1


class MNISTnet(nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.conv4 = nn.Conv2d(16, 8, 3)
        self.conv5 = nn.Conv2d(8, 4, 3)
        self.conv6 = nn.Conv2d(4, 2, 3)


        self.fc1 = nn.Linear(200, 128)
        self.fc2 = nn.Linear(128, 10)

    def Initalize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, CONV_INIT_STD)
                module.bias.data.normal_(0.0, CONV_INIT_STD)

            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, LINEAR_INIT_STD)
                module.bias.data.normal_(0.0, LINEAR_INIT_STD)



    def forward(self, x, returnMaps: bool = False):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        x = self.conv4(x)
        x = relu(x)
        x = self.conv5(x)
        x = relu(x)
        x = self.conv6(x)
        x = relu(x)

        if(returnMaps):
            conv5 = x.clone()


        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        output = functional.log_softmax(x, dim=1)

        if returnMaps:
            return output, conv5
        
        return output