import torch
import torch.nn as nn
import torch.nn.functional as functional

relu = functional.relu


class MNISTnet(nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.conv4 = nn.Conv2d(16, 8, 3)
        self.conv5 = nn.Conv2d(8, 4, 3)

        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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


        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        output = functional.log_softmax(x, dim=1)
        return output

# class MNISTnet(nn.Module):
#     def __init__(self):
#         super(MNISTnet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = relu(x)
#         x = self.conv2(x)
#         x = relu(x)
#         x = functional.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = functional.log_softmax(x, dim=1)
#         return output