import torch
import torchvision
from torchvision import transforms as ttrans
from nnmodel import MNISTnet
from tqdm import tqdm
import numpy as np

DEVICE = "cuda:0"
EPOCHS = 30
NUM_WORKERS = 12
BATCH = int(60000 / NUM_WORKERS)

def main():

    dataset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=ttrans.ToTensor())
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, num_workers=10
    )
    network = MNISTnet()

    lossFunc = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(network.parameters(),lr = 0.001)

    network = network.to(DEVICE)
    network.Initalize()
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1,end_factor=0.1,total_iters=EPOCHS)


    for epochIndex in range(EPOCHS):
        print("Epoch: " + epochIndex.__str__())
        cumLoss = 0
        for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
            
            input = input.to(DEVICE)
            truth = truth.to(DEVICE)

            optimizer.zero_grad()
            out = network(input)
            loss = lossFunc(out,truth)

            loss.backward()
            optimizer.step()

            cumLoss += loss.item()

        scheduler.step()
        print("Loss " + (cumLoss/dataset.__len__()).__str__())

    torch.save(network.state_dict(),"./model.pt")


if __name__ == "__main__":
    main()
