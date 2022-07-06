import torch
import torchvision
from torchvision import transforms as ttrans
from nnmodel import MNISTnet
from tqdm import tqdm


DEVICE = "cuda:0"
EPOCHS = 30
BATCH = 1000
PRINT_PREDICTED = False

def main():

    dataset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=ttrans.ToTensor())
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, num_workers=2
    )
    network = MNISTnet()

    lossFunc = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(network.parameters(),lr = 0.005)

    network = network.to(DEVICE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


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

            if PRINT_PREDICTED:
                predicted = out.argmax(dim=1)
                print(predicted)

            cumLoss += loss.item()

        scheduler.step()
        print("Loss " + (cumLoss/dataset.__len__()).__str__())

    torch.save(network.state_dict(),"./model.pt")


if __name__ == "__main__":
    main()
