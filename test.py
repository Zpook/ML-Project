import torch
import torchvision
from torchvision import transforms as ttrans
from nnmodel import MNISTnet
from tqdm import tqdm


DEVICE = "cuda:0"
NUM_WORKERS = 12
BATCH = int(60000 / NUM_WORKERS)


def main():

    transforms = ttrans.ToTensor()

    dataset = torchvision.datasets.MNIST("./dataset/", train=False, download=True, transform=transforms)
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS
    )
    network = MNISTnet()

    stateDict = torch.load("./model.pt")
    network = network.to(DEVICE)
    network.eval()
    network.load_state_dict(stateDict,strict=True)

    errors = 0
    for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
        
        input = input.to(DEVICE)
        truth = truth.to(DEVICE)

        out = network(input)

        label = out.argmax(dim=1)

        errors += (label != truth).sum()

    datalen = dataset.__len__()
    accuracy = (datalen-errors)/datalen
    print("Accuracy: " + (accuracy*100).__str__() + "%")



if __name__ == "__main__":
    main()
