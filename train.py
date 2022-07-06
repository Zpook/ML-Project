import torch
import torchvision
from torchvision import transforms as ttrans


def main():

    transforms = torchvision.transforms.Compose(
        [ttrans.ToTensor(), ttrans.Normalize((0.1307), (0.3081))]
    )

    dataset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=transforms)
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2
    )

    for batchIndex, (input, truth) in enumerate(dataLoader):
        pass


if __name__ == "__main__":
    main()
