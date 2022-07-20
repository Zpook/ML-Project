import imp
import torch
import torchvision
from torchvision import transforms as ttrans
import numpy as np
from nnmodel import MNISTnet
from tqdm import tqdm
from FeatureSelect import MapDistances
import sklearn.tree
import pickle

DEVICE = "cuda:0"
NUM_WORKERS = 12
BATCH = int(10000 / NUM_WORKERS)

SELECT_FEATURES = 50
RANDOM_SELECT = False
AVERAGE_MAPS = False


if __name__ == "__main__":

    transforms = ttrans.ToTensor()
    dataset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=transforms)


    treeClassifier = sklearn.tree.DecisionTreeClassifier()

    transforms = ttrans.ToTensor()
    dataset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=transforms)
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS
    )
    network = MNISTnet()

    stateDict = torch.load("./model.pt")
    network = network.to(DEVICE)
    network.eval()
    network.load_state_dict(stateDict,strict=True)

    AllMaps = torch.tensor([])

    for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
        
        input = input.to(DEVICE)
        truth = truth.to(DEVICE)


        out, maps = network.forward(input,returnMaps=True)

        maps = maps.cpu().detach()
        AllMaps = torch.cat([AllMaps,maps])
        # scores = out.cpu().detach().numpy()


    treeClassifier.fit(AllMaps.flatten(1),dataset.train_labels)

    pickle.dump(treeClassifier,open("tree_classifier.pkl","wb"))