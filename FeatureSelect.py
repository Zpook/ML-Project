import torch
import torchvision
from torchvision import transforms as ttrans
import numpy as np
from nnmodel import MNISTnet
from tqdm import tqdm


SHOW_MAPS = False
SHOW_MAPS_NUMBER = 10

DEVICE = "cuda:0"
NUM_WORKERS = 12
BATCH = int(10000 / NUM_WORKERS)

SELECT_FEATURES = 50
RANDOM_SELECT = False
AVERAGE_MAPS = False

def MapDistances(map, inputs):
    diffs = inputs - map
    return torch.linalg.norm(diffs,ord=2,dim=(1,2))


def main():

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

    mapDict = {0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{}}

    for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
        
        input = input.to(DEVICE)
        truth = truth.to(DEVICE)


        out, maps = network.forward(input,returnMaps=True)
        maps = maps.cpu().detach().numpy()
        scores = out.cpu().detach().numpy()

        for index in range(maps.shape[0]):
            
            map = maps[index]
            score = scores[index]

            score = score[score.argmax()]

            mapDict[int(truth[index])][score] = map

    finalMap = {}

    for key in mapDict.keys():
        mapDict[key] = {k: mapDict[key][k] for k in sorted(mapDict[key],reverse=True)}

        selectionRange = list(mapDict[key].values())


        if RANDOM_SELECT:
            indexes = np.random.choice(selectionRange.__len__(),SELECT_FEATURES)
            selectionRange = np.array(selectionRange)[indexes.astype(int)]

        finalMap[key] = selectionRange[0:SELECT_FEATURES]

        if AVERAGE_MAPS:
            averaged = 0
            for matrix in finalMap[key]:
                averaged += matrix
            finalMap[key] = averaged / finalMap[key].__len__()


    torch.save(finalMap,"./Features.pt")




if __name__ == "__main__":
    main()