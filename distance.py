import torch
import torchvision
from torchvision import transforms as ttrans
from nnmodel import MNISTnet
from tqdm import tqdm
from matplotlib import pyplot as plt

SHOW_MAPS = False
SHOW_MAPS_NUMBER = 10

SHOW_DISTANCES = False
SHOW_DISTANCES_NUMBER = 0

DEVICE = "cuda:0"
NUM_WORKERS = 12
BATCH = int(10000 / NUM_WORKERS)

MAP_NORM_ORD = 2
DIST_NORM_ORD = 2

def MapDistances(singleMap, inputs):
    diffs = inputs - singleMap
    return torch.linalg.norm(diffs,ord=MAP_NORM_ORD,dim=(1,2))

def KDistances(doubleMap, inputs1, inputs2):
    dist1 = 0
    dist2 = 0

    for map in doubleMap:
        dist1 += MapDistances(map[0],inputs1)
        dist2 += MapDistances(map[1],inputs2)
    
    
    return dist1, dist2

def KClassif(allMaps,inputs):
    inputs1 = inputs[:,0,:,:]
    inputs2 = inputs[:,1,:,:]

    distances = []

    for number, doubleMap in allMaps.items():

        dist1,dist2 = KDistances(doubleMap,inputs1,inputs2)
        curDistances = torch.stack((dist1,dist2))
        curDistances = torch.linalg.norm(curDistances,ord=DIST_NORM_ORD,dim=(0))

        distances.append(curDistances)

    distances = torch.stack(distances,dim=1)
    classifications = torch.argmin(distances,dim=1)

    return classifications

def main():

    transforms = torchvision.transforms.Compose(
        [ttrans.ToTensor(), ttrans.Normalize((0.1307), (0.3081))]
    )

    dataset = torchvision.datasets.MNIST("./dataset/", train=False, download=True, transform=transforms)
    dataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS
    )
    network = MNISTnet()

    stateDict = torch.load("./model.pt")
    network = network.to(DEVICE)
    network.eval()
    network.load_state_dict(stateDict,strict=True)

    KNNMaps = torch.load("./KNN.pt")

    mapDict = {0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{}}

    CNNErrors = 0
    KNNErrors = 0
    for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
        
        input = input.to(DEVICE)
        truth = truth.to(DEVICE)

        out, mapsTorch = network.forward(input,returnMaps=True)
        maps = mapsTorch.cpu().detach().numpy()
        scores = out.cpu().detach().numpy()

        CNNLabels = out.argmax(dim=1)
        KNNLabels = KClassif(KNNMaps,mapsTorch.cpu().detach()).to(DEVICE)

        CNNErrors += (CNNLabels != truth).sum().cpu()
        KNNErrors += (KNNLabels != truth).sum().cpu()

        for index in range(maps.shape[0]):
            
            map = maps[index]
            score = scores[index]

            score = score[score.argmax()]

            mapDict[int(truth[index])][score] = map

    # for key in mapDict.keys():
    #     mapDict[key] = {k: mapDict[key][k] for k in sorted(mapDict[key],reverse=True)}

    datalen = dataset.__len__()
    CNNAccuracy = (datalen-CNNErrors)/datalen
    KNNAccuracy = (datalen-KNNErrors)/datalen

    print("CNN Accuracy: " + (CNNAccuracy*100).__str__() + "%")
    print("KNN Accuracy: " + (KNNAccuracy*100).__str__() + "%")


    if SHOW_MAPS:
        for index in range(SHOW_MAPS_NUMBER):
            
            fig, ax = plt.subplots(nrows=2,ncols=3)

            ax[0][0].imshow(list(mapDict[0].values())[index][0])
            ax[0][0].set_title(0)
            ax[1][0].imshow(list(mapDict[0].values())[index][1])

            ax[0][1].imshow(list(mapDict[1].values())[index][0])
            ax[0][1].set_title(1)
            ax[1][1].imshow(list(mapDict[1].values())[index][1])

            ax[0][2].imshow(list(mapDict[2].values())[index][0])
            ax[0][2].set_title(2)
            ax[1][2].imshow(list(mapDict[2].values())[index][1])

            plt.show()
    

    if SHOW_DISTANCES:

        plt.figure()

        KNNMap = KNNMaps[SHOW_DISTANCES_NUMBER]
        
        allPoints = []
        for key in mapDict.keys():
            allPoints.extend(list(mapDict[key].values()))

        inputs1 = torch.tensor(allPoints)[:,0,:,:]
        inputs2 = torch.tensor(allPoints)[:,1,:,:]

        distances1, distances2 = KDistances(KNNMap,inputs1, inputs2)

        plt.scatter(distances1,distances2,c="red")

        inputs1 = torch.tensor(list(mapDict[0].values()))[:,0,:,:]
        inputs2 = torch.tensor(list(mapDict[0].values()))[:,1,:,:]

        distances1, distances2 = KDistances(KNNMap,inputs1, inputs2)

        plt.scatter(distances1,distances2,c="green")

        plt.show()



if __name__ == "__main__":
    main()