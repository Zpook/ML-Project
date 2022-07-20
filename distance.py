from typing import Tuple

import torch
import torchvision
import torchmetrics
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torchvision import transforms as ttrans
from nnmodel import MNISTnet
from tqdm import tqdm
from matplotlib import pyplot as plt
import sklearn.tree
import pickle

PSNR_CALC = torchmetrics.PeakSignalNoiseRatio(100)
SSIM_CALC = ssim

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def KMeans(doubleMap, inputs1, inputs2):
    dist1 = 0
    dist2 = 0

    for map in doubleMap:
        
        dist1 += torch.linalg.norm(inputs1 - map[0],ord=FEATURE_NORM,dim=(1,2))
        dist2 += torch.linalg.norm(inputs2 - map[1],ord=FEATURE_NORM,dim=(1,2))
    
    
    return dist1, dist2

def KNN(doubleMap, inputs1, inputs2):
    dist1 = []
    dist2 = []

    for map in doubleMap:
        
        tempDist1 = torch.linalg.norm(inputs1 - map[0],ord=FEATURE_NORM,dim=(1,2))
        tempDist2 = torch.linalg.norm(inputs2 - map[1],ord=FEATURE_NORM,dim=(1,2))

        dist1.append(tempDist1)
        dist2.append(tempDist2)

    dist1 = torch.stack(dist1)
    dist2 = torch.stack(dist2)
    
    return dist1, dist2

def PSNRDistance(doubleMap, inputs1, inputs2):
    dist1 = torch.zeros(inputs1.shape[0])
    dist2 = torch.zeros(inputs1.shape[0])

    doubleMap = torch.tensor(doubleMap)
    for map in doubleMap:
        for index in range(inputs1.shape[0]):
            iterIn1 = inputs1[index]
            iterIn2 = inputs2[index]

            dist1[index] += PSNR_CALC(iterIn1,map[0])
            dist2[index] += PSNR_CALC(iterIn2,map[1])
    
    return dist1, dist2

def SSIMDistance(doubleMap, inputs1, inputs2):
    dist1 = torch.zeros(inputs1.shape[0])
    dist2 = torch.zeros(inputs1.shape[0])

    doubleMap = torch.tensor(doubleMap)
    for map in doubleMap:
        for index in range(inputs1.shape[0]):
            iterIn1 = inputs1[index]
            iterIn2 = inputs2[index]

            dist1[index] += SSIM_CALC(iterIn1,map[0])
            dist2[index] += SSIM_CALC(iterIn2,map[1])
    
    return dist1, dist2

def KMeansClassif(allMaps,inputs):
    inputs1 = inputs[:,0,:,:]
    inputs2 = inputs[:,1,:,:]

    distances = []

    for number, doubleMap in allMaps.items():

        dist1,dist2 = KMeans(doubleMap,inputs1,inputs2)
        curDistances = torch.stack((dist1,dist2))
        curDistances = torch.linalg.norm(curDistances,ord=DISTANCE_NORM,dim=(0))

        distances.append(curDistances)

    distances = torch.stack(distances,dim=1)
    classifications = torch.argmin(distances,dim=1)

    return classifications

def KNNClassif(allMaps,inputs):
    inputs1 = inputs[:,0,:,:]
    inputs2 = inputs[:,1,:,:]

    distances = []

    for number, doubleMap in allMaps.items():

        dist1,dist2 = KNN(doubleMap,inputs1,inputs2)
        curDistances = torch.stack((dist1,dist2))
        curDistances = torch.linalg.norm(curDistances,ord=DISTANCE_NORM,dim=(0))

        distances.append(curDistances)

    distances = torch.stack(distances,dim=0)

    distances = distances.view(-1,distances.shape[2])
    top = torch.topk(distances,k=KNN_K,dim=0,largest=False)
    top = (top.indices / N_FEATURES).floor()


    
    classifications = torch.mode(top,dim=0).values

    return classifications


from FeatureSelect import SELECT_FEATURES


DEVICE = "cuda:0"
NUM_WORKERS = 12
BATCH = int(10000 / NUM_WORKERS)
N_FEATURES = SELECT_FEATURES

FEATURE_NORM = 1
DISTANCE_NORM = 1

CLASSIFIER = KNNClassif
CALC_MAPDICT = False
KNN_K = 1

SHOW_MAPS = False
SHOW_MAPS_COUNT = 10

SHOW_KMEANS_DISTANCES = False
SHOW_KMEANS_DISTANCES_NUMBER = 1

SHOW_CUNFUSION = False

if SHOW_MAPS:
    CALC_MAPDICT = True



def main():

    treeClassifier : sklearn.tree.DecisionTreeClassifier = pickle.load(open("./tree_classifier.pkl","rb"))


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

    FeatureMaps = torch.load("./Features.pt")

    mapDict = {0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{}}

    allMaps = []
    allInputs = []

    AllCNNLabels = torch.tensor([]).to(DEVICE)
    AllKNNLabels = torch.tensor([]).to(DEVICE)
    AllTreeLabels = torch.tensor([]).to(DEVICE)
    AllTrueLabels = torch.tensor([]).to(DEVICE)

    confusionMatCalc = torchmetrics.ConfusionMatrix(10).to(DEVICE)

    CNNErrors = 0
    KNNErrors = 0
    TreeErrors = 0

    for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
        
        input = input.to(DEVICE)
        truth = truth.to(DEVICE)

        out, mapsTorch = network.forward(input,returnMaps=True)

        allMaps.append(mapsTorch.cpu())
        allInputs.append(input)

        maps = mapsTorch.cpu().detach().numpy()
        scores = out.cpu().detach().numpy()

        CNNLabels = out.argmax(dim=1)
        KNNLabels = CLASSIFIER(FeatureMaps,mapsTorch.cpu().detach()).to(DEVICE)
        treeLabels = torch.tensor(treeClassifier.predict(mapsTorch.cpu().detach().flatten(1))).to(DEVICE)


        AllCNNLabels = torch.cat([AllCNNLabels,CNNLabels])
        AllKNNLabels = torch.cat([AllKNNLabels,KNNLabels])
        AllTreeLabels = torch.cat([AllTreeLabels,treeLabels])
        AllTrueLabels = torch.cat([AllTrueLabels,truth])

        CNNErrors += (CNNLabels != truth).sum().cpu()
        KNNErrors += (KNNLabels != truth).sum().cpu()
        TreeErrors += (treeLabels != truth).sum().cpu()

        if CALC_MAPDICT:
            for index in range(maps.shape[0]):
                
                map = maps[index]
                score = scores[index]
                iterInput = input[index][0].cpu().numpy()

                score = score[score.argmax()]

                mapDict[int(truth[index])][score] = [iterInput,map]

    datalen = dataset.__len__()
    CNNAccuracy = (datalen-CNNErrors)/datalen
    KNNAccuracy = (datalen-KNNErrors)/datalen
    TreeAccuracy = (datalen-TreeErrors)/datalen


    print("CNN Accuracy: " + (CNNAccuracy*100).__str__() + "%")
    print("NN Accuracy: " + (KNNAccuracy*100).__str__() + "%")
    print("Tree Accuracy: " + (TreeAccuracy*100).__str__() + "%")

    allMaps = torch.cat(allMaps)
    allInputs = torch.cat(allInputs)

    if SHOW_CUNFUSION:
        
        CNNConfuseMatrix = confusionMatCalc(AllCNNLabels.to(torch.int),AllTrueLabels.to(torch.int))
        NNConfuseMatrix = confusionMatCalc(AllKNNLabels.to(torch.int),AllTrueLabels.to(torch.int))

        fig, ax = plt.subplots(nrows=2)

        plt.xticks(np.arange(0, 10, 1))
        ax[0].imshow(CNNConfuseMatrix.cpu().numpy())
        ax[0].set_title("CNN")
        ax[0].set_xticks(np.arange(0, 10, 1))
        ax[0].set_yticks(np.arange(0, 10, 1))

        ax[1].imshow(NNConfuseMatrix.cpu().numpy())
        ax[1].set_title("Nearest")
        ax[1].set_xticks(np.arange(0, 10, 1))
        ax[1].set_yticks(np.arange(0, 10, 1))

        plt.show()



    if SHOW_MAPS:
        for index in range(SHOW_MAPS_COUNT):
            
            fig, ax = plt.subplots(nrows=3,ncols=10)

            for index2 in range(10):

                ax[0][index2].imshow(list(mapDict[index2].values())[index][0])
                ax[0][index2].set_title(index2)
                ax[1][index2].imshow(list(mapDict[index2].values())[index][1][0])
                ax[2][index2].imshow(list(mapDict[index2].values())[index][1][1])

                ax[0][index2].get_xaxis().set_visible(False)
                ax[0][index2].get_yaxis().set_visible(False)

                ax[1][index2].get_xaxis().set_visible(False)
                ax[1][index2].get_yaxis().set_visible(False)

                ax[2][index2].get_xaxis().set_visible(False)
                ax[2][index2].get_yaxis().set_visible(False)

            plt.show()
    

    if SHOW_KMEANS_DISTANCES:

        plt.figure()

        KNNMap = FeatureMaps[SHOW_KMEANS_DISTANCES_NUMBER]

        inputs1 = torch.tensor(allMaps)[:,0,:,:].cpu()
        inputs2 = torch.tensor(allMaps)[:,1,:,:].cpu()

        distances1, distances2 = KMeans(KNNMap,inputs1, inputs2)

        plt.scatter(distances1,distances2,c="red")

        inputs1 = torch.tensor(KNNMap)[:,0,:,:].cpu()
        inputs2 = torch.tensor(KNNMap)[:,1,:,:].cpu()

        distances1, distances2 = KMeans(KNNMap,inputs1, inputs2)

        plt.scatter(distances1,distances2,c="green")

        plt.show()



if __name__ == "__main__":
    main()