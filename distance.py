import torch
import torchvision
import torchmetrics
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torchvision import transforms as ttrans
from nnmodel import MNISTnet
from tqdm import tqdm
from matplotlib import pyplot as plt

PSNR_CALC = torchmetrics.PeakSignalNoiseRatio(100)
SSIM_CALC = ssim

def NormDistance(doubleMap, inputs1, inputs2):
    dist1 = 0
    dist2 = 0

    for map in doubleMap:
        
        dist1 += torch.linalg.norm(inputs1 - map[0],ord=MAP_NORM_ORD,dim=(1,2))
        dist2 += torch.linalg.norm(inputs2 - map[1],ord=MAP_NORM_ORD,dim=(1,2))
    
    
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

def KClassif(allMaps,inputs, distFunc):
    inputs1 = inputs[:,0,:,:]
    inputs2 = inputs[:,1,:,:]

    distances = []

    for number, doubleMap in allMaps.items():

        dist1,dist2 = distFunc(doubleMap,inputs1,inputs2)
        curDistances = torch.stack((dist1,dist2))
        curDistances = torch.linalg.norm(curDistances,ord=KNN_NORM,dim=(0))

        distances.append(curDistances)

    distances = torch.stack(distances,dim=1)
    classifications = torch.argmin(distances,dim=1)

    return classifications




DEVICE = "cuda:0"
NUM_WORKERS = 12
BATCH = int(10000 / NUM_WORKERS)

MAP_NORM_ORD = 1
KNN_NORM = 1

MAP_DIST_FUNC = NormDistance
CALC_MAPDICT = False


SHOW_MAPS = True
SHOW_MAPS_NUMBER = 10

SHOW_DISTANCES = False
SHOW_DISTANCES_NUMBER = 0

SHOW_CUNFUSION = False



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

    allMaps = []
    allInputs = []

    AllCNNLabels = torch.tensor([]).to(DEVICE)
    AllKNNLabels = torch.tensor([]).to(DEVICE)
    AllTrueLabels = torch.tensor([]).to(DEVICE)

    confusionMatCalc = torchmetrics.ConfusionMatrix(10).to(DEVICE)

    CNNErrors = 0
    KNNErrors = 0
    for batchIndex, (input, truth) in tqdm(enumerate(dataLoader)):
        
        input = input.to(DEVICE)
        truth = truth.to(DEVICE)

        out, mapsTorch = network.forward(input,returnMaps=True)

        allMaps.append(mapsTorch)
        allInputs.append(input)

        maps = mapsTorch.cpu().detach().numpy()
        scores = out.cpu().detach().numpy()

        CNNLabels = out.argmax(dim=1)
        KNNLabels = KClassif(KNNMaps,mapsTorch.cpu().detach(), MAP_DIST_FUNC).to(DEVICE)

        AllCNNLabels = torch.cat([AllCNNLabels,CNNLabels])
        AllKNNLabels = torch.cat([AllKNNLabels,KNNLabels])
        AllTrueLabels = torch.cat([AllTrueLabels,truth])

        CNNErrors += (CNNLabels != truth).sum().cpu()
        KNNErrors += (KNNLabels != truth).sum().cpu()

        if CALC_MAPDICT:
            for index in range(maps.shape[0]):
                
                map = maps[index]
                score = scores[index]

                score = score[score.argmax()]

                mapDict[int(truth[index])][score] = map

    datalen = dataset.__len__()
    CNNAccuracy = (datalen-CNNErrors)/datalen
    KNNAccuracy = (datalen-KNNErrors)/datalen



    print("CNN Accuracy: " + (CNNAccuracy*100).__str__() + "%")
    print("KNN Accuracy: " + (KNNAccuracy*100).__str__() + "%")


    allMaps = torch.cat(allMaps)
    allInputs = torch.cat(allInputs)

    if SHOW_CUNFUSION:
        
        CNNConfuseMatrix = confusionMatCalc(AllCNNLabels.to(torch.int),AllTrueLabels.to(torch.int))
        KNNConfuseMatrix = confusionMatCalc(AllKNNLabels.to(torch.int),AllTrueLabels.to(torch.int))

        fig, ax = plt.subplots(nrows=2)

        plt.xticks(np.arange(0, 10, 1))
        ax[0].imshow(CNNConfuseMatrix.cpu().numpy())
        ax[0].set_title("CNN")
        ax[0].set_xticks(np.arange(0, 10, 1))
        ax[0].set_yticks(np.arange(0, 10, 1))

        ax[1].imshow(KNNConfuseMatrix.cpu().numpy())
        ax[1].set_title("KNN")
        ax[1].set_xticks(np.arange(0, 10, 1))
        ax[1].set_yticks(np.arange(0, 10, 1))

        plt.show()



    if SHOW_MAPS:
        for index in range(SHOW_MAPS_NUMBER):
            
            fig, ax = plt.subplots(nrows=2,ncols=10)

            for index2 in range(10):

                ax[0][index2].imshow(list(mapDict[index2].values())[index][0])
                ax[0][index2].set_title(index2)
                ax[1][index2].imshow(list(mapDict[index2].values())[index][1])

            plt.show()
    

    if SHOW_DISTANCES:

        plt.figure()

        KNNMap = KNNMaps[SHOW_DISTANCES_NUMBER]
        
        allPoints = []
        for key in mapDict.keys():
            allPoints.extend(list(mapDict[key].values()))

        inputs1 = torch.tensor(allPoints)[:,0,:,:]
        inputs2 = torch.tensor(allPoints)[:,1,:,:]

        distances1, distances2 = NormDistance(KNNMap,inputs1, inputs2)

        plt.scatter(distances1,distances2,c="red")

        inputs1 = torch.tensor(list(mapDict[0].values()))[:,0,:,:]
        inputs2 = torch.tensor(list(mapDict[0].values()))[:,1,:,:]

        distances1, distances2 = NormDistance(KNNMap,inputs1, inputs2)

        plt.scatter(distances1,distances2,c="green")

        plt.show()



if __name__ == "__main__":
    main()