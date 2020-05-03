import numpy as np

from layer import *
from network import *
from utility import *

def getNetwork(fMin = 50, fMax = 100, vThreshold = 25, tau = 10, minSupervisedCurrent = -1, maxSupervisedCurrent = 1):
    #IN
    #np.float32 fMin: mean firing rate with input 0, in Hz
    #np.float32 fMax: mean firing rate with input 1, in Hz
    #np.float32 vThreshold: threshold voltage V_t in mV
    #np.float32 tau: time constant for spike response
    #np.float32 minSupervisedCurrent: min supervised input current with 0, in μA
    #np.float32 maxSupervisedCurrent: max supervised input current with 1, in μA
    #OUT
    #network.supervised SNN: spiking neuron network
    neuronLayerList = []
    neuronLayerList.append(poissonInput(2, fMin = 50, fMax = 250))
    neuronLayerList.append(IF(10, vThreshold = 0.9))
    neuronLayerList.append(IF(1, vThreshold = 0.25))
    SNN = supervised(neuronLayerList, minSupervisedCurrent, maxSupervisedCurrent, synapseConfig = {'tau': tau})
    return SNN

def getDataset():
    #OUT
    #np.ndarray dataX, dtype = np.flaot32: input data
    #list dataY [np.ndarray dataYi, dtype = np.float32]: supervised input for each layer
    dataX = np.empty((4, 2), dtype = np.float32)
    dataX[0] = (1, 1)
    dataX[1] = (1, 0)
    dataX[2] = (0, 1)
    dataX[3] = (0, 0)

    dataY0 = [0 for i in range(4)]
    #supervised idata for layer 1. here I use ((~x0) and x1, x0 and (not x1)). Maybe (x0 nand x1, x0 or x1) is better
    dataY1 = np.array(     ((~dataX[:, 0].astype(np.bool) & dataX[:, 1].astype(np.bool)).astype(np.float32),
                            (dataX[:, 0].astype(np.bool) & ~dataX[:, 1].astype(np.bool)).astype(np.float32)), dtype = np.float32).transpose()
    dataY2 = (dataX[:, 0].astype(np.bool) ^ dataX[:, 1].astype(np.bool)).astype(np.float32)
    return dataX, [(dataY0[i], dataY1[i], dataY2[i]) for i in range(4)]

def getAverageRate(spikeRateListList, dataSize):
    #IN
    #list spikeRateListList [[np.ndarray spikeRate, dtype = np.float32]]: [[spikeRate for each layer] for each input data] 
    #OUT
    #list averageSpikeRateList, [np.ndarray averageSpikeRate, dtype = np.float32]: average spiking rate in last iteration in Hz
    averageSpikeRateList = []
    for layerIdx in range(len(spikeRateListList[0])):
        layerSize = spikeRateListList[0][layerIdx].size
        tempSpikeRateList = np.empty((dataSize, layerSize), dtype = np.float32)
        for dataIdx in range(dataSize):
            tempSpikeRateList[dataIdx] = spikeRateListList[dataIdx][layerIdx]

        tempAverageSpikeRate = np.mean(tempSpikeRateList, axis = 0)
        averageSpikeRateList.append(tempAverageSpikeRate)
    return averageSpikeRateList

def preTrain(SNN, dataX, dataY, forwardTime = 1000, refreshTime = 1000):
    #IN
    #network.supervised SNN: spiking neuron network
    #np.ndarray dataX, dtype = np.flaot32: input data
    #list dataY [np.ndarray dataYi, dtype = np.float32]: supervised input for each layer
    #int forwardTime: time to forward
    #int refreshTime: time to refresh
    #OUT
    #list spikeRateListList [[np.ndarray spikeRate, dtype = np.float32]]: [[spikeRate for each layer] for each input data] 
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)

    spikeRateListList = []
    for idx in idxList:
        SNN.bcmPreUpdate(dataX[idx], dataY[idx], forwardTime, refreshTime)
        spikeRateListList.append(SNN.spikeRateList)
    return spikeRateListList

def decodeTemporal(spikeList):
    time = []
    for i in range(len(spikeList)):
        if spikeList[i]:
            time.append(i)
    return time

def train(SNN, dataX, dataY, iterNum, forwardTime = 1000, refreshTime = 1000, learningRate = 0.1):
    # IN
    # network.supervised SNN: spiking neuron network
    # np.ndarray dataX, dtype = np.flaot32: input data
    # list dataY [np.ndarray dataYi, dtype = np.float32]: supervised input for each layer
    # int iterNum: iteration to train
    # int forwardTime: time to forward
    # int refreshTime: time to refresh
    # np.float32 learningRate: step size for changing weights
    # OUT
    # list spikeRateListList [[np.ndarray spikeRate, dtype = np.float32]]: [[spikeRate for each layer] for each input data]
    # dataSize = dataX.shape[0]
    # idxList = np.array(range(dataSize), dtype = np.int8)
    # spikeRateListList = preTrain(SNN, dataX, dataY, forwardTime, refreshTime)
    #
    # for iters in range(iterNum):
    #     averageSpikeRateList = getAverageRate(spikeRateListList, dataSize)
    #     spikeRateListList = []
    #     print('iter %d: ' %iters)
    #     np.random.shuffle(idxList)
    #     for idx in idxList:
    #         print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
    #         #forward
    #         spikeRate = SNN.bcmPreUpdate(dataX[idx], dataY[idx], forwardTime, refreshTime)
    #         spikeRateListList.append(SNN.spikeRateList)
    #         print('%.2f' %spikeRate, end = '')
    #         # print(SNN.spikeRateList)
    #         #update
    #         SNN.bcmUpdate(averageSpikeRateList, learningRate)
    #         print(', ', end = '')
    #         #predict (for debug)
    #         SNN.refresh(refreshTime)
    #         spike = SNN.batchedPredict(dataX[idx], forwardTime)
    #         print('%.2f' %(np.sum(spike).astype(np.float32) / forwardTime * 1000))
    #     SNN._printWeight()
    # return
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    # spikeRateListList = preTrain(SNN, dataX, dataY, forwardTime, refreshTime)

    for iters in range(iterNum):
        print('iter %d: ' %iters)
        np.random.shuffle(idxList)
        for idx in idxList:
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            #forward
            spikeRate = SNN.stdpTrain(dataX[idx], dataY[idx][2], forwardTime, refreshTime)

            print('%.2f' %(np.sum(spikeRate).astype(np.float32) / forwardTime * 1000))
            SNN.reset()
            spikeRate = SNN.batchedPredict(dataX[idx], forwardTime)
            print('%.2f' %(np.sum(spikeRate).astype(np.float32) / forwardTime * 1000))
    for iter in range(10):
        for idx in idxList:
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            SNN.reset()
            spikeRate = SNN.batchedPredict(dataX[idx], forwardTime)
            print('%.2f' %(np.sum(spikeRate).astype(np.float32) / forwardTime * 1000))
    return

def test(SNN, dataX, iterNum, forwardTime = 1000, refreshTime = 1000):
    #IN
    #network.supervised SNN: spiking neuron network
    #np.ndarray dataX, dtype = np.flaot32: input data
    #int iterNum: iteration to train
    #int forwardTime: time to forward
    #int refreshTime: time to refresh
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)

    for iters in range(iterNum):
        print('iter %d: ' %iters)
        np.random.shuffle(idxList)
        for idx in idxList:
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            SNN.reset()
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            print('%.2f' %(np.sum(spike).astype(np.float32) / forwardTime * 1000))
    return


if __name__ == '__main__':
    fMin = 10
    fMax = 50
    vThreshold = 15
    tau = 10
    minSupervisedCurrent = -8
    maxSupervisedCurrent = 8
    forwardTime = 1000
    refreshTime = 2000
    learningRate = 1e-8

    SNN = getNetwork(fMin, fMax, vThreshold, tau)
    dataX, dataY = getDataset()
    for i in range(4):
        print(dataX[i], dataY[i])
    train(SNN, dataX, dataY, iterNum = 100, forwardTime = forwardTime, refreshTime = refreshTime, learningRate = learningRate)
    # test(SNN, dataX, iterNum=10)
    SNN._printWeight()