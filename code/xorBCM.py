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
    neuronLayerList.append(poissonInput(2, fMin = fMin, fMax = fMax))
    neuronLayerList.append(supervisedLIF(2, vThreshold = vThreshold))
    # neuronLayerList.append(supervisedLIF(1, vThreshold = vThreshold))
    SNN = supervised(neuronLayerList, minSupervisedCurrent, maxSupervisedCurrent, synapseConfig = {'tau': tau})
    return SNN

def getDataset(layerSize):
    #OUT
    #np.ndarray dataX, dtype = np.flaot32: input data
    #list dataY [np.ndarray dataYi, dtype = np.float32]: supervised input for each layer
    dataX = np.empty((4, 2), dtype = np.float32)
    dataX[0] = (1, 1)
    dataX[1] = (1, 0)
    dataX[2] = (0, 1)
    dataX[3] = (0, 0)

    dataYList = []

    dataYList.append([0 for i in range(4)])
    #supervised idata for layer 1. here I use ((~x0) and x1, x0 and (not x1)). Maybe (x0 nand x1, x0 or x1) is better
    dataYList.append(np.array(     ((~dataX[:, 0].astype(np.bool) & dataX[:, 1].astype(np.bool)).astype(np.float32),
                                    (dataX[:, 0].astype(np.bool) & ~dataX[:, 1].astype(np.bool)).astype(np.float32)), dtype = np.float32).transpose())
    # dataYList.append(np.array(     ((~dataX[:, 0].astype(np.bool) | ~dataX[:, 1].astype(np.bool)).astype(np.float32),
    #                                 (dataX[:, 0].astype(np.bool) | dataX[:, 1].astype(np.bool)).astype(np.float32)), dtype = np.float32).transpose())
    dataYList.append((dataX[:, 0].astype(np.bool) ^ dataX[:, 1].astype(np.bool)).astype(np.float32))
    # dataY = [(dataY0[i], dataY1[i], dataY2[i]) for i in range(4)]
    dataY = [[dataYList[layer][i] for layer in range(layerSize)] for i in range(4)]
    return dataX, dataY

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

def preTrain(SNN, dataX, dataY, forwardTime = 1000):
    #IN
    #network.supervised SNN: spiking neuron network
    #np.ndarray dataX, dtype = np.flaot32: input data
    #list dataY [np.ndarray dataYi, dtype = np.float32]: supervised input for each layer
    #int forwardTime: time to forward
    #OUT
    #list spikeRateListList [[np.ndarray spikeRate, dtype = np.float32]]: [[spikeRate for each layer] for each input data] 
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)

    spikeRateListList = []
    for idx in idxList:
        SNN.bcmPreUpdate(dataX[idx], dataY[idx], forwardTime)
        spikeRateListList.append(SNN.spikeRateList)
    return spikeRateListList


def train(SNN, dataX, dataY, iterNum, forwardTime = 1000, learningRate = 0.1, layerConstrainList = None, trainLayerSet = None):
    #IN
    #network.supervised SNN: spiking neuron network
    #np.ndarray dataX, dtype = np.flaot32: input data
    #list dataY [np.ndarray dataYi, dtype = np.float32]: supervised input for each layer
    #int iterNum: iteration to train
    #int forwardTime: time to forward
    #np.float32 learningRate: step size for changing weights
    #list layerconstrainList [function layerConstrain]: constrains of weights for each layer
    #set trainLayerset: the index of layer that need to train
    #OUT
    #list spikeRateListList [[np.ndarray spikeRate, dtype = np.float32]]: [[spikeRate for each layer] for each input data]
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    spikeRateListList = preTrain(SNN, dataX, dataY, forwardTime)

    for iters in range(iterNum):
        print('iter %d: ' %iters)
        np.random.shuffle(idxList)
        for idx in idxList:
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            #forward

    return
            spikeRate = SNN.bcmPreUpdate(dataX[idx], dataY[idx], forwardTime)
            spikeRateListList.append(SNN.spikeRateList)
            print(spikeRate, end = '')
            # print(SNN.spikeRateList)
            #update
            SNN.bcmUpdate(averageSpikeRateList, learningRate, forwardTime, layerConstrainList, trainLayerSet)
            print(', ', end = '')
            #predict (for debug)
            SNN.reset()
            # SNN.refresh(refreshTime)
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            print(np.sum(spike, axis = 0).astype(np.float32) / forwardTime * 1000)
        SNN._printWeight()
        test(SNN, dataX, iterNum = 1, forwardTime = forwardTime)
    return SNN

def test(SNN, dataX, iterNum, forwardTime = 1000, plot = False):
    #IN
    #network.supervised SNN: spiking neuron network
    #np.ndarray dataX, dtype = np.flaot32: input data
    #int iterNum: iteration to train
    #int forwardTime: time to forward
    #bool plot: True: plot spike list; False: no plot
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)

    for iters in range(iterNum):
        print('iter %d: ' %iters)
        np.random.shuffle(idxList)
        for idx in idxList:
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            SNN.reset()
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            print(np.sum(spike, axis = 0).astype(np.float32) / forwardTime * 1000)
            if plot is True:
                plotSpikeList(SNN.spikeListList)
    return SNN


def layer1Constrain(weight):
    #IN
    #np.ndarray weight, shape = (2, 2), dtype = float32: synapse weights
    #OUT
    #np.ndarray weight, shape = (2, 2), dtype = float32: synapse weights with constrains
    tempWeight = np.empty_like(weight)
    tempWeight[0, 0] = (weight[0, 0] + weight[1, 1]) / 2
    tempWeight[1, 1] = (weight[0, 0] + weight[1, 1]) / 2
    tempWeight[0, 1] = (weight[0, 1] + weight[1, 0]) / 2
    tempWeight[1, 0] = (weight[0, 1] + weight[1, 0]) / 2
    return tempWeight

def layer2Constrain(weight):
    #IN
    #np.ndarray weight, shape = (2, 1), dtype = float32: synapse weights
    #OUT
    #np.ndarray weight, shape = (2, 1), dtype = float32: synapse weights with constrains
    tempWeight = np.empty_like(weight)
    tempWeight[0, 0] = (weight[0, 0] + weight[1, 0]) / 2
    tempWeight[1, 0] = (weight[0, 0] + weight[1, 0]) / 2
    return tempWeight


def trainLayer1(fMin, fMax, vThreshold, tau, minSupervisedCurrent, maxSupervisedCurrent, forwardTime, learningRate):
    #IN
    #np.float32 fMin: mean firing rate with input 0, in Hz
    #np.float32 fMax: mean firing rate with input 1, in Hz
    #np.float32 vThreshold: threshold voltage V_t in mV
    #np.float32 tau: time constant for spike response
    #np.float32 minSupervisedCurrent: min supervised input current with 0, in μA
    #np.float32 maxSupervisedCurrent: max supervised input current with 1, in μA
    #int forwardTime: time to forward
    #np.float32 learningRate: step size for changing weights
    #OUT
    #network.supervised SNN: 2-layer spiking neuron network
    SNN = getNetwork(fMin, fMax, vThreshold, tau, minSupervisedCurrent, maxSupervisedCurrent)
    dataX, dataY = getDataset(layerSize = 2)
    for i in range(4):
        print(dataX[i], dataY[i])
    SNN = train(SNN, dataX, dataY, iterNum = 19, forwardTime = forwardTime, learningRate = learningRate, layerConstrainList = [layer1Constrain], trainLayerSet = {0})
    SNN._printWeight()
    return SNN

def trainLayer2(SNN, vThreshold, forwardTime, learningRate):
    #IN
    #network.supervised SNN: 2-layer spiking neuron network
    #np.float32 vThreshold: threshold voltage V_t in mV
    #int forwardTime: time to forward
    #np.float32 learningRate: step size for changing weights
    #OUT
    #network.supervised SNN: 3-layer spiking neuron network
    SNN.extend(supervisedLIF(1, vThreshold = vThreshold))
    dataX, dataY = getDataset(layerSize = 3)
    for i in range(4):
        print(dataX[i], dataY[i])

    SNN = train(SNN, dataX, dataY, iterNum = 5, forwardTime = forwardTime, learningRate = learningRate, layerConstrainList = [layer1Constrain, layer2Constrain], trainLayerSet = {1})
    SNN._printWeight()
    return SNN



if __name__ == '__main__':
    np.random.seed(6983)
    fMin = 100
    fMax = 200
    vThreshold = 12
    tau = 8
    minSupervisedCurrent = -4
    maxSupervisedCurrent = 4
    forwardTime = 500
    learningRate = 8e-10

    print('train layer 1')
    SNN = trainLayer1(fMin, fMax, vThreshold, tau, minSupervisedCurrent, maxSupervisedCurrent, forwardTime, learningRate)
    print('train layer 2')
    SNN = trainLayer2(SNN, vThreshold, forwardTime, learningRate)
    dataX, _ = getDataset(layerSize = 3)
    print('test')
    SNN = test(SNN, dataX, iterNum = 5, forwardTime = 1000, plot = True)
