import numpy as np

from layer import *
from network import *
from utility import *
from picData import *

def getNetwork(kernel, fMin = 10, fMax = 200, capitance = 4, resistance = 16, vThreshold = 25, tau = 4, minSupervisedCurrent = -1, maxSupervisedCurrent = 1):
    #IN
    #np.ndarray kernel, shape = (3, 3), dtype = float32: weight of recepitive field
    #np.float32 fMin: mean firing rate with input 0, in Hz
    #np.float32 fMax: mean firing rate with input 1, in Hz
    #np.float32 capitance: C_m in μF
    #np.float32 resistance: R_m in kΩ
    #np.float32 vThreshold: threshold voltage V_t in mV
    #np.float32 tau: time constant for spike response
    #np.float32 minSupervisedCurrent: min supervised input current with 0, in μA
    #np.float32 maxSupervisedCurrent: max supervised input current with 1, in μA
    #OUT
    #network.supervised SNN: spiking neuron network

    neuronLayerList = []
    neuronLayerList.append(onOffLayer(7, 7, kernel, 8, fMin, fMax))
    neuronLayerList.append(supervisedLIF(7, capitance = capitance, resistance = resistance, vThreshold = vThreshold))
    SNN = supervised(neuronLayerList, minSupervisedCurrent, maxSupervisedCurrent, synapseConfig = {'tau': tau})
    return SNN


def test(SNN, dataX, iterNum, forwardTime = 1000, plot = False, legend = True):
    #IN
    #network.supervised SNN: spiking neuron network
    #np.ndarray dataX, dtype = np.float32: input data
    #int iterNum: iteration to train
    #int forwardTime: time to forward
    #bool plot: True: plot spike list; False: no plot
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)

    for iters in range(iterNum):
        print('iter %d: ' %iters)
        np.random.shuffle(idxList)
        for idx in idxList:
            print(' %d' %idx , end = '')
            SNN.reset()
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            # spike = np.sum(SNN.spikeListList[0], axis = 0)
            # print(np.sum(spike.reshape(7, 7, 2), axis = (0)))
            print(np.sum(spike, axis = 0).astype(np.float32) / forwardTime * 1000)
            if plot is True:
                plotSpikeList(SNN.spikeListList, legend = legend)
    return SNN

def SetLayer1Weight():
    #OUT
    #np.ndarray weight, dtype = float32: new synapse weights
    tempWeight = np.zeros((98, 7), dtype = np.float32)
    for row in range(7):
        rowWeight = np.zeros((2, 7, 7), dtype = np.float32)
        #on neuron
        rowWeight[1, row, :] = 1
        #off neuron
        if row != 0:
            rowWeight[0, row - 1, :] = 0.5
        if row != 6:
            rowWeight[0, row + 1, :] = 0.5
        tempWeight[:, row] = rowWeight.reshape(-1)
    return tempWeight


if __name__ == '__main__':
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = np.float32)
    fMin = 10
    fMax = 200
    capitance = 2
    resistance = 8
    vThreshold = 25
    tau = 16
    minSupervisedCurrent = -1
    maxSupervisedCurrent = 1

    SNN = getNetwork(kernel, fMin, fMax, capitance, resistance, vThreshold, tau, minSupervisedCurrent, maxSupervisedCurrent)
    SNN.synapseLayerList[-1].setWeight(SetLayer1Weight)

    #different angles
    dataX = np.array(lineData().dataX, dtype = np.float32)
    test(SNN, dataX, 1, forwardTime = 1000, plot = True, legend = False)

    #different positions
    dataX = np.array(extendLineData().dataX, dtype = np.float32)
    test(SNN, dataX, 1, forwardTime = 1000, plot = True, legend = False)