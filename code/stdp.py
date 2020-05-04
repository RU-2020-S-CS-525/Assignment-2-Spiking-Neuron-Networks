"""
This module contains classes that can learn the weights.
"""

import numpy as np
from layer import poissonInput
from layer import forwardLIF as LIF, supervisedLIF, synapseLayer
from network import supervised
from xorBCM import getDataset
import random


def getNetwork(fMin = 20, fMax = 100, capitance = 1, resistance = 1, vThreshold = 5, vRest = -65, tau = 10, minSupervisedCurrent =
               -1, maxSupervisedCurrent = 1, learningRate = 1, dt = 1e-3):
    #IN
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
    neuronLayerList.append(poissonInput(2, fMin = fMin, fMax = fMax))
    neuronLayerList.append(supervisedLIF(2, capitance = capitance, resistance = resistance, vThreshold = vThreshold, dt = dt, vRest =
                                        vRest))
    neuronLayerList.append(supervisedLIF(1, capitance = capitance, resistance = resistance, vThreshold = vThreshold, dt = dt, vRest =
                                        vRest))
    SNN = supervised(neuronLayerList, minSupervisedCurrent, maxSupervisedCurrent, learningRate = learningRate, dt = dt, synapseConfig
                     = {'tau': tau, 'dt': dt})
    return SNN


#def getDataset():
#    dataX = np.empty((4, 2), dtype = np.float16)
#    dataX[0] = (1, 1)
#    dataX[1] = (1, 0)
#    dataX[2] = (0, 1)
#    dataX[3] = (0, 0)
#    dataY = (dataX[:, 0].astype(np.bool) ^ dataX[:, 1].astype(np.bool)).astype(np.float16)
#    return dataX, dataY


def train(SNN, dataX, dataY, iterNum, stdp_config, forwardTime = 1000, refreshTime = 1000):
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    for iters in range(iterNum):
        try:
            print('iter %d: ' %iters)
            np.random.shuffle(idxList)
            for idx in idxList:
                print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
                spike = SNN.stdpTrain(dataX[idx], dataY[idx], stdp_config, forwardTime, refreshTime)
                print('%.2f' %(np.sum(spike).astype(np.float16)), end = ', ')
                #print('%.2f %.2f' %(np.sum(SNN.spikeListList[0][:,0]).astype(np.float16), np.sum(SNN.spikeListList[0][:,1]).astype(np.float16)), end = ', ')
                spike = SNN.batchedPredict(dataX[idx], forwardTime)
                print('%.2f' %(np.sum(spike).astype(np.float16)))
        except KeyboardInterrupt:
            SNN._printWeight()
            if input('exit training?') == 'y':
                return
    return


def test(SNN, dataX, forwardTime = 1):
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    np.random.shuffle(idxList)
    for j in range(3):
        for idx in idxList:
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            print('%.2f' %(np.sum(spike).astype(np.float16)))
    return



if __name__ == "__main__":
    random.seed(114514)
    np.random.seed(114514)

    n_iter = 100
    tau_x = 2e-2
    tau_y = 2e-2
    F_x = 0.01
    F_y = 1.05*F_x
    stdp_config = {
        'tau_x': tau_x,
        'tau_y': tau_y,
        'F_x': F_x,
        'F_y': F_y
    }
    input_neuron = poissonInput(size = 2)
    dataX, dataY = getDataset(3)
    time = 1
    dt = 1e-3
    learningRate = 10
    stepNum = int(time/dt)
    SNN = getNetwork(learningRate = learningRate, dt = dt)
    SNN._printWeight()
    print('---- START TRAINING ----')
    train(SNN, dataX, dataY, n_iter, stdp_config, forwardTime=1, refreshTime=1)
    SNN._printWeight()
    print('---- TESTING ----')
    test(SNN, dataX, forwardTime=1)
