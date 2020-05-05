import numpy as np
from layer import poissonInput, fixedNeuron
from layer import forwardLIF as LIF, supervisedLIF, synapseLayer
from network import supervised
from xorBCM import getDataset
from utility import plotSpike
import random
from utility import plotSpikeList

def getNetwork(fMin = 50, fMax = 250, capitance = 1, resistance = 1, vThreshold = 0.7, vRest = 0, tau = 20, minSupervisedCurrent =
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
    neuronLayerList.append(LIF(10, capitance = capitance, resistance = resistance, vThreshold = 0.9, dt = dt, vRest =
                                        vRest))
    neuronLayerList.append(LIF(1, capitance = capitance, resistance = resistance, vThreshold = 0.25, dt = dt, vRest =
                                        vRest))
    SNN = supervised(neuronLayerList, minSupervisedCurrent, maxSupervisedCurrent, learningRate = learningRate, dt = dt, synapseConfig
                     = {'tau': tau, 'dt': dt})
    return SNN


def beforeTrain(SNN, fMin = 20, fMax = 100, capitance = 1, resistance = 1, vThreshold = 5, vRest = -65, tau = 10, minSupervisedCurrent =
               -1, maxSupervisedCurrent = 1, learningRate = 1, dt = 1e-3):
    SNN.neuronLayerList.pop()
    SNN.neuronLayerList.append(fixedNeuron(1, fMin = fMin, fMax = fMax, capitance = capitance, resistance = resistance, vThreshold = vThreshold, dt = dt, vRest =
                                        vRest))


def afterTrain(SNN, fMin = 20, fMax = 100, capitance = 1, resistance = 1, vThreshold = 5, vRest = -65, tau = 10, minSupervisedCurrent =
               -1, maxSupervisedCurrent = 1, learningRate = 1, dt = 1e-3):
    SNN.neuronLayerList.pop()
    SNN.neuronLayerList.append(LIF(1, capitance = capitance, resistance = resistance, vThreshold = vThreshold, dt = dt, vRest =
                                        vRest))


def train(SNN, dataX, dataY, iterNum, stdp_config, forwardTime = 1000, refreshTime = 1000):
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    for iters in range(iterNum):
        try:
            print('iter %d: ' %iters)
            np.random.shuffle(idxList)
            for idx in idxList:
                SNN.reset()
                print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
                spike = SNN.spStdpTrain(dataX[idx], dataY[idx][2], stdp_config, forwardTime, refreshTime)
                print('%.2f' %(np.sum(spike).astype(np.float16)), end = ', ')
                SNN.reset()
                spike = SNN.batchedPredict(dataX[idx], forwardTime)
                print('%.2f' %(np.sum(spike).astype(np.float16)))
            SNN._printWeight()
        except KeyboardInterrupt:

            if input('exit training?') == 'y':
                return
    return


def test(SNN, dataX, iterNum, forwardTime = 1000, plot = False, legend = True, fn_save = None):
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    hit = 0
    testResult = [[None for i in range(dataSize)] for j in range(iterNum)]
    for j in range(iterNum):
        np.random.shuffle(idxList)
        for idx in idxList:
            SNN.reset()
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            print('%.2f' %(np.sum(spike).astype(np.float16)))
            rate = np.sum(spike, axis = 0).astype(np.float32) / forwardTime * 1
            print(rate)
            testResult[j][idx] = rate
            if plot is True:
                spikeListList = [SNN.spikeListList[0], SNN.spikeListList[-1]]
                plotSpikeList(spikeListList, legend = legend, fn_save = fn_save + '.input%d.iter%d' %(idx, j))
    return



if __name__ == "__main__":
    random.seed(114514)
    np.random.seed(114514)

    n_iter = 100
    tau_x = 5e-3
    tau_y = 5e-3
    F_x = 0.005
    F_y = 1.05*F_x
    stdp_config = {
        'tau_x': tau_x,
        'tau_y': tau_y,
        'F_x': F_x,
        'F_y': F_y
    }
    input_neuron = poissonInput(size = 2)
    dataX, dataY = getDataset(3)
    dt = 1e-3
    learningRate = 1
    SNN = getNetwork(learningRate = learningRate, dt = dt)
    print('---- START TRAINING ----')
    train(SNN, dataX, dataY, 30, stdp_config, forwardTime=2, refreshTime=1)
    SNN._printWeight()
    print('---- TESTING ----')
    SNN = test(SNN, dataX, iterNum = 10, forwardTime = 2, plot = True, fn_save = 'stdp')
