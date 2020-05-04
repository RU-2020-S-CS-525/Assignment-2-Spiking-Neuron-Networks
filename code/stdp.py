import numpy as np
from layer import poissonInput, fixedNeuron
from layer import forwardLIF as LIF, supervisedLIF, synapseLayer
from network import supervised
from xorBCM import getDataset
from utility import plotSpike
import random
from tqdm import tqdm


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
    neuronLayerList.append(LIF(2, capitance = capitance, resistance = resistance, vThreshold = vThreshold, dt = dt, vRest =
                                        vRest))
    neuronLayerList.append(supervisedLIF(1, capitance = capitance, resistance = resistance, vThreshold = vThreshold, dt = dt, vRest =
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
    for iters in tqdm(range(iterNum)):
        try:
            print('iter %d: ' %iters)
            np.random.shuffle(idxList)
            for idx in idxList:
                SNN.reset()
                print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
                spike = SNN.stdpTrain(dataX[idx], dataY[idx], stdp_config, forwardTime, refreshTime)
                print('%.2f' %(np.sum(spike).astype(np.float16)), end = ', ')
                SNN.reset()
                spike = SNN.batchedPredict(dataX[idx], forwardTime)
                print('%.2f' %(np.sum(spike).astype(np.float16)))
            #SNN._printWeight()
        except KeyboardInterrupt:
            SNN._printWeight()
            if input('exit training?') == 'y':
                return
    return


def test(SNN, dataX, n_iter_test = 10, forwardTime = 1):
    dataSize = dataX.shape[0]
    idxList = np.array(range(dataSize), dtype = np.int8)
    hit = 0
    for j in tqdm(range(n_iter_test)):
        np.random.shuffle(idxList)
        for idx in idxList:
            SNN.reset()
            print(' %d, %d: ' %(dataX[idx, 0].astype(np.int8), dataX[idx, 1].astype(np.int8)), end = '')
            spike = SNN.batchedPredict(dataX[idx], forwardTime)
            prediction = np.sum(spike) > 0
            print(prediction)
            if prediction == (dataX[idx, 0].astype(np.bool) ^ dataX[idx, 1].astype(np.bool)):
                hit += 1
            #print('%.2f' %(np.sum(spike).astype(np.float16)))
            #plotSpike(spike, dt = 1e-3)
    print('Accuracy: %.2f%%' %(100.0 * hit / n_iter_test / 4))
    return



if __name__ == "__main__":
    random.seed(114514)
    np.random.seed(114514)

    # network params
    fMin = 20
    fMax = 100
    forwardTime = 5

    # neuron params
    capitance = 1
    resistance = 1
    vThreshold = 5
    vRest = 0
    minSupervisedCurrent = -1
    maxSupervisedCurrent = 1

    n_iter = 26
    n_iter_test = 15
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
    dt = 1e-3
    learningRate = 1
    SNN = getNetwork(learningRate = learningRate, dt = dt, fMin = fMin, fMax = fMax, capitance = capitance, resistance = resistance,
                    vThreshold = vThreshold, vRest = vRest, minSupervisedCurrent = minSupervisedCurrent, maxSupervisedCurrent =
                     maxSupervisedCurrent)
    print('---- START TRAINING ----')
    #beforeTrain(SNN, fMin = fMin, fMax = fMax, dt = dt)
    train(SNN, dataX, dataY, n_iter, stdp_config, forwardTime = forwardTime, refreshTime=1)
    #afterTrain(SNN, learningRate = learningRate, dt = dt, fMin = fMin, fMax = fMax, capitance = capitance, resistance = resistance,
    #                vThreshold = vThreshold, vRest = vRest, minSupervisedCurrent = minSupervisedCurrent, maxSupervisedCurrent =
    #                 maxSupervisedCurrent)
    SNN._printWeight()
    print('---- TESTING ----')
    test(SNN, dataX, n_iter_test = n_iter_test, forwardTime = forwardTime)
