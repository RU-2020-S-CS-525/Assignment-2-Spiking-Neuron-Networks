import numpy as np

from layer import *
from utility import *

class supervised(object):
    #supervised network
    def __init__(self, neuronLayerList, minSupervisedCurrent = -1, maxSupervisedCurrent = 1, synapseConfig = None, dt = 1):
        #list neuronLayerList [layer]: neuron layers
        #np.float32 minSupervisedCurrent: min supervised input current with 0, in μA
        #np.float32 maxSupervisedCurrent: max supervised input current with 1, in μA
        #dict synapseConfig: configuration of synapseLayer
        #np.float32 dt: simulation step size in msec
        super(supervised, self).__init__()
        self.neuronLayerList = neuronLayerList
        self.minSupervisedCurrent = np.float32(minSupervisedCurrent)
        self.maxSupervisedCurrent = np.float32(maxSupervisedCurrent)
        self.synapseConfig = dict() if synapseConfig is None else synapseConfig
        self.dt = dt

        #int layerNum: number of layers, excluding input layer
        #np.float32 maxSupervisedCurrent: range of supervised input current, in μA
        #list synapseLayerList [layer]: neuron layers
        #list spikeListList [np.ndarray spikeList]: spikeList by layer
        self.layerNum = len(self.neuronLayerList) - 1
        self.diffSupervisedCurrent = self.maxSupervisedCurrent - self.minSupervisedCurrent
        self.synapseLayerList = [synapseLayer(self.neuronLayerList[i].size, self.neuronLayerList[i + 1].size, **self.synapseConfig) for i in range(self.layerNum)]
        self.spikeListList = [np.empty(0, dtype = np.bool) for i in range(self.layerNum + 1)]
        self.supervisedTrace = 0
        return


    def _getSupervisedCurrent(self, supervisedIData):
        #IN
        #np.ndarray supervisedIData, dtype = np.float32: supervised input data
        #OUT
        #np.ndarray supervisedCurrent, dtype = np.float32: supervised current to inject into neurons
        return supervisedIData * self.diffSupervisedCurrent + self.minSupervisedCurrent
    def _getSupervisedTrace(self, supervisedIData, time):

        d = poissonInput(self.neuronLayerList[-1].size, 50, 250)
        spike = d.forward(supervisedIData)
        self.supervisedTrace = self.supervisedTrace * 9 / 10 + spike.astype(np.int8)
        return self.supervisedTrace
    def _getSupervisedSpikes(self, supervisedIData, time):

        d = poissonInput(self.neuronLayerList[-1].size, 50, 250)
        return d.forward(supervisedIData)
    def _forward(self, iData, supervisedIDataList, stepIdx):
        #IN
        #np.ndarray iData, dtype = np.float32: input data
        #list supervisedIDataList, [np.ndarray supervisedIData, dtype = np.float32]: supervised input data for each layer
        #int stepIdx: step index
        #OUT
        #np.ndarray oData, dtype = np.float32: output data
        for layerIdx in range(self.layerNum):
            tempNeuronLayer = self.neuronLayerList[layerIdx]
            tempSynapseLayer = self.synapseLayerList[layerIdx]
            tempSpikeList = self.spikeListList[layerIdx]
            tempSupervisedIData = supervisedIDataList[layerIdx]

            supervisedCurrent = self._getSupervisedCurrent(tempSupervisedIData)
            oData = tempNeuronLayer.forward(iData, supervisedCurrent)
            tempSpikeList[stepIdx] = oData
            iData = tempSynapseLayer.forward(oData)

        supervisedCurrent = self._getSupervisedCurrent(supervisedIDataList[-1])
        oData = self.neuronLayerList[-1].forward(iData, supervisedCurrent)
        self.spikeListList[-1][stepIdx] = oData
        return oData

    def batchedForward(self, iData, supervisedIDataList, time):
        #IN
        #np.ndarray iData, dtype = np.float32: input data
        #list supervisedIDataList, [np.ndarray supervisedIData, dtype = np.float32]: supervised input data for each layer
        #int time: time to forward
        #OUT
        #np.ndarray oData, dtype = np.float32: output data
        stepNum = int(time / self.dt)
        self.spikeListList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.bool) for i in range(self.layerNum + 1)]

        for stepIdx in range(stepNum):
            self._forward(iData, supervisedIDataList, stepIdx)
        return self.spikeListList[-1]

    def stdpBatchForward(self, iData, time, supervisedIData):
        stepNum = int(time / self.dt)
        self.spikeListList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.bool) for i in range(self.layerNum + 1)]

        for stepIdx in range(stepNum):
            self._bpStdpForward(iData, stepIdx, supervisedIData)
            self.bpStdpUpdate(stepIdx, supervisedIData)
        return self.spikeListList[-1]


    def _bpStdpForward(self, iData, stepIdx, supervisedIDataList = None):
        #IN
        #np.ndarray iData, dtype = np.float32: input data
        #list supervisedIDataList, [np.ndarray supervisedIData, dtype = np.float32]: supervised input data for each layer
        #int stepIdx: step index
        #OUT
        #np.ndarray oData, dtype = np.float32: output data
        for layerIdx in range(self.layerNum):
            tempNeuronLayer = self.neuronLayerList[layerIdx]
            tempSynapseLayer = self.synapseLayerList[layerIdx]
            tempSpikeList = self.spikeListList[layerIdx]

            oData = tempNeuronLayer.forward(iData)
            tempSpikeList[stepIdx] = oData
            iData = tempSynapseLayer.forward(oData)

        oData = self.neuronLayerList[-1].forward(iData)
        self.spikeListList[-1][stepIdx] = oData
        return oData

    def _predict(self, iData, stepIdx):
        #IN
        #np.ndarray iData, dtype = np.float32: input data
        #int stepIdx: step index
        #OUT
        #np.ndarray oData, dtype = np.float32: output data
        for layerIdx in range(self.layerNum):
            tempNeuronLayer = self.neuronLayerList[layerIdx]
            tempSynapseLayer = self.synapseLayerList[layerIdx]
            tempSpikeList = self.spikeListList[layerIdx]

            oData = tempNeuronLayer.forward(iData)
            tempSpikeList[stepIdx] = oData
            iData = tempSynapseLayer.forward(oData)

        oData = self.neuronLayerList[-1].forward(iData)
        self.spikeListList[-1][stepIdx] = oData
        return oData

    def batchedPredict(self, iData, time):
        #IN
        #np.ndarray iData, dtype = np.float32: input data
        #int time: time to predict
        #OUT
        #np.ndarray oData, dtype = np.float32: output data

        stepNum = int(time / self.dt)
        self.spikeListList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.bool) for i in range(self.layerNum + 1)]

        for stepIdx in range(stepNum):
            self._predict(iData, stepIdx)
        return self.spikeListList[-1]

    def stdpBatchPerdict(self, iData, time):
        stepNum = int(time / self.dt)
        self.spikeListList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.bool) for i in range(self.layerNum + 1)]

        for stepIdx in range(stepNum):
            self._stdpForward(iData, stepIdx)
        return self.spikeListList[-1]


    def refresh(self, time):
        #IN
        #int time: time to refresh
        iData = np.zeros(self.neuronLayerList[0].size, dtype = np.float32)
        self.batchedPredict(iData, time)
        return

    def reset(self):
        #reset states to init
        for neuron in self.neuronLayerList:
            neuron.reset()
        for synapse in self.synapseLayerList:
            synapse.reset()
        return

    def _printWeight(self):
        #print weight of each layer
        for layerIdx in range(self.layerNum):
            self.synapseLayerList[layerIdx].printWeight()
        return

    def bcmPreUpdate(self, iData, supervisedIData, forwardTime):
        #IN
        #np.ndarray iData, dtype = np.float32: input data
        #list supervisedIDataList, [np.ndarray supervisedIData, dtype = np.float32]: supervised input data for each layer
        #int forwardTime: time to forward
        #OUT
        #np.ndarray spikeRateList[-1], dtype = np.float32: last layer spiking rate

        self.reset()
        self.batchedForward(iData, supervisedIData, forwardTime)
        self.spikeRateList = [np.empty(self.neuronLayerList[i].size, dtype = np.float32) for i in range(self.layerNum + 1)]
        for layerIdx in range(0, self.layerNum + 1):
            self.spikeRateList[layerIdx] = np.sum(self.spikeListList[layerIdx], axis = 0).astype(np.float32) / forwardTime * 1000
        return self.spikeRateList[-1]

    def bcmUpdate(self, averageSpikeRateList, learningRate, forwardTime, constrainList = None, trainLayerSet = None):
        #IN
        #list averageSpikeRateList, [np.ndarray averageSpikeRate, dtype = np.float32]: average spiking rate in last iteration in Hz
        #np.float32 learningRate: step size for changing weights
        #int forwardTime: time to forward
        #list constrainList [function layerConstrain]: constrains of weights for each layer. None: no constrain
        #set trainLayerset: the index of layer that need to train. None: all need to train
        if constrainList is None:
            constrainList = [None for i in range(self.layerNum)]
        if trainLayerSet is None:
            trainLayerSet is {i for i in range(self.layerNum)}
        for layerIdx in range(0, self.layerNum):
            tempSynapseLayer = self.synapseLayerList[layerIdx]
            postAverageSpikeRate = averageSpikeRateList[layerIdx + 1]
            prevSpikeRate = self.spikeRateList[layerIdx]
            postSpikeRate = self.spikeRateList[layerIdx + 1]
            if layerIdx in trainLayerSet:
                tempSynapseLayer.bcmUpdate(prevSpikeRate, postSpikeRate, postAverageSpikeRate, learningRate, forwardTime, constrainList[layerIdx])
        return

    def extend(self, tempNeuronLayer):
        #IN
        #layer tempNeuronLayer: layer to append
        tempSynapseLayer = synapseLayer(self.neuronLayerList[-1].size, tempNeuronLayer.size, **self.synapseConfig)
        self.neuronLayerList.append(tempNeuronLayer)
        self.synapseLayerList.append(tempSynapseLayer)
        self.layerNum = self.layerNum + 1
        return

    def bpStdpUpdate(self, time, supervisedIdata, learningRate = 0.00005):
        if time - 4 > 0:
            preTime = time - 4
        else:
            preTime = 0
        supervisedSpikes = self._getSupervisedSpikes(supervisedIdata, time)
        prevSpikes = self.spikeListList[self.layerNum - 1][preTime : time + 1]
        postSpikes = self.spikeListList[self.layerNum][preTime : time + 1]
        Xis = (np.sum(postSpikes, axis=0) >= 1).astype(np.int8)

        for i in range(supervisedIdata.size):
            if Xis[i].astype(np.int8) == 1 and supervisedSpikes[i].astype(np.int8) == 0:
                Xis[i] = -1
            elif Xis[i].astype(np.int8) == 0 and supervisedSpikes[i].astype(np.int8) == 1:
                Xis[i] = 1
            else:
                Xis[i] = 0
        Xis = self.synapseLayerList[-1].bpStdpUpdate(prevSpikes, Xis,
                                                   learningRate)
        for layerIdx in range(self.layerNum - 2, -1, -1):
            prevSpikes = self.spikeListList[layerIdx][preTime : time + 1]
            postSpikes = self.spikeListList[layerIdx + 1][preTime : time + 1]
            derivatives = np.sum(postSpikes, axis=0) > 0
            Xis = Xis * derivatives.astype(np.int8)
            Xis = self.synapseLayerList[layerIdx].bpStdpUpdate(prevSpikes, Xis,
                                                   learningRate)
        return

    def stdpUpdate(self, time, supervisedIdata, learningRate = 0.0001):
        for layerIdx in range(self.layerNum - 1):
            prevSpikes = self.spikeListList[layerIdx][time]
            postSpikes = self.spikeListList[layerIdx + 1][time]
            self.synapseLayerList[layerIdx].stdpUpdate(prevSpikes, postSpikes,
                                                   learningRate)
        return

    def reset(self):
        for layerIdx in range(self.layerNum):
            self.neuronLayerList[layerIdx].reset()
            self.synapseLayerList[layerIdx].reset()

        self.neuronLayerList[-1].reset()
        return

    def stdpTrain(self, iData, supervisedIData, forwardTime = 1000, refreshTime = 300):
        self.reset()
        spikeList = self.stdpBatchForward(iData, forwardTime, supervisedIData)
        return spikeList



if __name__ == '__main__':
    time = 1000
    dt = 1
    stepNum = int(time / dt)
    neuronLayerList = []
    neuronLayerList.append(poissonInput(2))
    neuronLayerList.append(supervisedLIF(4))
    neuronLayerList.append(supervisedLIF(1))
    net = supervised(neuronLayerList)

    iData = np.array([0, 1], dtype = np.float32)
    supervisedIData = np.array([1], dtype = np.float32)
    net._printWeight()
    net.batchedForward(iData, supervisedIData, 1000)
    plotSpikeList(net.spikeListList)