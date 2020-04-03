import numpy as np

from layer import poissonInput, feedForward, supervisedOutput, synapseLayer
from neuron import LIF, Izhikevich

from utility import plotNeuron, plotSpike

class supervised(object):
    #supervised network
    def __init__(self, neuronLayerList, synapseType = synapseLayer, synapseConfig = None, supervisedStrongth = 1 / 16, dt = 0.5):
        super(supervised, self).__init__()
        self.neuronLayerList = neuronLayerList
        self.synapseType = synapseType
        self.synapseConfig = dict() if synapseConfig is None else synapseConfig
        self.supervisedStrongth = supervisedStrongth
        self.dt = dt

        self.layerNum = len(self.neuronLayerList) - 1
        self.synapseLayerList = [self.synapseType(self.neuronLayerList[i].size, self.neuronLayerList[i + 1].size, **self.synapseConfig) for i in range(self.layerNum)]
        self.spikeList = [np.empty(0, dtype = np.bool) for i in range(self.layerNum + 1)]
        return


    def _forward(self, iData, supervisedIData, stepIdx):
        for i in range(self.layerNum):
            oData = self.neuronLayerList[i].forward(iData)
            self.spikeList[i][stepIdx] = oData
            iData = self.synapseLayerList[i].forward(oData)
        oData = self.neuronLayerList[-1].forward(iData, supervisedIData * self.supervisedStrongth)
        self.spikeList[-1][stepIdx] = oData
        return oData

    def forward(self, iData, supervisedIData, time):
        stepNum = np.int32(time / self.dt)
        self.spikeList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.bool) for i in range(self.layerNum + 1)]
        for stepIdx in range(stepNum):
            self._forward(iData, supervisedIData, stepIdx)
        return self.spikeList[-1]

    def __forwardDebug(self, iData, supervisedIData, stepIdx):
        for i in range(self.layerNum):
            self.currentList[i][stepIdx] = iData
            oData = self.neuronLayerList[i].forward(iData)
            self.spikeList[i][stepIdx] = oData
            self.votageList[i][stepIdx] = self.neuronLayerList[i].getTempVoltageList()
            iData = self.synapseLayerList[i].forward(oData)
        self.currentList[-1][stepIdx] = iData
        oData = self.neuronLayerList[-1].forward(iData, supervisedIData * self.supervisedStrongth)
        self.spikeList[-1][stepIdx] = oData
        self.votageList[-1][stepIdx] = self.neuronLayerList[-1].getTempVoltageList()
        return oData

    def _forwardDebug(self, iData, supervisedIData, time):
        stepNum = np.int32(time / self.dt)
        self.spikeList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.bool) for i in range(self.layerNum + 1)]
        self.votageList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.float16) for i in range(self.layerNum + 1)]
        self.currentList = [np.empty((stepNum, self.neuronLayerList[i].size), dtype = np.float16) for i in range(self.layerNum + 1)]
        for stepIdx in range(stepNum):
            self.__forwardDebug(iData, supervisedIData, stepIdx)
        return self.spikeList[-1]

    def predict(self, iData, time):
        return self.forward(iData, None, time)

    def ojaUpdate(self, time):
        for i in range(self.layerNum):
            prevSpike = self.spikeList[i]
            postSpike = self.spikeList[i + 1]
            prevSpikeFrq = np.sum(prevSpike, axis = 0).astype(np.float16) / time
            postSpikeFrq = np.sum(postSpike, axis = 0).astype(np.float16) / time
            self.synapseLayerList[i].ojaUpdate(prevSpikeFrq, postSpikeFrq)
        return


if __name__ == '__main__':
    neuronLayerList = []
    neuronLayerList.append(poissonInput(size = 2))
    neuronLayerList.append(feedForward(size = 4))
    neuronLayerList.append(supervisedOutput(size = 1))
    network = supervised(neuronLayerList)
    iData = np.array(range(2), dtype = np.float16)
    supervisedIData = (iData[0].astype(np.bool) ^ iData[1].astype(np.bool)).astype(np.float16)
    print(iData)
    print(supervisedIData)

    time = 1000
    for iters in range(20):
        spike = network.forward(iData, supervisedIData, time)
        print(np.sum(spike))
        # for i in range(len(neuronLayerList) - 1):
        #     print(network.synapseLayerList[i].weight[0] + network.synapseLayerList[i].weight[1])
        for i in range(len(neuronLayerList)):
            # plotNeuron(network.votageList[i], network.spikeList[i], network.currentList[i])
            plotSpike(network.spikeList[i])
        print('prev')
        for i in range(len(network.synapseLayerList)):
            print(network.synapseLayerList[i].weight)
        network.ojaUpdate(time)
        print('post')
        for i in range(len(network.synapseLayerList)):
            print(network.synapseLayerList[i].weight)


