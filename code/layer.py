import numpy as np

from neuron import LIF, Izhikevich

from utility import plotNeuron

class feedForward(object):
    #vanilla feedForward Layer
    def __init__(self, size, neuronClass = LIF, neuronConfig = None, dt = 0.5, fastFlag = 0):
        #int size: number of neurons
        #neuron neuronClass: model of neuron
        #dict neuronConfig: configuration of a neuron
        #np.float16 dt: simulation step size in msec
        #bool fastFlag: 0: no fast computing; 1: intergrate neurons into the layer; 2: intergrate uniformed neurons into the layer
        super(feedForward, self).__init__()
        self.size = size
        self.neuronClass = neuronClass
        self.neuronConfig = dict() if neuronConfig is None else neuronConfig
        self.dt = dt
        self.fastFlag = fastFlag

        #list neuronList: neurons in the layer
        self.neuronList = [self.neuronClass(**self.neuronConfig) for i in range(self.size)]

        #function forward:
        #IN
        #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        self.forward = self._getForwardFunc()
        return

    def _getForwardFunc(self):
        #OUT
        #function forward
        if self.neuronClass is LIF:
            if self.fastFlag == 2:
                print('here')
                self._tempVoltageList = np.full((self.size, ), self.neuronList[0].tempVoltage, dtype = np.float16)
                neuronForwardFunc = self.neuronList[0].getSimpleFastForwardFunc()
                def forwardFunc(tempCurrentList):
                    #IN
                    #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                    #OUT
                    #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                    self._tempVoltageList, spikeList = neuronForwardFunc(tempCurrentList, self._tempVoltageList)
                    return spikeList
                return forwardFunc

            elif self.fastFlag == 1:
                self._tempVoltageList = np.array([self.neuronList[i].tempVoltage for i in range(self.size)], dtype = np.float16)
                factor1List = np.array([self.neuronList[i].factor1 for i in range(self.size)], dtype = np.float16)
                factor2List = np.array([self.neuronList[i].factor2 for i in range(self.size)], dtype = np.float16)
                vThresholdList = np.array([self.neuronList[i].vThreshold for i in range(self.size)], dtype = np.float16)
                vRestList = np.array([self.neuronList[i].vRest for i in range(self.size)], dtype = np.float16)
                neuronForwardFunc = self.neuronList[0].getFastForwardFunc(factor1List, factor2List, vThresholdList, vRestList)
                def forwardFunc(tempCurrentList):
                    #IN
                    #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                    #OUT
                    #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                    self._tempVoltageList, spikeList = neuronForwardFunc(tempCurrentList, self._tempVoltageList)
                    return spikeList
                return forwardFunc

            else:
                self.fastFlag = 0
                def forwardFunc(tempCurrentList):
                    #IN
                    #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                    #OUT
                    #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                    spikeList = np.empty((self.size, ), dtype = np.bool)
                    for i, neuron in enumerate(self.neuronList):
                        spikeList[i] = neuron.forward(tempCurrentList[i])
                    return spikeList
                return forwardFunc

        else:
            self.fastFlag = 0
            def forwardFunc(tempCurrentList):
                #IN
                #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                #OUT
                #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                spikeList = np.empty((self.size, ), dtype = np.bool)
                for i, neuron in enumerate(self.neuronList):
                    spikeList[i] = neuron.forward(tempCurrentList[i])
                return spikeList
            return forwardFunc

    def getTempVoltageList(self):
        #OUT
        #np.ndarray tempVoltageList, dtype = np.float16, shape = (n, ): n different membrance potential in mV
        if self.fastFlag == 1 or self.fastFlag == 2:
            return self._tempVoltageList
        else:
            self._tempVoltageList = np.array([self.neuronList[i].tempVoltage for i in range(self.size)], dtype = np.float16)
            return self._tempVoltageList


if __name__ == '__main__':
    stepNum = 500
    layerSize = 3
    currentList = np.ones((stepNum, layerSize), dtype = np.float16)
    currentList[:, 0] = 0.3
    currentList[:, 1] = 0.5
    currentList[:, 2] = 0.7
    voltageList = np.empty_like(currentList, dtype = np.float16)
    spikeList = np.empty_like(voltageList, dtype = np.bool)
    layer = feedForward(layerSize, neuronClass = LIF, fastFlag = 2)
    for i in range(stepNum):
        spikeList[i] = layer.forward(currentList[i])
        voltageList[i] = layer.getTempVoltageList()
    plotNeuron(currentList, voltageList, spikeList)