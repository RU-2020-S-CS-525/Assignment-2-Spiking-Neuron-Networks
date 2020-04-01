import numpy as np

from neuron import LIF, Izhikevich

from utility import plotNeuron

class feedForward(object):
    #vanilla feedForward layer
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
        ####for ease of understanding, one may assume having a member function layer.forward(tempCurrentList) defined below
        ####    def forward(self, tempCurrentList):
        ####        spikeList = np.empty((self.size, ), dtype = np.bool)
        ####        for i, neuron in enumerate(self.neuronList):
        ####            spikeList[i] = neuron.forward(tempCurrentList[i])
        ####        return spikeList
        return

    def _getForwardFunc(self):
        #OUT
        #function forward
        if self.neuronClass is LIF:
            if self.fastFlag == 2:
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


class supervisedOutput(object):
    #feedForward layer with an extra supervised input 
    def __init__(self, size, neuronClass = LIF, neuronConfig = None, dt = 0.5, fastFlag = 0):
        #int size: number of neurons
        #neuron neuronClass: model of neuron
        #dict neuronConfig: configuration of a neuron
        #np.float16 dt: simulation step size in msec
        #bool fastFlag: 0: no fast computing; 1: intergrate neurons into the layer; 2: intergrate uniformed neurons into the layer
        super(supervisedOutput, self).__init__()
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
        ####for ease of understanding, one may assume having a member function layer.forward(tempCurrentList, supervisedCurrentList) defined below
        ####    def forward(self, tempCurrentList, supervisedCurrentList):
        ####        tempCurrentList = tempCurrentList + supervisedCurrentList
        ####        spikeList = np.empty((self.size, ), dtype = np.bool)
        ####        for i, neuron in enumerate(self.neuronList):
        ####            spikeList[i] = neuron.forward(tempCurrentList[i])
        ####        return spikeList
        return

    def _getForwardFunc(self):
        #OUT
        #function forward
        if self.neuronClass is LIF:
            if self.fastFlag == 2:
                self._tempVoltageList = np.full((self.size, ), self.neuronList[0].tempVoltage, dtype = np.float16)
                neuronForwardFunc = self.neuronList[0].getSimpleFastForwardFunc()
                def forwardFunc(tempCurrentList, supervisedCurrentList):
                    #IN
                    #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                    #np.ndarray supervisedCurrentList, dtype = np.float16, shape = (n, ): n different supervised input
                    #OUT
                    #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                    tempCurrentList = tempCurrentList + supervisedCurrentList
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
                    #np.ndarray supervisedCurrentList, dtype = np.float16, shape = (n, ): n different supervised input
                    #OUT
                    #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                    tempCurrentList = tempCurrentList + supervisedCurrentList
                    self._tempVoltageList, spikeList = neuronForwardFunc(tempCurrentList, self._tempVoltageList)
                    return spikeList
                return forwardFunc

            else:
                self.fastFlag = 0
                def forwardFunc(tempCurrentList, supervisedCurrentList):
                    #IN
                    #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                    #np.ndarray supervisedCurrentList, dtype = np.float16, shape = (n, ): n different supervised input
                    #OUT
                    #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                    tempCurrentList = tempCurrentList + supervisedCurrentList
                    spikeList = np.empty((self.size, ), dtype = np.bool)
                    for i, neuron in enumerate(self.neuronList):
                        spikeList[i] = neuron.forward(tempCurrentList[i])
                    return spikeList
                return forwardFunc

        else:
            self.fastFlag = 0
            def forwardFunc(tempCurrentList, supervisedCurrentList = None):
                #IN
                #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
                #np.ndarray supervisedCurrentList, dtype = np.float16, shape = (n, ): n different supervised input
                #OUT
                #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
                if supervisedCurrentList is not None:
                   tempCurrentList = tempCurrentList + supervisedCurrentList
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



class synapseLayer(object):
    #translate presynapse spike to postsynapse input current
    def __init__(self, prevSize, postSize, response = 40):
        #int prevSize: presynapse layer size
        #int postSize: postsynapse layer size
        #np.float16 response: input current for a spike
        super(synapseLayer, self).__init__()
        self.prevSize = prevSize
        self.postSize = postSize
        self.response = np.float16(response)

        #np.ndarray weight, dtype = np.float16, shape = (2, prevSize, postSize): excitatory and inhibitatory synapses.
        self.weight = self._initWeight()
        
        return

    def _initWeight(self):
        #OUT
        #np.ndarray weight, dtype = np.float16, shape = (2, prevSize, postSize): excitatory and inhibitatory synapses.
        self.weight = np.abs(np.random.normal(size = (2, self.prevSize, self.postSize))).astype(np.float16)
        self.normalize()
        self.weight[1] = -1 * self.weight[1]
        return self.weight


    def normalize(self):
        #normalize weights
        weightSquare = np.square(self.weight)
        self.weight = weightSquare / np.sum(weightSquare, axis = (0, 1))
        return

    def forward(self, spikeList):
        #IN
        #np.ndarray spikeList, dtype = np.bool, shape = (prevSize, ): True: fire; False: not fire
        #OUT
        #np.ndarray tempCurrentList, dtype = np.float16, shape = (postSize, ): n different input currents
        tempWeight = self.weight[0] + self.weight[1]
        unweightedInput = self.response * spikeList
        tempCurrentList = np.matmul(spikeList.reshape((1, self.prevSize)), tempWeight).reshape(self.postSize)
        return tempCurrentList
        



if __name__ == '__main__':
    stepNum = 500
    layerSize = 3
    currentList = np.ones((stepNum, layerSize), dtype = np.float16)
    currentList[:, 0] = 0.3
    currentList[:, 1] = 0.5
    currentList[:, 2] = 0.7
    supervisedCurrentList = np.ones((stepNum, ), dtype = np.float16) * 0.3
    supervisedCurrentList[: stepNum // 2] = 0
    voltageList = np.empty_like(currentList, dtype = np.float16)
    spikeList = np.empty_like(voltageList, dtype = np.bool)
    layer = supervisedOutput(layerSize, neuronClass = LIF, fastFlag = 2)
    for i in range(stepNum):
        spikeList[i] = layer.forward(currentList[i], supervisedCurrentList[i])
        voltageList[i] = layer.getTempVoltageList()
    plotNeuron(currentList, voltageList, spikeList)