import numpy as np

from neuron import LIF, Izhikevich

from utility import plotNeuron, plotSpike


class poissonInput(object):
    #fire with probability proportion to input
    def __init__(self, size, fMin = 10, fMax = 200, dt = 0.5):
        super(poissonInput, self).__init__()
        self.size = size
        self.fMin = np.float16(fMin)
        self.fMax = np.float16(fMax)
        self.dt = dt

        self.pMin = self.fMin / 1000 * self.dt
        self.pMax = self.fMax / 1000 * self.dt
        self.pDiff = self.pMax - self.pMin
    
    def forward(self, iData):
        tempState = np.random.rand(self.size)
        tempThreshold = iData * self.pDiff + self.pMin
        spikeList = tempState <= tempThreshold
        return spikeList

    def getTempVoltageList(self):
        return np.empty(self.size, dtype = np.float16)



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
    def __init__(self, prevSize, postSize, response = 1, learningRate = 1, dt = 0.5):
        #int prevSize: presynapse layer size
        #int postSize: postsynapse layer size
        #np.float16 response: input current for a spike
        #np.float16 dt: simulation step size in msec
        super(synapseLayer, self).__init__()
        self.prevSize = prevSize
        self.postSize = postSize
        self.response = np.float16(response) / dt
        self.learningRate = np.float16(learningRate)
        self.dt = dt

        #np.ndarray weight, dtype = np.float16, shape = (2, prevSize, postSize): excitatory and inhibitatory synapses.
        self.weight = self._initWeight()
        return

    def _initWeight(self):
        #OUT
        #np.ndarray weight, dtype = np.float16, shape = (2, prevSize, postSize): excitatory and inhibitatory synapses.
        self.weight = np.sort(np.abs(np.random.normal(size = (2, self.prevSize, self.postSize))).astype(np.float16), axis = 0)
        self.normalize()
        self.weight[0] = -1 * self.weight[0]
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
        unweightedInput = self.response * np.asarray(spikeList, dtype = np.float16)
        # print('here')
        # print(self.response)
        # print(spikeList)
        # print(np.asarray(spikeList, dtype = np.float16))
        # print(unweightedInput)
        tempCurrentList = np.matmul(spikeList.reshape((1, self.prevSize)), tempWeight).reshape(self.postSize)
        return tempCurrentList

    def ojaUpdate(self, prevSpikeFrq, postSpikeFrq):
        corrFrq = np.matmul(prevSpikeFrq.reshape(-1, 1), postSpikeFrq.reshape(1, -1))
        postSqFrq = np.square(postSpikeFrq)
        dW = np.empty_like(self.weight, dtype = np.float16)
        for j in range(self.prevSize):
            for i in range(self.postSize):
                dW[1, j, i] = self.learningRate * (corrFrq[j, i] - self.weight[1, j, i] * postSqFrq[i])
                dW[0, j, i] = self.learningRate * (self.weight[0, j, i] * postSqFrq[i] + corrFrq[j, i])
        self.weight = self.weight + dW
        self.normalize()
        return



if __name__ == '__main__':
    stepNum = 20000
    layerSize = 2
    iData = np.array(range(2), dtype = np.float16)
    spikeList = np.empty((stepNum, layerSize), dtype = np.bool)
    layer = poissonInput(layerSize, fMin = 10, fMax = 200)
    for i in range(stepNum):
        spikeList[i] = layer.forward(iData)
    print(np.sum(spikeList, axis = 0))
    plotSpike(spikeList)
