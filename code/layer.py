import numpy as np

from utility import *

class tmperalInput(object):
    def __init__(self, size, dt = 1):
        # IN
        # int size: number of neurons
        # np.float32 fMin: mean firing rate with input 0, in Hz
        # np.float32 fMax: mean firing rate with input 1, in Hz
        # np.float32 dt: simulation step size in msec
        super(tmperalInput, self).__init__()
        self.size = size
        self.dt = dt

        return

    def forward(self, iData, time):
        # generate input spikes
        # IN
        # np.ndarray iData, dtype = np.float32, shape = (n, ): n different inputs
        # OUT
        # np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        spikeList = np.zeros(self.size)
        if self.size == 1:
            if iData == 1:
                spikeList[0] = time % 5 == 0
            else:
                spikeList[0] = time % 5 == 2
        else:
            for i in range(self.size):
                if iData[i] == 1:
                    spikeList[i] = time % 5 == 0
                else:
                    spikeList[i] = time % 5 == 2
        return spikeList
    def reset(self):
        return

class poissonInput(object):
    #fire with probability proportion to input
    def __init__(self, size, fMin = 50, fMax = 250, dt = 1):
        #IN
        #int size: number of neurons
        #np.float32 fMin: mean firing rate with input 0, in Hz
        #np.float32 fMax: mean firing rate with input 1, in Hz
        #np.float32 dt: simulation step size in msec
        super(poissonInput, self).__init__()
        self.size = size
        self.fMin = np.float32(fMin)
        self.fMax = np.float32(fMax)
        self.dt = dt

        #np.float32 pMin: firing probability with input 0
        #np.float32 pMax: firing probability with input 1
        #np.float32 pDiff: firing probability range
        self.pMin = self.fMin / 1000 * self.dt
        self.pMax = self.fMax / 1000 * self.dt
        self.pDiff = self.pMax - self.pMin
        return
    
    def forward(self, iData, *supervisedCurrentList):
        #generate input spikes
        #IN
        #np.ndarray iData, dtype = np.float32, shape = (n, ): n different inputs
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        tempState = np.random.rand(self.size)
        tempThreshold = iData.astype(np.int8) * self.pDiff + self.pMin
        spikeList = tempState <= tempThreshold
        return spikeList
    def reset(self):
        return

class IF(object):
    def __init__(self, size, vRest = 0, vThreshold = 25, dt = 1):
        #int size: number of neurons
        #np.float32 capitance: C_m in μF
        #np.float32 resistance: R_m in kΩ
        #np.float32 vRest: rest voltage V_r in mV
        #np.float32 vThreshold: threshold voltage V_t in mV
        #np.float32 dt: simulation step size in msec
        super(IF, self).__init__()
        self.size = size
        self.vThreshold = np.float32(vThreshold)
        self.vRest = np.float32(vRest)
        self.dt = np.float32(dt)

        #np.ndarray tempVoltageList, dtype = np.float32, shape = (n, ): n different membrance potential in mV
        self.tempVoltageList = np.full(self.size, self.vRest, dtype = np.float32)
        return

    def forward(self, tempCurrentList):
        #generate spikes given injected currents
        #IN
        #np.ndarray tempCurrentList, dtype = np.float32, shape = (n, ): n different input currents in μA
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        self.tempVoltageList += tempCurrentList
        spikeList = self.tempVoltageList >= self.vThreshold
        self.tempVoltageList[spikeList] = self.vRest
        return spikeList

    def reset(self):
        self.tempVoltageList = np.full(self.size, self.vRest, dtype=np.float32)

class forwardLIF(object):
    #feedForward LIF layer
    def __init__(self, size, capitance = 1, resistance = 20, vRest = -65, vThreshold = 0, dt = 1):
        #int size: number of neurons
        #np.float32 capitance: C_m in μF
        #np.float32 resistance: R_m in kΩ
        #np.float32 vRest: rest voltage V_r in mV
        #np.float32 vThreshold: threshold voltage V_t in mV
        #np.float32 dt: simulation step size in msec
        super(forwardLIF, self).__init__()
        self.size = size
        self.capitance = np.float32(capitance)
        self.resistance = np.float32(resistance)
        self.vThreshold = np.float32(vThreshold)
        self.vRest = np.float32(vRest)
        self.dt = np.float32(dt)

        #np.ndarray tempVoltageList, dtype = np.float32, shape = (n, ): n different membrance potential in mV
        self.tempVoltageList = np.full(self.size, self.vRest, dtype = np.float32)

        #pre-computing for fast computing
        self.factor1 = self.dt / self.capitance
        self.factor2 = 1 - self.factor1 / self.resistance
        return

    def forward(self, tempCurrentList, supervisedCurrentList):
        #generate spikes given injected currents
        #IN
        #np.ndarray tempCurrentList, dtype = np.float32, shape = (n, ): n different input currents in μA
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        self.tempVoltageList = self.factor1 * tempCurrentList + self.tempVoltageList * self.factor2
        spikeList = self.tempVoltageList >= self.vThreshold
        self.tempVoltageList[spikeList] = self.vRest
        return spikeList

    def reset(self):
        self.tempVoltageList = np.full(self.size, self.vRest, dtype=np.float32)

class supervisedLIF(object):
    #feedForward LIF layer with supervised input
    def __init__(self, size, capitance = 1, resistance = 20, vRest = -65, vThreshold = 5, dt = 1):
        #int size: number of neurons
        #np.float32 capitance: C_m in μF
        #np.float32 resistance: R_m in kΩ
        #np.float32 vRest: rest voltage V_r in mV
        #np.float32 vThreshold: threshold voltage V_t in mV
        #np.float32 dt: simulation step size in msec
        super(supervisedLIF, self).__init__()
        self.size = size
        self.capitance = np.float32(capitance)
        self.resistance = np.float32(resistance)
        self.vThreshold = np.float32(vThreshold)
        self.vRest = np.float32(vRest)
        self.dt = np.float32(dt)

        #np.ndarray tempVoltageList, dtype = np.float32, shape = (n, ): n different membrance potential in mV
        self.tempVoltageList = np.full(self.size, self.vRest, dtype = np.float32)

        #pre-computing for fast computing
        self.factor1 = self.dt / self.capitance
        self.factor2 = 1 - self.factor1 / self.resistance
        return

    def forward(self, tempCurrentList, stepIdx, supervisedCurrentList = None):
        #generate spikes given injected currents
        #IN
        #np.ndarray tempCurrentList, dtype = np.float32, shape = (n, ): n different input currents in μA
        #np.ndarray supervisedCurrentList, dtype = np.float32, shape = (n, ): n different input currents in μA
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        if supervisedCurrentList is not None:
            tempCurrentList = tempCurrentList + supervisedCurrentList
        self.tempVoltageList = self.factor1 * tempCurrentList + self.tempVoltageList * self.factor2
        spikeList = self.tempVoltageList >= self.vThreshold
        self.tempVoltageList[spikeList] = self.vRest
        return spikeList

    def reset(self):
        self.tempVoltageList = np.full(self.size, self.vRest, dtype=np.float32)

class synapseLayer(object):
    #translate presynapse spike to postsynapse input current
    def __init__(self, prevSize, postSize, tau = 10, dt = 1):
        #int prevSize: presynapse layer size
        #int postSize: postsynapse layer size
        #np.float32 tau: time constant for spike response
        #np.float32 dt: simulation step size in msec
        super(synapseLayer, self).__init__()
        self.prevSize = prevSize
        self.postSize = postSize
        self.tau = np.float32(tau) / dt
        self.dt = dt

        #np.ndarray weight, dtype = np.float32, shape = (2, prevSize, postSize): inhibitatory and excitatory synapses.
        self.weight = self._initWeight()
        #print(np.sum(self.weight, axis = 0))

        #np.ndarray tempTraceList, dtype = np.float32, shape = (n, ): n different membrance potential in mV
        self.tempTraceList = np.zeros(self.prevSize, dtype = np.float32)
        #pre-computing for fast computing
        self.factor1 = 1 - 1 / self.tau
        self.prevSpikeTrace = np.zeros(self.prevSize, dtype = np.float32)
        self.postSpikeTrace = np.zeros(self.postSize, dtype = np.float32)
        self.epsilon = 4
        return


    def _initWeight(self):
        #OUT
        #np.ndarray weight, dtype = np.float32, shape = (prevSize, postSize):
        self.weight = np.abs(np.random.normal(size = (self.prevSize, self.postSize))).astype(np.float32)
        self.normalize()
        return self.weight


    def forward(self, tempSpikeList):
        #IN
        #np.ndarray tempSpikeList, dtype = np.bool, shape = (prevSize, ): True: fire; False: not fire
        #OUT
        #np.ndarray currentList, dtype = np.float32, shape = (postSize, ): postSize different input currents to next layer in μA
        # self.tempTraceList = self.tempTraceList * self.factor1 + tempSpikeList.astype(np.int8)
        # currentList = np.matmul(self.tempTraceList, self.weight)
        currentList = np.matmul(tempSpikeList.astype(np.int8), self.weight)
        return currentList


    def normalize(self):
        #normalize weights
        # print(self.weight)
        weightSquare = np.square(self.weight)
        self.weight = self.weight / np.sqrt(np.sum(weightSquare, axis = (0), keepdims = True))
        return

    def bcmUpdate(self, prevSpikeRate, postSpikeRate, postAverageSpikeRate, learningRate):
        #IN
        #np.ndarray prevSpikeRate dtype = np.float32, shape = (prevSize, ): prev layer spiking rate in Hz
        #np.ndarray postSpikeRate dtype = np.float32, shape = (postSize, ): post layer spiking rate in Hz
        #np.ndarray postAverageSpikeRate, dtype = np.float32, shape = (postSize, ): post layer average spiking rate in last iteration in Hz
        #np.float32 learningRate: step size for changing weights
        postDiffSpikeRate = postSpikeRate - postAverageSpikeRate
        # print(postDiffSpikeRate)
        dw = learningRate * np.matmul(prevSpikeRate.reshape(self.prevSize, 1), (postSpikeRate * postDiffSpikeRate).reshape(1, self.postSize))
        self.weight = self.weight + dw
        self.normalize()
        return

    def stdpUpdate(self, prevSpikeList, postSpikeList, learningRate = 0.005, supervisedSpikeList = None):
        dW = np.empty_like(self.weight, dtype = np.float64)
        self.prevSpikeTrace = self.prevSpikeTrace * self.factor1 + prevSpikeList.astype(np.int8)
        self.postSpikeTrace = self.postSpikeTrace * self.factor1 + postSpikeList.astype(np.int8)
        for j in range(self.prevSize):
            for i in range(self.postSize):
                dW[j, i] = prevSpikeList[j].astype(np.int8) * learningRate + \
                           postSpikeList[i].astype(np.int8) * learningRate + \
                           prevSpikeList[j].astype(np.int8) * 0.005 * 1.05 * self.postSpikeTrace[i] + \
                           postSpikeList[i].astype(np.int8) * 0.005 * self.prevSpikeTrace[j]
        self.weight += dW
        # if supervisedSpikeList == None:
        #
        # else:
        #     for j in range(self.prevSize):
        #         for i in range(self.postSize):
        #             dW[j, i] = (supervisedSpikeList[i].astype(np.int8) - postSpikeList[i].astype(np.int8)) * \
        #                        (learningRate + 0.005 * self.prevSpikeTrace[j])
        #     self.weight += dW
        # self.normalize()
        return

    def bpStdpUpdate(self, prevSpikeList, Xis, learningRate):

        nextXis = np.matmul(self.weight, Xis.reshape([self.postSize, 1])).reshape(self.prevSize)
        dW = np.matmul(np.sum(prevSpikeList, axis = 0).reshape([self.prevSize, 1]), Xis.reshape([1, self.postSize])) * learningRate
        self.weight += dW
        return nextXis

    def reset(self):
        self.tempTraceList = np.zeros(self.prevSize, dtype = np.float32)
        self.prevSpikeTrace = np.zeros(self.prevSize, dtype = np.float32)
        self.postSpikeTrace = np.zeros(self.postSize, dtype = np.float32)



if __name__ == '__main__':
    prevSize = 2
    postSize = 3
    dt = 1
    tau = 10
    time = 1000
    w = 0.5
    vThreshold = 25
    stepNum = int(time / dt)
    layer = supervisedLIF(postSize, vThreshold = vThreshold)
    synapse = synapseLayer(prevSize, postSize, tau = tau)
    inLayer = poissonInput(prevSize)

    prevSpike = np.zeros((stepNum, prevSize), dtype = np.bool)
    tempCurrentList = np.empty((stepNum, postSize), dtype = np.float32)
    spikeList = np.empty_like(tempCurrentList, dtype = np.bool)
    tempVoltageList = np.empty_like(tempCurrentList, dtype = np.float32)

    supervisedCurrentList = np.full_like(tempCurrentList, 0, dtype = np.float32)
    for step in range(stepNum):
        prevSpike[step] = inLayer.forward(np.asarray([0, 1]))
        tempCurrentList[step] = synapse.forward(prevSpike[step])
        spikeList[step] = layer.forward(tempCurrentList[step], supervisedCurrentList[step])
        tempVoltageList[step] = layer.tempVoltageList

    plotSpike(prevSpike)
    plotSpike(spikeList)
    plotVoltage(tempVoltageList)
    plotCurrent(tempCurrentList)