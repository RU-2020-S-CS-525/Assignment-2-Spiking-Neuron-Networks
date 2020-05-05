import numpy as np

from utility import *

class poissonInput(object):
    #fire with probability proportion to input
    def __init__(self, size, fMin = 50, fMax = 100, dt = 0.5):
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
        tempThreshold = iData * self.pDiff + self.pMin
        spikeList = tempState <= tempThreshold
        return spikeList

    def reset(self):
        return

class forwardLIF(object):
    #feedForward LIF layer
    def __init__(self, size, capitance = 1, resistance = 20, vRest = 0, vThreshold = 25, dt = 0.5):
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

    def forward(self, tempCurrentList, supervisedCurrentList=None):
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
        self.tempVoltageList = np.full(self.size, self.vRest, dtype = np.float32)
        return


class fixedNeuron(object):
    def __init__(self, size, fMin, fMax, capitance = 1, resistance = 20, vRest = 0, vThreshold = 25, dt = 0.5):
        super(fixedNeuron, self).__init__()
        self.dt = dt
        self.poisson_neuron = poissonInput(size = 1, fMin = fMin, fMax = fMax)
        self.lif = forwardLIF(size = 1, capitance = capitance, resistance = resistance, vRest = vRest, vThreshold = vThreshold)
        self.size = 1

    def forward(self, tempCurrentList, supervisedCurrentList):
        if supervisedCurrentList is not None:
            supervisedCurrentList = int(supervisedCurrentList + 1) >> 1
            spikeList = self.poisson_neuron.forward(supervisedCurrentList)
        else:
            spikeList = self.lif.forward(tempCurrentList)
        #print(supervisedCurrentList, spikeList)
        return spikeList

    def reset(self):
        pass


class supervisedLIF(object):
    #feedForward LIF layer with supervised input
    def __init__(self, size, capitance = 1, resistance = 20, vRest = 0, vThreshold = 25, dt = 0.5):
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

    def forward(self, tempCurrentList, supervisedCurrentList = None):
        #generate spikes given injected currents
        #IN
        #np.ndarray tempCurrentList, dtype = np.float32, shape = (n, ): n different input currents in μA
        #np.ndarray supervisedCurrentList, dtype = np.float32, shape = (n, ): n different input currents in μA
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        if supervisedCurrentList is not None:
            tempCurrentList = tempCurrentList + supervisedCurrentList
            #spikeList = supervisedCurrentList > 0
            #self.tempVoltageList[spikeList] = self.vRest
            #return spikeList
            #print(self.tempVoltageList)
        self.tempVoltageList = self.factor1 * tempCurrentList + self.tempVoltageList * self.factor2
        spikeList = self.tempVoltageList >= self.vThreshold
        self.tempVoltageList[spikeList] = self.vRest
        return spikeList

    def reset(self):
        self.tempVoltageList = np.full(self.size, self.vRest, dtype = np.float32)
        return


class synapseLayer(object):
    #translate presynapse spike to postsynapse input current
    def __init__(self, prevSize, postSize, tau = 10, dt = 0.5):
        #int prevSize: presynapse layer size
        #int postSize: postsynapse layer size
        #np.float32 tau: time constant for spike response
        #np.float32 dt: simulation step size in msec
        super(synapseLayer, self).__init__()
        self.prevSize = prevSize
        self.postSize = postSize
        self.tau = np.float32(tau)
        self.dt = dt

        #np.ndarray weight, dtype = np.float32, shape = (2, prevSize, postSize): inhibitatory and excitatory synapses.
        self.weight = self._initWeight()
        #print(np.sum(self.weight, axis = 0))

        #np.ndarray tempTraceList, dtype = np.float32, shape = (n, ): n different membrance potential in mV
        self.tempTraceList = np.zeros(self.prevSize, dtype = np.float32)

        #pre-computing for fast computing
        self.factor1 = 1 - self.dt / self.tau
        return


    def _initWeight(self):
        #OUT
        #np.ndarray weight, dtype = np.float32, shape = (prevSize, postSize):
        self.weight = np.abs(np.random.normal(size = (self.prevSize, self.postSize))).astype(np.float32)
        self.normalize()
        return self.weight

    def reset(self):
        self.tempTraceList = np.zeros(self.prevSize, dtype = np.float32)
        return


    def forward(self, tempSpikeList):
        #IN
        #np.ndarray tempSpikeList, dtype = np.bool, shape = (prevSize, ): True: fire; False: not fire
        #OUT
        #np.ndarray currentList, dtype = np.float32, shape = (postSize, ): postSize different input currents to next layer in μA
        self.tempTraceList = self.tempTraceList * self.factor1 + tempSpikeList.astype(np.int8)
        currentList = np.matmul(self.tempTraceList, self.weight)
        return currentList


    def normalize(self):
        #normalize weights
        # print(self.weight)
        weightSquare = np.square(self.weight)
        # print(np.sqrt(np.sum(weightSquare, axis = 0, keepdims = True)))
        self.weight = self.weight / np.sqrt(np.sum(weightSquare, axis = 0, keepdims = True))
        return

    def bcmUpdate(self, prevSpikeRate, postSpikeRate, postAverageSpikeRate, learningRate, forwardTime, constrain = None):
        #IN
        #np.ndarray prevSpikeRate dtype = np.float32, shape = (prevSize, ): prev layer spiking rate in Hz
        #np.ndarray postSpikeRate dtype = np.float32, shape = (postSize, ): post layer spiking rate in Hz
        #np.ndarray postAverageSpikeRate, dtype = np.float32, shape = (postSize, ): post layer average spiking rate in last iteration in Hz
        #np.float32 learningRate: step size for changing weights
        #int forwardTime: time to forward
        #function constrain: constrain of weights. None: no constrain
        postDiffSpikeRate = postSpikeRate - postAverageSpikeRate
        # print(postDiffSpikeRate)
        dw = learningRate * np.matmul(prevSpikeRate.reshape(self.prevSize, 1), (postSpikeRate * postDiffSpikeRate).reshape(1, self.postSize))
        if constrain is not None:
            dw = constrain(dw)
        self.weight = self.weight + dw * forwardTime
        self.normalize()
        return

    def stdpUpdate(self, prevSpikeList, postSpikeList, learningRate, tau_x, tau_y, F_x, F_y):
        stepNum, n_neuron_pre = prevSpikeList.shape
        _, n_neuron_post = postSpikeList.shape
        trace_pre = np.zeros((n_neuron_pre,1))
        trace_post = np.zeros((n_neuron_post,1))
        latency = 5
        for t in range(latency):
            postSpikeList = np.insert(postSpikeList, 0, False, axis = 0)
        for j in range(n_neuron_pre):
            for i in range(n_neuron_post):
                for t in range(stepNum):
                    trace_pre[j] = trace_pre[j] * (1 - self.dt / tau_x) + prevSpikeList[t][j]
                    trace_post[i] = trace_post[i] * (1 - self.dt / tau_y) + postSpikeList[t][i]
                    dw = F_x * trace_pre[j] * postSpikeList[t][i] - F_y * trace_post[i] * prevSpikeList[t][j]
                    if prevSpikeList[t][j] or postSpikeList[t][i]:
                        #print(trace_pre[j], trace_post[i], dw)
                        #print(j, i, dw)
                        pass
                    self.weight[j, i] += dw
                # self.normalize()
                #self.stdpNormalize()
        return

    def spStdpUpdate(self, prevSpikeList, postSpikeList, learningRate, tau_x, tau_y, F_x, F_y, supervisedSpikes = None):
        stepNum, n_neuron_pre = prevSpikeList.shape
        _, n_neuron_post = postSpikeList.shape
        trace_pre = np.zeros((n_neuron_pre,1))
        trace_post = np.zeros((n_neuron_post,1))
        if supervisedSpikes == None:
            for j in range(n_neuron_pre):
                for i in range(n_neuron_post):
                    for t in range(stepNum):
                        trace_pre[j] = trace_pre[j] * (1 - self.dt / tau_x) + prevSpikeList[t][j]
                        trace_post[i] = trace_post[i] * (1 - self.dt / tau_y) + postSpikeList[t][i]
                        dw = F_x * trace_pre[j] * postSpikeList[t][i] - F_y * trace_post[i] * prevSpikeList[t][j]
                        if prevSpikeList[t][j] or postSpikeList[t][i]:
                            #print(trace_pre[j], trace_post[i], dw)
                            #print(j, i, dw)
                            pass
                        self.weight[j, i] += dw
                    # self.normalize()
                    #self.stdpNormalize()
        else:
            for j in range(n_neuron_pre):
                for i in range(n_neuron_post):
                    for t in range(stepNum):
                        trace_pre[j] = trace_pre[j] * (1 - self.dt / tau_x) + prevSpikeList[t][j]
                        trace_post[i] = trace_post[i] * (1 - self.dt / tau_y) + postSpikeList[t][i]
                        dw = (supervisedSpikes[t][i].astype(np.int8) - postSpikeList[t][i].astype(np.int8)) * F_x * trace_pre[j]
                        if prevSpikeList[t][j] or postSpikeList[t][i]:
                            #print(trace_pre[j], trace_post[i], dw)
                            #print(j, i, dw)
                            pass
                        self.weight[j, i] += dw
        return

    def stdpNormalize(self):
        self.weight = np.abs(self.weight) / np.sum(np.abs(self.weight), axis = 0, keepdims = True)
        return


    def printWeight(self):
        print(self.weight)
        print()
        return

    def setWeight(self, settingFunction):
        self.weight = settingFunction()
        self.normalize()
        return

class simple2DConvolution(object):
    #2D convolutional layer, stride = 1, pad = 0
    def __init__(self, size, kernel):
        #int size: number of neurons
        #np.ndarray kernel, shape = (3, 3), dtype = float32: weight of recepitive field
        super(simple2DConvolution, self).__init__()
        self.size = size
        self.kernel = kernel
        self.kernelSize = self.kernel.shape
        return

    def forward(self, iData):
        #generate input spikes
        #IN
        #np.ndarray iData, dtype = np.float32, shape = (size[0] + 2, size[0] + 2): n different inputs
        #OUT
        #np.ndarray cData, dtype = np.float32, shape = (size): convoluted data
        cData = np.empty(self.size, dtype = np.float32)
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                cData[row, col] = np.sum(self.kernel * iData[row: row + self.kernelSize[0], col: col + self.kernelSize[1]])
        return cData

    def reset(self):
        return

class onOffLayer(object):
    def __init__(self, row, col, kernel, kernelMax, fMin, fMax, dt = 0.5):
        #int row: row of neurons
        #int col: column of neurons
        #np.ndarray kernel, shape = (3, 3), dtype = float32: weight of recepitive field
        #float32 kernelMax: maximum value after convolution
        #np.float32 fMin: mean firing rate with input 0, in Hz
        #np.float32 fMax: mean firing rate with input 1, in Hz
        #np.float32 dt: simulation step size in msec
        super(onOffLayer, self).__init__()
        self.row = row
        self.col = col
        self.size = row * col * 2
        self.kernel = kernel
        self.kernelMax = np.float32(kernelMax)
        self.convolutionPart = simple2DConvolution((self.row, self.col), kernel)
        self.poissonPart = poissonInput(self.size, fMin, fMax, dt)
        return

    def forward(self, iData):
        #generate input spikes
        #IN
        #np.ndarray iData, dtype = np.float32, shape = (n, ): n different inputs
        #OUT
        #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
        #compute convolution
        cData = self.convolutionPart.forward(iData)
        #compute on/off
        c0Data = -1 * np.copy(cData)
        c1Data = np.copy(cData)
        mask = cData < 0
        c1Data[mask] = 0
        c0Data[~mask] = 0
        #reshape to 1d
        iData = np.empty(self.size, dtype = np.float32)
        iData[: self.row * self.col] = c0Data.reshape(-1)
        iData[self.row * self.col :] = c1Data.reshape(-1)
        iData = iData / self.kernelMax
        #compute spike
        spikeList = self.poissonPart.forward(iData)
        return spikeList

    def reset(self):
        self.convolutionPart.reset()
        self.poissonPart.reset()


if __name__ == '__main__':
    prevSize = 2
    postSize = 3
    dt = 0.5
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
