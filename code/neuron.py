import numpy as np

from utility import plotNeuron

class LIF(object):
    #leaky integrate-and-fire model
    #C_m \frac{dV}{dt} = I(t) - frac{V_m (t)}{R_m}
    def __init__(self, capitance = 1, resistance = 20, vRest = -65, vThreshold = 5, dt = 0.5):
        #np.float16 capitance: C_m in μF
        #np.float16 resistance: R_m in kΩ
        #np.float16 vRest: rest voltage V_r in mV
        #np.float16 vThreshold: threshold voltage V_t in mV
        #np.float16 dt: simulation step size in msec
        super(LIF, self).__init__()
        self.capitance = np.float16(capitance)
        self.resistance = np.float16(resistance)
        self.vThreshold = np.float16(vThreshold)
        self.vRest = np.float16(vRest)
        self.dt = np.float16(dt)

        #np.float16 tempVoltage: membrance potential in mV
        self.tempVoltage = self.vRest

        #pre-computing for fast computing
        self.factor1 = self.dt / self.capitance
        self.factor2 = 1 - self.factor1 / self.resistance
        return

    def forward(self, tempCurrent):
        #IN
        #np.float16 tempCurrent: input current in μA
        #OUT
        #np.bool spike: spiking behavior
        self.tempVoltage = self.factor1 * tempCurrent + self.tempVoltage * self.factor2
        if self.tempVoltage >= self.vThreshold:
            self.tempVoltage = self.vRest
            return np.bool(True)
        else:
            return np.bool(False)

    def getFastForwardFunc(self, factor1List, factor2List, vThresholdList, vRestList):
        #IN
        #np.ndarray factor1List, dtpye = np.float16, shape = (n, ): n different factor1
        #np.ndarray factor2List, dtpye = np.float16, shape = (n, ): n different factor2
        #np.ndarray vThresholdList, dtpye = np.float16, shape = (n, ): n different vThreshold
        #np.ndarray vRestList, dtpye = np.float16, shape = (n, ): n different vRest
        #OUT
        #function fastForwardFunc
        def fastForwardFunc(tempCurrentList, tempVoltageList):
            #IN
            #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
            #np.ndarray tempVoltageList, dtype = np.float16, shape = (n, ): n different membrance potential in mV
            #OUT
            #np.ndarray tempVoltageList, dtype = np.float16, shape = (n, ): n different membrance potential in mV
            #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
            tempVoltageList = factor1List * tempCurrentList + tempVoltageList * factor2List
            spikeList = tempVoltageList >= vThresholdList
            tempVoltageList[spikeList] = vRestList[spikeList]
            return tempVoltageList, spikeList
        return fastForwardFunc

    def getSimpleFastForwardFunc(self):
        #OUT
        #function fastForwardFunc
        def fastForwardFunc(tempCurrentList, tempVoltageList):
            #IN
            #np.ndarray tempCurrentList, dtype = np.float16, shape = (n, ): n different input currents
            #np.ndarray tempVoltageList, dtype = np.float16, shape = (t, n): n different membrance potential in mV
            #OUT
            #np.ndarray tempVoltageList, dtype = np.float16, shape = (t, n): n different membrance potential in mV
            #np.ndarray spikeList, dtype = np.bool, shape = (n, ): True: fire; False: not fire
            tempVoltageList = self.factor1 * tempCurrentList + tempVoltageList * self.factor2
            spikeList = tempVoltageList >= self.vThreshold
            tempVoltageList[spikeList] = self.vRest
            return tempVoltageList, spikeList
        return fastForwardFunc

class Izhikevich(object):
    #Izhikevich model
    #frac{dv}{dt} = 0.04 v^2 + 5 v + 140 - u + I
    #frac{du}{dt} = a (b v - u)
    def __init__(self, a = 0.02, b = 0.2, c = -65, d = 8, vThreshold = 30, dt = 0.5):
        #np.float16 a: time scale of u
        #np.float16 b: sensitivity of u to v
        #np.float16 c: after-spike reset v
        #np.float16 d: after-spike reset u
        #np.float16 vThreshold: threshold voltage V_t in mV
        #np.float16 dt: simulation step size in msec
        super(Izhikevich, self).__init__()
        self.a = np.float16(a)
        self.b = np.float16(b)
        self.c = np.float16(c)
        self.d = np.float16(d)
        self.vThreshold = np.float16(vThreshold)
        self.dt = np.float16(dt)
        self.halfDt = self.dt / 2 #used for update v stably

        #np.float16 tempVoltage: membrance potential in mV
        #np.float16 tempU: recovary variable
        self.tempVoltage = self.c
        self.tempU = self.b * self.c
        return

    def forward(self, tempCurrent):
        #IN
        #np.float16 tempCurrent: input current in μA
        #OUT
        #np.bool spike: spiking behavior
        
        #update V first half
        dV = (0.04 * np.square(self.tempVoltage) + 5 * self.tempVoltage + 140 - self.tempU + tempCurrent) * self.halfDt
        self.tempVoltage = self.tempVoltage + dV

        #update U
        dU = self.a * (self.b * self.tempVoltage - self.tempU) * self.dt
        self.tempU = self.tempU + dU

        #update V second half
        dV = (0.04 * np.square(self.tempVoltage) + 5 * self.tempVoltage + 140 - self.tempU + tempCurrent) * self.halfDt
        self.tempVoltage = self.tempVoltage + dV

        #get spike and reset
        if self.tempVoltage >= self.vThreshold:
            self.tempVoltage = self.c
            self.tempU = self.tempU + self.d
            return np.bool(True)
        else:
            return np.bool(False)


if __name__ == '__main__':
    stepNum = 5000
    currentList = np.ones((stepNum, 1), dtype = np.float16) * 4
    voltageList = np.empty_like(currentList, dtype = np.float16)
    spikeList = np.empty_like(voltageList, dtype = np.bool)
    neuron = Izhikevich()
    for i in range(stepNum):
        spikeList[i, 0] = neuron.forward(currentList[i, 0])
        voltageList[i, 0] = neuron.tempVoltage
    plotNeuron(currentList, voltageList, spikeList)