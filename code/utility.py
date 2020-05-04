import numpy as np
import matplotlib.pyplot as plt

def plotSpike(spikeList, dt = 0.5, fn_save = None):
    #IN
    #np.ndarray spikeList, dtype = np.bool, shape = (t, n): True: fire; False: not fire
    #np.float32 dt: simulation step size in msec
    #str fn_save: file name; None: not save
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    stepNum, simulationNum = spikeList.shape
    if simulationNum > len(color):
        print('E: too many neurons')
        exit(-1)

    time = np.array(range(stepNum), dtype = np.float32) * dt
    pos = np.array(range(simulationNum))
    for i in range(simulationNum):
        point = plt.scatter(time[spikeList[:, i]], np.full(np.sum(spikeList[:, i]), pos[i], dtype = np.float32), c = color[i], marker = '.')
        point.set_label('neuron ' + str(i))
    plt.xlabel('time (msec)')
    plt.ylabel('neuron index')
    plt.legend(loc = 0)
    plt.title('spiking behavior')
    plt.tight_layout()
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save + '.spike.png')
    plt.show()
    return

def plotSpikeList(spikeListList, dt = 0.5, legend = True, fn_save = None):
    #IN
    #list spikeListList [np.ndarray spikeList]: spikeList by layer
    #np.float32 dt: simulation step size in msec
    #str fn_save: file name; None: not save
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    layerNum = len(spikeListList)
    if layerNum > len(color):
        print('E: too many layers')
        exit(-1)

    startPos = 0
    for layerIdx, spikeList in enumerate(spikeListList):
        stepNum, simulationNum = spikeList.shape
        time = np.array(range(stepNum), dtype = np.float32) * dt
        pos = np.array(range(simulationNum)) + startPos
        startPos = startPos + simulationNum
        for i in range(simulationNum):
            point = plt.scatter(time[spikeList[:, i]], np.full(np.sum(spikeList[:, i]), pos[i], dtype = np.float32), c = color[layerIdx], marker = '.')
            point.set_label('neuron ' + str(layerIdx) + '.' + str(i))
    plt.xlabel('time (msec)')
    plt.ylabel('neuron index')
    if legend:
        plt.legend(loc = 0)
    plt.title('spiking behavior')
    plt.tight_layout()
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save + '.spikeList.png')
    plt.show()
    return

def plotVoltage(voltageList, dt = 0.5, fn_save = None):
    #IN
    #np.ndarray voltageList, dtype = np.float32, shape = (t, n): n different membrance potentials in mV
    #np.float32 dt: simulation step size in msec
    #str fn_save: file name; None: not save
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    stepNum, simulationNum = voltageList.shape
    if simulationNum > len(color):
        print('E: too many neurons')
        exit(-1)

    time = np.array(range(stepNum), dtype = np.float32) * dt
    for i in range(simulationNum):
        line, = plt.plot(time, voltageList[:, i], c = color[i])
        line.set_label('neuron ' + str(i))
    plt.xlabel('time (msec)')
    plt.ylabel('voltage (mV)')
    plt.legend(loc = 0)
    plt.title('membrane potential')
    plt.tight_layout()
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save + '.voltage.png')
    plt.show()
    return

def plotCurrent(currentList, dt = 0.5, fn_save = None):
    #IN
    #np.ndarray currentList, dtype = np.float32, shape = (t, n): n different input currents in μA
    #np.float32 dt: simulation step size in msec
    #str fn_save: file name; None: not save
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    stepNum, simulationNum = currentList.shape
    if simulationNum > len(color):
        print('E: too many neurons')
        exit(-1)

    time = np.array(range(stepNum), dtype = np.float32) * dt
    for i in range(simulationNum):
        line, = plt.plot(time, currentList[:, i], c = color[i])
        line.set_label('neuron ' + str(i))
    plt.xlabel('time (msec)')
    plt.ylabel('voltage (mV)')
    plt.legend(loc = 0)
    plt.title('input currents')
    plt.tight_layout()
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save + '.current.png')
    plt.show()
    return