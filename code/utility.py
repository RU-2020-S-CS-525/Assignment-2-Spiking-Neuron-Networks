import numpy as np
import matplotlib.pyplot as plt

def plotNeuron(voltageList, spikeList, currentList = None, dt = 0.5, fn_save = None):
    #IN
    #np.ndarray voltageList, dtype = np.float16, shape = (t, n): n different membrance potential in mV
    #np.ndarray spikeList, dtype = np.bool, shape = (t, n): True: fire; False: not fire
    #np.ndarray currentList, dtype = np.float16, shape = (t, n): n different input currents
    #str fn_save: file name; None: not save
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    stepNum, simulationNum = voltageList.shape
    vThreshold = np.max(voltageList, axis = 0)
    if simulationNum > len(color):
        print('E: too many currents')
        exit(-1)

    time = np.array(range(stepNum), dtype = np.float16) * dt
    for i in range(simulationNum):
        line, = plt.plot(time, voltageList[:, i], c = color[i])
        point = plt.scatter(time[spikeList[:, i]], np.full(np.sum(spikeList[:, i]), vThreshold[i], dtype = np.float16), c = color[i], marker = 'o')
        line.set_label('neuron ' + str(i))
        point.set_label('spiking indicator')
    plt.xlabel('time (msec)')
    plt.ylabel('voltage (mV)')
    plt.legend(loc = 0)
    plt.title('membrane potential and spiking behavior')
    plt.tight_layout()
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save + '.voltage.png')
    plt.show()

    if currentList is not None:
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

def plotSpike(spikeList, dt = 0.5, fn_save = None):
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    stepNum, simulationNum = spikeList.shape
    if simulationNum > len(color):
        print('E: too many currents')
        exit(-1)

    time = np.array(range(stepNum), dtype = np.float16) * dt
    pos = np.array(range(simulationNum))
    for i in range(simulationNum):
        point = plt.scatter(time[spikeList[:, i]], np.full(np.sum(spikeList[:, i]), pos[i], dtype = np.float16), c = color[i], marker = '.')
        point.set_label('neuron ' + str(i))
    plt.xlabel('time (msec)')
    plt.ylabel('neuron index')
    plt.legend(loc = 0)
    plt.title('spiking behavior')
    plt.tight_layout()
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save + '.voltage.png')
    plt.show()
    return