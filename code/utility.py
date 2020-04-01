import numpy as np
import matplotlib.pyplot as plt

def plotNeuron(currentList, voltageList, spikeList, dt = 0.5, fn_save = None):
    #IN
    #np.ndarray currentList, dtype = np.float16, shape = (t, n): n different input currents
    #np.ndarray voltageList, dtype = np.float16, shape = (t, n): n different membrance potential in mV
    #np.ndarray spikeList, dtype = np.bool, shape = (t, n): True: fire; False: not fire
    #str fn_save: file name; None: not save
    color = ['b', 'g', 'r', 'c', 'm', 'y']
    stepNum, simulationNum = currentList.shape
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