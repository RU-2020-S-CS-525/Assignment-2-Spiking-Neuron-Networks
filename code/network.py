import numpy as np

from layer import feedForward, supervisedOutput, synapseLayer
from neuron import LIF, Izhikevich

from utility import plotNeuron

class supervised(object):
	#supervised network
	def __init__(self, neuronLayerList, synapseType = synapseLayer, synapseConfig = None):
		super(supervised, self).__init__()
		self.neuronLayerList = neuronLayerList
		self.layerNum = len(self.neuronLayerList) - 1
		self.synapseType = synapseType
		self.synapseConfig = dict() if synapseConfig is None else synapseConfig
		self.synapseLayerList = [self.synapseType(self.neuronLayerList[i].size, self.neuronLayerList[i + 1].size, **synapseConfig) for i in range(self.layerNum)]
		return


	def forward(self, iData, supervisedIData):
		for i in range(self.layerNum):
			oData = neuronLayerList[i].forward(iData)
			iData = synapseLayerList[i].forward(oData)
		oData = layerList[-1].forward(iData, supervisedIData)
		return oData

	def update(self):
		pass

	def predict(self, iData):
		return self.forward(iData, None)