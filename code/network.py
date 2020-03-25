import numpy as np

from layer import feedForward
from neuron import LIF, Izhikevich

from utility import plotNeuron

class supervised(object):
	#supervised network
	def __init__(self, layerList):
		super(supervised, self).__init__()
		self.layerList = layerList
		self.layerNum = len(layerList)
		self.weightList = [self._getInitWeight(self.layerList[i].size, self.layerList[i + 1].size) for i in range(len(layerList - 1))]
		return

	def _getInitWeight(self, prevSize, postSize):
		weight = np.empty((prevSize, postSize), dtype = np.float16)
		#do some init
		return weight

	def forward(self, iData, supervisedIData):
		for i in range(layerNum - 1):
			oData = layerList[i].forward(iData)
			iData = np.matmul(oData, weightList[i])
		oData = layerList[-1].forward(iData, supervisedIData)
		return oData

	def update(self):
		pass