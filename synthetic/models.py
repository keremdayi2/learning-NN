# python stuff
import argparse
import typing 
from typing import Union, Optional
from enum import Enum

# pytorch et. al
import torch
import torch.nn as nn

# two layer regression model
class TwoLayer(nn.Module):
	def __init__(self, dimension : int, hidden_size : int, activation : nn.Module):
		super().__init__()

		self.fc1 = nn.Linear(dimension, hidden_size)
		self.activation = activation
		self.fc2 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.activation(x)
		return self.fc2(x)

# N layer model that can output multiple dimensions
class NLayer(nn.Module):
	def __init__(self, dimensions : list[int], activations : list[nn.Module]):
		super().__init__()

		assert len(dimensions) == len(activations) + 2

		self.layers = nn.ModuleList()
		for i in range(1, len(dimensions) - 1):
			self.layers.append(nn.Linear(dimensions[i-1], dimensions[i]))
			self.layers.append(activations[i-1])

		self.layers.append(nn.Linear(dimensions[-2], dimensions[-1]))


	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x