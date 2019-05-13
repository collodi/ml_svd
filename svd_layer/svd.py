import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def _pair(x):
	if isinstance(x, tuple):
		return x
	else:
		return (x, x)

class SVDLayer(nn.Module):
	def __init__(self, in_size, out_size, bias=True):
		super().__init__()

		ih, iw = _pair(in_size)
		oh, ow = _pair(out_size)

		self.w1 = Parameter(torch.Tensor(oh, ih))
		# def.d transposed instead of transposing every forward call
		self.w2 = Parameter(torch.Tensor(iw, ow))

		if bias:
			self.bias = Parameter(torch.Tensor(oh, ow))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
		nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

		if self.bias is not None:
			fin1, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
			fin2, _ = nn.init._calculate_fan_in_and_fan_out(self.w2)

			bound = 1. / math.sqrt((fin1 + fin2) / 2.)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		a = self.w1.matmul(x)
		return a.matmul(self.w2)

	def __repr__(self):
		oh, ih = self.w1.shape
		iw, ow = self.w2.shape
		return f'SVDLayer with ({ih}, {iw}) -> ({oh}, {ow})'

class Net(nn.Module):
	def __init__(self):
		super().__init__()

		sizes = [
			(28, 28), (25, 25), (20, 20),
			(15, 15), (10, 10), (5, 10),
			(1, 10)
		]

		self.svds = nn.ModuleList([
			SVDLayer(a, b)
			for a, b in zip(sizes, sizes[1:])
		])

		self.svds.insert(2, nn.AlphaDropout(p=0.2))
		self.svds.insert(1, nn.AlphaDropout(p=0.2))

	def forward(self, x):
		for net in self.svds:
			x = F.selu(net(x))

		return x.squeeze(dim=1)
