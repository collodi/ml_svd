import math
import random

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
		y = a.matmul(self.w2)
		return y + self.bias

	def __repr__(self):
		oh, ih = self.w1.shape
		iw, ow = self.w2.shape
		return f'SVDLayer ({ih}, {iw}) -> ({oh}, {ow})'

class FoldedSVDLayer(nn.Module):
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

	def forward(self, x, a=None):
		if a is None:
			a = random.random()

		y = None
		if a < 0.5:
			a = self.w1.matmul(x)
			y = a.matmul(self.w2)
		else:
			b = x.matmul(self.w2)
			y = self.w1.matmul(b)

		return y + self.bias

	def __repr__(self):
		oh, ih = self.w1.shape
		iw, ow = self.w2.shape
		return f'FoldedSVDLayer ({ih}, {iw}) -> ({oh}, {ow})'

class StackedSVDLayer(nn.Module):
	def __init__(self, *sizes, bias=True):
		super().__init__()

		sizes = [ _pair(x) for x in sizes ]

		self.lprms = []
		self.rprms = []

		for (ih, iw), (oh, ow) in zip(sizes, sizes[1:]):
			self.lprms.append(Parameter(torch.Tensor(oh, ih)))
			# def.d transposed instead of transposing every forward call
			self.rprms.append(Parameter(torch.Tensor(iw, ow)))

		if bias:
			self.bias = Parameter(torch.Tensor(*sizes[-1]))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		for lprm, rprm in zip(self.lprms, self.rprms):
			nn.init.kaiming_uniform_(lprm, a=math.sqrt(5))
			nn.init.kaiming_uniform_(rprm, a=math.sqrt(5))

		if self.bias is not None:
			fin1, _ = nn.init._calculate_fan_in_and_fan_out(self.lprms[-1])
			fin2, _ = nn.init._calculate_fan_in_and_fan_out(self.rprms[-1])

			bound = 1. / math.sqrt((fin1 + fin2) / 2.)
			nn.init.uniform_(self.bias, -bound, bound)

	def finish_left(self, x, idx):
		for i < range(idx, len(self.lprms)):
			x = self.lprms[i].matmul(x)
		return x

	def finish_right(self, x, idx):
		for i < range(idx, len(self.rprms)):
			x = x.matmul(self.rprms[i])
		return x

	def forward(self, x):
		il, ir = 0, 0

		while True:
			if il == len(self.lprms):
				x = self.finish_right(x, ir)
				break
			elif ir == len(self.rprms):
				x = self.finish_left(x, il)
				break

			a = random.random()
			if a < .5:
				x = self.lprms[il].matmul(x)
				il += 1
			else:
				x = x.matmul(self.rprms[ir])
				ir += 1

		return x

	def __repr__(self):
		if len(self.lprms) == 0:
			return 'StackedSVDLayer (Empty)'

		_, ih = self.lprms[0].shape
		iw, _ = self.rprms[0].shape

		sizes = [ f'({ih}, {iw})' ]

		for lprm, rprm in zip(self.lprms, self.rprms):
			oh, _ = lprm.shape
			_, ow = rprm.shape

			sizes.append(f'({oh}, {ow})')

		return f"StackedSVDLayer {' -> '.join(sizes)}"

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
