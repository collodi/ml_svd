import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import svd_op
from tools import print_tensor, print_tensors

# width = number of input columns
# height = number of input rows
#
# `net` takes a concatenation of 2 vectors (say u and v) and outputs a single value
# which corresponds to the importance of the matrix produced by u * v^T.
#
# u is a column of U and v is a column of V
class SVDLayer(nn.Module):
	def __init__(self, h, w):
		super().__init__()

		self.net = nn.Linear(h + w, 1)
		self.wt = nn.Linear(w, w)

	def forward(self, x):
		U, E, V = svd_op.batch_svd(x)

		cols = svd_op.batch_svdcols(U, V)

		weights = self.net(cols).view(*E.shape)
		alpha = F.softmax(self.wt(weights), dim=-1) # dist. among cols
		beta = F.softmax(self.wt(weights), dim=-2) # dist. among rows

		E_hat = E.bmm(alpha) + beta.bmm(E)
		return svd_op.batch_unsvd(U, E_hat, V)

class SVDReduce(nn.Module):
	def __init__(self, in_size, out_size, bias=True):
		super().__init__()

		ih, iw = in_size
		oh, ow = out_size

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

class Net(nn.Module):
	def __init__(self):
		super().__init__()

		sizes = [
			(28, 28), (25, 25), (20, 20),
			(15, 15), (10, 10), (5, 10),
			(1, 10)
		]

		self.svds = nn.ModuleList([
			SVDReduce(a, b)
			for a, b in zip(sizes, sizes[1:])
		])

		self.svds.insert(2, nn.AlphaDropout(p=0.2))
		self.svds.insert(1, nn.AlphaDropout(p=0.2))

	def forward(self, x):
		for net in self.svds:
			x = F.selu(net(x))

		return x.squeeze(dim=1)
