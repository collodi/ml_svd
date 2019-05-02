import torch
import torch.nn as nn
import torch.nn.functional as F

import svd_op

# width = number of input columns
# height = number of input rows
#
# `net` takes a concatenation of 2 vectors (say u and v) and outputs a single value
# which corresponds to the importance of the matrix produced by u * v^T.
#
# u is a column of U and v is a column of V
class SVDLayer(nn.Module):
	def __init__(self, net):
		super().__init__()

		self.net = net

	def forward(self, X):
		U, E, V = svd_op.batch_svd(X)

		cols = svd_op.batch_svdcols(U, E, V)
		E_hat = self.net(cols).view(*E.shape)

		print('params')
		print([x for x in self.net.parameters()])

		return svd_op.batch_unsvd(U, E_hat, V)

class SVDReduce(nn.Module):
	def __init__(self, nrow, ncol):
		super().__init__()

		self.nrow = nrow
		self.ncol = ncol

	def forward(self, X):
		print(X)
		U, E, V = svd_op.batch_svd(X)

		U, V = U[:, :self.nrow, :], V[:, :self.ncol, :]
		return svd_op.batch_unsvd(U, E, V)

class Net(nn.Module):
	def __init__(self):
		super().__init__()

		seq1 = nn.Sequential(
			nn.Linear(56, 20),
			nn.Linear(20, 20),
			nn.Linear(20, 1)
		)

		seq2 = nn.Sequential(
			nn.Linear(30, 15),
			nn.Linear(15, 15),
			nn.Linear(15, 1)
		)

		self.net_ = nn.Sequential(
			SVDLayer(seq1), SVDReduce(15, 15),
			SVDLayer(seq2), SVDReduce(1, 3)
		)

	def forward(self, X):
		return self.net_(X).squeeze(dim=1)
