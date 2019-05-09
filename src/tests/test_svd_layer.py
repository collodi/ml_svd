import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import svd
from tools import print_tensor, print_tensors

WIDTH = 3

class SVDNet(nn.Module):
	def __init__(self, seq):
		super().__init__()

		self.net_ = svd.SVDLayer(seq)

	def forward(self, x):
		return self.net_(x).squeeze(dim=1)

def calc_E_hat(x, weight, bias, i, j):
	col = calc_col(x, i, j)
	a = weight[0].dot(col) + bias
	return a.exp()

def calcA(x, weight, bias, i, j):
	U, E, V = torch.svd(x)
	u, v = U[:, i], V[:, j]
	E_hat_ij = calc_E_hat(x, weight, bias, i, j)
	return (u.unsqueeze(0).t() * E_hat_ij).mm(v.unsqueeze(0))

def calc_col(x, i, j):
	U, E, V = torch.svd(x)
	u, v = U[:, i], V[:, j]
	return torch.cat((u, v), dim=0)

def calcAs(x, weight, bias):
	return [[calcA(x, weight, bias, i, j) for j in range(WIDTH)] for i in range(WIDTH)]

def main():
	has_cuda = torch.cuda.is_available()

	dev = torch.device('cuda' if has_cuda else 'cpu')
	default_tensor = torch.cuda.FloatTensor if has_cuda else torch.FloatTensor

	torch.set_default_dtype(torch.float32)
	torch.set_default_tensor_type(default_tensor)

	seq = nn.Linear(WIDTH * 2, 1)
	net = SVDNet(seq)

	x = torch.randn(1, WIDTH, WIDTH)
	print_tensor(x, 'x')

	y_hat = net(x)
	print_tensor(y_hat, 'y_hat')

	y_hat.sum().backward()
	print_tensor(seq.weight.grad, 'y_hat grad')

	### manually doing backward of SVDLayer
	#weight, bias = seq.weight, seq.bias
	#x = x[0].clone().detach().requires_grad_(True)

	#As = calcAs(x, weight, bias)
	#sumAs = torch.zeros((WIDTH, WIDTH))
	#for i in range(WIDTH):
	#	for j in range(WIDTH):
	#		sumAs = sumAs + As[i][j]

	#grad = torch.zeros(weight.size(1))
	#for i in range(WIDTH):
	#	for j in range(WIDTH):
	#		col = calc_col(x, i, j)
	#		grad += col * As[i][j].sum()

	#print_tensor(sumAs, 'sumAs')
	#print_tensor(grad, 'sumAs grad')
	### end - manually doing backward

if __name__ == '__main__':
	main()
