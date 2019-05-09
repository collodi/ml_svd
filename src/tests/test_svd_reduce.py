import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import svd
import tools
from tools import print_tensor, print_tensors

WIDTH = 12

def main():
	has_cuda = torch.cuda.is_available()

	dev = torch.device('cuda' if has_cuda else 'cpu')
	default_tensor = torch.cuda.FloatTensor if has_cuda else torch.FloatTensor

	torch.set_default_dtype(torch.float32)
	torch.set_default_tensor_type(default_tensor)

	x = torch.randn(1, WIDTH, WIDTH)

	svd_reduce = svd.SVDReduce((WIDTH, WIDTH), (5, 5))
	opt = optim.Adam(svd_reduce.parameters())

	nprms = tools.nparams(svd_reduce)
	print(f'number of params: {nprms}')

	for i in range(10):
		opt.zero_grad()
		print()

		y = svd_reduce(x)
		print(f'sum to minimize: {y.sum():.6f}')

		y.sum().backward()
		opt.step()

if __name__ == '__main__':
	main()
