from termcolor import cprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import tools
import loaders

# models
import conv
import svd

def main():
	has_cuda = torch.cuda.is_available()

	dev = torch.device('cuda' if has_cuda else 'cpu')
	default_tensor = torch.cuda.FloatTensor if has_cuda else torch.FloatTensor

	torch.set_default_dtype(torch.float32)
	torch.set_default_tensor_type(default_tensor)

	# flat = single color channel
	emnist_train, emnist_test = loaders.emnist('digits', 5, dev)
	emnist_flat_train, emnist_flat_test = loaders.emnist_flat('digits', 5, dev)

	#fake_train, fake_test = loaders.fake(5, dev)
	#fake_flat_train, fake_flat_test = loaders.fakeflat(5, dev)

	conv_net = conv.Net()
	svd_net = svd.Net()

	print(f'ConvNet # of params: {tools.nparams(conv_net)}')
	print(f'SVDNet # of params: {tools.nparams(svd_net)}')
	print()

	conv_opt = optim.Adam(conv_net.parameters())
	svd_opt = optim.Adam(svd_net.parameters())

	nepoch = 3
	for epoch in range(nepoch):
		print(f'--- epoch {epoch}')

		cprint('SVDNet', 'red')
		tools.train(svd_net, dev, emnist_flat_train, svd_opt)
		tools.test(svd_net, dev, emnist_flat_test)
		print()

		cprint('ConvNet', 'blue')
		tools.train(conv_net, dev, emnist_train, conv_opt)
		tools.test(conv_net, dev, emnist_test)
		print()

if __name__ == '__main__':
	main()
