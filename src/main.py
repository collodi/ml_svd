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

	emnist_train, emnist_test = loaders.emnist('digits', 5, dev)
	emnist_flat_train, emnist_flat_test = loaders.emnist_flat('digits', 5, dev)

	fake_train, fake_test = loaders.fakeflat(3, dev)

	conv_net = conv.Net()
	conv_opt = optim.Adam(conv_net.parameters(), lr=1e-4)

	svd_net = svd.Net()
	svd_opt = optim.Adam(svd_net.parameters(), lr=1e-4)

	print([x.size() for x in svd_net.parameters()])

	tools.train(svd_net, dev, fake_train, svd_opt)
	tools.test(svd_net, dev, fake_test)
	return

	nepoch = 5
	for epoch in range(nepoch):
		print(f'--- epoch {epoch}')

		print('SVDNet')
		tools.train(svd_net, dev, emnist_flat_train, svd_opt)
		tools.test(svd_net, dev, emnist_flat_test)
		print()

		print('ConvNet')
		tools.train(conv_net, dev, emnist_train, conv_opt)
		tools.test(conv_net, dev, emnist_test)
		print()

if __name__ == '__main__':
	main()
