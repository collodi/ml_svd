import numpy as np

import torch
import torch.nn.functional as F

from termcolor import cprint
from timeit import default_timer as timer

def nparams(net):
	return sum(torch.numel(x) for x in net.parameters())

def print_tensor(x, label=None, newline=True):
	if label:
		print(label)

	np.set_printoptions(precision=6, linewidth=1000, suppress=True)
	print(x.cpu().detach().numpy())
	if newline:
		print()

def print_tensors(X, label=None):
	if label:
		print(label)

	for x in X:
		print_tensor(x, newline=False)
	print()

def train(model, dev, loader, opt):
	model.train()

	total = len(loader)
	print_every = total / 10

	start_time = timer()
	for idx, (x, y) in enumerate(loader):
		x, y = x.to(dev), y.long().to(dev)
		opt.zero_grad()

		out = model(x)
		loss = F.cross_entropy(out, y)
		loss.backward()

		opt.step()
		if idx % print_every == 0:
			print('{:.0f}% -> loss: {:.6f}'.format(100. * idx / total, loss.item()))
			torch.cuda.empty_cache()

	print(f'training took {timer() - start_time:.2f} seconds')

def test(model, dev, loader):
	model.eval()

	loss = 0
	correct = 0
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(dev), y.long().to(dev)

			out = model(x)
			loss += F.cross_entropy(out, y, reduction='sum')
			correct += (y == out.argmax(dim=1)).sum().item()

	N = len(loader.dataset)

	loss /= N
	percent = correct / N

	cprint(f'\navg.loss: {loss:.4f}', attrs=['bold'])
	cprint(f'accuracy: {percent:.4f} ({correct} / {N})')
