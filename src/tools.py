import torch
import torch.nn.functional as F

def train(model, dev, loader, opt):
	model.train()

	total = len(loader)
	print_every = total / 20

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

	print('\navg.loss: {:.4f}'.format(loss))
	print(f'accuracy: {percent:.4f} ({correct} / {N})')
