import torch
import torchvision
import torchvision.transforms as T

EMNIST_ROOT = '../data/emnist'

def emnist(split, batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') else { 'num_workers': 4, 'pin_memory': True }

	transform = T.Compose([
		T.ToTensor()
	])

	train = torchvision.datasets.EMNIST(EMNIST_ROOT, split, train=True, download=True, transform=transform)
	test = torchvision.datasets.EMNIST(EMNIST_ROOT, split, train=False, download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, **kwargs)

	return train_loader, test_loader

def emnist_flat(split, batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') else { 'num_workers': 4, 'pin_memory': True }

	transform = T.Compose([
		T.ToTensor(),
		T.Lambda(lambda x: x.squeeze())
	])

	train = torchvision.datasets.EMNIST(EMNIST_ROOT, split, train=True, download=True, transform=transform)
	test = torchvision.datasets.EMNIST(EMNIST_ROOT, split, train=False, download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, **kwargs)

	return train_loader, test_loader

def fakeflat(batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') else { 'num_workers': 4, 'pin_memory': True }

	transform = T.Compose([
		T.ToTensor(),
		T.Lambda(lambda x: x.squeeze())
	])

	train = torchvision.datasets.FakeData(100, (1, 28, 28), 3, transform=transform)
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)

	return train_loader, train_loader
