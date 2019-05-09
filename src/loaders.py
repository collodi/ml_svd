import torch
import torchvision
import torchvision.transforms as T

from torchvision.datasets import FakeData, EMNIST
from torch.utils.data import DataLoader

FAKE_SIZE = 1000
EMNIST_ROOT = '../data/emnist'

transform = T.Compose([
	T.ToTensor()
])

flat_transform = T.Compose([
	T.ToTensor(),
	T.Lambda(lambda x: x.squeeze())
])

def emnist(split, batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') \
			else { 'num_workers': 4, 'pin_memory': True }

	train = EMNIST(EMNIST_ROOT, split, train=True, download=True, transform=transform)
	test = EMNIST(EMNIST_ROOT, split, train=False, download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, **kwargs)

	return train_loader, test_loader

def emnist_flat(split, batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') \
			else { 'num_workers': 4, 'pin_memory': True }

	train = EMNIST(EMNIST_ROOT, split, train=True, download=True, transform=flat_transform)
	test = EMNIST(EMNIST_ROOT, split, train=False, download=True, transform=flat_transform)

	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(test, batch_size=batch_size, **kwargs)

	return train_loader, test_loader

def fake(batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') \
			else { 'num_workers': 4, 'pin_memory': True }

	train = FakeData(FAKE_SIZE, (1, 28, 28), 3, transform=transform)
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)

	return train_loader, train_loader

def fakeflat(batch_size, dev):
	kwargs = {} if dev == torch.device('cpu') \
			else { 'num_workers': 4, 'pin_memory': True }

	train = FakeData(FAKE_SIZE, (1, 28, 28), 3, transform=flat_transform)
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)

	return train_loader, train_loader
