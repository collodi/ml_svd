import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.cv_drop = nn.Dropout2d(p=0.2)

		self.cv1 = nn.Conv2d(1, 10, 5, padding=2)
		self.cv2 = nn.Conv2d(10, 20, 5, padding=2)
		self.cv3 = nn.Conv2d(20, 20, 5, padding=2)
		self.cv4 = nn.Conv2d(20, 10, 5, padding=2)
		self.cv5 = nn.Conv2d(10, 5, 5, padding=2)

		self.fc = nn.Sequential(
				nn.AlphaDropout(p=0.25),
				nn.Linear(125, 80), nn.SELU(),
				nn.AlphaDropout(p=0.2),
				nn.Linear(80, 80), nn.SELU(),
				nn.Linear(80, 80), nn.SELU()
			)

		self.out = nn.Linear(80, 10)

	def forward(self, x):
		x = F.selu(self.cv1(x))
		x = self.cv_drop(x)
		x = F.adaptive_max_pool2d(x, 24)

		x = F.selu(self.cv2(x))
		x = self.cv_drop(x)
		x = F.adaptive_max_pool2d(x, 20)

		x = F.selu(self.cv3(x))
		x = self.cv_drop(x)
		x = F.adaptive_max_pool2d(x, 15)

		x = F.selu(self.cv4(x))
		x = F.adaptive_max_pool2d(x, 10)

		x = F.selu(self.cv5(x))
		x = F.adaptive_max_pool2d(x, 5)

		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return self.out(x)
