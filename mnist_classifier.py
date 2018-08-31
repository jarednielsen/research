# mnist_classifier.py

"""
Creates a classifier that works on MNIST images.
See http://kvfrans.com/variational-autoencoders-explained/
"""

import os
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

def get_data():
	root = './data'
	if not os.path.exists(root):
		os.mkdir(root)

	trans = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
	train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
	test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
	return train_set, test_set

def run_vae():
	train_set, test_set = get_data()
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

	net = Net()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(2):
		running_loss = 0.0

		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i%2000 == 1999:
				print('[{}, {}] loss: {}'.format(epoch+1, i+1, running_loss/2000))
				running_loss = 0.0

	print('Finished Training')

if __name__ == "__main__":
	run_vae()