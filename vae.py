# vae.py

"""
Creates a variational autoencoder that works on MNIST images.
See http://kvfrans.com/variational-autoencoders-explained/
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 100)
		self.fc2 = nn.Linear(100, 28*28)

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

def get_data():
	root = './data'
	if not os.path.exists(root):
		os.mkdir(root)

	trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
	train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
	test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
	return train_set, test_set

def test_loss(net, test_loader, criterion, n=1000):
	total_loss = 0
	iters = 0
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			inputs, labels = data
			outputs = net(inputs)
			batch_avg_loss = criterion(outputs, inputs.view(-1, 28*28))
			total_loss += batch_avg_loss.item()
			iters += 1
			if i >= n:
				break
	print('Test loss on the {n} test images: {acc}'.format(n=n, acc=total_loss/iters))

def show_sample(net, test_loader):
	with torch.no_grad():
		inputs, labels = iter(test_loader).next()
		outputs = net(inputs)
		original = inputs[1,0]
		reconstruction = outputs.view(-1, 28, 28)[1]

		return original, reconstruction

PICKLE_FILE = 'mnist_gan_examples.p'
def plot_samples(examples=None):
	if examples is None:
		examples = pickle.load(open(PICKLE_FILE, 'rb'))

	n = len(examples)
	fig, axs = plt.subplots(2, n, figsize=None)
	for i in range(n):
		axs[0,i].set_axis_off()
		axs[1,i].set_axis_off()
		axs[0,i].imshow(examples[i][0], cmap='Greys')
		axs[1,i].imshow(examples[i][1], cmap='Greys')
	plt.show()

def run_vae():
	examples = []

	train_set, test_set = get_data()
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

	net = Net()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	criterion = nn.MSELoss()

	for epoch in range(2):
		running_loss = 0.0

		for i, data in enumerate(train_loader):
			inputs, labels = data
			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, inputs.view(-1, 28*28))
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i%2000 == 1999:
				print('[{}, {}] train loss: {}'.format(epoch+1, i+1, running_loss/2000))
				test_loss(net=net, test_loader=test_loader, criterion=criterion)
				running_loss = 0.0
				original, reconstruction = show_sample(net, test_loader)
				examples.append((original, reconstruction))

	pickle.dump(examples, open(PICKLE_FILE, 'wb'))
	print('Finished Training')

if __name__ == "__main__":
	# run_vae()
	plot_samples()