# deconv.py

"""
Train a network to map a vector of all 1's to an image of a cat.
"""

from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IN = 10

class CatNet(nn.Module):
	"""
	Maps a (10,) to a (3, 50, 50) array.
	Should generate a cat image from an initial vector of 1's.
	"""
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(IN, 3*50*50)

	def forward(self, x):
		# try changing this to relu and see what happens!
		x = F.leaky_relu(self.fc1(x), negative_slope=0.01) 
		x = x.view(3, 50, 50)
		return x


def train_cat_net(cat, epochs=3000):
	"""
	Trains a neural network to generate a cat from an vector of 1's.
	Returns the network.

	Parameters:
		cat ((3,50,50) tensor): an RGB image of a cat.
	Returns:
		net (class): the network.
	"""
	net = CatNet()
	input = torch.ones(IN) + 0.0*torch.randn(IN)
	optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.9)
	criterion = nn.MSELoss()

	for epoch in range(epochs):
		for i in range(1):
			optimizer.zero_grad()
			output = net(input)
			loss = criterion(output, cat)
			loss.backward()
			optimizer.step()

		if epoch%100 == 99:
			n_output = np.uint8(output.detach())
			n_output = np.transpose(n_output, (1, 2, 0))
			img = Image.fromarray(n_output)
			img.save('intermediate_outputs/catnet_epoch_{}.jpeg'.format(epoch+1))

	return net

def read_img(filename):
	"""
	Takes a filename and returns a Numpy array representing the cat image.

	Parameters:
		filename (str): the filename.
	Returns:
		img ((3,40,40) tensor): the RGB image of the cat.
	"""
	img = Image.open(filename)
	img_arr = np.array(img) # (width, height, channels)
	img_arr = np.transpose(img_arr, (2, 0, 1)) # (channels, width, height)
	img_tensor = torch.from_numpy(img_arr.astype(np.float32))
	return img_tensor

def save_net(net, filename):
	"""
	Serialize the Torch network.

	Parameters:
		net (Torch network): the trained network.
		filename (str): the filename of the serialized model.
	"""
	torch.save(net.state_dict(), filename)
	

def load_net(filename):
	"""
	Deserialize the Torch network.

	Parameters:
		filename (str): the filename of the serialized model.
	Returns:
		net (Torch network): the network with pretrained weights.
	"""
	net = CatNet()
	net.load_state_dict(torch.load(filename))
	return net

def generate_cat(net):
	"""
	Generate a cat, given the trained network.

	Parameters:
		net (Torch network): the trained network.
	Returns:
		cat ((3,50,50) ndarray): an image of a cat.
	"""
	input = torch.ones(IN)
	output = net(input)
	return output

def plot_cat_comparison(original, synthetic):
	"""
	Plot two images side-by-side.
	The images must be in form (width, height, channels).
	"""
	fig, axs = plt.subplots(1, 2)
	for i in range(2):
		axs[i].set_axis_off()
	axs[0].imshow(original)
	axs[1].imshow(synthetic)
	plt.show()

def get_np_ch_last(img):
	"""
	Turns a (ch, width, height) Torch tensor into a (width, height, ch) ndarray.
	"""
	return np.transpose(img.detach().numpy(), (1, 2, 0)).astype(np.uint8)

def main():
	"""
	Reads in the cat image, trains a network, saves the network weights for
	easy restoration, restores the network weights, generates a cat, and
	plots the two side-by-side.
	"""
	CAT_FILENAME = 'jared_data/cat_small.jpeg'
	NET_FILENAME = 'jared_data/cat_net.'
	cat = read_img(CAT_FILENAME)
	net = train_cat_net(cat)
	save_net(net, NET_FILENAME)
	load_net(NET_FILENAME)
	generated_cat = generate_cat(net)
	plot_cat_comparison(original=get_np_ch_last(cat), 
		synthetic=get_np_ch_last(generated_cat))

if __name__ == "__main__":
	main()
