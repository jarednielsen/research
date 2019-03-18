import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from mag.experiment import Experiment
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split, TensorDataset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm, tqdm_notebook, trange
from time import sleep

from models import CifarVAENet


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-size', type=int, default=20, metavar='N',
                    help='how big is z')
parser.add_argument('--intermediate-size', type=int, default=128, metavar='N',
                    help='how big is linear around z')
# parser.add_argument('--widen-factor', type=int, default=1, metavar='N',
#                     help='how wide is the model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda') if args.cuda else torch.device('cpu')
args.device = device
print("DEVICE: {}".format(args.device))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

root_dir = "../data/cifar10/"
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
data_train = torchvision.datasets.CIFAR10(root_dir, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
data_test = torchvision.datasets.CIFAR10(root_dir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True,
                            **kwargs)
test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=True,
                            **kwargs)



model = CifarVAENet(args)
if args.cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = torch.Tensor(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item()
        if epoch == args.epochs and i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                   recon_batch[:n]])
            save_image(comparison.data.cpu(),
                       'snapshots/conv_vae/reconstruction_' + str(epoch) +
                       '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

model.load_state_dict(torch.load(model.model_path))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

sample = torch.Tensor(torch.randn(64, args.hidden_size))
if args.cuda:
    sample = sample.cuda()
sample = model.decode(sample).cpu()
save_image(sample.data.view(64, 3, 32, 32),
        'snapshots/conv_vae/sample_' + str(args.epochs) + '.png')
torch.save(model.state_dict(), model.model_path)
