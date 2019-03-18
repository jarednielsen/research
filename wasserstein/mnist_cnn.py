import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from layers import SinkhornDistance
import pickle
from wide_net import WideResNet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class resnt(nn.Module):
    def __init__(self):
        super(resnt, self).__init__()
        self.main_model = models.resnet101()
        self.fc1 = nn.Linear(1000,100)
    
    def forward(self, x):
        x = F.relu(self.main_model(x))
        x = self.fc1(x)
        return x
    
def train(args, model, device, train_loader, optimizer, epoch, use_sinkhorn=False):
    model.train()
    sinkhorn_distance = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        target_one_hot = torch.zeros(len(target), 100).scatter_(1, target.cpu().unsqueeze(1), 1.)
        target_one_hot = target_one_hot * (1 - 0.01) + 1 - target_one_hot*0.01/(len(target_one_hot) - 1) #label smoothing
        if use_sinkhorn:
            
            loss, _, _ = sinkhorn_distance(output.float().cpu(), target_one_hot.float().cpu())
        else:
            loss = F.cross_entropy(output.float().cpu(), target.cpu())

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, use_sinkhorn=False):
    model.eval()
    test_loss = 0
    correct = 0
    sinkhorn_distance = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target_one_hot = torch.zeros(len(target), 100).scatter_(1, target.cpu().unsqueeze(1), 1.)
            target_one_hot = target_one_hot * (1 - 0.01) + 1 - target_one_hot*0.01/(len(target_one_hot) - 1) #label smoothing
            if use_sinkhorn:
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                loss, _, _ = sinkhorn_distance(output.float().cpu(), target_one_hot.float().cpu())
            else:
                loss = F.cross_entropy(output.float().cpu(), target.cpu())
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if use_sinkhorn:
        with open("sinkhorn_results.pkl", "rb") as f:
            acc = pickle.load(f)
    else:
        with open("xent_results.pkl", "rb") as f:
            acc = pickle.load(f)

    acc.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if use_sinkhorn:
        with open("sinkhorn_results.pkl", "wb") as f:
            pickle.dump(acc, f)
    else:
        with open("xent_results.pkl", "wb") as f:
            pickle.dump(acc, f)

def reset_pkl(use_sinkhorn = False):
    acc = []
    if use_sinkhorn:
        with open("sinkhorn_results.pkl", "wb") as f:
            pickle.dump(acc, f)
    else:
        with open("xent_results.pkl", "wb") as f:
            pickle.dump(acc, f)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--use-sinkhorn', action='store_true', default=False,
                        help='Determine the loss to be used')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True,
                       transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # model = Net().to(device)
    model = WideResNet(depth=100, num_classes=100).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters())

    reset_pkl(use_sinkhorn=args.use_sinkhorn)

    for epoch in range(1, args.epochs + 1):
        if args.use_sinkhorn:
            train(args, model, device, train_loader, optimizer, epoch, use_sinkhorn=True)
            test(args, model, device, test_loader, use_sinkhorn=True)
        else:
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()