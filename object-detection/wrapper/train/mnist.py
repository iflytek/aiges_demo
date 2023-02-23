from __future__ import print_function

import argparse
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def val2():
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt'))  # 根据模型结构，调用存储的模型参数
    #
    # test dataset
    #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    f = open("../test_data/img_1.png", 'rb')
    b = f.read()
    stream = io.BytesIO(b)
    img = Image.open(stream).resize((28, 28)).convert("L")
    # testing.
    #
    with torch.no_grad():
        img = transform(img)
        img.unsqueeze_(0)
        img.to(device)
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probabili
        print(int(pred))


def val3():
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt'))  # 根据模型结构，调用存储的模型参数
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    #
    img = Image.open('../test_data/img_1.png')
    img = transform(img).unsqueeze(0)
    print(img.shape)
    img.to(device)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probabili
    print(pred)


def val():
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt'))  # 根据模型结构，调用存储的模型参数
    #
    # test dataset
    #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset)

    # testing
    #
    with torch.no_grad():
        model.eval()
        test(model, device, test_loader)


def test2(model, device, dataset):
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)

    data, labels_ground_truth = testloader.__iter__().next()

    labels_ground_truth = labels_ground_truth.numpy().copy()

    _pred = model(data).numpy().copy()
    labels_pred = np.argmax(_pred, axis=1)

    # result = confusion_matrix(labels_ground_truth, labels_pred)

    print(labels_pred)


def val4():
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt'))  # 根据模型结构，调用存储的模型参数
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    #
    img = Image.open('../test_data/img_1.png')
    img.show()
    img = transform(img)
    new_img_PIL = transforms.ToPILImage()(img).convert('L')
    new_img_PIL.show()
    img = img.unsqueeze(0)
    print(img.shape)
    img.to(device)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probabili
    print(pred)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = False

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    # main()
    val4()
