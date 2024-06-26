import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np


# input_size = 512
# hidden = [512, 256]
# num_classes = 10 or 43 or ...


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        #self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)
        #self.relu = nn.ReLU()
        #self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        #out = self.softmax(out)
        #out = F.log_softmax(out, dim=1)
        return out


def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    # transform to torch tensor
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)

    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion):
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long()) # F.nll_loss(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test(net, test_loader, epoch, criterion):
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0
    prediction = np.array([])

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            test_loss += criterion(output, target.long()).item() #F.nll_loss(output, target.long(), reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            if epoch % 50 == 0:
                prediction = np.append(prediction, pred.cpu().numpy().reshape(1,-1)[0])
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - Accuracy", "value": {}, "epoch": {}}}'.format(
        100. * correct / len(test_loader.dataset), epoch))

    if epoch % 50 == 0:
        print("\n---------------------------")
        print("Classifier's prediction: ")
        unique, counts = np.unique(prediction, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        print({k: v for k, v in sorted(counts_dict.items(), key=lambda item: item[1])})
        print("---------------------------\n")

    return test_acc
