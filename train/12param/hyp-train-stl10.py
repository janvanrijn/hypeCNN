from __future__ import print_function
import ConfigSpace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
from resnet_stl10 import ResNet18
from time import time

def get_hyperparameter_search_space(seed=None):
    """
    Neural Network search space based on a best effort using the scikit-learn
    implementation. Note that for state of the art performance, other packages
    could be preferred.

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('ResNet18_classifier', seed)
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=1, upper=200, default_value=150)
    batch_size = ConfigSpace.CategoricalHyperparameter(
        name='batch_size', choices=[8, 16, 32, 64, 128], default_value=128)
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)
    lr_decay = ConfigSpace.UniformIntegerHyperparameter(
        name='lr_decay', lower=2, upper=1000, log=True, default_value=10)
    patience = ConfigSpace.UniformIntegerHyperparameter(
        name='patience', lower=2, upper=200, log=False, default_value=10)
    tolerance = ConfigSpace.UniformFloatHyperparameter(
        name='tolerance', lower=1e-5, upper=1e-2, log=True, default_value=1e-4)
    resize_crop = ConfigSpace.CategoricalHyperparameter(
        name='resize_crop', choices=[True, False], default_value=False)
    h_flip = ConfigSpace.CategoricalHyperparameter(
        name='h_flip', choices=[True, False], default_value=False)
    v_flip = ConfigSpace.CategoricalHyperparameter(
        name='v_flip', choices=[True, False], default_value=False)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)

    cs.add_hyperparameters([
        batch_size,
        learning_rate_init,
        epochs,
        momentum,
        weight_decay,
        lr_decay,
        patience,
        tolerance,
        resize_crop,
        h_flip,
        v_flip,
        shuffle,
    ])

    return cs

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc, test_loss

def load_data(shuffle, batch_size, resize_crop, h_flip, v_flip):
    t_list = []
    if resize_crop:
        t_list.append(transforms.RandomCrop(96, padding=4))
    if h_flip:
        t_list.append(transforms.RandomHorizontalFlip())
    if v_flip:
        t_list.append(transforms.RandomVerticalFlip())

    t_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]

    transform_train = transforms.Compose(t_list)
    
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    
    trainset = datasets.STL10(root='/rigel/dsi/users/as5414/stl10', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    testset = datasets.STL10(root='/rigel/dsi/users/as5414/stl10', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=shuffle)
    return train_loader, test_loader

def run_train(seed):
    device = torch.device("cuda")
    model = ResNet18().to(device)
    #### read hyps here ####
    cs = get_hyperparameter_search_space(seed)
    hyps = cs.sample_configuration(1).get_dictionary()
    lr = hyps['learning_rate_init']
    mom = hyps['momentum']
    batch_size = hyps['batch_size']
    epochs = hyps['epochs']
    weight_decay = hyps['weight_decay']
    lr_decay = 1.0/hyps['lr_decay']
    patience = hyps['patience']
    tolerance = hyps['tolerance']
    resize_crop = hyps['resize_crop']
    h_flip = hyps['h_flip']
    v_flip = hyps['v_flip']
    shuffle = hyps['shuffle']

    train_loader, test_loader = load_data(shuffle, batch_size, resize_crop, h_flip, v_flip)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay, patience=patience, threshold=tolerance)

    acc_list = []
    loss_list = []
    time_list = []

    start = time()
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch)
        test_acc, test_loss = test(model, device, test_loader)
        scheduler.step(test_acc/100)
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        time_list.append(time()-start)
    return acc_list, loss_list, time_list, hyps

if __name__ == '__main__':
    for i in range(200,250):
        try:
            acc_list, loss_list, time_list, hyps = run_train(i)
            s = ''
            for j in range(len(acc_list)):
                s += str(i)+' '+str(acc_list[j])+' '+str(loss_list[j])+' '+str(time_list[j])+' '+str(j)+' '+str(hyps)+'\n'
        except:
            s = str(i)+' ERROR!\n'
        f = open('output.txt', 'a')
        f.write(s)
        print(s)
