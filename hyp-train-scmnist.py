from __future__ import print_function
import ConfigSpace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from resnet import ResNet18
from time import time
import numpy as np
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class SCmnistDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the metadata csv file.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.skin_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.skin_df)

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(np.asarray(self.skin_df.iloc[idx][:-1]).reshape((28,28,3))))
        label = self.skin_df.iloc[idx][-1]

        if self.transform:
            image = self.transform(image)

        return (image, label)

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
	# batch_size = ConfigSpace.UniformIntegerHyperparameter(
	#     name='batch_size', lower=1, upper=256, log=True, default_value=128)
	# learning_rate = ConfigSpace.CategoricalHyperparameter(
	#     name='learning_rate', choices=['constant', 'invscaling', 'adaptive'], default_value='constant')
	learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
		name='learning_rate_init', lower=1e-6, upper=1, log=True, default_value=1e-1)

	epochs = ConfigSpace.UniformIntegerHyperparameter(
		name='epochs', lower=1, upper=200, default_value=150)
	batch_size = ConfigSpace.CategoricalHyperparameter(
		name='batch_size', choices=[32, 64, 128, 256, 512], default_value=128)
	# shuffle = ConfigSpace.CategoricalHyperparameter(
	#     name='shuffle', choices=[True, False], default_value=True)
	momentum = ConfigSpace.UniformFloatHyperparameter(
		name='momentum', lower=0, upper=1, default_value=0.9)
	weight_decay = ConfigSpace.UniformFloatHyperparameter(
		name='weight_decay', lower=1e-6, upper=1e-2, log=True, default_value=5e-4)

	cs.add_hyperparameters([
		batch_size,
		learning_rate_init,
		epochs,
		# shuffle,
		momentum,
		weight_decay,
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

def test(model, device, test_loader, len_test):
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

	test_loss /= len_test
	test_acc = 100. * correct / len_test
	return test_acc, test_loss

def load_data(batch_size):
    root_dir = '/rigel/dsi/users/as5414/scmnist/'
    split = 0.9

    ### Change this - resize image
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=6),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    trainset = SCmnistDataset(csv_file=root_dir+'hmnist_28_28_RGB.csv', transform=transform_train)

    transform_test = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    testset = SCmnistDataset(csv_file=root_dir+'hmnist_28_28_RGB.csv', transform=transform_test)

    length = len(trainset)
    split_idx = int(split*length)
    indices = list(range(length))
    np.random.shuffle(indices)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    len_test = len(test_idx)
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

    test_loader = DataLoader(testset, batch_size=501, sampler=test_sampler)
    
    return train_loader, test_loader, len_test

def run_train(seed):
	device = torch.device("cuda")
	#print(device)
	model = ResNet18(7).to(device)
	#### read hyps here ####
	cs = get_hyperparameter_search_space(seed)
	hyps = cs.sample_configuration(1).get_dictionary()
	lr = hyps['learning_rate_init']
	mom = hyps['momentum']
	batch_size = hyps['batch_size']
	epochs = hyps['epochs']
	weight_decay = hyps['weight_decay']

	train_loader, test_loader, len_test = load_data(batch_size)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-6)

	acc_list = []
	loss_list = []
	time_list = []

	start = time()
	for epoch in range(epochs):
		train(model, device, train_loader, optimizer, epoch)
		test_acc, test_loss = test(model, device, test_loader, len_test)
		scheduler.step(test_acc/100)
		acc_list.append(test_acc)
		loss_list.append(test_loss)
		time_list.append(time()-start)
	return acc_list, loss_list, time_list, hyps

if __name__ == '__main__':
	for i in range(50):
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
