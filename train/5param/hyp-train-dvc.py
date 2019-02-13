from __future__ import print_function
import ConfigSpace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
from resnet import ResNet18
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
	# batch_size = ConfigSpace.UniformIntegerHyperparameter(
	#     name='batch_size', lower=1, upper=256, log=True, default_value=128)
	# learning_rate = ConfigSpace.CategoricalHyperparameter(
	#     name='learning_rate', choices=['constant', 'invscaling', 'adaptive'], default_value='constant')
	learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
		name='learning_rate_init', lower=1e-6, upper=1e-1, log=True, default_value=1e-1)

	epochs = ConfigSpace.UniformIntegerHyperparameter(
		name='epochs', lower=1, upper=100, default_value=80)
	batch_size = ConfigSpace.CategoricalHyperparameter(
		name='batch_size', choices=[8, 16, 32, 64, 128], default_value=128)
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

def load_data(shuffle, batch_size):
	root = '/rigel/dsi/users/as5414/dvc-processed/'
	transform_train = transforms.Compose([
	transforms.RandomCrop(96),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
	
	transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
	
	trainset = datasets.ImageFolder(root=root+'train/', transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
	testset = datasets.ImageFolder(root=root+'test/', transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=shuffle, num_workers=4)
	return train_loader, test_loader

def run_train(seed):
	device = torch.device("cuda")
	print(device)
	model = ResNet18(2).to(device)
	#### read hyps here ####
	cs = get_hyperparameter_search_space(seed)
	hyps = cs.sample_configuration(1).get_dictionary()
	lr = hyps['learning_rate_init']
	mom = hyps['momentum']
	batch_size = hyps['batch_size']
	epochs = hyps['epochs']
	shuffle = True #hyps['shuffle']
	weight_decay = hyps['weight_decay']

	train_loader, test_loader = load_data(shuffle, batch_size)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', min_lr=1e-6)

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
