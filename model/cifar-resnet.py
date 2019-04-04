from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
from .resnet import ResNet18
from time import time

# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Conv2d(3, 64, 5)
# 		self.pool = nn.MaxPool2d(3, stride=2)
# 		self.conv2 = nn.Conv2d(64, 64, 5)
# 		self.fc1 = nn.Linear(64*16, 512)
# 		self.drop = nn.Dropout2d()
# 		self.fc2 = nn.Linear(512, 10)

# 	def forward(self, x):
# 		x = self.pool(F.relu(self.conv1(x)))
# 		#print(x.shape)
# 		x = self.pool(F.relu(self.conv2(x)))
# 		x = x.view(-1, 16*64)
# 		x = F.relu(self.fc1(x))
# 		x = self.drop(x)
# 		x = self.fc2(x)
# 		return x

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
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

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
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), test_acc))
	return test_acc


lr = 0.1
momentum = 0.9
batch_size = 128
log_interval = 100

if torch.cuda.is_available():
	print('CUDA!!')
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

if sys.argv[1]=='0':
	print('MNIST')
	epochs = 5
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
					   transform=transforms.Compose([transforms.RandomCrop(32, padding=6), transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)),transforms.Lambda(lambda x: x.repeat(3, 1, 1) )])),
						batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, download=True,
						transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )])),
						batch_size=1000, shuffle=True)
	
if sys.argv[1]=='1':
	print('CIFAR10')
	epochs = 350
	#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
	
	transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
	
	trainset = datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True)
	testset = datasets.CIFAR10(root='./data', train=False,
										   download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
											 shuffle=False)

if sys.argv[1]=='2':
	print('FashionMNIST')
	epochs = 5
	train_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST('../data', train=True, download=True,
					   transform=transforms.Compose([transforms.RandomCrop(32, padding=6), transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)),transforms.Lambda(lambda x: x.repeat(3, 1, 1) )])),
						batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST('../data', train=False, download=True,
						transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )])),
						batch_size=1000, shuffle=True)


# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#model = Net().to(device)
model = ResNet18().to(device)
#model = torch.load('/Users/abhinavsharma/Desktop/AM/mnist-e1')
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
best_test_acc = 0

for epoch in range(1, epochs + 1):
	if sys.argv[1]=='1':
		if epoch==151:
			lr /= 10
		if epoch==251:
			lr /= 10
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
	start = time()
	train(model, device, train_loader, optimizer, epoch)
	print('Time: {}'.format(time()-start))
	test_acc = test(model, device, test_loader)
	if test_acc>best_test_acc and epoch%10==0:
		best_test_acc = test_acc
		torch.save(model, 'cifar10_'+str(epoch)+'_'+str(test_acc))

