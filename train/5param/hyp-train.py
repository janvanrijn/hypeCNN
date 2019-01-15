from __future__ import print_function
import ConfigSpace
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
import sys
# from resnet import ResNet18
# from time import time

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
    strategy = ConfigSpace.CategoricalHyperparameter(
        name='columntransformer__numeric__imputer__strategy', choices=['mean', 'median', 'most_frequent'])
    # hidden_layer_sizes = ConfigSpace.UniformIntegerHyperparameter(
    #     name='mlpclassifier__hidden_layer_sizes', lower=32, upper=2048, default_value=2048)
    # activation = ConfigSpace.CategoricalHyperparameter(
    #     name='mlpclassifier__activation', choices=['identity', 'logistic', 'tanh', 'relu'], default_value='relu')
    # solver = ConfigSpace.CategoricalHyperparameter(
    #     name='mlpclassifier__solver', choices=['lbfgs', 'sgd', 'adam'], default_value='adam')
    # alpha = ConfigSpace.UniformFloatHyperparameter(
    #     name='mlpclassifier__alpha', lower=1e-5, upper=1e-1, log=True, default_value=1e-4)
    batch_size = ConfigSpace.UniformIntegerHyperparameter(
        name='batch_size', lower=32, upper=4096, default_value=200)
    learning_rate = ConfigSpace.CategoricalHyperparameter(
        name='learning_rate', choices=['constant', 'invscaling', 'adaptive'], default_value='constant')
    learning_rate_init = ConfigSpace.UniformFloatHyperparameter(
        name='learning_rate_init', lower=1e-5, upper=1e-1, log=True, default_value=1e-04)
    # TODO: Sensible range??
    # power_t = ConfigSpace.UniformFloatHyperparameter(
    #     name='mlpclassifier__power_t', lower=1e-5, upper=1, log=True, default_value=0.5)
    epochs = ConfigSpace.UniformIntegerHyperparameter(
        name='epochs', lower=64, upper=1024, default_value=200)
    shuffle = ConfigSpace.CategoricalHyperparameter(
        name='shuffle', choices=[True, False], default_value=True)
    # tol = ConfigSpace.UniformFloatHyperparameter(
    #     name='mlpclassifier__tol', lower=1e-5, upper=1e-1, default_value=1e-4, log=True)
    # TODO: log-scale?
    momentum = ConfigSpace.UniformFloatHyperparameter(
        name='momentum', lower=0, upper=1, default_value=0.9)
    # nesterovs_momentum = ConfigSpace.CategoricalHyperparameter(
    #     name='mlpclassifier__nesterovs_momentum', choices=[True, False], default_value=True)
    # early_stopping = ConfigSpace.CategoricalHyperparameter(
    #     name='mlpclassifier__early_stopping', choices=[True, False], default_value=True)
    # validation_fraction = ConfigSpace.UniformFloatHyperparameter(
    #     name='mlpclassifier__validation_fraction', lower=0, upper=1, default_value=0.1)
    # beta_1 = ConfigSpace.UniformFloatHyperparameter(
    #     name='mlpclassifier__beta_1', lower=0, upper=1, default_value=0.9)
    # beta_2 = ConfigSpace.UniformFloatHyperparameter(
    #     name='mlpclassifier__beta_2', lower=0, upper=1, default_value=0.999)
    # n_iter_no_change = ConfigSpace.UniformIntegerHyperparameter(
    #     name='mlpclassifier__n_iter_no_change', lower=1, upper=1024, default_value=200)

    cs.add_hyperparameters([
        # strategy,
        # hidden_layer_sizes,
        # activation,
        # solver,
        # alpha,
        batch_size,
        learning_rate,
        learning_rate_init,
        # power_t,
        epochs,
        shuffle,
        # tol,
        momentum,
        # nesterovs_momentum,
        # early_stopping,
        # validation_fraction,
        # beta_1,
        # beta_2,
        # n_iter_no_change,
    ])

    # batch_size_condition = ConfigSpace.InCondition(batch_size, solver, ['sgd', 'adam'])
    # learning_rate_init_condition = ConfigSpace.InCondition(learning_rate_init, solver, ['sgd', 'adam'])
    # power_t_condition = ConfigSpace.EqualsCondition(power_t, solver, 'sgd')
    # shuffle_confition = ConfigSpace.InCondition(shuffle, solver, ['sgd', 'adam'])
    # tol_condition = ConfigSpace.InCondition(tol, learning_rate, ['constant', 'invscaling'])
    # momentum_confition = ConfigSpace.EqualsCondition(momentum, solver, 'sgd')
    # nesterovs_momentum_confition_solver = ConfigSpace.EqualsCondition(nesterovs_momentum, solver, 'sgd')
    # nesterovs_momentum_confition_momentum = ConfigSpace.GreaterThanCondition(nesterovs_momentum, momentum, 0)
    # nesterovs_momentum_conjunstion = ConfigSpace.AndConjunction(nesterovs_momentum_confition_solver,
    #                                                             nesterovs_momentum_confition_momentum)
    # early_stopping_condition = ConfigSpace.InCondition(early_stopping, solver, ['sgd', 'adam'])
    # validation_fraction_condition = ConfigSpace.EqualsCondition(validation_fraction, early_stopping, True)
    # beta_1_condition = ConfigSpace.EqualsCondition(beta_1, solver, 'adam')
    # beta_2_condition = ConfigSpace.EqualsCondition(beta_2, solver, 'adam')
    # n_iter_no_change_condition_solver = ConfigSpace.InCondition(n_iter_no_change, solver, ['sgd', 'adam'])

    # cs.add_condition(batch_size_condition)
    # cs.add_condition(learning_rate_init_condition)
    # cs.add_condition(power_t_condition)
    # cs.add_condition(shuffle_confition)
    # cs.add_condition(tol_condition)
    # cs.add_condition(momentum_confition)
    # cs.add_condition(nesterovs_momentum_conjunstion)
    # cs.add_condition(early_stopping_condition)
    # cs.add_condition(validation_fraction_condition)
    # cs.add_condition(beta_1_condition)
    # cs.add_condition(beta_2_condition)
    # cs.add_condition(n_iter_no_change_condition_solver)

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

def load_data():
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

def run_train(cs):
    cs = get_hyperparameter_search_space()
    hyps = cs.sample_configuration(1).get_dictionary()
    #### read hyps here ####

    if torch.cuda.is_available():
        print('CUDA!!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ResNet18().to(device)
    #model = torch.load('/Users/abhinavsharma/Desktop/AM/mnist-e1')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=hyps['momentum'])
    best_test_acc = 0

    for epoch in range(1, epochs + 1):
        if sys.argv[1]=='1':
            if epoch==151:
                lr /= 10
            if epoch==251:
                lr /= 10
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=hyps['momentum'], weight_decay=5e-4)
        start = time()
        train(model, device, train_loader, optimizer, epoch)
        print('Time: {}'.format(time()-start))
        test_acc = test(model, device, test_loader)
        if test_acc>best_test_acc and epoch%10==0:
            best_test_acc = test_acc
            torch.save(model, 'cifar10_'+str(epoch)+'_'+str(test_acc))

def get_config(seed):
    print(hyps)

get_config(int(sys.argv[1]))

# c = get_hyperparameter_search_space(1)
# print(c)
