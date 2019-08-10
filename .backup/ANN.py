import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp

torch.utils.backcompat.broadcast_warning.enabled = True


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(784, 250)  # input output
        self.A1 = nn.LeakyReLU()
        # input must be the same as previous ouput
        self.L2 = nn.Linear(250, 100)
        self.A2 = nn.LeakyReLU()
        self.L3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return F.log_softmax(x, dim=1)


def fitFunction(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    # model.train()

    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch  # Variable(X_barch).float()
            var_y_batch = y_batch  # Variable(y_barch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            var_y_batch = var_y_batch.squeeze_()
            loss = error(output, var_y_batch)

            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()

            if batch_idx % 50 == 0:
                print(
                    "Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%".format(
                        epoch,
                        batch_idx * len(X_batch),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        float(correct * 100) / float(32 * (batch_idx + 1)),
                    )
                )


"""    input = np.random.random((16))
    input = torch.Tensor(input) # input size
    output = model(input) # otput size (4)

    with torch.no_grad():
        input_next = np.random.random((16))
        input_next = torch.Tensor(input_next) # input size
        output_next = model(input_next) # I don't want the back propagation on this

    loss = F.smooth_l1_loss(output,output_next)

    # optimize step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""


if __name__ == "__main__":
    MULTI = True
    # Import DATA
    import torchvision.datasets as datasets

    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None
    )
    X_train = mnist_trainset.train_data.type(torch.FloatTensor).view(60000, 28 * 28)
    y_train = mnist_trainset.train_labels.type(torch.LongTensor).view(60000, 1)
    X_test = mnist_testset.test_data.type(torch.FloatTensor).view(10000, 28 * 28)
    y_test = mnist_testset.test_labels.type(torch.LongTensor).view(10000, 1)
    BATCH_SIZE = 32
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False
    )
    if MULTI:
        # multiprocessing settings
        mp.set_start_method("spawn")
        num_processes = 4

    model = NeuralNetwork()
    model.type(torch.FloatTensor)

    if MULTI:
        # for multiprocessing
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=fitFunction, args=(model, train_loader))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        fitFunction(model, train_loader)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp

torch.utils.backcompat.broadcast_warning.enabled = True


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(784, 250)  # input output
        self.A1 = nn.LeakyReLU()
        # input must be the same as previous ouput
        self.L2 = nn.Linear(250, 100)
        self.A2 = nn.LeakyReLU()
        self.L3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return F.log_softmax(x, dim=1)


def fitFunction(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    # model.train()

    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch  # Variable(X_barch).float()
            var_y_batch = y_batch  # Variable(y_barch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            var_y_batch = var_y_batch.squeeze_()
            loss = error(output, var_y_batch)

            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()

            if batch_idx % 50 == 0:
                print(
                    "Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%".format(
                        epoch,
                        batch_idx * len(X_batch),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        float(correct * 100) / float(32 * (batch_idx + 1)),
                    )
                )


"""    input = np.random.random((16))
    input = torch.Tensor(input) # input size
    output = model(input) # otput size (4)

    with torch.no_grad():
        input_next = np.random.random((16))
        input_next = torch.Tensor(input_next) # input size
        output_next = model(input_next) # I don't want the back propagation on this

    loss = F.smooth_l1_loss(output,output_next)

    # optimize step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""


if __name__ == "__main__":
    MULTI = True
    # Import DATA
    import torchvision.datasets as datasets

    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None
    )
    X_train = mnist_trainset.train_data.type(torch.FloatTensor).view(60000, 28 * 28)
    y_train = mnist_trainset.train_labels.type(torch.LongTensor).view(60000, 1)
    X_test = mnist_testset.test_data.type(torch.FloatTensor).view(10000, 28 * 28)
    y_test = mnist_testset.test_labels.type(torch.LongTensor).view(10000, 1)
    BATCH_SIZE = 32
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False
    )
    if MULTI:
        # multiprocessing settings
        mp.set_start_method("spawn")
        num_processes = 4

    model = NeuralNetwork()
    model.type(torch.FloatTensor)

    if MULTI:
        # for multiprocessing
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=fitFunction, args=(model, train_loader))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        fitFunction(model, train_loader)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp

torch.utils.backcompat.broadcast_warning.enabled = True


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(784, 250)  # input output
        self.A1 = nn.LeakyReLU()
        # input must be the same as previous ouput
        self.L2 = nn.Linear(250, 100)
        self.A2 = nn.LeakyReLU()
        self.L3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return F.log_softmax(x, dim=1)


def fitFunction(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    # model.train()

    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch  # Variable(X_barch).float()
            var_y_batch = y_batch  # Variable(y_barch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            var_y_batch = var_y_batch.squeeze_()
            loss = error(output, var_y_batch)

            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()

            if batch_idx % 50 == 0:
                print(
                    "Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%".format(
                        epoch,
                        batch_idx * len(X_batch),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        float(correct * 100) / float(32 * (batch_idx + 1)),
                    )
                )


"""    input = np.random.random((16))
    input = torch.Tensor(input) # input size
    output = model(input) # otput size (4)

    with torch.no_grad():
        input_next = np.random.random((16))
        input_next = torch.Tensor(input_next) # input size
        output_next = model(input_next) # I don't want the back propagation on this

    loss = F.smooth_l1_loss(output,output_next)

    # optimize step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""


if __name__ == "__main__":
    MULTI = True
    # Import DATA
    import torchvision.datasets as datasets

    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None
    )
    X_train = mnist_trainset.train_data.type(torch.FloatTensor).view(60000, 28 * 28)
    y_train = mnist_trainset.train_labels.type(torch.LongTensor).view(60000, 1)
    X_test = mnist_testset.test_data.type(torch.FloatTensor).view(10000, 28 * 28)
    y_test = mnist_testset.test_labels.type(torch.LongTensor).view(10000, 1)
    BATCH_SIZE = 32
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False
    )
    if MULTI:
        # multiprocessing settings
        mp.set_start_method("spawn")
        num_processes = 4

    model = NeuralNetwork()
    model.type(torch.FloatTensor)

    if MULTI:
        # for multiprocessing
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=fitFunction, args=(model, train_loader))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        fitFunction(model, train_loader)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp

torch.utils.backcompat.broadcast_warning.enabled = True


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.L1 = nn.Linear(784, 250)  # input output
        self.A1 = nn.LeakyReLU()
        # input must be the same as previous ouput
        self.L2 = nn.Linear(250, 100)
        self.A2 = nn.LeakyReLU()
        self.L3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.L1(x)
        x = self.A1(x)
        x = self.L2(x)
        x = self.A2(x)
        x = self.L3(x)
        return F.log_softmax(x, dim=1)


def fitFunction(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    # model.train()

    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch  # Variable(X_barch).float()
            var_y_batch = y_batch  # Variable(y_barch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            var_y_batch = var_y_batch.squeeze_()
            loss = error(output, var_y_batch)

            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()

            if batch_idx % 50 == 0:
                print(
                    "Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%".format(
                        epoch,
                        batch_idx * len(X_batch),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        float(correct * 100) / float(32 * (batch_idx + 1)),
                    )
                )


"""    input = np.random.random((16))
    input = torch.Tensor(input) # input size
    output = model(input) # otput size (4)

    with torch.no_grad():
        input_next = np.random.random((16))
        input_next = torch.Tensor(input_next) # input size
        output_next = model(input_next) # I don't want the back propagation on this

    loss = F.smooth_l1_loss(output,output_next)

    # optimize step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""


if __name__ == "__main__":
    MULTI = True
    # Import DATA
    import torchvision.datasets as datasets

    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=None
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=None
    )
    X_train = mnist_trainset.train_data.type(torch.FloatTensor).view(60000, 28 * 28)
    y_train = mnist_trainset.train_labels.type(torch.LongTensor).view(60000, 1)
    X_test = mnist_testset.test_data.type(torch.FloatTensor).view(10000, 28 * 28)
    y_test = mnist_testset.test_labels.type(torch.LongTensor).view(10000, 1)
    BATCH_SIZE = 32
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False
    )
    if MULTI:
        # multiprocessing settings
        mp.set_start_method("spawn")
        num_processes = 4

    model = NeuralNetwork()
    model.type(torch.FloatTensor)

    if MULTI:
        # for multiprocessing
        model.share_memory()
        processes = []
        for rank in range(num_processes):
            p = mp.Process(target=fitFunction, args=(model, train_loader))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        fitFunction(model, train_loader)
