
# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (relu)         W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (relu)         W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (relu)         W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (relu)         W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]






from util import readDatabase, showPerformance, showConfusionMatrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
from tqdm import trange
import numpy as np
import argparse


class Net(nn.Module):
    def __init__(self):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(Net, self).__init__()
        # Linear function.
        self.linear1 = nn.Linear(784, 200, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=200)
        self.linear2 = nn.Linear(200, 100, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=100)
        self.linear3 = nn.Linear(100, 60, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=60)
        self.linear4 = nn.Linear(60, 30, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=30)
        self.linear5 = nn.Linear(30, 10, bias=True)


        torch.nn.init.xavier_uniform(self.linear1.weight)
        torch.nn.init.xavier_uniform(self.linear2.weight)

        torch.nn.init.xavier_uniform(self.linear3.weight)

        torch.nn.init.xavier_uniform(self.linear4.weight)
        torch.nn.init.xavier_uniform(self.linear5.weight)


        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)

        x = self.linear4(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)

        x = self.linear5(x)

        return self.softmax(x)


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", required=False, help="show images (0 = False, 1 = True)")
args = vars(ap.parse_args())
verbose = args["verbose"]

if verbose is None:
    verbose = False
else:
    if verbose == '1':
        verbose = True
    else:
        verbose = False

print("Verbose".format(verbose))

# Read the training / testing dataset and labels
xTrain, yTrain, xTest, yTest = readDatabase()

# Network parameters
learningRate = 0.003

noOfEpochs = 10
batchSize = 100

numberOfClasses = 10

learningRateDecay =  learningRate / (noOfEpochs+1.)

featureSize = xTrain.shape[1]

yTrain = yTrain.values
yTest = yTest.values

showPlot = verbose

device = "cpu"
model = Net()
optimizer = optim.Adam(model.parameters(), lr=learningRate)
criterion = nn.CrossEntropyLoss(size_average=False)

model.train()


def train(model, optimizer, epoch, criterion, xTrain, yTrain):
    model.train()
    correct = 0.
    runningLoss = 0.0

    noOfSteps = math.ceil(xTrain.shape[0] / float(batchSize))



    t = trange(noOfSteps, desc='Bar desc', leave=True)

    for batchId in t:
        firstIndex = batchId * batchSize
        secondIndex = min((batchId + 1) * batchSize, xTrain.shape[0])
        optimizer.zero_grad()

        data = Variable(torch.FloatTensor(xTrain[firstIndex: secondIndex, :]))
        target = Variable(torch.LongTensor(yTrain[firstIndex: secondIndex]))

        output = model(data)
        loss = criterion(output, target)

        runningLoss += loss.item()

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        t.set_description("Epoch {} Loss {} Accuracy: {} ".format(epoch, round(runningLoss / secondIndex, 4),
                                                                  round(correct / secondIndex, 4)))
        t.refresh()  # to show immediately the update



    return runningLoss / xTrain.shape[0], correct / xTrain.shape[0]


def test(model, xTest, yTest):

    criterion = nn.CrossEntropyLoss(size_average=False)
    data = Variable(torch.FloatTensor(xTest))
    target = Variable(torch.LongTensor(yTest))

    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    predTemp = np.rot90(pred.numpy()).tolist()

    predictions = predTemp[0]
    correct = pred.eq(target.view_as(pred)).sum().item()

    loss = criterion(output, target)
    runningLoss = loss.item()

    return runningLoss / float(xTest.shape[0]), correct / float(xTest.shape[0]), predictions


losses = []
accuracies = []
for epoch in range(0, noOfEpochs):
    loss, acc = train(model, optimizer, epoch, criterion, xTrain, yTrain)
    losses.append(loss)
    accuracies.append(acc)
    learningRate = learningRate - learningRateDecay
    for param_group in optimizer.param_groups:
        param_group['lr'] = learningRate

testLoss, testAccuracy, predictions = test(model, xTest, yTest)

showPlot = True

showPerformance(testLoss, testAccuracy, noOfEpochs, losses, accuracies, plot=showPlot)

showConfusionMatrix(yTest, predictions)


