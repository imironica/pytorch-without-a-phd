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
        self.linear1 = nn.Linear(784, 50)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
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

noOfEpochs = 20
batchSize = 32

numberOfClasses = 10
featureSize = xTrain.shape[1]

yTrain = yTrain.values
yTest = yTest.values

showPlot = verbose

device = "cpu"
model = Net()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

model.train()


def train(model, optimizer, epoch, xTrain, yTrain):
    model.train()
    criterion = nn.CrossEntropyLoss(size_average=False)
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
    loss, acc = train(model, optimizer, epoch, xTrain, yTrain)
    losses.append(loss)
    accuracies.append(acc)

testLoss, testAccuracy, predictions = test(model, xTest, yTest)

showPlot = True

showPerformance(testLoss, testAccuracy, noOfEpochs, losses, accuracies, plot=showPlot)

showConfusionMatrix(yTest, predictions)