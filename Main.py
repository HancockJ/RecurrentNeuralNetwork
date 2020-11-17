import string
import torch
import torch.nn as nn
import random

DEVICE = torch.device("cpu")

with open('data/tiny-shakespeare.txt', 'r', encoding='utf-8-sig') as f:
    inputData = f.read()

# Get all characters from string.printable
allCharacters = string.printable

print(allCharacters)
numCharacters = len(allCharacters)


class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, layerCount, outputSize):
        super(RNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount

        self.embed = nn.Embedding(inputSize, hiddenSize)
        self.LSTM = nn.LSTM(hiddenSize, hiddenSize, layerCount, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, data, hidden, cell):
        output = self.embed(data)
        output, (hidden, cell) = self.LSTM(output.unsqueeze(1), (hidden, cell))
        output = self.fc(output.reshape(output.shape[0], -1))
        return output, (hidden, cell)

    def initializeHiddenAndCell(self, batchSize):
        hidden = torch.zeros(self.layerCount, batchSize, self.hiddenSize).to(DEVICE)
        cell = torch.zeros(self.layerCount, batchSize, self.hiddenSize).to(DEVICE)
        return hidden, cell


class generateShakespeareText():
    def __init__(self):
        self.sequence = 250
        self.batchSize = 1
        self.epochCount = 5000
        self.printStatus = 50
        self.hiddenSize = 256
        self.layerCount = 2
        self.learnRate = .001

        self.RNN = RNN(numCharacters, self.hiddenSize, self.layerCount, numCharacters).to(DEVICE)

    def createCharTensor(self, batchString):
        tensor = torch.zeros(len(batchString)).long()
        for i in range(len(batchString)):
            tensor[i] = allCharacters.index(batchString[i])
        return tensor

    def getNewBatch(self):
        pass
        start = random.randint(0, len(inputData) - self.sequence)
        end = start + self.sequence + 1
        batchText = inputData[start:end]
        batchInput = torch.zeros(self.batchSize, self.sequence)
        batchTarget = torch.zeros(self.batchSize, self.sequence)

        for i in range(self.batchSize):
            batchInput[i, :] = self.createCharTensor(batchText[:-1])
            batchTarget[i, :] = self.createCharTensor(batchText[1:])
        return batchInput.long(), batchTarget.long()

    # Confidence is the amount of randomness added in to not always get highest probability output
    def generateText(self, generationString="Carl", predictionLength=1000, confidence=0.85):
        hidden, cell = self.RNN.initializeHiddenAndCell(self.batchSize)
        initialInput = self.createCharTensor(generationString)
        prediction = generationString

        for i in range(len(generationString) - 1):
            _, (hidden, cell) = self.RNN(initialInput[i].view(1).to(DEVICE), hidden, cell)

        lastChar = initialInput[-1]
        for i in range(predictionLength):
            output, (hidden, cell) = self.RNN(lastChar.view(1).to(DEVICE), hidden, cell)
            outputDistance = output.data.view(-1).div(confidence).exp()
            topChar = torch.multinomial(outputDistance, 1)[0]  #
            predictedChar = allCharacters[topChar]
            prediction += predictedChar
            lastChar = self.createCharTensor(predictedChar)
        return prediction

    def trainRNN(self):
        print("Starting up RNN training...")

        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.epochCount + 1):
            inp, target = self.getNewBatch()
            hidden, cell = self.RNN.initializeHiddenAndCell(self.batchSize)

            self.RNN.zero_grad()
            loss = 0
            inp = inp.to(DEVICE)
            target = target.to(DEVICE)

            for i in range(self.sequence):
                # Gets the predicted output
                output, (hidden, cell) = self.RNN(inp[:, i], hidden, cell)
                # Calculates cross entropy loss by comparing predicted value to target value
                loss += criterion(output, target[:, i])

            # Calculating final loss average
            loss.backward()
            loss = loss.item() / self.sequence

            if epoch % self.printStatus == 0:
                print("Current Loss = " + str(loss))
                print(self.generateText())


# genText = generateShakespeareText()
# genText.trainRNN()
