import torch
import torch.nn as nn
import random

# I ran my project on GoogleColabs while editing it on my local computer.
# This is the reason for the different file locations
if torch.cuda.is_available():
    fileLocation = '/content/drive/MyDrive/tiny-shakespeare.txt'
    DEVICE = torch.device("cuda")
else:
    fileLocation = 'data/tiny-shakespeare.txt'
    DEVICE = torch.device("cpu")

# Reads in file
with open(fileLocation, 'r', encoding='utf-8-sig') as f:
    inputData = f.read()


# Takes all input text and creates an alphabetically sorted string of all UNIQUE characters
def getAllCharacters(inputText):
    chars = sorted(set(inputText))  # Creates a sorted string of all unique characters in text
    newText = ""
    for i in chars:
        newText += i
    return newText


# Global character variables
allCharacters = getAllCharacters(inputData)
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
        output = self.embed(data)  # Embedding creates a look up table with a dictionary & size
        print(output.shape)
        output, (hidden, cell) = self.LSTM(output.unsqueeze(1), (hidden, cell))  # Runs data through LSTM
        print(hidden.shape)
        print(cell.shape)
        output = self.fc(output.reshape(output.shape[0], -1))  # Runs data through the fully connected later (nn.Linear)
        return output, (hidden, cell)

    def initializeHiddenAndCell(self, batchSize):  # Setting Tensors for hidden layers and cell state to 0.
        hidden = torch.zeros(self.layerCount, batchSize, self.hiddenSize).to(DEVICE)
        cell = torch.zeros(self.layerCount, batchSize, self.hiddenSize).to(DEVICE)
        return hidden, cell


class generateShakespeareText:
    def __init__(self):
        self.sequence = 50
        self.batchSize = 2
        self.epochCount = 5000
        self.printStatus = 250  # How often we will display loss & current text generation
        self.hiddenSize = 256
        self.layerCount = 2
        self.learnRate = .003

        self.RNN = RNN(numCharacters, self.hiddenSize, self.layerCount, numCharacters).to(DEVICE)

    def createCharTensor(self, batchString):  # slice of characters (current batch) to a One-Hot Tensor
        tensor = torch.zeros(len(batchString)).long()
        for i in range(len(batchString)):
            tensor[i] = allCharacters.index(batchString[i])
        return tensor

    def getNewBatch(self):
        # Gets a random start and stop index from data to create a batch for training
        start = random.randint(0, len(inputData) - self.sequence)
        end = start + self.sequence + 1
        batchText = inputData[start:end]
        # Initializing input text and target text
        batchInput = torch.zeros(self.batchSize, self.sequence)
        batchTarget = torch.zeros(self.batchSize, self.sequence)

        for i in range(self.batchSize):
            # The target text will always be shifted over 1 since we are attempting to predict the NEXT character
            batchInput[i, :] = self.createCharTensor(batchText[:-1])
            batchTarget[i, :] = self.createCharTensor(batchText[1:])
        return batchInput.long(), batchTarget.long()

    # Confidence is the amount of randomness added in to not always get highest probability output; 1 = normal
    def generateText(self, generationString="Carl", predictionLength=100, confidence=1):
        hidden, cell = self.RNN.initializeHiddenAndCell(batchSize=self.batchSize)
        initialInput = self.createCharTensor(generationString)
        prediction = generationString

        for i in range(len(generationString) - 1):
            _, (hidden, cell) = self.RNN(initialInput[i].view(1).to(DEVICE), hidden, cell)

        lastChar = initialInput[-1]
        for i in range(predictionLength):
            output, (hidden, cell) = self.RNN(lastChar.view(1).to(DEVICE), hidden, cell)
            outputDistance = output.data.view(-1).div(confidence).exp()
            topChar = torch.multinomial(outputDistance, 1)[0]
            predictedChar = allCharacters[topChar]
            prediction += predictedChar
            lastChar = self.createCharTensor(predictedChar)
        return prediction

    def trainRNN(self):
        print("Starting up RNN training...")
        self.RNN = RNN(numCharacters, self.hiddenSize, self.layerCount, numCharacters).to(DEVICE)

        optimizer = torch.optim.Adam(self.RNN.parameters(), lr=self.learnRate)
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
            optimizer.step()
            loss = loss.item() / self.sequence

            if epoch % self.printStatus == 0:
                print("|||| Current Loss = " + str(loss) + " | Epoch = " + str(epoch) + " ||||")
                print(self.generateText())


class HyperParameters:
    def __init__(self):
        self.sequence = 420     # The size of each sequence ran (Text data is sequential since it goes in order)
        self.batchSize = 1      # How much the training data is divided into samples (1 = all training data)
        self.epochCount = 5000  # Simply the amount of iterations of training we do
        self.printStatus = 250  # How often we will display loss & current text generation
        self.hiddenSize = 256   # Amount of hidden features in hidden state
        self.layerCount = 2     # Number of recurrent layers stacked together
        self.learnRate = .003   # How much we will adjust model based on loss calculations
        self.dropRate = .5      # Regularization method to exclude LSTM units that may over-fit


genText = generateShakespeareText()
genText.trainRNN()
