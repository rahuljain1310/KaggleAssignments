import csv
import numpy as np
# from ..PNN import PNN
# import Operators as op

### =======================================================================================================
### Import Data from CSV
### =======================================================================================================
data = []
Classes = ['male','female']

with open('GenderVoiceRecognition/voice.csv', 'r') as csvfile: 
  csvreader = csv.reader(csvfile) 
  next(csvreader, None)
  for row in csvreader:
    rowX = []
    for x in row[0:-1]:
      rowX.append(float(x))
    rowX.append(Classes.index(row[-1]))
    data.append(rowX)
data = np.array(data)

np.random.shuffle(data)
X = op.stdAugData(data[:,0:-1])
Y = op.intData(data[:,-1])

Samples = X.shape[0]
TrainingSamples = int(0.8*Samples)
ValidationSamples = int(0.9*Samples)

XTrain,XValidate,XTest = np.split(X, [TrainingSamples,ValidationSamples])
YTrain,YValidate,YTest = np.split(Y, [TrainingSamples,ValidationSamples]) 

assert XTrain.shape[0]+XValidate.shape[0]+XTest.shape[0] == Samples

### =======================================================================================================
### Parzan Window Method
### =======================================================================================================

parzanNetwork = ProbabilisticNeuralNetwork()
parzanNetwork.Train(XTrain,YTrain,Classes)
parzanNetwork.OptimizeSigma(XValidate,YValidate,np.arange(1e-2,2e-1,1e-2))

print("Training Set")
parzanNetwork.getPNNLoss(XTrain,YTrain,True)
print("Cross Validation Set")
parzanNetwork.getPNNLoss(XValidate,YValidate,True)
print("Test Set")
parzanNetwork.getPNNLoss(XTest,YTest,True)