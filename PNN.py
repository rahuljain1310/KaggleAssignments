import csv
import numpy as np
import math
import Operators as op

class ProbabilisticNeuralNetwork():
  def __init__(self):
    self.W = None
    self.patternCategoryMatrix = None
    self.optimizedSigma = 0.5
    self.classes = None
    self.ParzanWindow = lambda x,sigma : math.exp((x-1)/(sigma*sigma))

  def getPatternCategoryMatrix(self,Y):
    TrainingSamples = Y.shape[0]
    patternCategoryMatrix = np.zeros((TrainingSamples, len(self.classes)))
    patternCategoryMatrix[np.arange(TrainingSamples),Y] = 1
    return patternCategoryMatrix

  def Train(self,XTrain,YTrain,classes):
    self.classes = classes
    self.W = op.NormalizePattern(XTrain).T
    self.patternCategoryMatrix = self.getPatternCategoryMatrix(YTrain)

  def OptimizeSigma(self, XValidate, YValidate, rge):
    minLoss = math.inf
    for sigma in rge:
      loss,_ = self.getPNNLoss(XValidate,YValidate)
      if loss<minLoss:
        self.optimizedSigma = sigma
        minLoss = loss

  def getProbabilityMatrix(self,X,sigma):
    X = op.NormalizePattern(X)
    s = np.vectorize(self.ParzanWindow)(X.dot(self.W),sigma)
    s = s.dot(self.patternCategoryMatrix)
    s = op.getProbability(s)
    return s

  def getPNNresult(self,X,sigma):
    s = self.getProbabilityMatrix(X,sigma)
    return op.classify(s)

  def getPNNLoss(self,X,Y,printLoss = False):
    s = self.getProbabilityMatrix(X,self.optimizedSigma)
    result = op.classify(s)
    rm = np.zeros(s.shape)
    rm[np.arange(Y.shape[0]),Y] = 1
    loss = np.linalg.norm(s-rm,axis=1)
    loss = loss.mean()
    Wrong = np.count_nonzero(Y-result)
    if printLoss:
      print("Loss: {0}, WrongPredictions: {1}".format(loss,Wrong))
    else:
      return loss,Wrong


