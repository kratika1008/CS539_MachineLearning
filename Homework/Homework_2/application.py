import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression

# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE

def get_alpha(Xtest,Ytest,Xtrain,Ytrain):
    n_epochs = 500
    alphaList = []
    testLoss=[]
    min_testLoss=0
    for i in range(n_epochs):
        current_alpha = i/n_epochs
        alphaList.append(current_alpha)
        w = train(Xtrain,Ytrain,current_alpha,n_epochs)
        testYhat = compute_yhat(Xtest,w)
        current_testloss = compute_L(testYhat,Ytest)
        testLoss.append(current_testloss)
    min_testLoss = min(testLoss)
    index = testLoss.index(min_testLoss)
    optimum_alpha = alphaList[index]
    return optimum_alpha

def get_epochs(Xtest,Ytest,Xtrain,Ytrain,alpha):
    epochsList = []
    testLoss=[]
    trainLoss=[]
    Yhat = []
    min_testLoss=0.
    min_trainLoss=0.
    for i in range(500):
        current_epoch = i
        epochsList.append(current_epoch)
        w = train(Xtrain,Ytrain,alpha,current_epoch)
        trainYhat = compute_yhat(Xtrain,w)
        current_trainloss = compute_L(trainYhat,Ytrain)
        trainLoss.append(current_trainloss)
        testYhat = compute_yhat(Xtest,w)
        Yhat.append(testYhat)
        current_testloss = compute_L(testYhat,Ytest)
        testLoss.append(current_testloss)
    min_trainLoss = min(trainLoss)
    min_testLoss = min(testLoss)
    index = testLoss.index(min_testLoss)
    predicted_Yhat = Yhat[index]
    optimum_epoch = epochsList[index]

    return optimum_epoch,min_trainLoss,min_testLoss,predicted_Yhat

def get_alpha_epochs_relation(Xtrain,Ytrain,Xtest,Ytest):
    epochsList = []
    alphaList = []
    trainLoss=[]
    testLoss=[]
    for i in range(1,10):
        alpha = 1/i
        for j in range(1,10):
            epoch = j*10
            epochsList.append(epoch)
            alphaList.append(alpha)
            w = train(Xtrain,Ytrain,alpha,epoch)
            trainYhat = compute_yhat(Xtrain,w)
            current_trainloss = compute_L(trainYhat,Ytrain)
            trainLoss.append(current_trainloss)
            testYhat = compute_yhat(Xtest,w)
            current_testloss = compute_L(testYhat,Ytest)
            testLoss.append(current_testloss)
    min_trainLoss = min(trainLoss)
    min_testLoss = min(testLoss)
    index = testLoss.index(min_testLoss)
    return alphaList,epochsList,trainLoss,testLoss



optimum_alpha = get_alpha(Xtest,Ytest,Xtrain,Ytrain)
optimum__epochs,min_trainLoss,min_testLoss,predicted_Yhat = get_epochs(Xtest,Ytest,Xtrain,Ytrain,optimum_alpha)
alphaList,epochsList,trainLoss,testLoss = get_alpha_epochs_relation(Xtrain,Ytrain,Xtest,Ytest)

Yhat_test_predicted = [i for i in predicted_Yhat]

print('Predicted value for Y on test dataset: ',Yhat_test_predicted)
print('Optimum alpha for minimum Test Loss: ',optimum_alpha)
print('Optimum number of epochs for minimum Test Loss: ',optimum__epochs)
print('Minimum Test Loss: ',min_testLoss[0])
print('Minimum Train Loss: ',min_trainLoss[0])


#########################################
