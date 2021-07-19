#!/usr/bin/env python
# coding: utf-8

#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab1 Backpropagation
#Date: 2021/07/16
#Subject: Implement Two Layer Neural Network(Feedforward) with Backpropagation without using ML framework.
#Email: oscarchen.cs10@nycu.edu.tw

import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import random, uniform
import math


#GenerateData: Used to collect two problem (XOR, Linear) data generator
class GenerateData:
    @staticmethod
    def generate_linear(n=100): #Get Linear Point Data & Label
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0] - pt[1]) / 1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)
    
    @staticmethod
    def generate_XOR_easy(): #Get XOR Point Data & Label
        inputs = []
        labels = []

        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue

            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape(21, 1)


# show_result: show the comparing pic between ground truth & predicted data
def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth", fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
            
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if abs(pred_y[i]) >=0.9:
            plt.plot(x[i][0], x[i][1], 'bo')
            print("print Blue", pred_y[i])
            
        else:
            plt.plot(x[i][0], x[i][1], 'ro')
            print("print Red", pred_y[i])
            
    plt.show()


#show_accuracy: show the accuracy of model predicted result
#Accuracy: TP+TN/TP+TN+FP+FN
def show_accuracy(Y_train, Y_predict):
    correct = 0
    for i in range(len(Y_predict)):
        if Y_train[i][0]==0 and (Y_predict[i] < 0.9):
            correct += 1
        elif Y_train[i][0]==1 and (Y_predict[i] >= 0.9):
            correct += 1
    print(f"Accuracy--> {round(correct/len(Y_predict), 3)*100}%")


#SimpleNN: the class of a Neural Network
class SimpleNN:
    def __init__(self):
        self.network = list()
        self.neuronNum = list()
        self.l_rate = 0.01
        self.n_epoch = 1000
        self.haveLearningCurve = True
        self.showLoss = 1
        seed(1) #make the fixed random

    #addLayer: Add layer in network
    #Notice: You should add at least one layer have n_inputs(the dimensions of input data)
    def addLayer(self, n_neuron, **karg):
        if "n_inputs" in karg.keys():
            #This is input layer
            n_inputs = karg['n_inputs']
            inputLayer = [{"weight":[uniform(-1, 1)  for __ in range(n_inputs+1)]}for _ in range(n_neuron)] #weight[-1] is for Bias
            self.network.append(inputLayer)
            self.neuronNum.append(n_inputs)
            self.neuronNum.append(n_neuron)
        else:
            #This is hidden/output layer
            try:
                if len(self.neuronNum)>0:
                    n_previousNeuron = self.neuronNum[-1]
                    middleLayer = [{"weight":[uniform(-1, 1) for __ in range(n_previousNeuron+1)]}for _ in range(n_neuron)]
                    self.network.append(middleLayer)
                    self.neuronNum.append(n_neuron)
                else:
                    raise Exception('Error Layer! (You must check last layer.(Or you miss n_inputs parameter there?))')

            except Exception as error:
                print('Caught this error: ' + repr(error))

    #setting: set some hyperparameters
    #l_rate -> float : Learning rate
    #n_epoch -> int : Max Learning Iteration
    #haveLearningCurve -> bool : Whether showing the learning curve or not
    #showLoss -> int : the frequency of showing error value in training phase
    def setting(self, l_rate, n_epoch, haveLearningCurve=True, showLoss=1):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.haveLearningCurve = haveLearningCurve
        self.showLoss = showLoss
        print("----Training Setting----")
        print(f"Epoch:{self.n_epoch}")
        print(f"Learning Rate:{self.l_rate}")
        print(f"Learning Curve:{self.haveLearningCurve}")
        print(f"Show Loss Frequency:{self.showLoss}")
        print("------------------------")
        
    #describe: Showing the description of NN
    def describe(self):
        print("----Network Description----")
        print(f"total {len(self.network)} layers")
        print(f"{len(self.network)-1} hidden layer")
        print("one output layer")
        for index, layer in enumerate(self.network):
            print(f"-Layer {index+1}-")
            print(f"{len(layer)} neurons")
            for neu_index, neuron in enumerate(layer):
                print(f"neuron{neu_index}:")
                print(f"weight:{neuron['weight'][:-1]}")
                print(f"bias:{neuron['weight'][-1]}")
        print("---------------------------")
    
    #activateProcess: Caculate "net" value
    #weights(List): the weight between inputs and neurons. Notice: weights[-1] is the bias. 
    #inputs(List): X_train or the outputs of previous layer
    def activateProcess(self, weights, inputs):
        bias = weights[-1]
        value = -bias
        for i in range(len(weights)-1):
            value += weights[i]*inputs[i]
        return value
    
    #Activation Function
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    #Forward phase
    def foward(self, rowData):
        inputs = rowData #get input data(by one data) Ex. x[0]~x[n]
        for layer in self.network:
            new_inputs = list()
            for neuron in layer:
                output = self.activateProcess(neuron["weight"], inputs)
                output_sigmoid = self.sigmoid(output)
                neuron["output"] = output_sigmoid
                new_inputs.append(output_sigmoid)
            inputs = new_inputs
        return inputs
    
    #the derivation of sigmoid
    def derivate_sigmoid(self, x):
        return x * (1.0 - x)
    
    #Backward phase
    #groundTruth: y(List), y[0]~y[n]
    def backwardPropagation(self, groundTruth):
        for i in reversed(range(len(self.network))):
            if i == (len(self.network)-1):
                #It is a  output layer!
                for index, neuron in enumerate(self.network[i]):
                    diff = groundTruth[index]-neuron['output']
                    neuron['delta'] = diff  #The Derivative of MSE -> -(T-Y)
            else:
                #It is not a ouput layer!
                for index, neuron in enumerate(self.network[i]):
                    error = 0
                    for nextNeuron in self.network[i+1]:
                        error += nextNeuron['weight'][index] * nextNeuron['delta']
                    neuron['delta'] = error

            for neuron in self.network[i]:
                neuron['delta'] = neuron['delta'] * self.derivate_sigmoid(neuron['output'])
    
    #updateWeight: Update all weights(& bias)
    #initInputs: Original input(X_train)
    def updateWeight(self, initInputs):
        inputs = initInputs
        tempInput = list()
        for nowIndex, layer in enumerate(self.network):
            if nowIndex != 0:
                # Not First Hidden layer
                inputs = tempInput.copy()
                tempInput.clear()
            for neuron in layer:
                for index, inputItem in enumerate(inputs):
                    neuron['weight'][index] +=  self.l_rate * neuron['delta'] * inputItem #For weight
                neuron['weight'][-1] += -(self.l_rate) * neuron['delta'] # For Bias
                tempInput.append(neuron['output'])

    #learningCurve: Plot a learning curve (Error Value to Iteration(Epoch))        
    def learningCurve(self):
        plt.title("Learning Curve", fontsize = 18)
        plt.plot([x for x in range(1,self.n_epoch+1)], self.errorList,color='blue', label="learning curve")
        plt.legend(loc = 'upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    #trainNetwork: Start to train model
    def trainNetwork(self, X_train, Y_train):
        print("----Training Setting----")
        print(f"Epoch:{self.n_epoch}")
        print(f"Learning Rate:{self.l_rate}")
        print(f"Learning Curve:{self.haveLearningCurve}")
        print(f"Show Loss Frequency:{self.showLoss}")
        print("------------------------")
        self.describe()
        
        errorList = list()
        for epoch in range(self.n_epoch):
            sum_error = 0
            for index, row in enumerate(X_train):
                #row: [0.1, 0.2]
                Y_predict = self.foward(row) #n_outputs dimension #forward phase
                sum_error += sum([(Y_train[index][i]-Y_predict[i])**2 for i in range(self.neuronNum[-1])]) 
                self.backwardPropagation(Y_train[index]) #backward phase
                self.updateWeight(row)
            error = sum_error/len(X_train)
            errorList.append(round(error, 3))
            if not (epoch%self.showLoss):
                print('==epoch=%d, lrate=%.3f, error=%.3f==' % (epoch, self.l_rate, error)) #print error value (MSE)
        self.errorList = errorList
        if self.haveLearningCurve:
            self.learningCurve()
    
    #predict: Predict the data (Using forward phase)
    def predict(self, X_test):
        result = list()
        for index, row in enumerate(X_test):
            print(f"Predict:{index}, {row}")
            Y_predict = self.foward(row)
            result.append(Y_predict[0])
        return result


#Main Function

choose = input("Please enter the problem number you want to execute(0: XOR / 1:Linear(n=900)):")   

if choose == "0":
    #XOR Problem:
    X_xor_train, Y_xor_train = GenerateData.generate_XOR_easy() #Generate Training Data (XOR)
    xorNN = SimpleNN() #Implement a NN object

    #Create first hidden layer <with 2 neuron>(You need to input n_inputs parameter, it means the dimension of input data)
    xorNN.addLayer(n_neuron=2, n_inputs=2) 
    #Create second hidden layer <with 2 neuron>
    xorNN.addLayer(n_neuron=2) 
    #Create one output layer <with 1 neuron>
    xorNN.addLayer(n_neuron=1)

    xorNN.describe() #Show the NN description

    #Setting some hyperparameter (l_rate: learning rate, n_epoch:Max Iteration, 
    #                             haveLearningCurve: need to show learning curve?,
    #                             showLoss: the frequency of showing error value in training phase) 
    xorNN.setting(l_rate=4, n_epoch=2000, showLoss=10) 

    xorNN.trainNetwork(X_xor_train, Y_xor_train) #Start to train model

    Y_xor_predict = xorNN.predict(X_xor_train) #Start to predict the data

    show_result(X_xor_train, Y_xor_train, Y_xor_predict) #show the comparing pic
    show_accuracy(Y_xor_train, Y_xor_predict) #show the accuracy

elif choose == "1":
    #linear Problem
    X_linear_train, Y_linear_train = GenerateData.generate_linear(100) #Generate Training Data (Linear)
    linearNN = SimpleNN() #Implement a NN object

    #Create first hidden layer <with 2 neuron>(You need to input n_inputs parameter, it means the dimension of input data)
    linearNN.addLayer(n_neuron=2, n_inputs=2)
    #Create second hidden layer <with 2 neuron>
    linearNN.addLayer(n_neuron=2)
    #Create one output layer <with 1 neuron>
    linearNN.addLayer(n_neuron=1)

    linearNN.describe() #Show the NN description

    #Setting some hyperparameter (l_rate: learning rate, n_epoch:Max Iteration, 
    #                             haveLearningCurve: need to show learning curve?,
    #                             showLoss: the frequency of showing error value in training phase) 
    linearNN.setting(l_rate=0.2, n_epoch=3000, haveLearningCurve=True, showLoss=5)

    linearNN.trainNetwork(X_linear_train, Y_linear_train) #Start to train model

    Y_linear_predict = linearNN.predict(X_linear_train) #Start to predict the data

    show_result(X_linear_train, Y_linear_train, Y_linear_predict) #show the comparing pic
    show_accuracy(Y_linear_train, Y_linear_predict) #show the accuracy
else:
    print("You enter the wrong number!")