import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pylab as pl
import os
import time
import math

from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split

#convert arff to cvs
def arff_to_csv(filename):
	if filename.find('.arff') < 0:
		print("the file is not .arff file")
		return
	data = []
	arff_file = open(filename, 'r')
	for line in arff_file:
		if not (line.startswith("%")):
			if not (line.startswith("@")):
				if not (line.startswith("\n")):
					line = line.strip("\n")
					cs = line.split(',')
					data.append(cs)
	df = pd.DataFrame(data=data, index=None, columns=None)
	csv_file = filename[:filename.find('.arff')] + '.csv'
	df.to_csv(csv_file, index = None)

    
#load data from cvs and convert it to list, convert the string to number without preprocessing
def loadData(cvsfile):
    l=[]
    with open(cvsfile) as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
    l.remove(l[0])
    l=np.array(l)
    l_float = l.astype(np.float)
    return l_float

#split dataset into training set and test set
def splitData(X, split):
    train, test = train_test_split(X, test_size = split)
    return train, test

#using MinMaxScaler to process the dataset 
def preprocessing_with_MinMaxScaler(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_data = min_max_scaler.fit_transform(X[:,:-1])
    X_label = X[:,-1].reshape((len(X),1))
    new_X = np.hstack((X_data, X_label))
    return new_X
    
def preprocessing_with_scale(X):
    X_data =  preprocessing.scale(X[:,:-1])
    X_label = X[:,-1].reshape((len(X),1))
    new_X = np.hstack((X_data, X_label))
    return new_X
    
#using standard scaler to scale the dataset     
def preprocessing_with_StandardScaler(X):
    sc =  preprocessing.StandardScaler()
    sc.fit(X[:,:-1])
    X_data = sc.transform(X[:,:-1])
    X_label = X[:,-1].reshape((len(X),1))
    new_X = np.hstack((X_data, X_label))
    return new_X
    
    
# PCA extract principal features
from sklearn.decomposition import PCA
def pca(X, n):
    pca = PCA(n_components=n)
    pca.fit(X[:,:-1])
    features = pca.transform(X[:,:-1])
    print(f'Explained Variance: {pca.explained_variance_ratio_}')
    X_label = X[:,-1].reshape((len(X),1))
    return np.hstack((features,X_label)) 

#remove the last feature column    
def remove_feature_column(X, column):
    return np.delete(X, column, 1)
  
#Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
			#print(f'i={i},actual={actual[i]}, predicted={predicted[i]}\n')
	return correct / float(len(actual))

#plot true label and its prediction confidence
def plot_confidence(actual, predictions, probabilities):    
    map_probabilities = list()
    for i in range(len(actual)):
        if actual[i] == 1.0:
            map_probabilities.append(probabilities[i])
        else:
            map_probabilities.append(1 - probabilities[i])
    pl.scatter(actual, map_probabilities)
    pl.xlabel("True classification")
    pl.ylabel("prediction confidence")
    pl.show()
    
#generate new features by column1/column2
def generate_new_feature(X, column1, column2):
    new_array_X = np.insert(X, 0, X[:,column1]/X[:,column2], axis=1)
    return new_array_X
    
#generate new feature by multiply all its column 
def generate_cross_feature(X):
    cross_feature = np.ones(X.shape[0])
    for i in range(X.shape[1] - 1):
        cross_feature = cross_feature * X[:,i]
    new_array_X = np.insert(X, 0, cross_feature, axis = 1)
    return new_array_X
    
#relabel the dataset by comparing the label value to the threshold
def relabel(X, threshold=5):
    for i in range(0, X.shape[0]):
        if X[i, -1] >= threshold:
            X[i, -1] = 0
        else:
            X[i, -1] = 1
            
#relabel the dataset to 0 and 1
def relabel_01(X):
    for i in range(0, X.shape[0]):
        if X[i, -1] == 2:
            X[i, -1] = 1
        else:
            X[i, -1] = 0
            
#Make a prediction with weights
def predict(row, weights):
	score = weights[0]
	for i in range(len(row)-1):
		score += weights[i + 1] * row[i]
	return step(score), score
	#return step(sum)

#step function
def step(z):
    if z >= 0.0:
        return 1.0 
    else:
        return 0.0
    
#sigmoid function
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))
 
#calculate confidence interval
def confidence_interval(accuracy, n):
    #1.64 (90%)
    #1.96 (95%)
    #2.33 (98%)
    #2.58 (99%)
    const = 1.96 
    range = const * math.sqrt((accuracy * (1 - accuracy))/n)
    return accuracy - range, accuracy + range
    
# Make a prediction with weights
def predict2(row, weights):
    z = weights.T.dot(row)
    return step(z)
    
    
#update weights using stochastic gradient descent, train is list
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	errors = np.zeros(n_epoch)
	for epoch in range(n_epoch):
		for row in train:
			prediction, score = predict(row, weights)
			error = row[-1] - prediction
			if error != 0:
				errors[epoch] += 1;
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		#print(f'errors[{epoch}]={errors[epoch]}, len={len(train)}\n')
	return weights, errors/len(train)

#update weights using stochastic gradient descent, train is numpy.array
def train_weights2(train, l_rate, n_epoch):
	#add one column 1 for train
	intercept = np.ones((train.shape[0], 1))
	train = np.hstack((intercept, train))
	#The last column is label, so create (total column -1) dimension weights 
	weights = np.zeros(train.shape[1] - 1)
	errors = np.zeros(n_epoch)
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict2(row[:-1], weights)
			error = row[-1] - prediction
			if error != 0:
				errors[epoch] += 1;
			errors[epoch] += error
			weights = weights + l_rate * error * row[:-1]
			
	return weights, errors/len(train)
	
# Perception Algorithm With Stochastic Gradient Descent, train and test are list 
def perceptron(train, test, l_rate, n_epoch):
	probabilities = list()
	predictions = list()
	weights, loss = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction, score = predict(row, weights)
		probabilities.append(sigmoid(score))
		predictions.append(prediction)
	return(predictions, loss, probabilities)

#Perceptron Algorithm With Stochastic Gradient Descent, train and test are numpy.array
def perceptron2(train, test, l_rate, n_epoch):
	predictions = list()
	weights, loss = train_weights2(train, l_rate, n_epoch)
	print("finish train_weights2\n")
	intercept = np.ones((test.shape[0], 1))
	test = np.hstack((intercept, test))
	for row in test:
		prediction = predict2(row[:-1], weights)
		predictions.append(prediction)

	return(predictions, loss)
        
arff_to_csv("dataset_8_liver-disorders.arff")
dataset = loadData("dataset_8_liver-disorders.csv")
#remove last column
#dataset = remove_feature_column(dataset, dataset.shape[1] - 1)
#generate a new feature using dataset[3]/dataset[2], and insert the new feature in dataset[0]
#new_dataset = generate_new_feature(dataset, 3, 2)


#dataset = pca(dataset, 4)
#relabel the dataset
#relabel(new_dataset)
#relabel(dataset)
new_dataset = dataset
#new_dataset = generate_cross_feature(dataset)
#new_dataset = generate_new_feature(new_dataset, 4, 3)
relabel_01(new_dataset)
#normalize data
new_dataset = preprocessing_with_StandardScaler(new_dataset)


n_epoch = 1000
lr = 10
accuracys = []


'''
#plot the true label and its probability 
train, test = splitData(new_dataset, 0.1) 
predictions, loss, probabilities = perceptron(train.tolist(), test.tolist(), lr, n_epoch)
actual = [i[len(test[0]) -1] for i in test]
#plot scattor with the classification and probabilities
plot_confidence(actual, predictions, probabilities)

#plot the loss change with epochs
epochs = np.linspace(1, n_epoch, n_epoch, endpoint=True)
pl.plot(epochs, loss)
pl.xlabel('epoch')
pl.ylabel('loss')
pl.show()
os._exit(0)
'''

'''
#plot two features with label
colors = list()
label_index = new_dataset.shape[1] - 1
for i in range(len(new_dataset)):
    if new_dataset[i,label_index] == 1:
        colors.append('blue')
    else:
        colors.append('red')

print(new_dataset[:,3])
pl.scatter(new_dataset[:,3], new_dataset[:,4], c = colors)
pl.xlabel("sgot")
pl.ylabel("gammagt")
pl.title("Sgot vs Gammagt")
pl.show()

os._exit(0)
'''


#calculate the confidence_interval
for i in range(50):   
    train, test = splitData(new_dataset, 0.1)    
    start_time = time.time()
    predictions, loss, probabilitys = perceptron(train.tolist(), test.tolist(), lr, n_epoch)
    end_time = time.time()
    #print("prediction executed  %s second" %(end_time - start_time))   
    actual = [i[len(test[0]) -1] for i in test]
    #actual = [i[6] for i in test]  
    accuracy = accuracy_metric(actual, predictions)
    accuracys.append(accuracy)
print(np.mean(accuracys))
#calculate the confidence interval
lower, upper = confidence_interval(np.mean(accuracys), len(test))

print(f'There is 95% likehood that confidence interval [{lower},{upper}] covers the true classifiction accuracy of the unseen data')
os._exit(0)

'''
#try different threshold of column-6 as the target label 
accuracy_with_threshold=[]
for threshold in range(15):
    new_dataset = generate_cross_feature(dataset)
    relabel(new_dataset, threshold)
    new_dataset = preprocessing_with_StandardScaler(new_dataset)

    for i in range(50):   
        train, test = splitData(new_dataset, 0.1)    
        start_time = time.time()
        predictions, loss = perceptron(train.tolist(), test.tolist(), lr, n_epoch)
        end_time = time.time()
        #print("prediction executed  %s second" %(end_time - start_time))   
        actual = [i[len(test[0]) -1] for i in test]
        #actual = [i[6] for i in test]  
        accuracy = accuracy_metric(actual, predictions)
        accuracys.append(accuracy)
    print(np.mean(accuracys))
    accuracy_with_threshold.append(np.mean(accuracys))    
#plot the accuracy with the threshold
threshold = np.arange(15)
pl.plot(threshold, accuracy_with_threshold)
pl.xlabel('threshold')
pl.ylabel('accuracy')
pl.show()
os._exit(0)
'''

'''
#using perceptron in sklearn
train, test = loadData4_2("dataset_8_liver-disorders.csv", 0.1)
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=5000, eta0=0.001, shuffle=False)
#train_array = np.array(train)
#test_array = np.array(test)
X_train = train[:,:-1]
Y_train = train[:,-1]

X_test = test[:,:-1]
Y_test = test[:,-1]
print(X_train)
ppn.fit(X_train, Y_train)
acc = ppn.score(X_test,Y_test)
print(acc)
'''









