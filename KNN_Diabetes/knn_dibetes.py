from numpy import *
import operator
import csv
import pdb
import time
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in range(m):
        for j in range(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def loadData(split):
    l=[]
    with open('pima-indians-diabetes-database/diabetes.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=array(l)
    train, test = train_test_split(l, test_size = split)
    train_label=toInt(train[:,-1])
    train_data=preprocessing.scale(train[:,:-1])
    print(train[:,:-1])
    test_label = toInt(test[:,-1])
    test_data = preprocessing.scale(test[:,:-1])
    return train_data, train_label, test_data, test_label
    
#inX:1*n  dataSet:m*n   labels:m*1  
def classify(inX, dataSet, labels, k): 
	inX=mat(inX)
	dataSet=mat(dataSet)
	labels=mat(labels)
	dataSetSize = dataSet.shape[0]                  
	diffMat = tile(inX, (dataSetSize,1)) - dataSet   
	sqDiffMat = array(diffMat)**2
	sqDistances = sqDiffMat.sum(axis=1)                  
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()            
	classCount={}                                      
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i],0]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv', 'w', newline = '') as myFile:    
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)
        

def Test():
	start_time = time.time()
	trainData, trainLabel, testData, testLabel = loadData(0.35)
	end_time = time.time()
	print("loading data costs %s second" %(end_time - start_time))
	m,n=shape(testData)
	errorCount=0
	resultList=[]
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	start_time = time.time()
	for i in range(m):
		classifierResult = classify(testData[i], trainData, trainLabel.transpose()[0:s], 9)
		resultList.append(classifierResult)
		print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0,i]))
		if (classifierResult != testLabel[0,i]): 
			errorCount += 1.0
		if classifierResult == testLabel[0,i] == 1.0:
			TP += 1
		if classifierResult ==1.0 and testLabel[0,i] != classifierResult:
			FP += 1
		if testLabel[0,i] == classifierResult == 0.0:
		 	TN += 1
		if classifierResult == 0.0 and testLabel[0,i] != classifierResult:
			FN += 1
		print ("\nthe total number is %d, the errors is: %d" % (i+1,errorCount))
		print ("\nthe accuracy rate is: %f" % (1 - errorCount/float(m)))
	end_time = time.time()
	print("KNN excuted  %s second" %(end_time - start_time))
	print("TP:%d, FP:%d, TN:%d, FN:%d" %(TP, FP, TN, FN))
	saveResult(resultList)
def main():
	Test()
main()
