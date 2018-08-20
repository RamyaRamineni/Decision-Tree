#Creation of decision Tree and Accuracy of Tree after pruning

from math import log
import pandas as pd
import numpy as np
import operator
import random
from random import randrange
import sys

######## GLOBAL VARIABLES #####################

pruningCount = 0 #Count of pruning so that it prunes only one instance of the variable
count = 0    #Count of labels
countNode = 0 #Count of number of nodes
leafNode = 0  #Count of leaf nodes

#Function Name : setGlobalVar

def setGlobalVar():
    global pruningCount
    global count 
    pruningCount = 1
    count = 1

#Function Name: createDataSet
#Input:
#	trainingSet : Path of the training data set
#Output:
#	dataFrame   : dataFrame created by pandas
#	attributes  : Column Names

def createDataSet(dataSet):
    separator = '.'
    dataSetSheetName = dataSet.split(separator, 1)[0]    
    dataFrame = pd.read_excel(dataSet, sheet_name = dataSetSheetName)
    dataFrame.dropna(how="all", inplace=True);           #Remove Nan from dataframe
    attributes = dataFrame.columns.values.tolist()
    return dataFrame, attributes

#Function Name: getEntropy
#Input:
#	dataSet : to which entropy is calculated
#Output:
#	H   : Entropy


def getEntropy(dataSet):
    instancesNum = len(dataSet)
    countLabel = {}
    for i in dataSet: 
        label = i[-1]
        if label not in countLabel.keys(): countLabel[label] = 0
        countLabel[label] += 1
    H = 0.0
    for key in countLabel:
        p = float(countLabel[key])/instancesNum 
        H = H - (p * log(p,2)) 
    return H

#Function Name: splittingData
#Input:
#	dataSet : list of data
#       num     : number
#       value   : value
#Output:
#	dataSubSet   : subset of dataFrame 

def splittingData(dataSet, num, value):
    dataSubSet = []
    for record in dataSet:
        if record[num] == value:
            recordSubSet = record[:num]     
            recordSubSet.extend(record[num+1:])
            dataSubSet.append(recordSubSet)       
    return dataSubSet

#Function Name: splittingLabel
#Input:
#	dataSet : list of data
#Output:
#	labelNum   : number given to one of the label with high IG 
    
def splittingLabel(dataSet):
    columns = len(dataSet[0]) - 1      
    H = getEntropy(dataSet)
    currentIG = 0.0
    labelNum = -1
    for i in range(columns):        
        netH = 0.0
        for j in [0,1]:
            dataSubSet = splittingData(dataSet, i, j)
            p = len(dataSubSet)/float(len(dataSet))
            netH = netH + (p * getEntropy(dataSubSet))    
        IG = H - netH     
        if (IG >= currentIG):       
            currentIG = IG         
            labelNum = i
    return labelNum                      

#Function Name: numFeatures
#Input:
#	classList : list of class values
#Output:
#	sortedFeature   : feature after sorting (0 or 1) 
    
def numFeatures(classList):
    attList = {}
    for val in classList:
        if val not in attList.keys(): attList[val] = 0
        attList[val] = attList[val] + 1
    sortedFeature = sorted(attList.items())
    return sortedFeature[0][0]

#Function Name: createDecisionTree
#Input:
#	dataSet : list of data
#       labelsList: list of labels
#Output:
#	decTree   : constructed decision tree

def createDecisionTree(dataSet,labelsList):
    lastColumn = [column[-1] for column in dataSet]
    if len(lastColumn)>0:
        if lastColumn.count(lastColumn[0]) == len(lastColumn): 
            return lastColumn[0]
    if len(dataSet)>0:   
        if len(dataSet[0]) == 1:
            print("##############################################################################################")
            return numFeatures(lastColumn)
        selectLabel = splittingLabel(dataSet)
        selectedLabel = labelsList[selectLabel]
        decTree = {selectedLabel:{}}
        del(labelsList[selectLabel])
        for value in [0,1]:
            subLabels = labelsList[:]       
            decTree[selectedLabel][value] = createDecisionTree(splittingData(dataSet, selectLabel, value), subLabels) # Recursively call fucntion
    else:
        decTree = {}
    return decTree

#Function Name: printDecisionTree
#Input:
#	tree  : decision tree
#       height: height of the tree
#       node  : key
#Output: prints tree

def printDecisionTree(tree, height, node):    
    if not(tree == {}):
        for key, value in tree.items():        
            if key in [0,1]:            
                for i in range(height - 1):                
                    print("| ", end="")            
                print(node, "= ", end="")
                print(key, ": ", end="")        
            else:
                if height > 0:
                    print(" ")
                height = height + 1
            if isinstance(value, dict):           
                printDecisionTree(value, height, key)
            else:            
                print(value)
    else:
          print(" ")

#Function Name: countTreeNodes
#Input:
#	tree  : decision tree
#       height: height of the tree
#       node  : key
#Output:
#       total number of nodes
#       leafNode: number of leaf nodes

def countTreeNodes(tree, height, node):
    global countNode
    global leafNode    
    for key, value in tree.items():
        flag = False
        if key in [0,1]:            
            for i in range(height - 1):
                flag = True              
            if flag == True:
                countNode += 1         
        else:            
            height = height + 1
        if isinstance(value, dict):
            countTreeNodes(value, height, key)
        elif key in [0,1]:
             leafNode = leafNode + 1
    return (((countNode+2)/2) + leafNode), leafNode #as each node is counted twice and root is not counted.
 

# Function Name : calculateParentLabel
# Input :
# Output : Value of the label

def calculateParentLabel():
    value = 0
    if random.random() < 0.6:
        value = 0.0
    else:
        value = 1.0
    return value

# Function Name : pruneTree
# Input :
#	decisionTree : Tree to be pruned
#	field  : Node to be pruned
#       countOfLabels : Count of label
# Output :
#       decisionTree : Pruned Tree

def pruneTree(decisionTree, field, countOfLabel):
    global pruningCount
    global count
    for key, value in decisionTree.items():
        if key == field:
            if pruningCount == 1 and count == countOfLabel:
                pruningCount += 1
                decisionTree.pop(key, None) # delete the node
                break
            else:
                count = count + 1
                pruneTree(value, field, countOfLabel)
        elif isinstance(value, dict):
            pruneTree(value,field, countOfLabel)
    return decisionTree

# Function Name : assignLabel
# Input : 
#	decisionTree : Tree from empty values needs to be removed
#   keyParent  : Parent Node
# Output : DecisionTree with classLabel asisgned to parent post pruning

def assignLabel(decisionTree,keyParent):
    for key,value in decisionTree.items():
        if isinstance(value,dict):
            keyParent = key
            value = assignLabel(value,keyParent)
        if value == {}:
            del decisionTree[key]
            parentLabel = calculateParentLabel()
            decisionTree[keyParent] = parentLabel
    return decisionTree

#Function Name: traverseTree
#Input:
#       decTree     : decision tree
#	labelsList  : list of labels
#       rowlist     : list of each record
#Output:
#	prediction   : class value

def traverseTree(decTree, labelsList, rowList):
    if not (decTree == {}) :
        i = 0
        for key in decTree:
            if i == 0:
                value = key            
                i = i + 1
        subTree = decTree[value]    
        getIndex = labelsList.index(value)   
        key = rowList[getIndex]   
        subDecTree = subTree[key]    
        if isinstance(subDecTree, dict):        
            prediction = traverseTree(subDecTree, labelsList, rowList)
        else:        
            prediction = subDecTree
        return prediction
    else:
        return 0

#Function Name: calculateAccuracy
#Input:
#       list1, list2     : lists among which accuracy is calculated 
#Output:
#	accu   : accuracy

def calculateAccuracy(list1,list2):
    len1 = len(list1)    
    len2 = len(list2)
    count = 0
    if len1 != len2:
        print("cannot check accuracy")
        return 0
    else:
        for i in range(0,len1):
            if list1[i] == list2[i]:
                count += 1
        accu = (count/len1)*100
    return accu

		
    
#Function Name : main
#Input : 
#	trainingSet 	: The data to train the model
#	testSet     	: Data Set used to find accuracy
#	validationSet	: To be used during pruning
#	pruningFactor	: This will give a count of number of node sto eb pruned
#Output : None

def main(trainingSet, testSet, validationSet, pruningFactor):
    trainDataFrame,trainAttributes = createDataSet(trainingSet)
    trainDataSet = trainDataFrame.values.tolist() #Convert DataFrame values toList
    trainAttributesList = list(trainAttributes)

    testDataFrame,testAttributes = createDataSet(testSet)
    testDataSet = testDataFrame.values.tolist() #Convert DataFrame values toList
    testAttributesList = list(testAttributes)

    validationDataFrame,validationAttributes = createDataSet(validationSet)
    validationDataSet = validationDataFrame.values.tolist() #Convert DataFrame values toList
    validationAttributesList = list(validationAttributes)
    
    decisionTree = createDecisionTree(trainDataSet, trainAttributesList)
    
    print("Tree before pruning")
    printDecisionTree(decisionTree, 0, None)
    
    totalTreeNodes, leafNodes = countTreeNodes(decisionTree, 0, None)
    trainAttributesList = list(trainAttributes)

    
    print("Pre-Pruned Accuracy")
    print("-------------------------------------")
    print("Number of training instances = ", len(trainDataSet))
    print("Number of training attributes = ",len(trainAttributesList))
    print("Total number of nodes in the tree = ",totalTreeNodes)
    print("Number of leaf nodes in the tree = ", leafNodes)
    trainClassList = [column[-1] for column in trainDataSet]    
    trainClassListOther = []
    for l in trainDataSet:        
        n = traverseTree(decisionTree,trainAttributesList,l)
        trainClassListOther.append(n)
    trainAccu = calculateAccuracy(trainClassList,trainClassListOther)
    print("Accuracy of the model on the training dataset = ", trainAccu ,"%")
    print(" ")
    print(" ")    

    print("Number of validation instances",len(validationDataSet))
    print("Number of validation attributes", len(validationAttributesList))
    validationClassList = [column[-1] for column in validationDataSet]    
    validationClassListOther = []
    for l in validationDataSet:        
        n = traverseTree(decisionTree,validationAttributesList,l)
        validationClassListOther.append(n)
    validationAccu = calculateAccuracy(validationClassList,validationClassListOther)
    print("Accuracy of the model on the validation dataset", validationAccu, "%")
    print(" ")
    print(" ")

    print("Number of testing instances",len(testDataSet))
    print("Number of testing attributes", len(testAttributesList))
    testClassList = [column[-1] for column in testDataSet]    
    testClassListOther = []
    for l in testDataSet:        
        n = traverseTree(decisionTree,testAttributesList,l)
        testClassListOther.append(n)
    testAccu = calculateAccuracy(testClassList,testClassListOther)
    print("Accuracy of the model on the testing dataset", testAccu, "%")
    print(" ")
    print(" ")

    nodesToPrune = int(pruningFactor * totalTreeNodes)
    count = 1
    root = list(decisionTree.keys())[0]
   
    prunedTreeLabels = decisionTree.copy()
    while (count <= nodesToPrune): 
        randIndex = randrange(0, len(trainAttributes)-1)
        label = trainAttributes[randIndex]
        if label != root: #Prune only if label is not root node
            setGlobalVar()
            strTree = str(decisionTree)
            countOfLabel = strTree.count(label)
            prunedTree = pruneTree(decisionTree,label,countOfLabel)
            treeTemp = prunedTree.copy()
            prunedTreeLabels = assignLabel(treeTemp,root)           
        count = count + 1
        
    print("Tree after pruning")
    printDecisionTree(prunedTreeLabels, 0, None) 
        
 
    totalNodecount,totalLeafCount = countTreeNodes(prunedTreeLabels, 0, None)

    
    trainDataSet = trainDataFrame.values.tolist() #Convert DataFrame values toList
    print(" ")
    print("Post-Pruned Accuracy")
    print("-------------------------------------")
    print("Number of training instances = ", len(trainDataSet))
    print("Number of training attributes = ",len(trainAttributesList))
    print("Total number of nodes in the tree = ",(totalNodecount-totalTreeNodes))
    print("Number of leaf nodes in the tree = ", (totalLeafCount-leafNodes))
    trainClassList = [column[-1] for column in trainDataSet]    
    trainClassListOther = []
    for l in trainDataSet:        
        n = traverseTree(prunedTreeLabels,trainAttributesList,l)
        trainClassListOther.append(n)
    prunedTrainAccu = calculateAccuracy(trainClassList,trainClassListOther)
    print("Accuracy of the model on the training dataset = ", prunedTrainAccu ,"%")
    print(" ")
    print(" ")  

    print("Number of validation instances",len(validationDataSet))
    print("Number of validation attributes", len(validationAttributesList))
    validationClassList = [column[-1] for column in validationDataSet]      
    validationClassListOther = []
    for l in validationDataSet:        
        n = traverseTree(prunedTreeLabels,validationAttributesList,l)
        validationClassListOther.append(n)
    PrunedValidationAccu = calculateAccuracy(validationClassList,validationClassListOther)
    print("Accuracy of the model on the validation dataset ", PrunedValidationAccu, "%")
    print(" ")
    print(" ")

    print("Number of testing instances",len(testDataSet))
    print("Number of testing attributes", len(testAttributesList))
    testClassList = [column[-1] for column in testDataSet]    
    testClassListOther = []
    for l in testDataSet:        
        n = traverseTree(prunedTreeLabels,testAttributesList,l)
        testClassListOther.append(n)
    prunedTestAccu = calculateAccuracy(testClassList,testClassListOther)
    print("Accuracy of the model on the testing dataset", prunedTestAccu, "%")
    print(" ")
    print(" ")


#Function to read input from the command line and call main function

if __name__ == "__main__":
	
	if (len(sys.argv)) < 5 :   #Print error if number of arguments is less than 6
		print("Invalid number of arguments")
	else:
		trainingSet = sys.argv[1]
		testSet = sys.argv[2]
		validationSet = sys.argv[3]
		pruningFactor = float(sys.argv[4]) 		
		main(trainingSet, testSet, validationSet, pruningFactor) # Main function call
		
