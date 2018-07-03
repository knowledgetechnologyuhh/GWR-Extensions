# -*- coding: utf-8 -*-
"""
Demo with Associative GWR

@last-modified: 2 July 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Weber, C., Wermter, S. (2015) Self-Organizing Neural Integration of Pose-Motion Features for Human Action Recognition. Frontiers in Neurorobotics, 9(3).
"""

import csv
from agwr import AssociativeGWR
import numpy as np
import matplotlib.pyplot as plt
import cPickle

# Main ########################################################################

if __name__ == "__main__":
    # Set working path
    #os.getcwd()
    dataFlag = 1         # Import dataset from file
    importFlag = 0       # Import saved network
    trainFlag = 1        # Train AGWR with imported dataset
    saveFlag = 0         # Save trained network to file
    testFlag = 1         # Compute classification accuracy
    plotFlag = 1         # Plot 2D map
    
    if (dataFlag):
        # Load data set
        reader = csv.reader(open("iris.csv","rU"),delimiter=',')
        x = list(reader)
        dataSet = np.array(x).astype('float')
        size = dataSet.shape
    
        # Pre-process samples and labels
        labelSet = dataSet[:,size[1]-1]
        dataSet = dataSet[:,0:size[1]-1]
        dimension = len(dataSet[1,:])
    
        # Data normalization
        oDataSet = np.copy(dataSet)
        for i in range(0, size[1]-1):
            maxColumn = max(dataSet[:,i])
            minColumn = min(dataSet[:,i])
            for j in range(0, size[0]):
                oDataSet[j,i] = ( dataSet[j,i] - minColumn ) / ( maxColumn - minColumn )

    if (importFlag):
        """try load self.name.txt"""
        file = open("myAGWR"+'.network','r')
        dataPickle = file.read()
        file.close()
        myAGWR = AssociativeGWR()
        myAGWR.__dict__ = cPickle.loads(dataPickle)

    if (trainFlag):
        initNeurons = 1                 # Weight initialization (0: random, 1: sequential)
        numberOfEpochs = 25             # Number of training epochs
        insertionThreshold = 0.85       # Activation threshold for node insertion
        learningRateBMU = 0.1           # Learning rate of the best-matching unit (BMU)
        learningRateNeighbors = 0.01    # Learning rate of the BMU's topological neighbors
    
        myAGWR = AssociativeGWR()
        myAGWR.initNetwork(oDataSet, labelSet, initNeurons)
        myAGWR.trainAGWR(oDataSet, labelSet, numberOfEpochs, insertionThreshold, learningRateBMU, learningRateNeighbors)

    if (saveFlag):
        file = open("myAGWR"+'.network','w')
        file.write(cPickle.dumps(myAGWR.__dict__))
        file.close()

    if (testFlag):
        bmus, blabels, activations = myAGWR.predictAGWR(oDataSet, myAGWR.weights, myAGWR.alabels)
    
        print "Test accuracy: " + str(myAGWR.computeAccuracy(labelSet,blabels))

    if (plotFlag):
        # Plot network
        # This just plots the first two dimensions of the weight vectors.
        # For better visualization, PCA over weight vectors must be performed.
        classLabels = 1
        ccc = ['black','blue','red'] #'green','yellow','cyan','magenta','0.75','0.15','1'
        fig = plt.figure()
        for ni in range(len(myAGWR.weights)):
            plindex = np.argmax(myAGWR.alabels[ni])
            if (classLabels):
                plt.scatter(myAGWR.weights[ni,0], myAGWR.weights[ni,1], color=ccc[plindex])
            else:
                plt.scatter(myAGWR.weights[ni,0], myAGWR.weights[ni,1])
            for nj in range(len(myAGWR.weights)):
                if (myAGWR.edges[ni,nj]>0):
                    plt.plot([myAGWR.weights[ni,0], myAGWR.weights[nj,0]], [myAGWR.weights[ni,1], myAGWR.weights[nj,1]], 'gray', alpha=.3)
        plt.show()
